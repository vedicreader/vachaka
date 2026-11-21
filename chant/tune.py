"Stage 5: the tune model. A metre's shape and its syllables in; a tune code per pāda and each syllable's duration and pitch out."
import math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fastcore.utils import *
from score import SVARA

NUC = 'a ā i ī u ū ṛ ṝ ḷ e ai o au'.split()
CODA = {'ṃ': 1, 'ḥ': 2}
MAXP = 6                                           # pādas per verse the model knows (paṅkti has 5)
SIZES = (2, MAXP, MAXP, 9, 32, 2, 2, 64, len(NUC), 4, 4, 48, len(SVARA))
MI = 7                                             # the metre-name column, hidden at random in training

def meter_vocab(recs): return ['', *sorted({r['meter'] for r in recs} - {''})]

def feats(rows, meter, mv):
    "Thirteen integers per syllable: the metre by its shape and its name; place in eighths of the pāda, so an unheard pāda length still has a known place."
    nuc = lambda r: next((NUC.index(u) for u in r['ph'] if u in NUC), 0)
    ons = lambda r: min(3, next((i for i,u in enumerate(r['ph']) if u in NUC), 0))
    cod = lambda r: CODA.get(r['ph'][-1], 3 if r['ph'][-1] not in NUC else 0)
    c = lambda v, n: min(v, n - 1)
    f = [[r['heavy'], c(r['pada_end'], MAXP), c(r['pada'], MAXP), round(8*r['pos']/max(1, r['q'] - 1)), c(r['pos_end'], 32), r['end'], r['wstart'],
          mv.index(meter) if meter in mv else 0, nuc(r), ons(r), cod(r), c(r['mora'], 48), SVARA.index(r.get('sv', ''))] for r in rows]
    return torch.tensor(f, dtype=torch.long)

# --- tune codes: a reciter sings each metre to a few tunes; regression on all of them gives their flat average ---------------
def contour(st, n=8):
    "One pāda's pitch resampled to `n` points, gaps filled; None if fewer than half its syllables are voiced."
    st = np.asarray(st, float); ok = ~np.isnan(st)
    if ok.sum() < max(2, len(st)/2): return None
    st = np.interp(np.arange(len(st)), np.flatnonzero(ok), st[ok])
    return np.interp(np.linspace(0, len(st) - 1, n), np.arange(len(st)), st)

def kmeans(X, k, iters=50, seed=0):
    "Plain k-means; returns the centres and each row's code."
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None])**2).sum(-1).argmin(1)
        C = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return C, a

def pada_codes(vs, k=6, seed=0):
    "Cluster every pāda contour of `vs` into `k` tunes; sets `v['codes']` (one per pāda, -1 if unvoiced) and returns the centres."
    cs = [(i, p, c) for i, v in enumerate(vs) for p in range(v['rows'][0]['npada'])
          if (c := contour([s for r, s in zip(v['rows'], v['st']) if r['pada'] == p])) is not None]
    C, a = kmeans(np.stack([c for *_, c in cs]), k, seed=seed)
    for v in vs: v['codes'] = [-1]*v['rows'][0]['npada']
    for (i, p, _), j in zip(cs, a): vs[i]['codes'][p] = int(j)
    return C

# --- the model -------------------------------------------------------------------------------------------------------------
class Tune(nn.Module):
    "A transformer over one verse's syllables, a head choosing a tune per pāda, and a head giving each syllable's log-ms and semitones."
    def __init__(self, k=1, d=128, layers=4, heads=4, maxlen=128):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n, d) for n in SIZES)
        self.pos = nn.Embedding(maxlen, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, 0.1, batch_first=True, norm_first=True), layers)
        self.code_emb, self.code_head = nn.Embedding(k, d), nn.Linear(d, k)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def encode(self, x, pad):
        h = sum(e(x[..., i]) for i,e in enumerate(self.emb)) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.enc(h, src_key_padding_mask=pad)

    def code_logits(self, h, x, pad):
        "Logits over tunes for each pāda, from the mean of its syllables: [B, MAXP, k]."
        P = F.one_hot(x[..., 2], MAXP).float() * (~pad)[..., None]
        return self.code_head(P.transpose(1, 2) @ h / P.sum(1)[..., None].clamp(min=1))

    def forward(self, x, pad, code):
        "`code` [B, MAXP]: the tune of each pāda (the truth in training, a choice when chanting)."
        h = self.encode(x, pad)
        c = torch.gather(code.clamp(0, self.code_emb.num_embeddings - 1), 1, x[..., 2])
        return self.out(h + self.code_emb(c)), self.code_logits(h, x, pad)

# --- training --------------------------------------------------------------------------------------------------------------
def item(v, mv):
    "One training item from a verse that stages 3 and 4 filled: rows with `ms`, `st` per syllable, `codes` per pāda (from `pada_codes`)."
    rows = v['rows']
    return (feats(rows, v['meter'], mv), torch.tensor([math.log(max(30, r['ms'][1]-r['ms'][0])) for r in rows]),
            torch.tensor(v['st'], dtype=torch.float), torch.tensor((v.get('codes') or [0]*rows[0]['npada'])[:MAXP]))

def batch(items, dev, drop=0.0):
    "Pad items to one batch; with probability `drop` a verse's metre name is hidden, so the model also learns metres from shape alone."
    n = max(len(f) for f,*_ in items)
    x = torch.zeros(len(items), n, len(SIZES), dtype=torch.long); y = torch.full((len(items), n, 2), float('nan'))
    pad = torch.ones(len(items), n, dtype=torch.bool); code = torch.full((len(items), MAXP), -1, dtype=torch.long)
    for i,(f,d,s,c) in enumerate(items):
        k = len(f); x[i,:k], y[i,:k,0], y[i,:k,1], pad[i,:k], code[i,:len(c)] = f, d, s, False, c
        if drop and random.random() < drop: x[i,:,MI] = 0
    return x.to(dev), y.to(dev), pad.to(dev), code.to(dev)

def loss_fn(m, x, y, pad, code, wc=0.3):
    "Masked MSE on log-ms and semitones (unvoiced syllables skip pitch), plus cross-entropy on the pāda tunes."
    p, lg = m(x, pad, code)
    mk = ~pad[..., None] & ~torch.isnan(y)
    mse = ((p - torch.nan_to_num(y))**2)[mk].mean()
    return mse + wc*F.cross_entropy(lg.flatten(0, 1), code.flatten(), ignore_index=-1) if m.code_head.out_features > 1 else mse

def device(): return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def fit(items, k=1, epochs=20, bs=64, lr=3e-4, drop=0.3, dev=None, val=0.05, seed=0, quiet=False):
    "Train on `items`, holding out `val`; `k` tunes per pāda (1 = plain regression). Returns the model and the loss history."
    dev = dev or device(); random.seed(seed); torch.manual_seed(seed); items = items[:]; random.shuffle(items)
    nv = max(1, int(len(items)*val)); va, tr = items[:nv], items[nv:]
    m = Tune(k).to(dev); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs*math.ceil(len(tr)/bs))
    hist = []
    for ep in range(epochs):
        m.train(); random.shuffle(tr)
        for i in range(0, len(tr), bs):
            l = loss_fn(m, *batch(tr[i:i+bs], dev, drop))
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        hist.append((ep, l.item(), evaluate(m, va, dev)))
        if not quiet: print(*hist[-1])
    return m, hist

@torch.no_grad()
def evaluate(m, items, dev, bs=256):
    m.eval()
    return float(np.mean([loss_fn(m, *batch(items[i:i+bs], dev)).item() for i in range(0, len(items), bs)]))

@torch.no_grad()
def predict(m, rows, meter, mv, codes=None, temp=1.0, seed=None):
    "Each syllable's ms and semitones. `codes`: a tune per pāda; None samples them from the model at `temp` (0 = most likely)."
    m.eval(); dev = next(m.parameters()).device
    x, pad = feats(rows, meter, mv)[None].to(dev), torch.zeros(1, len(rows), dtype=torch.bool, device=dev)
    if codes is None:
        lg = m.code_logits(m.encode(x, pad), x, pad)[0]
        g = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        codes = lg.argmax(-1) if temp == 0 else torch.multinomial((lg/temp).softmax(-1).cpu(), 1, generator=g)[:, 0]
    code = torch.as_tensor(codes, dtype=torch.long, device=dev).reshape(1, -1)
    code = F.pad(code, (0, MAXP - code.shape[1]))
    p = m(x, pad, code)[0][0].cpu()
    return p[:, 0].exp().tolist(), p[:, 1].tolist(), code[0, :rows[0]['npada']].tolist()
