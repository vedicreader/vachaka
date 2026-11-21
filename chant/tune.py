"Stage 5: the tune model. Metre and syllables in; each syllable's duration and pitch out."
import json, math, random, numpy as np, torch, torch.nn as nn
from fastcore.utils import *

NUC = 'a ā i ī u ū ṛ ṝ ḷ e ai o au'.split()
CODA = {'ṃ': 1, 'ḥ': 2}

def meter_vocab(recs): return ['', *sorted({r['meter'] for r in recs} - {''})]

def feats(rows, meter, mv):
    "Eight integer features per syllable: weight, pāda, place, pāda length, pāda end, word start, metre, nucleus, onset size, coda."
    nuc = lambda r: next((NUC.index(u) for u in r['ph'] if u in NUC), 0)
    ons = lambda r: min(3, next((i for i,u in enumerate(r['ph']) if u in NUC), 0))
    cod = lambda r: CODA.get(r['ph'][-1], 3 if r['ph'][-1] not in NUC else 0)
    f = [[r['heavy'], r['pada'], min(r['pos'], 31), min(r['q'], 31), r['end'], r['wstart'],
          mv.index(meter) if meter in mv else 0, nuc(r), ons(r), cod(r)] for r in rows]
    return torch.tensor(f, dtype=torch.long)

SIZES = (2, 4, 32, 32, 2, 2, 64, len(NUC), 4, 4)

class Tune(nn.Module):
    "A small transformer over one verse's syllables: about 0.8M parameters at d=128."
    def __init__(self, d=128, layers=4, heads=4, maxlen=128):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n, d) for n in SIZES)
        self.pos = nn.Embedding(maxlen, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, 0.1, batch_first=True, norm_first=True), layers)
        self.head = nn.Linear(d, 2)                  # log duration (ms), semitones from the tonic

    def forward(self, x, pad):
        h = sum(e(x[..., i]) for i,e in enumerate(self.emb)) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.enc(h, src_key_padding_mask=pad))

def batch(items, dev):
    "Pad a list of (features, log-dur, semitones) to one batch with masks; nan semitones (unvoiced) are masked."
    n = max(len(f) for f,_,_ in items)
    x = torch.zeros(len(items), n, len(SIZES), dtype=torch.long); y = torch.full((len(items), n, 2), float('nan'))
    pad = torch.ones(len(items), n, dtype=torch.bool)
    for i,(f,d,s) in enumerate(items): k = len(f); x[i,:k], y[i,:k,0], y[i,:k,1], pad[i,:k] = f, d, s, False
    return x.to(dev), y.to(dev), pad.to(dev)

def loss_fn(p, y, pad):
    "Masked mean squared error, duration and pitch weighted alike (log-ms and semitones have similar spreads)."
    m = ~pad[..., None] & ~torch.isnan(y)
    return ((p - torch.nan_to_num(y))**2)[m].mean()

def device(): return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def fit(items, epochs=20, bs=64, lr=3e-4, dev=None, val=0.05, seed=0):
    "Train on `items` [(features, log-dur, semitones)], holding out `val` of them; returns the model and the loss history."
    dev = dev or device(); random.Random(seed).shuffle(items)
    k = max(1, int(len(items)*val)); va, tr = items[:k], items[k:]
    m = Tune().to(dev); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs*math.ceil(len(tr)/bs))
    hist = []
    for ep in range(epochs):
        m.train(); random.shuffle(tr)
        for i in range(0, len(tr), bs):
            x, y, pad = batch(tr[i:i+bs], dev); l = loss_fn(m(x, pad), y, pad)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        hist.append((ep, l.item(), evaluate(m, va, dev))); print(*hist[-1])
    return m, hist

@torch.no_grad()
def evaluate(m, items, dev, bs=256):
    m.eval()
    return float(np.mean([loss_fn(m(*(lambda x,y,p: (x,p))(*b)), b[1], b[2]).item() for b in (batch(items[i:i+bs], dev) for i in range(0, len(items), bs))]))

@torch.no_grad()
def predict(m, rows, meter, mv):
    "Each syllable's duration in ms and pitch in semitones from the tonic."
    m.eval(); dev = next(m.parameters()).device
    p = m(feats(rows, meter, mv)[None].to(dev), torch.zeros(1, len(rows), dtype=torch.bool, device=dev))[0].cpu()
    return p[:, 0].exp().tolist(), p[:, 1].tolist()

def item(v, mv):
    "One training item from a verse record that stages 3 and 4 filled: rows with `ms`, and `st` per syllable."
    rows = v['rows']
    return feats(rows, v['meter'], mv), torch.tensor([math.log(max(30, r['ms'][1]-r['ms'][0])) for r in rows]), torch.tensor(v['st'])
