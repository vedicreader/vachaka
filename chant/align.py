"Stage 3: phone and syllable timings inside a verse clip, by forced alignment of its romanised letters."
import unicodedata, numpy as np
from fastcore.utils import *

SR = 16000
def letters(u): return ''.join(c for c in unicodedata.normalize('NFD', u.lower()) if 'a' <= c <= 'z')

class Aligner:
    "MMS forced aligner (wav2vec2, 1,100 languages, romanised letters). It knows no Sanskrit, but it only has to place known letters."
    def __init__(self, dev='cpu'):
        from torchaudio.pipelines import MMS_FA as B
        self.m, self.tok, self.ali, self.dev = B.get_model(with_star=False).to(dev).eval(), B.get_tokenizer(), B.get_aligner(), dev

    def emission(self, wav):
        import torch
        with torch.inference_mode(): e, _ = self.m(torch.as_tensor(wav, dtype=torch.float32)[None].to(self.dev))
        return e[0].cpu()

    def __call__(self, wav, rows):
        "`rows` from `score.verse_rows`, `wav` 16 kHz mono; adds `ms=[s, e]` per syllable and `phs=[[unit, s, e]]` per phone."
        em = self.emission(wav)
        spans = self.ali(em, self.tok([letters(''.join(r['ph'])) for r in rows]))
        return place(rows, spans, len(wav)/(SR/1000)/em.shape[0], [sp.score for o in spans for sp in o])

def place(rows, spans, fms, scores=()):
    "Turn letter spans into phone and syllable spans; each runs to the start of the next, so gaps go to the sound before."
    lt = [(sp.start*fms, sp.end*fms) for o in spans for sp in o]
    starts, k = [], 0
    for r in rows:
        for u in r['ph']: starts.append(lt[k][0]); k += len(letters(u))
    ends, out, i = starts[1:] + [lt[-1][1]], [], 0
    for r in rows:
        phs = [[u, round(starts[i+j]), round(ends[i+j])] for j,u in enumerate(r['ph'])]; i += len(phs)
        out.append(dict(r, phs=phs, ms=[phs[0][1], phs[-1][2]]))
    return out, float(np.mean(scores)) if len(scores) else None

def gl_ratio(rows):
    "Mean guru duration over mean laghu duration in one verse; a sung śloka sits near 1.5 to 2, a bad alignment does not."
    d = lambda h: [r['ms'][1]-r['ms'][0] for r in rows if r['heavy'] == h]
    g, l = d(1), d(0)
    return np.mean(g)/np.mean(l) if g and l else None

def ok(rows, score, lo=1.2, min_score=0.5):
    "Keep a verse only if the aligner was sure and the timing agrees with the scansion."
    r = gl_ratio(rows)
    return score is not None and score >= min_score and r is not None and r >= lo
