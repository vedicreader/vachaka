"Stage 6: hear a tune before any voice is trained, through scyllasband's singing flow with our own phones and timing."
import numpy as np
from score import brk
from fastcore.utils import *

# Sanskrit phone -> scyllasband's 87 IPA tokens. It has no retroflex row and no aspiration, so those are approximate:
# ṭ ḍ ṇ fall to t d n, ṣ to ʃ, an aspirate to its stop plus h. Combining marks (ː, ̪) own no frames.
SCY = dict(a=['ɐ'], ā=['a','ː'], i=['i'], ī=['i','ː'], u=['u'], ū=['u','ː'], ṛ=['ɾ','i'], ṝ=['ɾ','i','ː'], ḷ=['l','i'],
           e=['e','ː'], ai=['a','i'], o=['o','ː'], au=['a','u'],
           k=['k'], kh=['k','h'], g=['ɡ'], gh=['ɡ','h'], ṅ=['ŋ'], c=['c'], ch=['c','h'], j=['ɟ'], jh=['ɟ','h'], ñ=['ɲ'],
           ṭ=['t'], ṭh=['t','h'], ḍ=['d'], ḍh=['d','h'], ṇ=['n'], t=['t','̪'], th=['t','̪','h'], d=['d','̪'], dh=['d','̪','h'], n=['n'],
           p=['p'], ph=['p','h'], b=['b'], bh=['b','h'], m=['m'], y=['j'], r=['ɾ'], l=['l'], v=['v'],
           ś=['ç'], ṣ=['ʃ'], s=['s'], h=['h'], ḻ=['l'], ṃ=['m'], ḥ=['h'])
VOW = frozenset('aeiouɐ')
MARK = frozenset('ː̪')
HOP_MS = 512/24000*1000                              # one latent frame, 21.3 ms

def plan(rows, ms, midi, cons=4):
    "Phones, frames per phone and the melody [T, 3] for one phrase: a syllable's consonants take `cons` frames each, its vowel the rest."
    ph, dur, mel, starts = ['<sil>'], [0], [], []
    for r, d, n in zip(rows, ms, midi):
        if r['wstart'] and len(ph) > 1: ph.append('<sil>'); dur.append(0); starts.append(len(ph))
        toks = [t for u in r['ph'] for t in SCY.get(u, [])]
        f = max(len(toks), round(d/HOP_MS)); nc = sum(t not in VOW and t not in MARK for t in toks)
        c = min(cons, max(1, (f//2)//max(1, nc))); nv = sum(t in VOW for t in toks)
        left = f - c*nc
        for t in toks:
            ph.append(t); dur.append(0 if t in MARK else c if t not in VOW else left//nv + (left % nv if t == next(x for x in toks if x in VOW) else 0))
        mel += [[(n - 60)/24, k/f, 1.0] for k in range(f)]
    ph.append('<sil>'); dur.append(4); mel += [[0, 0, 0]]*4
    return ph, starts, dur, np.array(mel, np.float32)

def render(bundle, rows, ms, midi, voice='orpheus', language='it', seed=7, decoder='bigvgan'):
    "One phrase as 24 kHz audio. `bundle`: a downloaded scyllasband bundle dir (python -m scyllasband download --flavor onnx)."
    from scyllasband.engine import Engine, Sentence
    eng = Engine(bundle, backend='onnx')
    ph, st, dur, mel = plan(rows, ms, midi)
    res = eng.synthesize_sentence(Sentence(phones=ph, word_starts=st), voice=voice, language=language,
                                  delivery=(np.zeros((1, 5), np.float32), np.zeros((1, 5), bool)), steps=4, sampler='euler',
                                  rng=np.random.default_rng(seed), durations=dur, melody=mel, delivery_mode='ballad')
    return eng.decode(res.latents, voice=voice, language=language, decoder=decoder)

def phrases(rows, ms, limit=8500):
    "Split a verse at half-verse ends, where a reciter breathes; a half longer than `limit` ms (scyllasband trained on 9 s) at pāda ends."
    padas, cur = [], []
    for r, d in zip(rows, ms):
        cur.append((r, d))
        if r['end']: padas.append(cur); cur = []
    if cur: padas.append(cur)
    out, ph = [], []
    for p in padas:
        if ph and (sum(d for _, d in ph + p) > limit or brk(ph[-1][0])): out.append(ph); ph = []
        ph = ph + p
    return out + ([ph] if ph else [])
