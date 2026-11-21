"Stage 4: pitch per syllable in semitones from the reciter's tonic, and the ceiling test."
import numpy as np, pyworld as pw
from fastcore.utils import *

FP = 5.0                                            # WORLD frame period, ms

def world(y, sr):
    "WORLD analysis: f0 (0 where unvoiced), spectral envelope, aperiodicity, frame times."
    x = np.asarray(y, np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=65, f0_ceil=500, frame_period=FP)
    return f0, pw.cheaptrick(x, f0, t, sr), pw.d4c(x, f0, t, sr), t

def tonic(f0s):
    "The reciter's tonic in Hz: the median of every voiced frame of a recording."
    v = np.concatenate([f[f > 0] for f in f0s])
    return float(np.exp(np.median(np.log(v))))

def semis(hz, ton): return 12*np.log2(hz/ton)

def syl_pitch(f0, rows, ton, core=(0.35, 0.9)):
    "Median semitones of each syllable over the latter part of its span (the vowel), or nan if unvoiced."
    out = []
    for r in rows:
        a, b = r['ms']; a, b = a + core[0]*(b-a), a + core[1]*(b-a)
        f = f0[int(a/FP):max(int(a/FP)+1, int(b/FP))]
        out.append(float(np.median(semis(f[f > 0], ton))) if (f > 0).any() else float('nan'))
    return out

def score_f0(f0, rows, ton, notes=None, glide=6):
    "A score's f0: one flat note per syllable (rounded semitones unless `notes` given), joined by a `glide`-frame slide."
    notes = notes if notes is not None else [round(s) if s == s else 0 for s in syl_pitch(f0, rows, ton)]
    lf = np.zeros(len(f0))
    for r, n in zip(rows, notes): lf[int(r['ms'][0]/FP):int(r['ms'][1]/FP)] = np.log(ton) + n*np.log(2)/12
    lf[lf == 0] = np.log(ton)
    lf = np.convolve(np.pad(lf, glide, mode='edge'), np.ones(glide)/glide, 'same')[glide:-glide]
    return np.where(f0 > 0, np.exp(lf), 0.0)

def ceiling(y, sr, rows, ton=None):
    "Three versions of one clip: WORLD's copy, the copy on its own score's notes, and the tonic used."
    f0, sp, ap, _ = world(y, sr)
    ton = ton or tonic([f0])
    return pw.synthesize(f0, sp, ap, sr, FP), pw.synthesize(score_f0(f0, rows, ton), sp, ap, sr, FP), ton
