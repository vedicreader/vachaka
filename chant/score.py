"Stage 1: a verse to a chant score: syllables, weights, pāda places, phones, beats and notes."
import json, unicodedata, bisect
from operator import itemgetter
from fastcore.utils import *
from ganapati.metre import syllables, detect_meter, metrical_text, deva2iast, _phones

def units(s): return [u for _,u in _phones(s)]
def nvow(w): return sum(k == 'V' for k,_ in _phones(w))

def word_of(words):
    "Word index for each syllable: a syllable belongs to the word that holds its vowel."
    return [i for i,w in enumerate(words) for _ in range(nvow(w))]

VEDIC = [(24, 'gāyatrī', 8), (28, 'uṣṇih', 7), (32, 'anuṣṭubh', 8), (36, 'bṛhatī', 9), (40, 'paṅkti', 8), (44, 'triṣṭubh', 11), (48, 'jagatī', 12)]
def vedic_meter(n, tol=2):
    "A Vedic metre by syllable count, `(name, pāda length)`, or None. Uṣṇih (8+8+12) and bṛhatī (8+8+12+8) get a mean pāda length."
    return next(((nm, q) for k, nm, q in VEDIC if abs(n - k) <= tol), None)

_IV, _MV = set('अआइईउऊऋॠऌॡएऐओऔॐ'), set('ािीुूृॄॢॣेैोौ')
_CONS = {chr(c) for c in range(0x915, 0x93A)} | set('ळक़ख़ग़ज़ड़ढ़फ़य़')
_MARK = {'॒': 'A', '॑': 'S', '᳚': 'S'}
SVARA = ['', 'U', 'A', 'S', 'P']

def svaras(text):
    "One svara per syllable from the accent marks: A anudātta (॒), S svarita (॑ ᳚), U udātta (unmarked), P pracaya (unmarked after a svarita)."
    s, mk = unicodedata.normalize('NFC', text), []
    for i, ch in enumerate(s):
        nx = s[i+1:i+3].replace('़', '')[:1]
        if ch in _IV or ch in _MV or (ch in _CONS and nx != '्' and nx not in _MV): mk.append('')
        elif ch in _MARK and mk: mk[-1] = _MARK[ch]
    out = []
    for m in mk: out.append(m or ('P' if out and out[-1] in 'SP' else 'U'))
    return out if any(mk) else None

def verse_rows(text, vedic=False):
    "One row per syllable of a verse: syllable, weight, pāda and place counted from both ends, mātrā place, word start, svara, phones."
    t = metrical_text(text)
    sy, m = syllables(t), detect_meter(t)
    if not sy: return None, []
    vm = vedic_meter(len(sy)) if vedic else None
    q = vm[1] if vm else (m.per_pada if m else None) or max(1, round(len(sy)/4))
    np_ = max(1, -(-len(sy) // q))
    ws = t.split()
    wi = word_of(ws)
    if len(wi) != len(sy): ws, wi = [s for s,_ in sy], list(range(len(sy)))   # a word scanned differently: one word per syllable
    sv = (svaras(t) if vedic else None) or []
    sv = sv if len(sv) == len(sy) else [''] * len(sy)
    n, ws0 = len(sy), [i for i in range(len(sy)) if i == 0 or wi[i] != wi[i-1]]
    cuts = sorted({min(ws0, key=lambda w: abs(w - k*n/np_)) for k in range(1, np_)} - {0}) if vm else [k*q for k in range(1, np_)]
    bounds = [0, *cuts, n]
    rows, mora = [], 0
    for i, (s, h) in enumerate(sy):
        pada = bisect.bisect_right(cuts, i); a, b = bounds[pada], bounds[pada+1]; pos, qn = i - a, b - a
        if pos == 0: mora = 0
        rows.append(dict(syl=s, heavy=int(h), pada=pada, npada=len(bounds)-1, pada_end=len(bounds)-2-pada, pos=pos, q=qn, pos_end=qn-1-pos,
                         end=int(pos == qn - 1), mora=mora, sv=sv[i], wstart=int(i == 0 or wi[i] != wi[i-1]), word=wi[i],
                         wtext=deva2iast(ws[wi[i]]), ph=units(s)))
        mora += 1 + h
    return (AttrDict(name=vm[0], per_pada=q) if vm else m), rows

def brk(r): return r['end'] and (r['pada'] % 2 == 1 or r['pada_end'] == 0)

# The flat baseline tune. A learned tune (tune.py) replaces it; keep it to hear what the model adds.
def rule_tune(rows, tonic=57, beat_ms=230):
    "Beats and MIDI notes by rule: laghu 1 beat, guru 2, pāda end held, a break (half-verse or verse end) falls a tone."
    out = []
    for r in rows:
        b = 1 + r['heavy'] + r['end']*(1 + brk(r))
        n = tonic - 2*brk(r)
        out.append(dict(r, beats=b, ms=b*beat_ms, midi=n))
    return out

def roman(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c))

def scylla_score(rows, bpm=None, beat_ms=230, style='ballad'):
    "A scyllasband score JSON: one note per syllable, labelled `word~n`, rests at half-verse ends."
    bpm, words, notes = bpm or round(60000/beat_ms), groupby(rows, itemgetter('word')), []
    names = 'C C# D D# E F F# G G# A A# B'.split()
    for k, rs in words.items():
        w = roman(rs[0]['wtext'])
        for j,r in enumerate(rs):
            notes.append([w if len(rs) == 1 else f'{w}~{j+1}', f"{names[r['midi']%12]}{r['midi']//12-1}", r['beats']])
            if brk(r): notes.append(['-', 'rest', 1])
    return dict(text=' '.join(roman(rs[0]['wtext']) for rs in words.values()), bpm=bpm, style=style, notes=notes)

if __name__ == '__main__':
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else 'तपःस्वाध्यायनिरतं तपस्वी वाग्विदां वरम् । नारदं परिपप्रच्छ वाल्मीकिर्मुनिपुङ्गवम् ॥ १ ॥'
    m, rows = verse_rows(v)
    print(m.name, m.variant, m.scan)
    for r in rule_tune(rows)[:10]: print(r['syl'], r['heavy'], r['pada'], r['pos'], r['wstart'], r['ph'], r['beats'], r['midi'])
    print(json.dumps(scylla_score(rule_tune(rows)), ensure_ascii=False)[:400])
