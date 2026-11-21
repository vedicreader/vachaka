"Stage 1: a verse to a chant score: syllables, weights, pāda places, phones, beats and notes."
import json, unicodedata
from operator import itemgetter
from fastcore.utils import *
from ganapati.metre import syllables, detect_meter, metrical_text, deva2iast, _phones

def units(s): return [u for _,u in _phones(s)]
def nvow(w): return sum(k == 'V' for k,_ in _phones(w))

def word_of(words):
    "Word index for each syllable: a syllable belongs to the word that holds its vowel."
    return [i for i,w in enumerate(words) for _ in range(nvow(w))]

def verse_rows(text):
    "One row per syllable of a verse: syllable, weight, pāda, place in pāda, word start, phones."
    t = metrical_text(text)
    sy, m = syllables(t), detect_meter(t)
    if not sy: return None, []
    q = (m.per_pada if m else None) or max(1, round(len(sy)/4))
    ws = t.split()
    wi = word_of(ws)
    if len(wi) != len(sy): ws, wi = [s for s,_ in sy], list(range(len(sy)))   # a word scanned differently: one word per syllable
    rows = [dict(syl=s, heavy=int(h), pada=min(i//q, 3), pos=i%q, q=q, end=int(i%q == q-1),
                 wstart=int(i == 0 or wi[i] != wi[i-1]), word=wi[i], wtext=deva2iast(ws[wi[i]]), ph=units(s)) for i,(s,h) in enumerate(sy)]
    return m, rows

# The flat baseline tune. A learned tune (tune.py) replaces it; keep it to hear what the model adds.
def rule_tune(rows, tonic=57, beat_ms=230):
    "Beats and MIDI notes by rule: laghu 1 beat, guru 2, pāda end held, half-verse end falls a tone."
    out = []
    for r in rows:
        b = 1 + r['heavy'] + r['end']*(1 + (r['pada'] % 2))
        n = tonic - 2*(r['end'] and r['pada'] % 2 == 1)
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
            if r['end'] and r['pada'] % 2: notes.append(['-', 'rest', 1])
    return dict(text=' '.join(roman(rs[0]['wtext']) for rs in words.values()), bpm=bpm, style=style, notes=notes)

if __name__ == '__main__':
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else 'तपःस्वाध्यायनिरतं तपस्वी वाग्विदां वरम् । नारदं परिपप्रच्छ वाल्मीकिर्मुनिपुङ्गवम् ॥ १ ॥'
    m, rows = verse_rows(v)
    print(m.name, m.variant, m.scan)
    for r in rule_tune(rows)[:10]: print(r['syl'], r['heavy'], r['pada'], r['pos'], r['wstart'], r['ph'], r['beats'], r['midi'])
    print(json.dumps(scylla_score(rule_tune(rows)), ensure_ascii=False)[:400])
