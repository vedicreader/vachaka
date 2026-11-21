"Stage 2: vedicreader titles to timed verses, then one clean mono clip per verse."
import json, subprocess, urllib.request
from pathlib import Path
from fastcore.utils import *
from ganapati.metre import detect_meter, metrical_text

def timed(l): return 's' in l and 'e' in l and l.get('role', 'verse') in ('verse', 'audio')

def verses(doc):
    "Timed lines of a title grouped into verses: a verse ends at a line with `॥`."
    out, cur = [], []
    for sec in doc.get('sections', []):
        for l in filter(timed, sec.get('lines', [])):
            cur.append(l)
            if '॥' in l['t']: out.append(cur); cur = []
    return out

def verse_rec(doc, ls, i):
    "One record per verse: id, audio, span in ms, lines and the metre ganapati finds."
    t = ' '.join(l['t'] for l in ls)
    m = detect_meter(metrical_text(t))
    return dict(id=f"{doc['name']}_{i:03d}", audio=doc['audio'], s=ls[0]['s'], e=ls[-1]['e'], text=t,
                lines=[dict(t=l['t'], s=l['s'], e=l['e'], w=l.get('w', [])) for l in ls],
                meter=str(m.name.value if m and m.name else ''), variant=(m.variant if m else None) or '')

def scan_titles(root, glob='itihasa/valmiki/*/*.json', credit='Ghanapāṭhī V. Śrīrāma'):
    "Every timed verse of the titles under `root` whose credit names the permitted recording."
    for p in sorted(Path(root).glob(glob)):
        d = json.loads(p.read_text())
        if isinstance(d.get('audio'), str) and credit in d.get('credit', ''):
            yield from (verse_rec(d, ls, i) for i,ls in enumerate(verses(d)))

def fetch(url, dst):
    "Download a recording once."
    dst = Path(dst)
    if not dst.exists(): dst.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(url, dst)
    return dst

def cut(src, dst, s, e, sr=24000, pad=150):
    "One verse as mono wav at `sr`, with `pad` ms either side so no consonant is lost."
    a = max(0, s - pad)/1000
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{a:.3f}', '-to', f'{(e+pad)/1000:.3f}', '-i', str(src),
                    '-ac', '1', '-ar', str(sr), str(dst)], check=True)
    return a*1000                                    # the clip's offset in the recording, in ms

if __name__ == '__main__':
    import sys, collections
    recs = list(scan_titles(sys.argv[1] if len(sys.argv) > 1 else '../../vedicreader/static/vedic_texts'))
    hrs = sum(r['e']-r['s'] for r in recs)/3.6e6
    print(len(recs), 'verses', round(hrs, 1), 'h')
    for k,v in collections.Counter(r['meter'] or '-' for r in recs).most_common(8): print(f'{v:6d}  {k}')
    print(json.dumps(recs[5], ensure_ascii=False)[:300])
