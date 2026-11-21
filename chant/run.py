"The pipeline on the Mac: verses, clips, timings, pitch, the tune model, and the voice model's training list."
import json, csv, math, numpy as np, soundfile as sf, librosa, torch
from pathlib import Path
from itertools import groupby
from fastcore.utils import *
from data import scan_titles, fetch, cut
from score import verse_rows
from align import Aligner, ok, SR as ASR
from pitch import world, tonic, syl_pitch, svara_check, FP
import tune

def prep_one(rec, src, out, ali, sr=24000):
    "Clip one verse, place its phones, take its f0; returns the record with rows and f0 path, or None if it fails the checks."
    m, rows = verse_rows(rec['text'], vedic=rec.get('vedic', False))
    if not rows: return None
    wav = out/'wav'/f"{rec['id']}.wav"; wav.parent.mkdir(parents=True, exist_ok=True)
    cut(src, wav, rec['s'], rec['e'], sr)
    y, _ = sf.read(wav)
    rows, score = ali(librosa.resample(y, orig_sr=sr, target_sr=ASR), rows)
    if not ok(rows, score): return None
    f0, *_ = world(y, sr); fp = out/'f0'/f"{rec['id']}.npy"; fp.parent.mkdir(exist_ok=True); np.save(fp, f0)
    return dict(rec, rows=rows, align=score, f0=str(fp), wav=str(wav))

def prep(root, out, limit=None, dev='cpu', **scan):
    "Stages 2 to 4 for every permitted verse, one recording at a time; the tonic is per recording. `scan` goes to `scan_titles`. Writes verses.jsonl."
    out, ali, n = Path(out), Aligner(dev), 0
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'verses.jsonl', 'a') as fh:
        for url, rs in groupby(scan_titles(root, **scan), key=lambda r: r['audio']):
            src = fetch(url, out/'mp3'/url.rsplit('/', 1)[-1])
            vs = [v for r in rs if (v := prep_one(r, src, out, ali))]
            if not vs: continue
            ton = tonic([np.load(v['f0']) for v in vs])
            for v in vs: fh.write(json.dumps(dict(v, tonic=ton, st=syl_pitch(np.load(v['f0']), v['rows'], ton)), ensure_ascii=False) + '\n')
            n += len(vs); print(url.rsplit('/', 1)[-1], len(vs), 'kept', n, 'in all')
            if limit and n >= limit: break

def train(out, epochs=20, meters=None, k=6, vedic=False):
    "Stage 5 on verses.jsonl: `k` tunes per pāda, `meters` keeps those metres, `vedic` adds the Veda verses. Saves tune.pt."
    vs = [json.loads(l) for l in open(Path(out)/'verses.jsonl')]
    vs = [v for v in vs if (not meters or v['meter'] in meters) and (vedic or not v.get('vedic'))]
    if vedic: print('svara check (udātta above anudātta):', svara_check(vs))
    mv, C = tune.meter_vocab(vs), tune.pada_codes(vs, k)
    m, hist = tune.fit([tune.item(v, mv) for v in vs], k=k, epochs=epochs)
    torch.save(dict(state=m.state_dict(), mv=mv, k=k, centres=C, hist=hist), Path(out)/'tune.pt')
    return m, mv

def export(out, name='transcriptions.csv'):
    "Stage 7's training list: one row per verse, its phones (SP for silence) and their seconds, beside the wav."
    out = Path(out)
    with open(out/name, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['name', 'ph_seq', 'ph_dur'])
        for v in map(json.loads, open(out/'verses.jsonl')):
            seq, dur, t = [], [], 0
            for u, s, e in (p for r in v['rows'] for p in r['phs']):
                if s > t: seq.append('SP'); dur.append((s - t)/1000)
                seq.append(u); dur.append((e - s)/1000); t = e
            seq.append('SP'); dur.append(max(0.05, sf.info(v['wav']).duration - t/1000))
            w.writerow([Path(v['wav']).stem, ' '.join(seq), ' '.join(f'{d:.3f}' for d in dur)])

if __name__ == '__main__':
    import sys
    cmd, *a = sys.argv[1:]
    dict(prep=lambda root, out, n=None: prep(root, out, n and int(n), tune.device() if tune.device() != 'mps' else 'cpu'),
         train=lambda out, ep=20: train(out, int(ep), {'anuṣṭubh'}), export=export)[cmd](*a)
