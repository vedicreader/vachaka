# chant: a Sanskrit chant voice trained on our own recordings

Plain scripts, outside `nbs/`, as in ganapati's `evals/`. They are not part of the package.

## The goal

A small model that chants a laukika verse in the Rāmāyaṇa reciters' voice. It is trained on the recordings that their publication gave us permission to use.
It must sound like a reciter who sings, not like a speech model.

## What the other code bases give us

| Source | What we take | What we leave |
|---|---|---|
| [scyllasband](https://github.com/lowkeytea/scyllasband) | The design: a score goes in (phones, frames per phone, a note per frame), a flow model makes latents, and a vocoder makes audio. See `singing.py` `plan_song`, `frame_melody`, and `engine.py` `synthesize_sentence`. | Its weights. There is no training code. The PyTorch files hold weights, not the classes that use them. The 87 phones have no retroflex row and no aspiration. Its 14 voices are fixed. |
| ganapati | `syllables` (weight per syllable), `_phones` (IAST units), `detect_meter` (metre, pāda length). | Nothing. It is the score's source. |
| vedicreader | 19,489 timed Rāmāyaṇa verses, 55.8 h, from the Ghanapāṭhī recording: 16,138 anuṣṭubh, 452 upajāti, 233 vaṃśastha, 2,486 with no metre. Each line has `s`, `e` and word timings. | Veda and sūkta audio for the tune. It has a different tune. It can help the voice later (see stage 7). |
| vachaka | Su-śrotā (`check.Ear`), to listen back and score each render. | Vāgdhenu. |

## The design: two models and a vocoder

```
verse text ──ganapati──> syllables, weights, pādas
                              │
                     TUNE MODEL (stage 5, 0.8M params)
                              │
              duration and pitch for each syllable  = a score
                              │
                     VOICE MODEL (stage 7) ──> mel ──> vocoder ──> audio
```

- The tune model learns how these reciters sing each metre. It is small and trains on the Mac.
- The voice model learns their voice from the same audio. The score tells it what to sing, so it does not guess.
- One tune model serves all metres: the metre is an input.

## The stages

Each stage has a gate. Do not start the next stage until the gate passes.

### Stage 1: a verse to a score (`score.py`)

```sh
python score.py "तपःस्वाध्यायनिरतं तपस्वी वाग्विदां वरम् । नारदं परिपप्रच्छ वाल्मीकिर्मुनिपुङ्गवम् ॥"
```

- `verse_rows` gives one row per syllable: syllable, heavy, pāda, place in pāda, word start, phones.
- A syllable belongs to the word that holds its vowel. Sanskrit moves consonants across word gaps (`va-ra-mnā`), so we count vowels, not letters.
- `rule_tune` is a flat baseline: laghu 1 beat, guru 2, pāda end held, half-verse end falls one tone. Keep it so you can hear what the learned model adds.
- `scylla_score` writes the score as scyllasband JSON.

Learn: how a score is built from scansion.

### Stage 2: timed verses and clips (`data.py`)

```sh
python data.py ../../vedicreader/static/vedic_texts     # counts verses and metres
```

- `scan_titles` keeps only the titles whose credit names the permitted recording.
- `verses` groups timed lines into verses; a verse ends at `॥`.
- `cut` writes one mono wav per verse, with 150 ms padding at each end.

Learn: your data is already 90% labelled. The word timings came from audio_alignment.

### Stage 3: phone timings (`align.py`)

- The MMS forced aligner (torchaudio) places romanised letters in the audio. It knows no Sanskrit, but it only has to put known letters in their order.
- `place` turns letter spans into phone spans (`dh` = 2 letters) and syllable spans.
- `ok` rejects a verse when the aligner is unsure, or when gurus are not at least 1.2 times as long as laghus.

Gate: listen to 20 verses with their syllable marks in Audacity (export the spans as labels). Most marks must sit on the syllable onsets.

### Stage 4: pitch and the ceiling test (`pitch.py`)

- `world` analyses a clip with the WORLD vocoder: f0, spectral envelope, aperiodicity.
- `tonic` is the median pitch of one recording. Pitch is stored in semitones from it, so the two reciters and all sargas share one scale.
- `syl_pitch` takes each syllable's median over its vowel part.
- `ceiling` makes two copies of a clip: WORLD's plain copy, and a copy on flat notes, one per syllable.

Gate (the most important one): listen to the flat-note copy.
- If it still sounds like a beautiful chant, a score can carry this chant. Go on.
- If it sounds dead, the beauty is in the slides and ornaments between notes. Then the voice model needs the real f0 curve as input, not notes. Change stage 5 to predict a curve.

This gate costs one day and no training.

### Stage 5: the tune model (`tune.py`)

```sh
python run.py prep ../../vedicreader/static/vedic_texts out 2000    # stages 2 to 4 on 2,000 verses
python run.py train out 20                                          # anuṣṭubh only
```

- Inputs per syllable: weight, pāda, place, pāda length, pāda end, word start, metre, vowel, onset size, coda.
- A 4-layer transformer sees the whole verse, so a syllable knows what comes before and after it.
- Outputs: log duration in ms, pitch in semitones. The loss is masked MSE; unvoiced syllables do not count for pitch.
- Start with anuṣṭubh only (83% of the data). Add the triṣṭubh family later. It shares one model through the metre input.

Gate: on held-out sargas, the predicted guru/laghu ratio and the pitch at pāda ends match the reciters. Then hear it in stage 6.

### Stage 6: hear the tune before you have a voice (`proto.py`)

```sh
pip install -e path/to/scyllasband && python -m scyllasband download --flavor onnx
```

```python
from score import verse_rows, rule_tune
from proto import render, phrases
m, rows = verse_rows(text); rows = rule_tune(rows)          # or the tune model's ms and midi
for p in phrases(rows, [r['ms'] for r in rows]):
    audio = render('scyllasband/models/onnx', [r for r,_ in p], [d for _,d in p], [r['midi'] for r in p])
```

- `plan` builds the same three inputs that scyllasband's `plan_song` builds: phones, frames per phone, melody `[T, 3]` = `((midi-60)/24, phase in note, 1)`.
- We pass our own phones and frames, so the timing is exactly the tune model's, syllable by syllable.
- A phrase is a half-verse, where a reciter breathes. Long metres split at pāda ends, because scyllasband trained on clips of 9 s or less.
- The voice is a European singer and the retroflexes are approximate. Judge the tune, not the voice. Never publish this audio.

Learn: the interface. Your own voice model in stage 7 takes the same three inputs.

### Stage 7: the voice model

```sh
python run.py export out            # out/transcriptions.csv: name, ph_seq, ph_dur
```

- Train an acoustic model that takes phones, durations and f0, and gives a mel spectrogram. Then a vocoder.
- Use a code base that has training code and an f0 input. OpenVPI's DiffSinger is the closest that I know: a singing model with phone durations, f0 and an NSF-HiFiGAN vocoder at 44.1 kHz. Check its current docs for the exact list format before you use `export`.
- Our phones are IAST units (`dh`, `ṭ`, `ā`). Lossless. No IPA mapping is needed for our own model.
- Cut the clips at 44.1 kHz for this stage (`cut(..., sr=44100)`).
- The Veda recordings can join here. With f0 as an input, their svara tune does not leak. Give each reciter a speaker id.
- Hardware: the Mac is enough for stages 1 to 6 and for inference. For the acoustic model, I estimate one day on one rented GPU. I have not measured this.

Gate: Su-śrotā CER below 5% on held-out verses (`vachaka.check.cer`), and you like it more than the recording's WORLD copy from stage 4.

## What was tested, and how

No audio could reach the container where this was written. archive.org is blocked, and the LFS audio is pointers only. So:

| Part | Test | Result |
|---|---|---|
| `score.py` | Rāmāyaṇa 1.1.1 (anuṣṭubh) and Annapūrṇāṣṭakam 2 (śārdūlavikrīḍita) | Metres found; words rebuilt from the source, not from syllables |
| `data.py` | Every Rāmāyaṇa title in vedicreader | 19,489 verses, 55.8 h, in 6.5 s |
| `align.place` | Fake letter spans | Phone and syllable spans correct; `dh` takes two letters |
| `align.Aligner` | API only (dictionary, tokenizer, aligner signature) | Model download blocked. Not run on audio. |
| `pitch.py` | A synthetic voice on notes 0, +2, +2, −1 | Read back 0.12, 1.93, 1.93, −0.98; the flat-note copy reads 0, 2, 2, −1 |
| `tune.py` | 3,000 real verses, made-up targets with noise | Loss 1.18 to 0.059 in 6 epochs (36 s, CPU); 13 ms mean error on a śloka |
| `proto.plan` | Against scyllasband's phone list | All phones in its vocabulary; phrases 6.6 to 7.6 s. Render not run (bundle download blocked). |
| `run.py` | 38 verses, synthetic audio, stub aligner | prep, train and export run end to end |

## Install

```sh
uv pip install -e ../../ganapati librosa soundfile pyworld "setuptools<81" torch==2.5.1 torchaudio==2.5.1
```

pyworld 0.3.5 imports `pkg_resources`, so it needs setuptools older than 81.
