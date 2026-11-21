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
- One tune model serves all metres. The next section says why, and how.

## Model structure per metre

### The data decides the structure

Recorded hours per metre in vedicreader (the Veda metre is by syllable count, ±2):

| laukika | verses | h | Veda | verses | h |
|---|---|---|---|---|---|
| anuṣṭubh | 19,764 | 57.6 | triṣṭubh | 4,679 | 20.4 |
| no metre found | 3,086 | 13.0 | paṅkti | 2,958 | 12.5 |
| upajāti | 481 | 1.8 | gāyatrī | 3,417 | 8.8 |
| śikhariṇī | 97 | 1.0 | anuṣṭubh | 2,708 | 8.4 |
| vaṃśastha | 247 | 0.9 | jagatī | 1,113 | 5.2 |
| indra/upendravajrā | 158 | 0.6 | uṣṇih, bṛhatī | 2,327 | 7.7 |
| śārdūlavikrīḍita | 2 | 0.0 | | | |

- One model per metre works only for anuṣṭubh. Every other laukika metre has less than 2 h.
- So: one model for all metres, which describes a metre by its shape, not only by its name.
- 97.6% of the Veda verses carry svara marks: 19,602 of 19,631 parse to one svara per syllable.

### Four choices, each tested on made-up tunes over real verses

| Problem | Structure | Test | Result |
|---|---|---|---|
| A reciter sings one metre to several tunes. Plain regression gives their flat average. | Tune codes: k-means on each pāda's pitch contour gives a code; the model chooses a code per pāda, then sings the syllables to it (`pada_codes`, `Tune.code_logits`). | Two tunes per pāda, rising and falling | k-means found them exactly. Pitch range per pāda: plain 0.24 st, with codes 2.92 st (truth 3.0). Error 0.88 → 0.27 st. |
| A rare metre has a name the model barely knows. | Hide the metre name in 30% of training verses (`drop`), so the model also learns from shape. | vaṃśastha held out | 0.54 → 0.40 st, 166 → 120 ms |
| A pāda length never heard (vaṃśastha's 12). | Place in eighths of the pāda, not a count; places counted from both ends. | Same | 0.40 → 0.31 st, 120 → 45 ms (seen metres: 0.24 st, 20 ms) |
| Gāyatrī has 3 pādas, paṅkti 5; the written gāyatrī mantra has 23 syllables. | `pada_end` (pādas left), `pos_end`; Vedic pādas cut at the word start nearest each even split. | Ṛgveda 3.62.10 | `ta tsa vi tu rva re ṇyaṃ / bha rgo de va sya dhī ma hi / dhi yo yo naḥ pra co da yāt` |

The features per syllable: weight, pāda, pādas left, place (eighths), places left, pāda end, word start, metre name, vowel, onset size, coda, mātrā place, svara.

### What each metre gets

- **anuṣṭubh.** Learned from 57.6 h. Use `k` 4 to 8 tune codes. Listen to the codes' centres (`centres` in `tune.pt`): each one should be a tune that you can hear in the recording.
- **triṣṭubh family (upajāti, indravajrā, vaṃśastha) and śikhariṇī.** Same model. It learns from 2 to 3 h of shared shape, plus anuṣṭubh.
- **śārdūlavikrīḍita, mālinī, vasantatilakā.** No data. The model sings them from shape. Judge the result by ear. To fix this, record 20 minutes in each and add them.
- **āryā (counted in mātrās).** No audio. `mora` is a feature now, but nothing teaches it yet.
- **gāyatrī and the other Veda metres.** Their pitch comes from the svara marks, so the svara is an input (`sv`: U, A, S, P). Use the Veda audio for three things, never as output:
  1. `svara_check`: ground truth for stage 4. Udātta must read above anudātta. A unit test gives 1.00 for pitch with 1 st noise and 0.49 for random pitch.
  2. Durations: guru and laghu in chant, pāda ends and breaths (`run.train(out, vedic=True)`).
  3. The voice model in stage 7, with f0 as input.
- **A laukika gāyatrī** (a deity gāyatrī, 3×8 without marks) shares the 8-syllable pāda with anuṣṭubh, and `pada_end` gives it the cadence on its third pāda.

### When to change the structure

- If the codes' centres all sound alike, the reciters use one tune: set `k=1`.
- If one tune per pāda is too coarse (the tune changes inside a pāda), replace the codes with a flow-matching pitch head. scyllasband's pitch predictor is this kind (a rectified flow), so that it does not average. Try this only after the codes fail by ear.

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
python run.py train out 20                                          # anuṣṭubh only, 6 tune codes
```

- A 4-layer transformer sees the whole verse, so a syllable knows what comes before and after it.
- A code head chooses a tune for each pāda. The syllable head gives log duration in ms and pitch in semitones for that tune.
- The loss is masked MSE plus cross-entropy on the codes. Unvoiced syllables do not count for pitch.
- `predict` samples the codes (`temp`), so two renders of one verse can use different tunes, as a reciter does.
- Start with anuṣṭubh only. Add the other metres and the Veda verses after.

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
| `tune.py` | 3,000 real verses, made-up targets with noise | First version: loss 1.18 to 0.059 in 6 epochs (36 s, CPU), 13 ms mean error on a śloka. Codes, hidden names and eighths: see the table under Model structure. |
| `score.svaras` | Every recorded Veda verse; Ṛgveda 3.62.10 | 19,602 of 19,631 give one svara per syllable; the gāyatrī's accents match its Śākala marks |
| `pitch.svara_check` | Made-up pitch from svaras, then random pitch | 1.00 (1 st noise), 0.98 (3 st), 0.49 (random) |
| `proto.plan` | Against scyllasband's phone list | All phones in its vocabulary; phrases 6.6 to 7.6 s. Render not run (bundle download blocked). |
| `run.py` | 28 Rāmāyaṇa and 35 Yajurveda verses, synthetic audio, stub aligner | prep, train with 3 codes and Veda verses, and export run end to end |

## Install

```sh
uv pip install -e ../../ganapati librosa soundfile pyworld "setuptools<81" torch==2.5.1 torchaudio==2.5.1
```

pyworld 0.3.5 imports `pkg_resources`, so it needs setuptools older than 81.
