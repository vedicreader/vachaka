# Release notes

<!-- do not remove -->

## 0.1.2

- Words no longer end abruptly: each piece keeps 180 ms past its voice (Vāgdhenu's gate kept 60 ms) faded over 120 ms, and a part ends on half a second of silence (`Chanter(tail=, fade=, pad=)`).
- Sound treatments for someone who is not a sound engineer, all loudness-matched to -16 LUFS and none moving a syllable in time: `warm` (a little warmth, a soft de-esser, a small room's echo), `phone` (compression and a lift around 3 kHz for small speakers) and `plain`. `Recitation.save(fx=)`, `polish(src, dst, fx)`, and `vachaka-recite --fx`, which defaults to `warm`.

## 0.1.1

- A section is recited in parts, each in one voice, instead of all in one. The caller can pass the parts, `dict(lines=[…], meter=…)`, with metres from a better detector (VedicReader passes ganapati's): a metre in Vāgdhenu's bank is chanted against its reference, another vṛtta against vasantatilakā, and the āryā family or no metre is read. A commentary section that opens with verses now chants them and reads its prose.
- Without parts, a section splits into verses at double daṇḍas and Vāgdhenu's detector names each, but only a verse that closes on a double daṇḍa is chanted: its anuṣṭubh test passed a 32-syllable prose sentence (Nyāyavṛtti 1.1), and missed a mālinī that ganapati finds.
- `Chanter.bank_ref` finds the bank's reference for a metre in any spelling. Results carry `parts` per section, and `kind` is `mixed` when they differ.

## 0.1.0

- First release. `vachaka.core.Chanter` chants a verse from its lines against Vāgdhenu's reference for the detected metre, and reads prose in chunks of at most 24 akṣaras against the anuṣṭubh or gadya reference; each render comes back as a `Recitation` with one `(start_ms, end_ms)` per piece. Runs on CUDA, on Apple silicon (DiT on MPS, BigVGAN on the CPU, since MPS rejects its transposed convolution) or on the CPU.
- `vachaka.check.Ear` loads Su-śrotā under upstream NeMo by rewriting the checkpoint's config for the AI4Bharat fork. `hear` transcribes, `cer` scores a transcript against its text with spaces, avagraha and daṇḍas ignored, `best_of` keeps the best of a few seeds, and `words` force-aligns each piece's words inside its span at 40 ms.
- `vachaka.recite` turns display lines (daṇḍas, verse numbers and accents left in) into pieces, chooses chant or prose by whether the section scans, and returns one audio file with `dict(s, e, w)` per line. `vachaka-recite jobs.jsonl results.jsonl` runs a batch and resumes where it stopped.
- Vāgdhenu, BigVGAN and IndicF5 are fetched at pinned commits into `~/.cache/vachaka` on first use, so the package has no git dependencies.
