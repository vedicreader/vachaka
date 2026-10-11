# Release notes

<!-- do not remove -->

## 0.1.0

- First release. `vachaka.core.Chanter` chants a verse from its lines against Vāgdhenu's reference for the detected metre, and reads prose in chunks of at most 24 akṣaras against the anuṣṭubh or gadya reference; each render comes back as a `Recitation` with one `(start_ms, end_ms)` per piece. Runs on CUDA, on Apple silicon (DiT on MPS, BigVGAN on the CPU, since MPS rejects its transposed convolution) or on the CPU.
- `vachaka.check.Ear` loads Su-śrotā under upstream NeMo by rewriting the checkpoint's config for the AI4Bharat fork. `hear` transcribes, `cer` scores a transcript against its text with spaces, avagraha and daṇḍas ignored, `best_of` keeps the best of a few seeds, and `words` force-aligns each piece's words inside its span at 40 ms.
- `vachaka.recite` turns display lines (daṇḍas, verse numbers and accents left in) into pieces, chooses chant or prose by whether the section scans, and returns one audio file with `dict(s, e, w)` per line. `vachaka-recite jobs.jsonl results.jsonl` runs a batch and resumes where it stopped.
- Vāgdhenu, BigVGAN and IndicF5 are fetched at pinned commits into `~/.cache/vachaka` on first use, so the package has no git dependencies.
