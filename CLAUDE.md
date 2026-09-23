# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

**This is rrherr's fork of openmirlab/lv-chordia**, made for
bluegrass-karaoke's `bgk chords`; README's "This fork" section lists what
differs. Where the upstream notes below mention org canon or org decisions,
they describe upstream's reasoning, not rules this fork follows (for
example, this fork runs on MPS).

## What this package is

`lv-chordia` is an **inference-only** chord recognition package: given
decoded audio or an audio file, it returns a time-aligned chord sequence as
JSON. It
packages the pre-trained ensemble models from the ISMIR 2019 paper
"Large-Vocabulary Chord Transcription via Chord Structure Decomposition".

There is no training or evaluation code in this repository, and no
dependency on any training dataset. If you find yourself adding a training
loop, a dataset loader, or an eval/benchmark script here, stop -- that
doesn't belong in this package.

## Weights hosting: documented size-based exception to org constitution article 4

Unlike most other openmirlab inference packages, this repo does **not**
download its weights at runtime. The pre-trained ensemble
(`lv_chordia/cache_data/*.sdict`, 5 files, ~28MB total, 5.5MB each) is committed
directly to git and shipped inside the built wheel/sdist via
`pyproject.toml`'s wheel/sdist includes as package data, found at runtime
through `importlib.resources` (`config.CHECKPOINT_DIR`) -- this is the package's
pre-existing, original design, not a recent regression.

This is a **deliberate, documented exception** to the org's default weights
contract (constitution article 4: weights are normally downloaded at
runtime, never committed to git), confirmed acceptable on 2026-07-12. The
justification is size: 28MB total is small enough that bundling costs
little and buys a fully-offline install with zero download/caching/sha256
machinery -- the same size-vs-simplicity tradeoff behind
drum-classifier-infer's bundled checkpoint (there the driver was license
instead of size, but the org-level precedent -- bundling is fine when the
weight is genuinely small -- is the same one applied here). This is not a
defect to migrate away from; do not treat it as a TODO.

**Still do not delete or otherwise touch `lv_chordia/cache_data/*.sdict` or any
git-tracked weight file casually** -- if a future change genuinely needs to
move to runtime download (e.g. the ensemble grows well past this size, or
the org tightens the exception threshold), build the downloader, host the
weights (the org's usual pattern is a versioned external host + sha256
verification, as in bs-roformer-infer/melband-roformer-infer), then update
this note, `pyproject.toml`'s packaging config, and README's Scope section
together -- but that is a deliberate future call, not a standing violation
to clear.

## Entry points and the live import graph

- CLI: `lv_chordia/cli.py` (`lv-chordia` console script) -> `chord_recognition()`.
  The CLI alone handles URLs (`audio_utils.py`, imported lazily).
- Python API (arrays): `recognize(ensemble, audio, chords, beats=None, *, downbeats=True)`
  = `probabilities()` (CQT + five networks, averaged) then `decode()` (the
  HMM over a vocabulary list, optionally on a beat grid). `load_ensemble()`
  loads the five checkpoints once.
- Python API (one-shot): `chord_recognition()` / `chord_recognition_json()`
  (alias) -- loads a throwaway ensemble per call; `recognize_with_ensemble()`
  decodes the file with librosa and calls `recognize()` with
  `chord_list(chord_dict_name)`.
- Python API (resident): `lv_chordia.LVChordiaSession` (`session.py`) -- loads
  the ensemble once at `load()` and reuses it across `infer()` calls; the
  chord dictionary is a per-call choice that only drives the decoder.

The whole package is the inference path: `chord_recognition.py` ->
`chordnet_ismir_naive.py` (model) -> `network.py` (device placement,
checkpoint loading), `extractors/xhmm_ismir.py` (decoder) ->
`complex_chord.py` (chord encoding), plus `config/`, `settings.py`,
`device_utils.py`, `session.py`, `cli.py`, `audio_utils.py`. The upstream
`mir/` toolkit, `extractors/cqt.py` and ChordNet's training code were
removed in this fork; don't reintroduce training or dataset tooling.

## The upstream `mir/` subpackage

Gone in this fork. `NetworkBehavior`/`NetworkInterface` (the only parts the
inference path used) live in `lv_chordia/network.py`; the CQT that
`extractors/cqt.py`'s `CQTV2` computed through a `DataEntry` is
`chord_recognition.cqt()`, with identical parameters.

## Device handling -- do not touch casually

GPU/CPU selection defaults to automatic: `NetworkBehavior.__init__`
(`network.py`) checks `torch.cuda.device_count() > 0` and moves the
model to `.cuda()` if so, and this default **must stay untouched** -- GPU
support is a hard requirement of this package.

As of 2026-07, that default has an explicit, opt-in override:
`chord_recognition(..., device=...)` / `lv-chordia --device ...` accept
`'cpu'`, `'cuda'`, `'cuda:N'`, `'mps'` (this fork; RuntimeError if MPS is
unavailable), or `'auto'` (never MPS),
resolved by `device_utils.resolve_device()` and threaded through
`NetworkBehavior`/`ChordNet`/`ChordNetCNN`. Passing nothing (`device=None`,
no `--device` flag) is byte-for-byte the same auto-detect as before this
change -- the override is additive, not a replacement of the default. An
explicit `'cuda:N'` is validated against `torch.cuda.device_count()` and is
passed into model and tensor `.to()` calls without changing process-global
CUDA state.

Do not simplify, remove, or hardcode the *default* auto-detect to CPU or
GPU. Do not add `torch.cuda.set_device`: explicit indexes are carried on
individual model/tensor operations and must not mutate global CUDA state.

## Accuracy rule

Any change touching the inference path (`cli.py`, `chord_recognition.py`,
`chordnet_ismir_naive.py`, `network.py`, `extractors/xhmm_ismir.py`,
`settings.py`, `audio_utils.py`, `complex_chord.py`) must produce
byte-identical chord recognition JSON output on the regression fixture
before and after, and keep the decoder goldens (captured from unmodified
upstream code, see `tests/test_recognize.py`) passing. Verify with:

```bash
pytest tests/test_chord_recognition_regression.py tests/test_recognize.py -v
```

This compares the CLI's output on the tracked `test_data/yellow.wav` against
`tests/fixtures/expected_chords_yellow.json`. If a change is expected to
alter model output (e.g. retraining, a genuine bug fix in decoding), update
the fixture deliberately and say so in the commit message -- don't let it
change silently.

## Running tests

```bash
uv sync --extra dev   # or: pip install -e ".[dev]"
pytest tests/ -v
```

No network access or GPU is required; model weights (`lv_chordia/cache_data/*.sdict`)
and the test audio (`test_data/yellow.wav`) are tracked in the repo.

## Versioning

The package version is single-sourced from `lv_chordia.__version__` in
`lv_chordia/__init__.py` (`[tool.hatch.version] path = ...` in
`pyproject.toml` reads it at build time). Don't add a second, hand-edited
version field to `pyproject.toml`.
