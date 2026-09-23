"""
Regression tests for the array entry point (recognize / probabilities / decode)
and the vocabulary-list + beat-grid decoder.

The expected outputs were captured from UNMODIFIED upstream code
(openmirlab/lv-chordia@aa6841b, which took a template file and read beats from
a DataEntry) by driving XHMMDecoder.decode_to_chordlab(entry, probs, False,
use_beats=..., use_downbeats=...) directly, so they pin today's decoder to the
upstream decoding math:

- expected_decodes_yellow.json: test_data/yellow.wav's real ensemble output,
  decoded with each bundled dictionary, and with synthetic beat grids of 2, 3
  and 4 beats per bar (beats only, and beats + downbeats).
- expected_decodes_synthetic.json + synthetic_probabilities.npz: 900 frames of
  seeded, piecewise-constant probabilities peaked on chords drawn from the full
  vocabulary. One pop clip barely separates the vocabularies (every dictionary
  decodes yellow.wav identically); these 30 cases all differ from one another.

Reads: lv_chordia/chord_recognition.py, lv_chordia/extractors/xhmm_ismir.py
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

from lv_chordia import (
    FRAME_SECONDS,
    SAMPLE_RATE,
    chord_list,
    decode,
    load_ensemble,
    probabilities,
    recognize,
)
from lv_chordia.extractors.xhmm_ismir import XHMMDecoder, beat_frames

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_AUDIO = Path(__file__).resolve().parent.parent / "test_data" / "yellow.wav"

# The two non-bundled vocabularies the fixtures were captured with (as template files upstream).
VOCABULARIES = {
    "triads": ["C:maj", "C:min", "N"],
    "sevenths": ["C:maj", "C:min", "C:7", "C:min7", "N"],
}


def vocabulary(name):
    return VOCABULARIES[name] if name in VOCABULARIES else chord_list(name)


def split_case(case):
    """'submission/downbeats3' -> ('submission', '3', True); 'full' -> ('full', None, False)."""
    name, _, grid = case.partition("/")
    if not grid:
        return name, None, False
    if grid.startswith("downbeats"):
        return name, grid[len("downbeats"):], True
    return name, grid[len("beats"):], False


def decode_case(probs, fixture, case):
    name, meter, downbeats = split_case(case)
    beats = None if meter is None else [tuple(b) for b in fixture["beats"][meter]]
    return decode(probs, vocabulary(name), beats, downbeats=downbeats)


@pytest.fixture(scope="module")
def ensemble():
    return load_ensemble(False, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def yellow_audio():
    audio, _ = librosa.load(str(TEST_AUDIO), sr=SAMPLE_RATE, mono=True)
    return audio


@pytest.fixture(scope="module")
def yellow_probs(ensemble, yellow_audio):
    return probabilities(ensemble, yellow_audio)


YELLOW = json.loads((FIXTURES / "expected_decodes_yellow.json").read_text())
SYNTHETIC = json.loads((FIXTURES / "expected_decodes_synthetic.json").read_text())


@pytest.mark.parametrize("case", sorted(YELLOW["cases"]))
def test_decode_matches_upstream_on_yellow(yellow_probs, case):
    assert decode_case(yellow_probs, YELLOW, case) == YELLOW["cases"][case]


@pytest.mark.parametrize("case", sorted(SYNTHETIC["cases"]))
def test_decode_matches_upstream_on_synthetic_probabilities(case):
    with np.load(FIXTURES / "synthetic_probabilities.npz") as data:
        probs = [data[f"head{i}"] for i in range(6)]
    assert decode_case(probs, SYNTHETIC, case) == SYNTHETIC["cases"][case]


def test_recognize_on_librosa_decode_matches_the_path_api_fixture(ensemble, yellow_audio):
    """The array entry point fed librosa's decode must equal the CLI's byte-identical golden output."""
    expected = json.loads((FIXTURES / "expected_chords_yellow.json").read_text())
    assert recognize(ensemble, yellow_audio, chord_list("submission")) == expected


def test_recognize_output_is_contiguous_and_covers_the_audio(ensemble, yellow_audio):
    beats = [tuple(b) for b in YELLOW["beats"]["3"]]
    segments = recognize(ensemble, yellow_audio, VOCABULARIES["triads"], beats)
    assert segments[0]["start_time"] == 0.0
    for before, after in zip(segments, segments[1:]):
        assert before["end_time"] == after["start_time"]
        assert before["chord"] != after["chord"]
    assert segments[-1]["end_time"] == pytest.approx(len(yellow_audio) / SAMPLE_RATE, abs=FRAME_SECONDS + 0.01)
    assert {s["chord"] for s in segments} <= {"N"} | {f"{r}:{q}" for r in
        ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"] for q in ["maj", "min"]}


def test_recognize_rejects_non_mono_audio(ensemble):
    with pytest.raises(ValueError, match="mono"):
        recognize(ensemble, np.zeros((2, SAMPLE_RATE), dtype=np.float32), ["C:maj"])


def test_vocabulary_chords_must_be_on_root_c():
    with pytest.raises(ValueError, match="root C"):
        XHMMDecoder(["C:maj", "G:min"])


def test_vocabulary_ignores_lines_without_a_colon():
    with_blank = XHMMDecoder(["C:maj", "", "N", "C:min"]).known_chord_array
    assert with_blank == XHMMDecoder(["C:maj", "C:min"]).known_chord_array
    assert [name for _, name in with_blank][:3] == ["N", "C:maj", "C#:maj"]


def test_beat_frames_without_a_grid_allows_changes_everywhere():
    assert (beat_frames(None, 50, FRAME_SECONDS, downbeats=True) == 1).all()


def test_beat_frames_blocks_changes_between_beats_and_ranks_downbeats():
    frame = FRAME_SECONDS
    beats = [(10 * frame, 1), (20 * frame, 2), (30 * frame, 3), (40 * frame, 4), (50 * frame, 1)]
    codes = beat_frames(beats, 60, frame, downbeats=True)
    assert (codes[:10] == 1).all() and (codes[51:] == 1).all()  # outside the grid: free
    assert (codes[11:20] == 0).all() and (codes[41:50] == 0).all()  # between beats: no change
    assert [codes[10], codes[20], codes[30], codes[40], codes[50]] == [2, 4, 3, 4, 2]
    beats_only = beat_frames(beats, 60, frame, downbeats=False)
    assert [beats_only[i] for i in (10, 20, 30, 40, 50)] == [1] * 5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_probabilities_match_cpu(yellow_probs, yellow_audio):
    """On Apple Silicon the float32 ensemble must agree with CPU: probabilities within 1e-4,
    and the decoded chords identical on the reference clip."""
    mps_probs = probabilities(load_ensemble(False, device=torch.device("mps")), yellow_audio)
    for cpu_head, mps_head in zip(yellow_probs, mps_probs):
        np.testing.assert_allclose(mps_head, cpu_head, atol=1e-4)
    assert decode(mps_probs, chord_list("submission")) == decode(yellow_probs, chord_list("submission"))
