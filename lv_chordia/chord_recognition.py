"""
The inference pipeline: audio in, time-aligned chord JSON out.

This is the one real entry point of the package (cli.py just wraps it;
session.py holds a preloaded ensemble across calls). The pipeline runs a
5-model ensemble (ChordNet, defined in chordnet_ismir_naive.py) over CQT
features (cqt()) loaded via network.NetworkInterface,
averages the ensemble's per-frame probabilities, and decodes them into chord
segments with an HMM (extractors/xhmm_ismir.py) driven by a chosen chord
dictionary (lv_chordia/data/*_chord_list.txt). Split into load_ensemble()
(the expensive part: five torch.load calls, vocabulary independent) and
recognize() (per-call on decoded audio: CQT + inference + HMM decode, with an
optional beat grid); recognize_with_ensemble() decodes a file for recognize(),
and chord_recognition() composes it with load_ensemble() for one-shot callers. Accuracy-sensitive:
any change here must keep tests/test_chord_recognition_regression.py passing
byte-for-byte.

Reads: chordnet_ismir_naive.py, network.py, extractors/xhmm_ismir.py, settings.py, device_utils.py
"""

from .chordnet_ismir_naive import ChordNet
from .network import NetworkInterface
from .extractors.xhmm_ismir import XHMMDecoder
import numpy as np
import librosa
from .settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
from .config import model_names
from .device_utils import resolve_device
import logging
from typing import List, Dict, Iterable, Optional, Sequence, Tuple, Union
import importlib.resources

MODEL_NAMES = model_names()

logger = logging.getLogger(__name__)

#: Sample rate recognize() expects its mono audio at.
SAMPLE_RATE = DEFAULT_SR
#: Seconds per network frame (hop 512 at 22.05 kHz).
FRAME_SECONDS = DEFAULT_HOP_LENGTH / DEFAULT_SR


def load_ensemble(use_gpu: Optional[bool] = None, *, device=None) -> List[NetworkInterface]:
    """
    Load all five ChordNet ensemble members from their bundled checkpoints, once.

    This is the expensive part of the pipeline (five torch.load calls over
    cache_data/*.sdict) and it is vocabulary independent: the vocabulary
    only drives the HMM decoder built per call in recognize(). Callers that
    recognize chords repeatedly from a resident process should load the
    ensemble once (or hold an LVChordiaSession, which does exactly that) and
    pass it to recognize() per call.

    Args:
        use_gpu: The resolved GPU flag NetworkBehavior consumes -- True forces
            GPU, False forces CPU, None preserves auto-detection. Produce it
            from a caller-facing device string with
            device_utils.resolve_use_gpu().
        device: Optional explicit ``torch.device``.  ``cuda:N`` is forwarded
            unchanged so model construction and tensors use that index.

    Returns:
        The five loaded NetworkInterface ensemble members, in MODEL_NAMES order.
    """
    return [
        NetworkInterface(ChordNet(use_gpu=use_gpu, device=device), model_name, load_checkpoint=False)
        for model_name in MODEL_NAMES
    ]


def chord_list(chord_dict_name: str) -> List[str]:
    """The vocabulary of a bundled chord dictionary ('submission', 'ismir2017', 'full' or 'extended'):
    chord names on root C, one per line of lv_chordia/data/<name>_chord_list.txt."""
    text = importlib.resources.files("lv_chordia.data").joinpath(f"{chord_dict_name}_chord_list.txt").read_text()
    return text.splitlines()


def cqt(audio: np.ndarray) -> np.ndarray:
    """The network input: |hybrid CQT| of mono audio at SAMPLE_RATE, frames x 288 bins, float32."""
    result = librosa.core.hybrid_cqt(audio,
                                     bins_per_octave=36,
                                     fmin=librosa.note_to_hz('F#0'),
                                     n_bins=288,
                                     tuning=None,
                                     hop_length=DEFAULT_HOP_LENGTH).T
    return abs(result).astype(np.float32)


def recognize(
    ensemble: List[NetworkInterface],
    audio: np.ndarray,
    chords: Iterable[str],
    beats: Optional[Sequence[Tuple[float, int]]] = None,
    *,
    downbeats: bool = True,
) -> List[Dict[str, Union[float, str]]]:
    """
    Recognize chords in decoded audio with an already-loaded ensemble.

    Computes CQT features, runs the five preloaded ensemble members, averages
    their per-frame probabilities (probabilities()), and decodes them to chord
    segments (decode()). No model weights are loaded here.

    Args:
        ensemble: The loaded ensemble members from load_ensemble().
        audio: Mono audio at SAMPLE_RATE (22.05 kHz), 1-D; converted to float32.
        chords: The vocabulary, as chord names on root C in Harte notation
            (e.g. ``["C:maj", "C:min", "C:7"]``, or ``chord_list("submission")``);
            the decoder transposes each to all twelve roots and always adds ``N``.
            Lines without a colon (such as ``N``) are ignored.
        beats: Optional beat grid, ``[(time_seconds, position_in_bar), ...]`` in
            ascending time, position 1 = downbeat (beat_this's ``.beats`` rows).
            Between the first and last beat, chords then change only on beats.
        downbeats: With ``beats``, make changes cheapest on downbeats, then on
            the middle beat of even bars, then on other beats.

    Returns:
        ``[{"start_time": s, "end_time": e, "chord": label}, ...]``, contiguous
        from 0 to the end of the audio, times rounded to two decimals.

    Raises:
        ValueError: audio is not 1-D, or a vocabulary chord is not on root C.
    """
    return decode(probabilities(ensemble, audio), chords, beats, downbeats=downbeats)


def probabilities(ensemble: List[NetworkInterface], audio: np.ndarray) -> List[np.ndarray]:
    """The ensemble's frame probabilities for mono audio at SAMPLE_RATE: the five members'
    six heads (triad, bass, 7th, 9th, 11th, 13th), each averaged over the members."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono (1-D), got shape {audio.shape}")
    features = cqt(audio)
    probs = []
    for net in ensemble:
        logger.info('Inference: %s', net.save_name)
        probs.append(net.inference(features))
    return [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]


def decode(
    probs: Sequence[np.ndarray],
    chords: Iterable[str],
    beats: Optional[Sequence[Tuple[float, int]]] = None,
    *,
    downbeats: bool = True,
) -> List[Dict[str, Union[float, str]]]:
    """Decode probabilities() output to chord segments; arguments as for recognize()."""
    hmm = XHMMDecoder(list(chords))
    chordlab = hmm.decode_to_chordlab(list(probs), FRAME_SECONDS, beats=beats, downbeats=downbeats)
    return [
        {
            'start_time': float(f"{segment[0]:.2f}"),
            'end_time': float(f"{segment[1]:.2f}"),
            'chord': str(segment[2]),
        }
        for segment in chordlab
    ]


def recognize_with_ensemble(
    ensemble: List[NetworkInterface],
    audio_path: str,
    chord_dict_name: str = 'submission',
) -> List[Dict[str, Union[float, str]]]:
    """
    Run chord recognition on one audio file using an already-loaded ensemble.

    Decodes the file with librosa at SAMPLE_RATE, mono, and calls recognize()
    with the named bundled chord dictionary and no beat grid.

    Args:
        ensemble: The loaded ensemble members from load_ensemble().
        audio_path: Path to a local audio file (the CLI also accepts URLs).
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')

    Returns:
        List of chord annotations -- same shape as chord_recognition().
    """
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return recognize(ensemble, audio, chord_list(chord_dict_name))


def chord_recognition(audio_path: str, chord_dict_name: str = 'submission', device: Optional[str] = None) -> List[Dict[str, Union[float, str]]]:
    """
    Perform chord recognition on an audio file and return results as JSON.

    Takes a local file; the lv-chordia CLI downloads URLs before calling this.

    Args:
        audio_path: Path to a local audio file.
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')
        device: Optional device override -- one of 'cpu', 'cuda', 'cuda:N', or
            'auto'. None (the default) preserves today's behavior exactly:
            auto-detect GPU via torch.cuda.device_count() > 0. 'cpu' forces
            CPU even when CUDA is available; 'cuda'/'cuda:N' force GPU,
            raising RuntimeError if no CUDA device is visible. Only
            CUDA availability is validated before model loading; 'cuda:N'
            is forwarded to model construction unchanged. 'mps' runs on Apple
            Silicon, raising RuntimeError if MPS is not available.

    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)

    Raises:
        RuntimeError: device requests CUDA but none is visible, or requests
            an out-of-range CUDA index.
        ValueError: device is not a recognized string.

    Examples:
        >>> # Local file
        >>> results = chord_recognition("song.mp3")
        >>> print(results)
        [
            {"start_time": 0.0, "end_time": 2.5, "chord": "C:maj"},
            {"start_time": 2.5, "end_time": 5.0, "chord": "F:maj"},
            ...
        ]

        >>> # Force CPU even on a CUDA-capable machine
        >>> results = chord_recognition("song.mp3", device="cpu")
    """
    # Resolve device before doing any work, so an invalid/unavailable device
    # request fails fast rather than after loading models.
    resolved_device = resolve_device(device)
    use_gpu = None if resolved_device is None else resolved_device.type == "cuda"

    # One-shot convenience path: a throwaway ensemble is loaded per call, so
    # every call pays the five torch.load calls. Callers that recognize chords
    # repeatedly should hold an LVChordiaSession (session.py) instead.
    ensemble = load_ensemble(use_gpu, device=resolved_device)
    return recognize_with_ensemble(ensemble, audio_path, chord_dict_name)


def chord_recognition_json(audio_path: str, chord_dict_name: str = 'submission', device: Optional[str] = None) -> List[Dict[str, Union[float, str]]]:
    """
    Alias for chord_recognition function for backward compatibility.

    Args:
        audio_path: Path to the input audio file
        chord_dict_name: Chord dictionary to use ('submission', 'ismir2017', or 'full')
        device: Optional device override -- see chord_recognition() for the
            accepted values and defaults.

    Returns:
        List of chord annotations as dictionaries with keys:
        - start_time: Start time in seconds (float)
        - end_time: End time in seconds (float)
        - chord: Chord label (string)
    """
    return chord_recognition(audio_path, chord_dict_name, device)

