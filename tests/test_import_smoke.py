"""
Import smoke tests for lv-chordia.

lv-chordia is inference-only: importing the top-level package must not
depend on any training/eval module, and must not silently swallow import
errors (see lv_chordia/__init__.py -- the old blanket
``try/except ImportError: pass`` was removed on purpose).
"""

import importlib

import pytest


def test_package_imports_cleanly():
    """Importing lv_chordia must succeed without a blanket except clause hiding errors."""
    import lv_chordia

    assert lv_chordia.__version__
    assert hasattr(lv_chordia, "chord_recognition")
    assert callable(lv_chordia.chord_recognition)


def test_public_submodules_present():
    import lv_chordia

    assert set(lv_chordia.__all__) >= {
        "chord_recognition",
        "recognize",
        "load_ensemble",
        "__version__",
    }
    # __all__ must no longer advertise the removed training-only `datasets` module.
    assert "datasets" not in lv_chordia.__all__


def test_datasets_module_removed():
    """The training-only `datasets` module was deleted; it must not be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lv_chordia.datasets")


@pytest.mark.parametrize(
    "removed_module",
    [
        "lv_chordia.results",
        "lv_chordia.results_ismir2017",
        "lv_chordia.chordnet_ismir_naive_eval",
        "lv_chordia.test_for_all",
        "lv_chordia.storage_creation",
        "lv_chordia.train_eval_test_split",
        # The vendored mir toolkit and the DataEntry-based CQT extractor (rrherr fork).
        "lv_chordia.mir",
        "lv_chordia.extractors.cqt",
    ],
)
def test_training_eval_scripts_removed(removed_module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(removed_module)


def test_cli_entry_point_importable():
    from lv_chordia.cli import main

    assert callable(main)


def test_chord_recognition_function_importable():
    from lv_chordia.chord_recognition import chord_recognition, chord_recognition_json

    assert callable(chord_recognition)
    assert callable(chord_recognition_json)


def test_inference_path_needs_no_training_or_io_dependencies():
    """The rrherr fork dropped h5py, pretty_midi, pydub and joblib: importing the package and
    its CLI must not pull them (or the URL downloader) in."""
    import subprocess
    import sys

    code = (
        "import sys, lv_chordia, lv_chordia.cli; "
        "print(sorted(m for m in ('h5py', 'pretty_midi', 'pydub', 'joblib', 'lv_chordia.audio_utils') "
        "if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
