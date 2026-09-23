"""Runtime metadata and read-only resolver for bundled checkpoints.

The five ensemble checkpoints ship as package data in lv_chordia/cache_data/.
"""

import importlib.resources
from pathlib import Path
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


#: Where the bundled checkpoints live: package data, found through importlib.resources.
CHECKPOINT_DIR = Path(str(importlib.resources.files("lv_chordia").joinpath("cache_data")))


def checkpoint_entries() -> tuple[dict, ...]:
    """Read the package TOML that names every shipped ensemble checkpoint."""
    config = importlib.resources.files(__package__).joinpath("checkpoints.toml")
    with config.open("rb") as handle:
        document = tomllib.load(handle)
    return tuple(document["models"]["lv_chordia"]["artifacts"])


def model_names() -> tuple[str, ...]:
    """Return NetworkInterface save names from the TOML artifact names."""
    return tuple(Path(entry["name"]).stem for entry in checkpoint_entries())


def resolve_checkpoint_paths() -> tuple[Path, tuple[dict, ...]]:
    """Resolve bundled checkpoint paths without downloading or creating paths."""
    root = CHECKPOINT_DIR
    entries = tuple(
        {**entry, "path": root / entry["name"], "cached": (root / entry["name"]).is_file()}
        for entry in checkpoint_entries()
    )
    return root, entries
