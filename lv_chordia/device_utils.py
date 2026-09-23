"""Validate public device requests while preserving the legacy auto default.

Explicit 'mps' runs on Apple Silicon (rrherr fork; upstream rejects it). 'auto'
never selects MPS: it keeps the legacy CUDA-or-CPU auto-detect.
"""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(device: Optional[str] = None) -> Optional[torch.device]:
    """Return an explicit torch device, or ``None`` for legacy auto-selection.

    ``None`` and ``"auto"`` deliberately leave device selection to the
    original ``NetworkBehavior`` CUDA auto-detect. Explicit requests are
    validated before model construction so they never silently fall back.

    ``"mps"`` is honoured when ``torch.backends.mps.is_available()``; the
    ensemble's float32 probabilities match CPU to within about 1e-6. Supported
    devices: 'cpu', 'cuda', 'cuda:N', 'mps', 'auto', or None.
    """
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' was requested but MPS is not available.")
        return torch.device("mps")
    if device is None or device == "auto":
        return None
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.device_count() == 0:
            raise RuntimeError("device='cuda' was requested but no CUDA device is visible.")
        return torch.device("cuda")
    if device.startswith("cuda:"):
        index_text = device[len("cuda:") :]
        if not index_text.isdigit():
            raise ValueError("Invalid device %r: expected 'cuda:N' with a non-negative index." % device)
        index = int(index_text)
        available = torch.cuda.device_count()
        if index >= available:
            raise RuntimeError(
                "device=%r was requested but only %d CUDA device(s) are visible."
                % (device, available)
            )
        return torch.device("cuda", index)
    raise ValueError(
        "Invalid device %r: expected 'cpu', 'cuda', 'cuda:N', 'mps', 'auto', or None."
        % device
    )


def resolve_use_gpu(device: Optional[str] = None) -> Optional[bool]:
    """Compatibility facade for legacy callers that consume a GPU boolean."""
    resolved = resolve_device(device)
    return None if resolved is None else resolved.type == "cuda"
