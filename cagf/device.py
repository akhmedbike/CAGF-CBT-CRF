"""Single source of truth for accelerator selection.

All trainers call ``pick_device()`` instead of hard-coding ``cuda``/``cpu`` so the
same code runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA) and CPU-only machines.
Preference order: MPS > CUDA > CPU.
"""
from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> str:
    """Return the best available accelerator device string.

    Parameters
    ----------
    prefer:
        Optional explicit override (e.g. ``"cpu"`` for deterministic unit tests).
        When given and non-empty it wins over auto-detection.

    Notes
    -----
    MPS (Metal Performance Shaders) is the Apple-Silicon GPU backend used on
    M-series Macs; it is what makes the ablation grid (5 configs x 5 seeds,
    up to 200 epochs each) tractable overnight on an M3 Pro.
    """
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
