"""Run provenance metadata for experiment directories (revision spec 0.2).

Every CV run directory gets a ``run_meta.json`` recording how and where the
numbers were produced: git commit (and whether the tree was dirty), start
date, python/torch versions, device, and the driver-specific fields the caller
passes in (k, strategy, seeds, configs, corpus paths).

This exists because one archived artefact (the KazRoBERTa noise-floor) turned
out to contain two byte-identical runs, i.e. one measurement rather than two;
run-level provenance makes such an artefact impossible to produce silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _git(field: str, args: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(['git', *args], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent)
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


def collect_run_meta(device: Optional[str] = None, **extra) -> dict:
    """Assemble the provenance dict. ``extra`` fields are added verbatim."""
    meta: dict = {
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python': sys.version.split()[0],
    }
    try:
        import torch
        meta['torch'] = torch.__version__
    except ImportError:
        meta['torch'] = None
    commit = _git('commit', ['rev-parse', 'HEAD'])
    dirty = _git('dirty', ['status', '--porcelain'])
    meta['git_commit'] = commit or 'unknown'
    meta['git_dirty'] = bool(dirty)
    if device:
        meta['device'] = device
    else:
        try:
            from .device import pick_device
            meta['device'] = pick_device()
        except Exception:
            meta['device'] = None
    meta.update(extra)
    return meta


def write_run_meta(out_dir: str | Path, device: Optional[str] = None, **extra) -> Path:
    """Write ``<out_dir>/run_meta.json`` (overwrites on resume)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'run_meta.json'
    path.write_text(json.dumps(collect_run_meta(device=device, **extra),
                               ensure_ascii=False, indent=2), encoding='utf-8')
    return path
