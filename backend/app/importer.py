"""Tensor import logic — supports .pt/.pth, .npy/.npz, .csv."""
from __future__ import annotations

import io
import csv
from pathlib import Path
from typing import Union

import numpy as np
import torch


MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def _from_numpy(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor."""
    return torch.from_numpy(arr).float()


def load_tensor_file(
    file_bytes: bytes,
    filename: str,
    key: str | None = None,
) -> dict:
    """
    Load a tensor from raw bytes.

    Returns a dict with:
        tensor   : torch.Tensor | None
        keys     : list[str]   (non-empty when file is a dict/state-dict and key not specified)
        key_used : str | None
    """
    if len(file_bytes) > MAX_BYTES:
        raise ValueError(f"File too large ({len(file_bytes) // (1024*1024)} MB). Limit is 50 MB.")

    suffix = Path(filename).suffix.lower()

    # ── PyTorch format (.pt / .pth) ───────────────────────────────────────────
    if suffix in (".pt", ".pth"):
        buf = io.BytesIO(file_bytes)
        try:
            obj = torch.load(buf, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback for older checkpoints that don't support weights_only
            buf.seek(0)
            obj = torch.load(buf, map_location="cpu")  # noqa: S614

        if isinstance(obj, torch.Tensor):
            return {"tensor": obj.float(), "keys": [], "key_used": None}

        if isinstance(obj, dict):
            tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
            if not tensor_keys:
                raise ValueError("File contains a dict but no tensor values were found.")
            if key is not None:
                if key not in obj:
                    raise KeyError(f"Key '{key}' not found. Available: {tensor_keys}")
                return {"tensor": obj[key].float(), "keys": tensor_keys, "key_used": key}
            # No key supplied yet — return key list so frontend can ask
            return {"tensor": None, "keys": tensor_keys, "key_used": None}

        raise ValueError(f"Unsupported object type in checkpoint: {type(obj).__name__}")

    # ── NumPy .npy ────────────────────────────────────────────────────────────
    if suffix == ".npy":
        arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        return {"tensor": _from_numpy(arr), "keys": [], "key_used": None}

    # ── NumPy .npz ────────────────────────────────────────────────────────────
    if suffix == ".npz":
        npz = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        available = list(npz.files)
        if not available:
            raise ValueError("Empty .npz file.")
        if key is not None:
            if key not in npz:
                raise KeyError(f"Key '{key}' not found. Available: {available}")
            return {"tensor": _from_numpy(npz[key]), "keys": available, "key_used": key}
        if len(available) == 1:
            k = available[0]
            return {"tensor": _from_numpy(npz[k]), "keys": available, "key_used": k}
        # Multiple arrays — ask frontend which one
        return {"tensor": None, "keys": available, "key_used": None}

    # ── CSV ──────────────────────────────────────────────────────────────────
    if suffix == ".csv":
        text = file_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = []
        for row in reader:
            if not row or (len(row) == 1 and row[0].strip() == ""):
                continue
            try:
                rows.append([float(x) for x in row])
            except ValueError as exc:
                raise ValueError(
                    f"CSV parse error — non-numeric value: {exc}"
                ) from exc
        if not rows:
            raise ValueError("CSV file is empty or contains no numeric data.")
        arr = np.array(rows, dtype=np.float32)
        return {"tensor": torch.from_numpy(arr), "keys": [], "key_used": None}

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Supported: .pt, .pth, .npy, .npz, .csv"
    )
