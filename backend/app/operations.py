import torch
from typing import List, Optional, Union


def _to_tensor(data: List, dtype: str) -> torch.Tensor:
    return torch.tensor(data, dtype=getattr(torch, dtype))


def create_tensor(op: str, shape: List[int], dtype: str = "float32"):
    torch_dtype = getattr(torch, dtype)
    if op == "ones":
        return torch.ones(shape, dtype=torch_dtype)
    elif op == "zeros":
        return torch.zeros(shape, dtype=torch_dtype)
    else:
        raise ValueError("op must be 'ones' or 'zeros'")


# ── Binary ops ────────────────────────────────────────────────────────────────

def add_tensors(a: List, b: List, dtype: str = "float32"):
    return _to_tensor(a, dtype) + _to_tensor(b, dtype)


def sub_tensors(a: List, b: List, dtype: str = "float32"):
    return _to_tensor(a, dtype) - _to_tensor(b, dtype)


def mul_tensors(a: List, b: List, dtype: str = "float32"):
    return _to_tensor(a, dtype) * _to_tensor(b, dtype)


def div_tensors(a: List, b: List, dtype: str = "float32"):
    return _to_tensor(a, dtype) / _to_tensor(b, dtype)


def matmul_tensors(a: List, b: List, dtype: str = "float32"):
    return torch.matmul(_to_tensor(a, dtype), _to_tensor(b, dtype))


# ── Unary ops ─────────────────────────────────────────────────────────────────

def abs_tensor(t: List, dtype: str = "float32"):
    return torch.abs(_to_tensor(t, dtype))


def neg_tensor(t: List, dtype: str = "float32"):
    return torch.neg(_to_tensor(t, dtype))


def clamp_tensor(
    t: List,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    dtype: str = "float32",
):
    return torch.clamp(_to_tensor(t, dtype), min=min_val, max=max_val)


# ── Reduction ops ─────────────────────────────────────────────────────────────

def sum_tensor(t: List, dim: Optional[int] = None, keepdim: bool = False, dtype: str = "float32"):
    tensor = _to_tensor(t, dtype)
    if dim is None:
        return tensor.sum()
    return tensor.sum(dim=dim, keepdim=keepdim)


def tensor_stats(t: List, dtype: str = "float32") -> dict:
    tensor = _to_tensor(t, dtype)
    flat = tensor.flatten()
    return {
        "mean":  float(flat.mean()),
        "std":   float(flat.std()) if flat.numel() > 1 else 0.0,
        "min":   float(flat.min()),
        "max":   float(flat.max()),
        "sum":   float(flat.sum()),
        "norm":  float(torch.linalg.norm(flat)),
        "numel": int(flat.numel()),
        "rank":  int(tensor.dim()),
    }


# ── Shape & indexing ops ──────────────────────────────────────────────────────

def reshape_tensor(t: List, shape: List[int], dtype: str = "float32"):
    return _to_tensor(t, dtype).reshape(shape)


def transpose_tensor(t: List, dim0: int = 0, dim1: int = 1, dtype: str = "float32"):
    return torch.transpose(_to_tensor(t, dtype), dim0, dim1)


def flatten_tensor(t: List, start_dim: int = 0, end_dim: int = -1, dtype: str = "float32"):
    return torch.flatten(_to_tensor(t, dtype), start_dim=start_dim, end_dim=end_dim)


def squeeze_tensor(t: List, dim: Optional[Union[int, List[int]]] = None, dtype: str = "float32"):
    tensor = _to_tensor(t, dtype)
    if dim is None:
        return tensor.squeeze()
    if isinstance(dim, list):
        out = tensor
        for d in sorted(dim, reverse=True):
            out = out.squeeze(d)
        return out
    return tensor.squeeze(dim)


def unsqueeze_tensor(t: List, dim: int, dtype: str = "float32"):
    return torch.unsqueeze(_to_tensor(t, dtype), dim)


def permute_tensor(t: List, dims: List[int], dtype: str = "float32"):
    return torch.permute(_to_tensor(t, dtype), tuple(dims))


def tile_tensor(t: List, dims: List[int], dtype: str = "float32"):
    return torch.tile(_to_tensor(t, dtype), tuple(dims))


def repeat_tensor(t: List, sizes: List[int], dtype: str = "float32"):
    return _to_tensor(t, dtype).repeat(*sizes)


def narrow_tensor(t: List, dim: int, start: int, length: int, dtype: str = "float32"):
    return torch.narrow(_to_tensor(t, dtype), dim=dim, start=start, length=length)


def chunk_tensor(t: List, chunks: int, dim: int = 0, dtype: str = "float32") -> List[torch.Tensor]:
    """Split tensor into up to `chunks` pieces along `dim`.
    May return fewer than `chunks` if the dim size is not evenly divisible.
    """
    return list(torch.chunk(_to_tensor(t, dtype), chunks=chunks, dim=dim))


def cat_tensors(tensors: List[List], dim: int = 0, dtype: str = "float32"):
    """Concatenate a sequence of tensors along an existing dimension.
    The first element of `tensors` is treated as the primary tensor.
    """
    ts = [_to_tensor(x, dtype) for x in tensors]
    return torch.cat(ts, dim=dim)


def stack_tensors(tensors: List[List], dim: int = 0, dtype: str = "float32"):
    """Stack a sequence of tensors along a NEW dimension.
    All tensors must have the same shape.
    The first element of `tensors` is treated as the primary tensor.
    """
    ts = [_to_tensor(x, dtype) for x in tensors]
    return torch.stack(ts, dim=dim)
