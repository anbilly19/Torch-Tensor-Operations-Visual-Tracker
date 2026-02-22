import torch
from typing import List, Optional


def create_tensor(op: str, shape: List[int], dtype: str = "float32"):
    torch_dtype = getattr(torch, dtype)
    if op == "ones":
        tensor = torch.ones(shape, dtype=torch_dtype)
    elif op == "zeros":
        tensor = torch.zeros(shape, dtype=torch_dtype)
    else:
        raise ValueError("op must be 'ones' or 'zeros'")
    return tensor


# ── Binary ops ────────────────────────────────────────────────────────────────

def add_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    return t_a + t_b


def sub_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    return t_a - t_b


def mul_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    return t_a * t_b


def div_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    return t_a / t_b


def matmul_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    return torch.matmul(t_a, t_b)


# ── Unary ops ─────────────────────────────────────────────────────────────────

def abs_tensor(t: List, dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    return torch.abs(tensor)


def neg_tensor(t: List, dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    return torch.neg(tensor)


def clamp_tensor(
    t: List,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    dtype: str = "float32",
):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    return torch.clamp(tensor, min=min_val, max=max_val)


# ── Reduction ops ─────────────────────────────────────────────────────────────

def sum_tensor(t: List, dim: Optional[int] = None, keepdim: bool = False, dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    if dim is None:
        return tensor.sum()
    return tensor.sum(dim=dim, keepdim=keepdim)


def tensor_stats(t: List, dtype: str = "float32") -> dict:
    """Compute summary statistics over all elements of the tensor.

    Returns a flat dict of scalar floats:
        mean, std, min, max, sum, norm (L2), numel, rank
    """
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    flat = tensor.flatten()
    return {
        "mean":  float(flat.mean()),
        "std":   float(flat.std()) if flat.numel() > 1 else 0.0,
        "min":   float(flat.min()),
        "max":   float(flat.max()),
        "sum":   float(flat.sum()),
        "norm":  float(torch.linalg.norm(flat)),   # L2
        "numel": int(flat.numel()),
        "rank":  int(tensor.dim()),
    }


# ── Shape ops ─────────────────────────────────────────────────────────────────

def reshape_tensor(t: List, shape: List[int], dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    return tensor.reshape(shape)


def transpose_tensor(t: List, dim0: int = 0, dim1: int = 1, dtype: str = "float32"):
    """Swap two dimensions. Default dim0=0, dim1=1 gives classic 2-D transpose."""
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    return torch.transpose(tensor, dim0, dim1)
