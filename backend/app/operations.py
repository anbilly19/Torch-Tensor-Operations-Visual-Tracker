import torch
import torch.nn.functional as F
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
    return list(torch.chunk(_to_tensor(t, dtype), chunks=chunks, dim=dim))

def cat_tensors(tensors: List[List], dim: int = 0, dtype: str = "float32"):
    return torch.cat([_to_tensor(x, dtype) for x in tensors], dim=dim)

def stack_tensors(tensors: List[List], dim: int = 0, dtype: str = "float32"):
    return torch.stack([_to_tensor(x, dtype) for x in tensors], dim=dim)


# ── Layer ops (shape-altering) ───────────────────────────────────────────────────

def linear_layer(
    t: List,
    out_features: int,
    dtype: str = "float32",
):
    """
    F.linear(input, weight, bias=None)
    weight is auto-initialised with kaiming_uniform.
    input shape: [*, in_features]  ->  output: [*, out_features]
    """
    tensor = _to_tensor(t, dtype)
    in_features = tensor.shape[-1]
    weight = torch.nn.init.kaiming_uniform_(
        torch.empty(out_features, in_features, dtype=tensor.dtype)
    )
    bias = torch.zeros(out_features, dtype=tensor.dtype)
    return F.linear(tensor, weight, bias)


def conv1d_layer(
    t: List,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dtype: str = "float32",
):
    """
    F.conv1d  input: [N, C_in, L]  ->  [N, C_out, L_out]
    L_out = floor((L + 2*padding - kernel_size) / stride) + 1
    Weight auto-initialised with kaiming_uniform.
    """
    tensor = _to_tensor(t, dtype)
    in_channels = tensor.shape[1]
    weight = torch.nn.init.kaiming_uniform_(
        torch.empty(out_channels, in_channels, kernel_size, dtype=tensor.dtype)
    )
    return F.conv1d(tensor, weight, stride=stride, padding=padding)


def conv2d_layer(
    t: List,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dtype: str = "float32",
):
    """
    F.conv2d  input: [N, C_in, H, W]  ->  [N, C_out, H_out, W_out]
    H_out = floor((H + 2*padding - kernel_size) / stride) + 1
    Weight auto-initialised with kaiming_uniform.
    """
    tensor = _to_tensor(t, dtype)
    in_channels = tensor.shape[1]
    weight = torch.nn.init.kaiming_uniform_(
        torch.empty(out_channels, in_channels, kernel_size, kernel_size, dtype=tensor.dtype)
    )
    return F.conv2d(tensor, weight, stride=stride, padding=padding)


def maxpool1d_layer(
    t: List,
    kernel_size: int = 2,
    stride: Optional[int] = None,
    dtype: str = "float32",
):
    """
    F.max_pool1d  input: [N, C, L]  ->  [N, C, L_out]
    stride defaults to kernel_size when None.
    """
    tensor = _to_tensor(t, dtype)
    return F.max_pool1d(tensor, kernel_size=kernel_size, stride=stride)


def maxpool2d_layer(
    t: List,
    kernel_size: int = 2,
    stride: Optional[int] = None,
    dtype: str = "float32",
):
    """
    F.max_pool2d  input: [N, C, H, W]  ->  [N, C, H_out, W_out]
    stride defaults to kernel_size when None.
    """
    tensor = _to_tensor(t, dtype)
    return F.max_pool2d(tensor, kernel_size=kernel_size, stride=stride)


def avgpool1d_layer(
    t: List,
    kernel_size: int = 2,
    stride: Optional[int] = None,
    dtype: str = "float32",
):
    """
    F.avg_pool1d  input: [N, C, L]  ->  [N, C, L_out]
    """
    tensor = _to_tensor(t, dtype)
    return F.avg_pool1d(tensor, kernel_size=kernel_size, stride=stride)


def avgpool2d_layer(
    t: List,
    kernel_size: int = 2,
    stride: Optional[int] = None,
    dtype: str = "float32",
):
    """
    F.avg_pool2d  input: [N, C, H, W]  ->  [N, C, H_out, W_out]
    """
    tensor = _to_tensor(t, dtype)
    return F.avg_pool2d(tensor, kernel_size=kernel_size, stride=stride)


def adaptive_avgpool2d_layer(
    t: List,
    output_size: List[int],
    dtype: str = "float32",
):
    """
    F.adaptive_avg_pool2d  input: [N, C, H, W]  ->  [N, C, output_size[0], output_size[1]]
    output_size can contain None to keep that dimension unchanged.
    """
    tensor = _to_tensor(t, dtype)
    return F.adaptive_avg_pool2d(tensor, output_size=tuple(output_size))


def embedding_layer(
    t: List,
    embed_dim: int,
    vocab_size: Optional[int] = None,
    dtype: str = "float32",
):
    """
    F.embedding  input: [N, seq_len] (int64 indices)  ->  [N, seq_len, embed_dim]
    vocab_size defaults to max(input) + 1 if not provided.
    Weight initialised from N(0,1).
    """
    indices = torch.tensor(t, dtype=torch.long)
    vs = vocab_size if vocab_size is not None else int(indices.max().item()) + 1
    weight = torch.randn(vs, embed_dim)
    return F.embedding(indices, weight)


def scaled_dot_product_attention_layer(
    query: List,
    key: List,
    value: List,
    dtype: str = "float32",
):
    """
    F.scaled_dot_product_attention
    query: [N, H, T, D]  key: [N, H, S, D]  value: [N, H, S, Dv]
    output: [N, H, T, Dv]
    """
    q = _to_tensor(query, dtype)
    k = _to_tensor(key,   dtype)
    v = _to_tensor(value, dtype)
    return F.scaled_dot_product_attention(q, k, v)
