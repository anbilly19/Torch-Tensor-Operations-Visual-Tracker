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

def add_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    result = t_a + t_b
    return result

def matmul_tensors(a: List, b: List, dtype: str = "float32"):
    t_a = torch.tensor(a, dtype=getattr(torch, dtype))
    t_b = torch.tensor(b, dtype=getattr(torch, dtype))
    result = torch.matmul(t_a, t_b)
    return result

def sum_tensor(t: List, dim: Optional[int] = None, keepdim: bool = False, dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    if dim is None:
        result = tensor.sum()
    else:
        result = tensor.sum(dim=dim, keepdim=keepdim)
    return result

def reshape_tensor(t: List, shape: List[int], dtype: str = "float32"):
    tensor = torch.tensor(t, dtype=getattr(torch, dtype))
    result = tensor.reshape(shape)
    return result