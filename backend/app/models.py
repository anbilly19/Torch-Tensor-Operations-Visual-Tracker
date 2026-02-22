from pydantic import BaseModel, Field
from typing import List, Optional, Union


# ── Tensor creation ───────────────────────────────────────────────────────────

class CreateRequest(BaseModel):
    op: str = Field(..., description="'ones' or 'zeros'")
    shape: List[int] = Field(..., description="Tensor dimensions, e.g. [2,3]")
    dtype: Optional[str] = "float32"


# ── Binary op requests ────────────────────────────────────────────────────────

class BinaryOpRequest(BaseModel):
    """Used for add, sub, mul, div."""
    tensor_a: List
    tensor_b: List
    dtype: Optional[str] = "float32"


class MatMulRequest(BaseModel):
    tensor_a: List
    tensor_b: List
    dtype: Optional[str] = "float32"


# ── Unary op requests ─────────────────────────────────────────────────────────

class UnaryOpRequest(BaseModel):
    """Used for abs, neg."""
    tensor: List
    dtype: Optional[str] = "float32"


class ClampRequest(BaseModel):
    tensor: List
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    dtype: Optional[str] = "float32"


# ── Reduction requests ────────────────────────────────────────────────────────

class SumRequest(BaseModel):
    tensor: List
    dim: Optional[int] = None
    keepdim: bool = False
    dtype: Optional[str] = "float32"


class StatsRequest(BaseModel):
    tensor: List
    dtype: Optional[str] = "float32"


class StatsResponse(BaseModel):
    mean:  float
    std:   float
    min:   float
    max:   float
    sum:   float
    norm:  float
    numel: int
    rank:  int


# ── Shape & indexing requests ─────────────────────────────────────────────────

class ReshapeRequest(BaseModel):
    tensor: List
    shape: List[int]
    dtype: Optional[str] = "float32"


class TransposeRequest(BaseModel):
    tensor: List
    dim0: int = 0
    dim1: int = 1
    dtype: Optional[str] = "float32"


class FlattenRequest(BaseModel):
    tensor: List
    start_dim: int = 0
    end_dim: int = -1
    dtype: Optional[str] = "float32"


class SqueezeRequest(BaseModel):
    tensor: List
    dim: Optional[Union[int, List[int]]] = None
    dtype: Optional[str] = "float32"


class UnsqueezeRequest(BaseModel):
    tensor: List
    dim: int
    dtype: Optional[str] = "float32"


class PermuteRequest(BaseModel):
    tensor: List
    dims: List[int]
    dtype: Optional[str] = "float32"


class TileRequest(BaseModel):
    tensor: List
    dims: List[int]
    dtype: Optional[str] = "float32"


class RepeatRequest(BaseModel):
    tensor: List
    sizes: List[int]
    dtype: Optional[str] = "float32"


class NarrowRequest(BaseModel):
    tensor: List
    dim: int
    start: int
    length: int
    dtype: Optional[str] = "float32"


class ChunkRequest(BaseModel):
    tensor: List
    chunks: int
    dim: int = 0
    dtype: Optional[str] = "float32"


class CatRequest(BaseModel):
    """Concatenate along an existing dim. `tensors[0]` is the primary tensor."""
    tensors: List[List]
    dim: int = 0
    dtype: Optional[str] = "float32"


class StackRequest(BaseModel):
    """Stack along a NEW dim. All tensors must have the same shape. `tensors[0]` is the primary tensor."""
    tensors: List[List]
    dim: int = 0
    dtype: Optional[str] = "float32"


# ── Response models ───────────────────────────────────────────────────────────

class TensorResponse(BaseModel):
    data: List
    shape: List[int]
    dtype: str
    operation: str


class TensorsResponse(BaseModel):
    """Used by ops that return multiple tensors (e.g. chunk)."""
    tensors: List[TensorResponse]
    operation: str


class GraphResponse(BaseModel):
    image: str  # base64 PNG


class OperationStep(BaseModel):
    op: str
    params: dict = {}
    tensor_b: Optional[List] = None


class CumulativeGraphRequest(BaseModel):
    original_tensor: List
    operations: List[OperationStep]
