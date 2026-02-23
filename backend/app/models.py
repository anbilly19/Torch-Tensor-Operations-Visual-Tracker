from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


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
    tensors: List[List]
    dim: int = 0
    dtype: Optional[str] = "float32"


class StackRequest(BaseModel):
    tensors: List[List]
    dim: int = 0
    dtype: Optional[str] = "float32"


# ── Layer op requests ───────────────────────────────────────────────────────────

class LinearRequest(BaseModel):
    tensor: List
    out_features: int
    dtype: Optional[str] = "float32"


class Conv1dRequest(BaseModel):
    tensor: List
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    dtype: Optional[str] = "float32"


class Conv2dRequest(BaseModel):
    tensor: List
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    dtype: Optional[str] = "float32"


class MaxPool1dRequest(BaseModel):
    tensor: List
    kernel_size: int = 2
    stride: Optional[int] = None
    dtype: Optional[str] = "float32"


class MaxPool2dRequest(BaseModel):
    tensor: List
    kernel_size: int = 2
    stride: Optional[int] = None
    dtype: Optional[str] = "float32"


class AvgPool1dRequest(BaseModel):
    tensor: List
    kernel_size: int = 2
    stride: Optional[int] = None
    dtype: Optional[str] = "float32"


class AvgPool2dRequest(BaseModel):
    tensor: List
    kernel_size: int = 2
    stride: Optional[int] = None
    dtype: Optional[str] = "float32"


class AdaptiveAvgPool2dRequest(BaseModel):
    tensor: List
    output_size: List[int] = Field(..., description="[H_out, W_out]")
    dtype: Optional[str] = "float32"


class EmbeddingRequest(BaseModel):
    tensor: List
    embed_dim: int
    vocab_size: Optional[int] = None
    dtype: Optional[str] = "float32"


class SDPARequest(BaseModel):
    """Scaled Dot-Product Attention."""
    query: List
    key:   List
    value: List
    dtype: Optional[str] = "float32"


# ── Response models ───────────────────────────────────────────────────────────

class TensorResponse(BaseModel):
    data: List
    shape: List[int]
    dtype: str
    operation: str


class TensorsResponse(BaseModel):
    tensors: List[TensorResponse]
    operation: str


# ── Import response ──────────────────────────────────────────────────────────

class ImportResponse(BaseModel):
    """Returned by POST /import-tensor.

    Two modes:
    1. data is populated  -> tensor loaded, ready to use
    2. data is None       -> file is a dict; frontend must re-upload with ?key=
    """
    data:         Optional[List]     = None
    shape:        Optional[List[int]] = None
    dtype:        Optional[str]      = None
    operation:    str                = "import"
    keys:         List[str]          = []
    key_used:     Optional[str]      = None


# ── Interactive graph response ─────────────────────────────────────────────────

class TensorStats(BaseModel):
    mean:  float
    std:   float
    min:   float
    max:   float
    sum:   float
    norm:  float
    numel: int
    rank:  int


class GraphNode(BaseModel):
    id:       str
    type:     str
    label:    str
    shape:    Optional[List[int]] = None
    stats:    Optional[TensorStats] = None
    docs_url: Optional[str] = None


class GraphEdge(BaseModel):
    id:     str
    source: str
    target: str


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ── Cumulative graph request ───────────────────────────────────────────────────

class OperationStep(BaseModel):
    op: str
    params: dict = {}
    tensor_b: Optional[List] = None


class CumulativeGraphRequest(BaseModel):
    original_tensor: List
    operations: List[OperationStep]
