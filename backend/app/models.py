from pydantic import BaseModel, Field
from typing import List, Optional

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


# ── Reduction / shape requests ────────────────────────────────────────────────

class SumRequest(BaseModel):
    tensor: List
    dim: Optional[int] = None
    keepdim: bool = False
    dtype: Optional[str] = "float32"


class ReshapeRequest(BaseModel):
    tensor: List
    shape: List[int]
    dtype: Optional[str] = "float32"


# ── Response models ───────────────────────────────────────────────────────────

class TensorResponse(BaseModel):
    data: List
    shape: List[int]
    dtype: str
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
