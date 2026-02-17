from pydantic import BaseModel, Field
from typing import List, Union, Optional

# For tensor data we accept nested lists
TensorData = List

class CreateRequest(BaseModel):
    op: str = Field(..., description="'ones' or 'zeros'")
    shape: List[int] = Field(..., description="Tensor dimensions, e.g. [2,3]")
    dtype: Optional[str] = "float32"  # optional, default float32

class BinaryOpRequest(BaseModel):
    tensor_a: List  # nested list
    tensor_b: List
    dtype: Optional[str] = "float32"

class MatMulRequest(BaseModel):
    tensor_a: List
    tensor_b: List
    dtype: Optional[str] = "float32"

class SumRequest(BaseModel):
    tensor: List
    dim: Optional[int] = None  # if None, sum all elements
    keepdim: bool = False
    dtype: Optional[str] = "float32"

class ReshapeRequest(BaseModel):
    tensor: List
    shape: List[int]
    dtype: Optional[str] = "float32"

# Response model
class TensorResponse(BaseModel):
    data: List  # result as nested list
    shape: List[int]
    dtype: str
    operation: str

class GraphRequest(BaseModel):
    operation: str
    tensors: List[List]  # list of tensor data (nested lists)
    params: Optional[dict] = {}  # optional parameters (e.g., shape for reshape)

class GraphResponse(BaseModel):
    image: str  # base64 PNG

class OperationStep(BaseModel):
    op: str                 # operation name, e.g., "add", "matmul"
    params: dict = {}       # parameters like dim, shape, etc.
    tensor_b: Optional[List] = None  # for binary ops, second tensor

class CumulativeGraphRequest(BaseModel):
    original_tensor: List
    operations: List[OperationStep]