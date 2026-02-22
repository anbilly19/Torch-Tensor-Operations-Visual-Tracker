from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import models, operations, graph

app = FastAPI(title="PyTorch Tensor Operations API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resp(tensor, operation: str) -> dict:
    """Build a TensorResponse dict from a torch.Tensor."""
    return {
        "data": tensor.tolist(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
        "operation": operation,
    }


@app.get("/")
def root():
    return {"message": "PyTorch Tensor Operations API. Use /docs for interactive docs."}


# ── Tensor creation ───────────────────────────────────────────────────────────

@app.post("/create", response_model=models.TensorResponse)
def create_tensor(req: models.CreateRequest):
    try:
        t = operations.create_tensor(req.op, req.shape, req.dtype)
        return _resp(t, f"create_{req.op}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Binary ops ────────────────────────────────────────────────────────────────

@app.post("/add", response_model=models.TensorResponse)
def add_tensors(req: models.BinaryOpRequest):
    try:
        return _resp(operations.add_tensors(req.tensor_a, req.tensor_b, req.dtype), "add")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sub", response_model=models.TensorResponse)
def sub_tensors(req: models.BinaryOpRequest):
    try:
        return _resp(operations.sub_tensors(req.tensor_a, req.tensor_b, req.dtype), "sub")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/mul", response_model=models.TensorResponse)
def mul_tensors(req: models.BinaryOpRequest):
    try:
        return _resp(operations.mul_tensors(req.tensor_a, req.tensor_b, req.dtype), "mul")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/div", response_model=models.TensorResponse)
def div_tensors(req: models.BinaryOpRequest):
    try:
        return _resp(operations.div_tensors(req.tensor_a, req.tensor_b, req.dtype), "div")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/matmul", response_model=models.TensorResponse)
def matmul(req: models.MatMulRequest):
    try:
        return _resp(operations.matmul_tensors(req.tensor_a, req.tensor_b, req.dtype), "matmul")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Unary ops ─────────────────────────────────────────────────────────────────

@app.post("/abs", response_model=models.TensorResponse)
def abs_tensor(req: models.UnaryOpRequest):
    try:
        return _resp(operations.abs_tensor(req.tensor, req.dtype), "abs")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/neg", response_model=models.TensorResponse)
def neg_tensor(req: models.UnaryOpRequest):
    try:
        return _resp(operations.neg_tensor(req.tensor, req.dtype), "neg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clamp", response_model=models.TensorResponse)
def clamp_tensor(req: models.ClampRequest):
    try:
        return _resp(
            operations.clamp_tensor(req.tensor, req.min_val, req.max_val, req.dtype),
            "clamp",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Reduction ops ─────────────────────────────────────────────────────────────

@app.post("/sum", response_model=models.TensorResponse)
def sum_tensor(req: models.SumRequest):
    try:
        return _resp(operations.sum_tensor(req.tensor, req.dim, req.keepdim, req.dtype), "sum")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stats", response_model=models.StatsResponse)
def tensor_stats(req: models.StatsRequest):
    """Return summary statistics for the given tensor (read-only, no state change)."""
    try:
        return operations.tensor_stats(req.tensor, req.dtype)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Shape ops ─────────────────────────────────────────────────────────────────

@app.post("/reshape", response_model=models.TensorResponse)
def reshape(req: models.ReshapeRequest):
    try:
        return _resp(operations.reshape_tensor(req.tensor, req.shape, req.dtype), "reshape")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/transpose", response_model=models.TensorResponse)
def transpose(req: models.TransposeRequest):
    try:
        return _resp(
            operations.transpose_tensor(req.tensor, req.dim0, req.dim1, req.dtype),
            "transpose",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Graph ─────────────────────────────────────────────────────────────────────

@app.post("/cumulative-graph", response_model=models.GraphResponse)
def cumulative_graph(req: models.CumulativeGraphRequest):
    try:
        image = graph.generate_shape_graph(
            req.original_tensor,
            [step.dict() for step in req.operations],
        )
        return {"image": image}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
