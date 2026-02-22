import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import models, operations, graph

app = FastAPI(title="PyTorch Tensor Operations API")

# ── CORS ───────────────────────────────────────────────────────────────────
_raw = os.environ.get("ALLOWED_ORIGINS", "")
_explicit = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_explicit if _explicit else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resp(tensor, operation: str) -> dict:
    return {
        "data": tensor.tolist(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).split(".")[-1],
        "operation": operation,
    }


def _resp_many(tensors, operation: str) -> dict:
    return {
        "tensors": [_resp(t, operation) for t in tensors],
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
        return _resp(operations.clamp_tensor(req.tensor, req.min_val, req.max_val, req.dtype), "clamp")
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
    try:
        return operations.tensor_stats(req.tensor, req.dtype)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Shape & indexing ops ──────────────────────────────────────────────────────

@app.post("/reshape", response_model=models.TensorResponse)
def reshape(req: models.ReshapeRequest):
    try:
        return _resp(operations.reshape_tensor(req.tensor, req.shape, req.dtype), "reshape")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transpose", response_model=models.TensorResponse)
def transpose(req: models.TransposeRequest):
    try:
        return _resp(operations.transpose_tensor(req.tensor, req.dim0, req.dim1, req.dtype), "transpose")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/flatten", response_model=models.TensorResponse)
def flatten(req: models.FlattenRequest):
    try:
        return _resp(operations.flatten_tensor(req.tensor, req.start_dim, req.end_dim, req.dtype), "flatten")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/squeeze", response_model=models.TensorResponse)
def squeeze(req: models.SqueezeRequest):
    try:
        return _resp(operations.squeeze_tensor(req.tensor, req.dim, req.dtype), "squeeze")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/unsqueeze", response_model=models.TensorResponse)
def unsqueeze(req: models.UnsqueezeRequest):
    try:
        return _resp(operations.unsqueeze_tensor(req.tensor, req.dim, req.dtype), "unsqueeze")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/permute", response_model=models.TensorResponse)
def permute(req: models.PermuteRequest):
    try:
        return _resp(operations.permute_tensor(req.tensor, req.dims, req.dtype), "permute")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tile", response_model=models.TensorResponse)
def tile(req: models.TileRequest):
    try:
        return _resp(operations.tile_tensor(req.tensor, req.dims, req.dtype), "tile")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/repeat", response_model=models.TensorResponse)
def repeat(req: models.RepeatRequest):
    try:
        return _resp(operations.repeat_tensor(req.tensor, req.sizes, req.dtype), "repeat")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/narrow", response_model=models.TensorResponse)
def narrow(req: models.NarrowRequest):
    try:
        return _resp(operations.narrow_tensor(req.tensor, req.dim, req.start, req.length, req.dtype), "narrow")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chunk", response_model=models.TensorsResponse)
def chunk(req: models.ChunkRequest):
    try:
        ts = operations.chunk_tensor(req.tensor, req.chunks, req.dim, req.dtype)
        return _resp_many(ts, "chunk")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cat", response_model=models.TensorResponse)
def cat(req: models.CatRequest):
    try:
        return _resp(operations.cat_tensors(req.tensors, req.dim, req.dtype), "cat")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stack", response_model=models.TensorResponse)
def stack(req: models.StackRequest):
    try:
        return _resp(operations.stack_tensors(req.tensors, req.dim, req.dtype), "stack")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Layer ops (shape-altering) ───────────────────────────────────────────────────

@app.post("/linear", response_model=models.TensorResponse)
def linear(req: models.LinearRequest):
    try:
        return _resp(operations.linear_layer(req.tensor, req.out_features, req.dtype), "linear")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/conv1d", response_model=models.TensorResponse)
def conv1d(req: models.Conv1dRequest):
    try:
        return _resp(
            operations.conv1d_layer(req.tensor, req.out_channels, req.kernel_size, req.stride, req.padding, req.dtype),
            "conv1d",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/conv2d", response_model=models.TensorResponse)
def conv2d(req: models.Conv2dRequest):
    try:
        return _resp(
            operations.conv2d_layer(req.tensor, req.out_channels, req.kernel_size, req.stride, req.padding, req.dtype),
            "conv2d",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/maxpool1d", response_model=models.TensorResponse)
def maxpool1d(req: models.MaxPool1dRequest):
    try:
        return _resp(operations.maxpool1d_layer(req.tensor, req.kernel_size, req.stride, req.dtype), "maxpool1d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/maxpool2d", response_model=models.TensorResponse)
def maxpool2d(req: models.MaxPool2dRequest):
    try:
        return _resp(operations.maxpool2d_layer(req.tensor, req.kernel_size, req.stride, req.dtype), "maxpool2d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/avgpool1d", response_model=models.TensorResponse)
def avgpool1d(req: models.AvgPool1dRequest):
    try:
        return _resp(operations.avgpool1d_layer(req.tensor, req.kernel_size, req.stride, req.dtype), "avgpool1d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/avgpool2d", response_model=models.TensorResponse)
def avgpool2d(req: models.AvgPool2dRequest):
    try:
        return _resp(operations.avgpool2d_layer(req.tensor, req.kernel_size, req.stride, req.dtype), "avgpool2d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/adaptive-avgpool2d", response_model=models.TensorResponse)
def adaptive_avgpool2d(req: models.AdaptiveAvgPool2dRequest):
    try:
        return _resp(operations.adaptive_avgpool2d_layer(req.tensor, req.output_size, req.dtype), "adaptive_avgpool2d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/embedding", response_model=models.TensorResponse)
def embedding(req: models.EmbeddingRequest):
    try:
        return _resp(operations.embedding_layer(req.tensor, req.embed_dim, req.vocab_size, req.dtype), "embedding")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sdpa", response_model=models.TensorResponse)
def sdpa(req: models.SDPARequest):
    try:
        return _resp(operations.scaled_dot_product_attention_layer(req.query, req.key, req.value, req.dtype), "sdpa")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Interactive graph ───────────────────────────────────────────────────────────

@app.post("/cumulative-graph", response_model=models.GraphResponse)
def cumulative_graph(req: models.CumulativeGraphRequest):
    try:
        data = graph.generate_graph_data(
            req.original_tensor,
            [step.dict() for step in req.operations],
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
