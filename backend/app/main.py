from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import models, operations
from . import models, operations, graph

app = FastAPI(title="PyTorch Tensor Operations API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PyTorch Tensor Operations API. Use /docs for interactive docs."}

@app.post("/create", response_model=models.TensorResponse)
def create_tensor(req: models.CreateRequest):
    try:
        tensor = operations.create_tensor(req.op, req.shape, req.dtype)
        return {
            "data": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1],
            "operation": f"create_{req.op}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add", response_model=models.TensorResponse)
def add_tensors(req: models.BinaryOpRequest):
    try:
        tensor = operations.add_tensors(req.tensor_a, req.tensor_b, req.dtype)
        return {
            "data": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1],
            "operation": "add"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/matmul", response_model=models.TensorResponse)
def matmul(req: models.MatMulRequest):
    try:
        tensor = operations.matmul_tensors(req.tensor_a, req.tensor_b, req.dtype)
        return {
            "data": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1],
            "operation": "matmul"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sum", response_model=models.TensorResponse)
def sum_tensor(req: models.SumRequest):
    try:
        tensor = operations.sum_tensor(req.tensor, req.dim, req.keepdim, req.dtype)
        return {
            "data": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1],
            "operation": "sum"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reshape", response_model=models.TensorResponse)
def reshape(req: models.ReshapeRequest):
    try:
        tensor = operations.reshape_tensor(req.tensor, req.shape, req.dtype)
        return {
            "data": tensor.tolist(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1],
            "operation": "reshape"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/cumulative-graph", response_model=models.GraphResponse)
def cumulative_graph(req: models.CumulativeGraphRequest):
    try:
        image = graph.generate_shape_graph(
            req.original_tensor,
            [step.dict() for step in req.operations]
        )
        return {"image": image}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))