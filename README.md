# Torch Tensor Operations Visual Tracker

An interactive full-stack web application for exploring PyTorch tensor operations step-by-step — with live shape tracking, computation graph visualization, per-operation stats, and inline PyTorch documentation links.

---

## Features

- **21 tensor operations** across four categories: elementwise, reduction, shape & indexing, and joining
- **Computation graph** — DAG rendered with Graphviz, showing tensor shapes at every step
- **Live statistics** — mean, std, min, max, sum, norm, numel, rank updated after each op
- **Operation history** — timestamped log of every applied operation
- **PyTorch docs pill** — one-click link to the official PyTorch API page for the selected op
- **Shape presets** — quick-select buttons in the tensor creator for common demo shapes

---

## Project Structure

```
.
├── frontend/                     # React + Vite + Tailwind UI
│   ├── api/
│   │   └── client.js             # Axios HTTP client
│   ├── src/
│   │   ├── components/
│   │   │   ├── TensorCreator.jsx     # Create tensors with shape presets
│   │   │   ├── OperationPanel.jsx    # Select & configure ops, docs link
│   │   │   ├── TensorDisplay.jsx     # Grid view of current tensor values
│   │   │   ├── GraphDisplay.jsx      # Computation graph (base64 PNG)
│   │   │   ├── StatsPanel.jsx        # Summary statistics
│   │   │   └── HistoryList.jsx       # Timestamped op log
│   │   ├── stores/
│   │   │   └── tensorStore.js        # Zustand global state
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── backend/                      # FastAPI + PyTorch backend
    ├── app/
    │   ├── main.py               # Routes
    │   ├── models.py             # Pydantic request / response models
    │   ├── operations.py         # PyTorch operation implementations
    │   └── graph.py              # Graphviz DAG builder
    └── pyproject.toml
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI framework | React 18 + Vite |
| State management | Zustand |
| Styling | Tailwind CSS + shadcn/ui |
| HTTP client | Axios |
| API server | FastAPI |
| Tensor ops | PyTorch |
| Graph rendering | Graphviz (Python) |
| Data validation | Pydantic v2 |

---

## Supported Operations

### Elementwise / Scalar

| Op | Endpoint | Notes |
|---|---|---|
| Add | `POST /add` | Element-wise, broadcasts |
| Subtract | `POST /sub` | Element-wise |
| Multiply | `POST /mul` | Element-wise |
| Divide | `POST /div` | Element-wise |
| Matrix Multiply | `POST /matmul` | `torch.matmul` — supports batched |
| Absolute Value | `POST /abs` | |
| Negate | `POST /neg` | |
| Clamp | `POST /clamp` | Optional `min_val`, `max_val` |

### Reduction

| Op | Endpoint | Notes |
|---|---|---|
| Sum | `POST /sum` | Optional `dim`, `keepdim` |

### Shape & Indexing

| Op | Endpoint | Notes |
|---|---|---|
| Reshape | `POST /reshape` | `shape: [int, ...]` |
| Transpose | `POST /transpose` | `dim0`, `dim1` |
| Flatten | `POST /flatten` | `start_dim`, `end_dim` |
| Squeeze | `POST /squeeze` | Optional `dim`; squeezes all size-1 dims if omitted |
| Unsqueeze | `POST /unsqueeze` | `dim` |
| Permute | `POST /permute` | `dims: [int, ...]` |
| Tile | `POST /tile` | `dims: [int, ...]` repeat counts per axis |
| Repeat | `POST /repeat` | `sizes: [int, ...]` copies per axis |
| Narrow | `POST /narrow` | `dim`, `start`, `length` |
| Chunk | `POST /chunk` | `chunks`, `dim`; returns `TensorsResponse` (may be < `chunks` pieces) |

### Joining

| Op | Endpoint | Notes |
|---|---|---|
| Cat | `POST /cat` | Concatenate along existing dim; `tensors[0]` is current tensor |
| Stack | `POST /stack` | Stack along new dim; all tensors must share shape |

### Utilities

| Op | Endpoint | Notes |
|---|---|---|
| Stats | `POST /stats` | Read-only; returns mean, std, min, max, sum, norm, numel, rank |
| Create | `POST /create` | `op: "ones" \| "zeros"`, `shape` |
| Cumulative graph | `POST /cumulative-graph` | Returns base64 PNG of full op DAG |

---

## Installation

### Backend

```bash
cd backend
pip install -e .
```

Requires Python ≥ 3.10 and the `graphviz` system binary:

```bash
# macOS
brew install graphviz

# Ubuntu / Debian
sudo apt install graphviz
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running

```bash
# Terminal 1 — backend
cd backend
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs

# Terminal 2 — frontend
cd frontend
npm run dev
# UI available at http://localhost:5173
```

---

## Usage

1. **Create a tensor** — pick `ones` or `zeros`, type a shape or click a preset (`2×2`, `3×3`, `4×3`, `2×1×2`, `1×3×3`)
2. **Select an operation** — grouped dropdown; a **PyTorch docs** pill appears next to the label linking to the official API page
3. **Configure parameters** — contextual inputs appear for the chosen op (dim, shape, extra tensors, etc.)
4. **Apply** — the result tensor, stats, and computation graph all update instantly
5. **Chunk** — after a chunk op, an info box shows all chunk shapes; `currentTensor` follows `chunk[0]`
6. **Cat / Stack** — extra tensors textarea is pre-seeded with a same-shape tensor; current tensor is always `tensors[0]`

---

## API Response Models

```json
// TensorResponse (single tensor)
{ "data": [[1,2],[3,4]], "shape": [2,2], "dtype": "float32", "operation": "add" }

// TensorsResponse (chunk)
{ "tensors": [ { "data": ..., "shape": ..., "dtype": ..., "operation": "chunk" }, ... ], "operation": "chunk" }

// StatsResponse
{ "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "sum": 9.0, "norm": 3.0, "numel": 9, "rank": 2 }

// GraphResponse
{ "image": "data:image/png;base64,..." }
```

---

## Configuration

| Setting | Default | Location |
|---|---|---|
| Backend port | `8000` | `uvicorn` CLI |
| CORS origins | `*` | `backend/app/main.py` |
| Default dtype | `float32` | Per-request param |
| API base URL | `http://localhost:8000` | `frontend/api/client.js` |

---

## Production Build

```bash
cd frontend
npm run build    # outputs to frontend/dist/
npm run preview  # local preview of production build
```
