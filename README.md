# Torch Tensor Operations Visual Tracker

An interactive full-stack web application for exploring PyTorch tensor operations step-by-step — with live shape tracking, computation graph visualization, per-operation stats, and inline PyTorch documentation links.

> **Live demo** — the published URL is listed under the [Deployments](../../deployments) section of this repository (look for the latest `Production` deployment from Vercel).

---

## Features

- **31 tensor operations** across five categories: elementwise, reduction, shape & indexing, joining, and layers
- **Tensor import** — load tensors from `.pt`, `.pth`, `.npy`, `.npz`, or `.csv` files via drag-and-drop; supports state-dict key selection for multi-tensor files
- **Interactive computation graph** — DAG rendered with React Flow; hover tensor nodes for live stats, click op nodes to open PyTorch docs
- **Live statistics** — mean, std, min, max, sum, norm, numel, rank updated after each op
- **Operation history** — timestamped log of every applied operation
- **PyTorch docs pill** — one-click link to the official PyTorch API page for the selected op
- **Shape presets** — quick-select buttons in the tensor creator for common demo shapes
- **Fully responsive** — works on mobile, tablet, and desktop; controls panel slides in as a drawer on small screens

---

## Project Structure

```
.
├── frontend/                     # React + Vite + Tailwind UI
│   ├── api/
│   │   └── client.js             # Axios HTTP client
│   ├── src/
│   │   ├── components/
│   │   │   ├── TensorCreator.jsx     # Create / Import tab — file dropzone, key picker
│   │   │   ├── OperationPanel.jsx    # Select & configure ops, docs link
│   │   │   ├── TensorDisplay.jsx     # Shape / rank / numel / dtype summary
│   │   │   ├── GraphDisplay.jsx      # Interactive React Flow computation graph
│   │   │   ├── TensorStats.jsx       # Summary statistics
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
    │   ├── importer.py           # Tensor file loader (.pt/.pth/.npy/.npz/.csv)
    │   └── graph.py              # React Flow graph data builder
    └── pyproject.toml
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI framework | React 18 + Vite |
| Graph rendering | React Flow (@xyflow/react) |
| State management | Zustand |
| Styling | Tailwind CSS + shadcn/ui |
| HTTP client | Axios |
| API server | FastAPI |
| Tensor ops | PyTorch |
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
| Chunk | `POST /chunk` | `chunks`, `dim`; returns `TensorsResponse` |

### Joining

| Op | Endpoint | Notes |
|---|---|---|
| Cat | `POST /cat` | Concatenate along existing dim; `tensors[0]` is current tensor |
| Stack | `POST /stack` | Stack along new dim; all tensors must share shape |

### Layers (shape-altering)

| Op | Endpoint | Notes |
|---|---|---|
| Linear | `POST /linear` | `out_features`; weight auto-init Kaiming uniform |
| Conv1D | `POST /conv1d` | `out_channels`, `kernel_size`, `stride`, `padding` |
| Conv2D | `POST /conv2d` | Same as above, 2D |
| MaxPool1D | `POST /maxpool1d` | `kernel_size`, `stride` |
| MaxPool2D | `POST /maxpool2d` | `kernel_size`, `stride` |
| AvgPool1D | `POST /avgpool1d` | `kernel_size`, `stride` |
| AvgPool2D | `POST /avgpool2d` | `kernel_size`, `stride` |
| Adaptive AvgPool2D | `POST /adaptive-avgpool2d` | `output_size: [H, W]` |
| Embedding | `POST /embedding` | Integer indices in; `embed_dim`, optional `vocab_size` |
| Scaled Dot-Product Attention | `POST /sdpa` | Separate `query`, `key`, `value` inputs |

### Utilities

| Op | Endpoint | Notes |
|---|---|---|
| Stats | `POST /stats` | Read-only; returns mean, std, min, max, sum, norm, numel, rank |
| Create | `POST /create` | `op: "ones" \| "zeros"`, `shape` |
| **Import** | `POST /import-tensor` | `multipart/form-data`; optional `?key=` for dict files |
| Cumulative graph | `POST /cumulative-graph` | Returns `{ nodes, edges }` for React Flow |

---

## Tensor Import

The **Import** tab in the Tensor Source panel lets you load a real tensor from disk instead of creating one from scratch.

### Supported formats

| Format | Extension | Notes |
|---|---|---|
| PyTorch checkpoint | `.pt`, `.pth` | Single tensor or `state_dict`; key picker shown for dicts |
| NumPy array | `.npy` | Single array, auto-converted to `float32` |
| NumPy archive | `.npz` | Single array auto-loaded; key picker for multi-array archives |
| CSV | `.csv` | Rows × columns of numeric values; no header row |

### Key selection flow

When a `.pt` / `.pth` / `.npz` file contains multiple tensors (e.g. a model state dict), the backend returns the list of available keys. The UI shows a dropdown — select the tensor you want and hit **Load selected tensor**. The file is re-uploaded with `?key=<name>` appended.

### File size limit

50 MB enforced server-side before any parsing begins.

---

## Responsive Layout

The UI adapts across all screen sizes:

| Breakpoint | Layout |
|---|---|
| Mobile (`< lg`) | Single column; controls panel hidden by default, opens as a left-side drawer via **Controls** button in the header |
| Desktop (`lg+`) | Original side-by-side layout — `w-72` controls column + vertical separator + `flex-1` output column |

The computation graph canvas also scales: `320px` on mobile, `420px` on tablet, `480px` on desktop. The MiniMap is hidden on mobile to save space.

---

## Installation

### Backend

```bash
cd backend
pip install -e .
```

Requires Python ≥ 3.10.

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

1. **Create or import a tensor** — use the **Create** tab (ones/zeros + shape presets) or the **Import** tab (drag-and-drop a `.pt`, `.pth`, `.npy`, `.npz`, or `.csv` file)
2. **Select an operation** — grouped dropdown with five categories; a **PyTorch docs** pill links to the official API page
3. **Configure parameters** — contextual inputs appear for the chosen op (dim, shape, extra tensors, conv params, etc.)
4. **Apply** — the result shape, stats, and computation graph all update instantly
5. **Graph** — hover blue tensor nodes for full stats popover; click green op nodes to open PyTorch docs; pan, zoom, and minimap supported
6. **Chunk** — after a chunk op, an info box shows all chunk shapes; `currentTensor` follows `chunk[0]`
7. **SDPA** — current tensor is used as Query; provide Key and Value tensors in the param panel
8. **Mobile** — tap **Controls** in the header to slide open the operations panel

---

## API Response Models

```json
// TensorResponse
{ "data": [[1,2],[3,4]], "shape": [2,2], "dtype": "float32", "operation": "add" }

// TensorsResponse (chunk)
{ "tensors": [ { "data": ..., "shape": ..., "dtype": ..., "operation": "chunk" } ], "operation": "chunk" }

// ImportResponse — single tensor
{ "data": [[...]], "shape": [3,4], "dtype": "float32", "operation": "import", "keys": [], "key_used": null }

// ImportResponse — dict file (key selection required)
{ "data": null, "shape": null, "dtype": null, "operation": "import", "keys": ["layer1.weight", "layer1.bias", ...], "key_used": null }

// StatsResponse
{ "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "sum": 9.0, "norm": 3.0, "numel": 9, "rank": 2 }

// GraphResponse
{ "nodes": [ { "id": "t0", "type": "tensor", "label": "Tensor 0", "shape": [2,2], "stats": {...} } ],
  "edges": [ { "id": "e0_in", "source": "t0", "target": "op0" } ] }
```

---

## Configuration

| Setting | Default | Location |
|---|---|---|
| Backend port | `8000` | `uvicorn` CLI |
| CORS origins | `*` (dev) / regex Vercel (prod) | `backend/app/main.py` |
| Default dtype | `float32` | Per-request param |
| API base URL | `http://localhost:8000` | `VITE_API_URL` env var |
| Import file size limit | `50 MB` | `backend/app/importer.py` |

---

## Production Build

```bash
cd frontend
npm run build    # outputs to frontend/dist/
npm run preview  # local preview of production build
```
