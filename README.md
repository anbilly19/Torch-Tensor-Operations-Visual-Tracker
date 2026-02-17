# PyTorch Tensor Playground

A full-stack web application for visualizing PyTorch tensor operations and their computation graphs in real-time.

## Overview

**PyTorch Tensor Playground** is an interactive tool that allows you to:
- Create tensors with various initialization methods
- Apply tensor operations step by step
- Visualize the computation graph showing tensor shapes through the pipeline
- Track operation history with timestamps

## Project Structure

```
pytorch-viz/
├── frontend/          # Vue 3 + Vite web application
│   ├── src/
│   │   ├── components/     # Vue components for UI
│   │   ├── stores/         # Pinia state management
│   │   ├── assets/         # CSS and static files
│   │   └── main.js         # Entry point
│   ├── api/
│   │   └── client.js       # Axios HTTP client for backend
│   ├── package.json
│   └── vite.config.js
└── backend/           # FastAPI Python backend
    ├── app/
    │   ├── main.py         # FastAPI application and routes
    │   ├── models.py       # Pydantic request/response models
    │   ├── operations.py   # PyTorch tensor operations
    │   └── graph.py        # Computation graph visualization
    └── pyproject.toml
```

## Features

### Frontend (`frontend/`)
- **Vue 3** with `<script setup>` syntax
- **Pinia** for state management with `useTensorStore`
- **Axios** for API communication
- **Vite** for fast development and optimized builds

**Components:**
- `TensorCreator.vue` - Create tensors (ones/zeros with custom shapes)
- `OperationPanel.vue` - Apply operations with dynamic parameters
- `TensorDisplay.vue` - View current tensor values in grid format
- `GraphDisplay.vue` - Visualize computation graph as PNG image
- `HistoryList.vue` - Track operation history with timestamps

### Backend (`backend/`)
- **FastAPI** REST API server
- **PyTorch** for tensor operations
- **Graphviz** for computation graph generation
- **CORS** enabled for frontend access

**Supported Operations:**
- `create` - Initialize tensor (ones/zeros)
- `add` - Element-wise addition
- `matmul` - Matrix multiplication
- `sum` - Reduce with optional dimension
- `reshape` - Change tensor shape
- `transpose` - Swap dimensions
- `cumulative-graph` - Generate visual graph of all operations

## Installation

### Backend Setup

```bash
cd backend
pip install -e .
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The application will be available at `http://localhost:5173`

## Usage

1. **Create a Tensor**: Select initialization method (ones/zeros) and specify shape
2. **Apply Operations**: Choose an operation and provide required parameters
3. **View Results**: The current tensor is displayed in the right panel
4. **Inspect Graph**: After applying operations, see the computation graph
5. **Check History**: All operations are logged with timestamps

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/create` | Create initial tensor |
| POST | `/add` | Add two tensors |
| POST | `/matmul` | Matrix multiply |
| POST | `/sum` | Sum tensor elements |
| POST | `/reshape` | Reshape tensor |
| POST | `/transpose` | Transpose dimensions |
| POST | `/cumulative-graph` | Generate computation graph |

## Configuration

### Backend
- Default port: `8000`
- CORS origins: `*` (configure for production)
- PyTorch dtype: `float32` (default, customizable per request)

### Frontend
- API base URL: `http://localhost:8000` (configured in `frontend/api/client.js`)
- Build output: `dist/`

## Technologies Used

### Frontend
- Vue 3
- Pinia
- Axios
- Vite

### Backend
- FastAPI
- PyTorch
- Graphviz
- Pydantic

## Development

### Build Frontend for Production

```bash
cd frontend
npm run build
```

### Preview Production Build

```bash
cd frontend
npm run preview
```

