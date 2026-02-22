import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// ── Endpoint map ──────────────────────────────────────────────────────────────
const ENDPOINTS = {
  // Elementwise / scalar
  add:       '/add',
  sub:       '/sub',
  mul:       '/mul',
  div:       '/div',
  matmul:    '/matmul',
  abs:       '/abs',
  neg:       '/neg',
  clamp:     '/clamp',
  // Reduction
  sum:       '/sum',
  // Shape & indexing
  reshape:   '/reshape',
  transpose: '/transpose',
  flatten:   '/flatten',
  squeeze:   '/squeeze',
  unsqueeze: '/unsqueeze',
  permute:   '/permute',
  tile:      '/tile',
  repeat:    '/repeat',
  narrow:    '/narrow',
  chunk:     '/chunk',
  cat:       '/cat',
  stack:     '/stack',
};

const BINARY_OPS = new Set(['add', 'sub', 'mul', 'div', 'matmul']);
const UNARY_OPS  = new Set(['abs', 'neg']);

// ── Request body builder ──────────────────────────────────────────────────────
function buildBody(opName, tensor, params) {
  if (BINARY_OPS.has(opName)) return { tensor_a: tensor, tensor_b: params.tensor_b };
  if (UNARY_OPS.has(opName))  return { tensor };

  switch (opName) {
    case 'clamp':
      return {
        tensor,
        ...(params.min_val !== undefined && { min_val: params.min_val }),
        ...(params.max_val !== undefined && { max_val: params.max_val }),
      };
    case 'sum':
      return { tensor, dim: params.dim ?? null, keepdim: params.keepdim ?? false };
    case 'reshape':
      return { tensor, shape: params.shape };
    case 'transpose':
      return { tensor, dim0: params.dim0 ?? 0, dim1: params.dim1 ?? 1 };
    case 'flatten':
      return { tensor, start_dim: params.start_dim ?? 0, end_dim: params.end_dim ?? -1 };
    case 'squeeze':
      return { tensor, ...(params.dim !== undefined && params.dim !== '' && { dim: params.dim }) };
    case 'unsqueeze':
      return { tensor, dim: params.dim ?? 0 };
    case 'permute':
      return { tensor, dims: params.dims };
    case 'tile':
      return { tensor, dims: params.dims };
    case 'repeat':
      return { tensor, sizes: params.sizes };
    case 'narrow':
      return { tensor, dim: params.dim ?? 0, start: params.start ?? 0, length: params.length ?? 1 };
    case 'chunk':
      return { tensor, chunks: params.chunks ?? 2, dim: params.dim ?? 0 };
    case 'cat':
      return { tensors: params.tensors, dim: params.dim ?? 0 };
    case 'stack':
      return { tensors: params.tensors, dim: params.dim ?? 0 };
    default:
      throw new Error(`Unsupported operation: ${opName}`);
  }
}

// ── Public API ─────────────────────────────────────────────────────────────────
export default {
  createTensor(op, shape) {
    return apiClient.post('/create', { op, shape });
  },

  applyOperation(opName, tensor, params = {}) {
    const url = ENDPOINTS[opName];
    if (!url) throw new Error(`Unknown operation: ${opName}`);
    return apiClient.post(url, buildBody(opName, tensor, params));
  },

  fetchStats(tensor) {
    return apiClient.post('/stats', { tensor });
  },

  generateCumulativeGraph(originalTensor, operations) {
    return apiClient.post('/cumulative-graph', {
      original_tensor: originalTensor,
      operations,
    });
  },
};
