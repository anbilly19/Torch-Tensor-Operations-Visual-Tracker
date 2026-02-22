import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// ── Endpoint map ──────────────────────────────────────────────────────────────
const ENDPOINTS = {
  add:       '/add',
  sub:       '/sub',
  mul:       '/mul',
  div:       '/div',
  matmul:    '/matmul',
  abs:       '/abs',
  neg:       '/neg',
  clamp:     '/clamp',
  sum:       '/sum',
  reshape:   '/reshape',
  transpose: '/transpose',
};

const BINARY_OPS = new Set(['add', 'sub', 'mul', 'div', 'matmul']);
const UNARY_OPS  = new Set(['abs', 'neg']);

// ── Request body builder ───────────────────────────────────────────────────
function buildBody(opName, tensor, params) {
  if (BINARY_OPS.has(opName)) return { tensor_a: tensor, tensor_b: params.tensor_b };
  if (UNARY_OPS.has(opName))  return { tensor };
  if (opName === 'clamp') {
    return {
      tensor,
      ...(params.min_val !== undefined && { min_val: params.min_val }),
      ...(params.max_val !== undefined && { max_val: params.max_val }),
    };
  }
  if (opName === 'sum')       return { tensor, dim: params.dim ?? null, keepdim: params.keepdim ?? false };
  if (opName === 'reshape')   return { tensor, shape: params.shape };
  if (opName === 'transpose') return { tensor, dim0: params.dim0 ?? 0, dim1: params.dim1 ?? 1 };
  throw new Error(`Unsupported operation: ${opName}`);
}

// ── Public API ────────────────────────────────────────────────────────────────
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
