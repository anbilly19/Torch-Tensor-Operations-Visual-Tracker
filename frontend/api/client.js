import axios from 'axios';

// Change this to your backend URL (FastAPI default is 8000)
const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' }
});

export default {
  // Create tensor
  createTensor(op, shape) {
    return apiClient.post('/create', { op, shape });
  },

  // Apply a single operation (for immediate result)
  applyOperation(opName, tensor, params = {}) {
    // Map opName to the correct endpoint
    const endpoints = {
      add: '/add',
      matmul: '/matmul',
      sum: '/sum',
      reshape: '/reshape',
      transpose: '/transpose'  // you may need to add this endpoint
    };
    const url = endpoints[opName];
    if (!url) throw new Error(`Unsupported operation: ${opName}`);

    // Build request body based on operation
    let body = {};
    if (opName === 'add' || opName === 'matmul') {
      body = {
        tensor_a: tensor,
        tensor_b: params.tensor_b
      };
    } else if (opName === 'sum') {
      body = {
        tensor: tensor,
        dim: params.dim,
        keepdim: params.keepdim
      };
    } else if (opName === 'reshape') {
      body = {
        tensor: tensor,
        shape: params.shape
      };
    } else if (opName === 'transpose') {
      body = {
        tensor: tensor,
        dim0: params.dim0,
        dim1: params.dim1
      };
    }
    return apiClient.post(url, body);
  },

  // Generate cumulative graph
  generateCumulativeGraph(originalTensor, operations) {
    return apiClient.post('/cumulative-graph', {
      original_tensor: originalTensor,
      operations: operations
    });
  }
};