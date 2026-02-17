import { defineStore } from 'pinia';
import api from '../../api/client';

export const useTensorStore = defineStore('tensor', {
  state: () => ({
    originalTensor: null,        // the first tensor created
    currentTensor: null,          // latest result (as nested list)
    operations: [],                // list of applied operations
    graphImage: null,              // base64 image of cumulative graph
    history: []                    // simple text history
  }),

  actions: {
    async createTensor(op, shape) {
      try {
        const response = await api.createTensor(op, shape);
        const tensor = response.data.data;
        this.originalTensor = tensor;
        this.currentTensor = tensor;
        this.operations = [];
        this.graphImage = null;
        this.addHistory(`Created tensor with ${op} shape ${shape}`);
      } catch (error) {
        console.error('Creation failed:', error);
        throw error;
      }
    },

    async applyOperation(opName, params = {}) {
      if (!this.currentTensor) {
        alert('Please create a tensor first.');
        return;
      }

      try {
        // Call the backend to get the result
        const response = await api.applyOperation(opName, this.currentTensor, params);
        const resultTensor = response.data.data;

        // Record the operation
        const opRecord = {
          op: opName,
          params: params,
          tensor_b: params.tensor_b // for binary ops
        };
        this.operations.push(opRecord);
        this.currentTensor = resultTensor;
        this.addHistory(`Applied ${opName}`);

        // Regenerate cumulative graph
        await this.generateGraph();
      } catch (error) {
        console.error('Operation failed:', error);
        alert('Operation failed: ' + error.message);
      }
    },

    async generateGraph() {
      if (!this.originalTensor || this.operations.length === 0) {
        this.graphImage = null;
        return;
      }
      try {
        const response = await api.generateCumulativeGraph(this.originalTensor, this.operations);
        this.graphImage = response.data.image;
      } catch (error) {
        console.error('Graph generation failed:', error);
      }
    },

    addHistory(description) {
      this.history.push({
        time: new Date().toLocaleTimeString(),
        desc: description
      });
    },

    reset() {
      this.originalTensor = null;
      this.currentTensor = null;
      this.operations = [];
      this.graphImage = null;
      this.history = [];
    }
  }
});