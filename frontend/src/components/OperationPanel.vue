<template>
  <div class="operation-panel">
    <h2>Apply Operation</h2>
    <select v-model="selectedOp">
      <option value="add">Add (requires second tensor)</option>
      <option value="matmul">Matrix Multiply (requires second tensor)</option>
      <option value="sum">Sum (optional dimension)</option>
      <option value="reshape">Reshape</option>
      <option value="transpose">Transpose</option>
    </select>

    <!-- Dynamic parameters based on selectedOp -->
    <div v-if="selectedOp === 'add' || selectedOp === 'matmul'" class="params">
      <label>Second tensor (as JSON list):</label>
      <textarea v-model="tensorB" placeholder="e.g. [[1,2],[3,4]]"></textarea>
    </div>

    <div v-if="selectedOp === 'sum'" class="params">
      <label>Dimension (leave empty for all):</label>
      <input v-model.number="sumDim" type="number" />
      <label>Keep dimensions:</label>
      <input v-model="keepdim" type="checkbox" />
    </div>

    <div v-if="selectedOp === 'reshape'" class="params">
      <label>New shape (comma-separated):</label>
      <input v-model="reshapeShape" placeholder="e.g. 2,3" />
    </div>

    <div v-if="selectedOp === 'transpose'" class="params">
      <label>dim0:</label>
      <input v-model.number="dim0" type="number" />
      <label>dim1:</label>
      <input v-model.number="dim1" type="number" />
    </div>

    <button @click="apply">Apply</button>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useTensorStore } from '../stores/tensorStore';

const store = useTensorStore();
const selectedOp = ref('add');
const tensorB = ref('[[5,6],[7,8]]');
const sumDim = ref(null);
const keepdim = ref(false);
const reshapeShape = ref('2,3');
const dim0 = ref(0);
const dim1 = ref(1);

async function apply() {
  let params = {};
  if (selectedOp.value === 'add' || selectedOp.value === 'matmul') {
    try {
      params.tensor_b = JSON.parse(tensorB.value);
    } catch (e) {
      alert('Invalid JSON for second tensor');
      return;
    }
  } else if (selectedOp.value === 'sum') {
    params.dim = sumDim.value;
    params.keepdim = keepdim.value;
  } else if (selectedOp.value === 'reshape') {
    params.shape = reshapeShape.value.split(',').map(Number);
  } else if (selectedOp.value === 'transpose') {
    params.dim0 = dim0.value;
    params.dim1 = dim1.value;
  }
  await store.applyOperation(selectedOp.value, params);
}
</script>

<style scoped>
.operation-panel {
  margin-bottom: 20px;
}
.params {
  margin: 10px 0;
  padding: 10px;
  background: #f9f9f9;
  border-radius: 4px;
}
label {
  display: block;
  margin: 5px 0;
}
textarea {
  width: 100%;
  font-family: monospace;
}
</style>