<template>
  <div class="creator">
    <h2>Create Tensor</h2>
    <select v-model="selectedOp">
      <option value="ones">Ones</option>
      <option value="zeros">Zeros</option>
    </select>
    <input v-model="shapeInput" placeholder="shape (comma-separated)" />
    <button @click="create">Create</button>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useTensorStore } from '../stores/tensorStore';

const store = useTensorStore();
const selectedOp = ref('ones');
const shapeInput = ref('3,3');

async function create() {
  const shape = shapeInput.value.split(',').map(Number);
  await store.createTensor(selectedOp.value, shape);
}
</script>

<style scoped>
.creator {
  margin-bottom: 20px;
}
select, input {
  margin-right: 10px;
  padding: 5px;
}
</style>