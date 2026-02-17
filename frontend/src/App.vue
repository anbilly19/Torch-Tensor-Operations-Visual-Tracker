<template>
  <div class="app">
    <h1>🔬 PyTorch Tensor Playground</h1>
    <p>Create a tensor, then apply operations step by step. The graph updates after each step.</p>

    <div class="main-panel">
      <div class="left-panel">
        <TensorCreator />
        <OperationPanel />
        <HistoryList />
      </div>
      <div class="right-panel">
        <TensorDisplay :tensor="store.currentTensor" title="Current Tensor" />
        <GraphDisplay :image="store.graphImage" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { useTensorStore } from './stores/tensorStore';
import TensorCreator from './components/TensorCreator.vue';
import OperationPanel from './components/OperationPanel.vue';
import TensorDisplay from './components/TensorDisplay.vue';
import GraphDisplay from './components/GraphDisplay.vue';
import HistoryList from './components/HistoryList.vue';

const store = useTensorStore();
</script>

<style>
/* Global styles */
body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 20px;
  background: #f0f2f5;
}
.app {
  max-width: 1400px;
  margin: 0 auto;
}
.main-panel {
  display: flex;
  gap: 20px;
  margin-top: 20px;
}
.left-panel {
  flex: 1;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.right-panel {
  flex: 2;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>