<template>
  <div class="history">
    <h3>History</h3>
    <ul>
      <li v-for="(item, idx) in store.history" :key="idx">
        {{ item.time }}: {{ item.desc }}
      </li>
    </ul>
    <button @click="store.reset()">Reset All</button>
  </div>
</template>

<script setup>
import { useTensorStore } from '../stores/tensorStore';
const store = useTensorStore();
</script>

<style scoped>
.history {
  margin-top: 20px;
}
ul {
  list-style: none;
  padding: 0;
  max-height: 200px;
  overflow-y: auto;
}
li {
  padding: 5px;
  border-bottom: 1px solid #eee;
  font-size: 0.9em;
}
</style>