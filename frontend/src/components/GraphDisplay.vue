<template>
  <div class="graph-display">
    <h3>Computation Graph</h3>
    <div v-if="!image" class="empty">No graph yet. Apply an operation.</div>
    <img v-else :src="image" alt="Computation graph" />
  </div>
</template>

<script setup>
defineProps({
  image: String
});
</script>

<style scoped>
img {
  max-width: 100%;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.empty {
  color: #999;
  font-style: italic;
}
</style>