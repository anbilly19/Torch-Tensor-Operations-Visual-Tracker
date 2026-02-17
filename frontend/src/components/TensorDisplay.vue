<template>
  <div class="tensor-display">
    <h3>{{ title }}</h3>
    <div v-if="!tensor" class="empty">No tensor</div>
    <div v-else class="tensor-grid">
      <div v-for="(row, i) in tensor" :key="i" class="row">
        <span v-for="(val, j) in row" :key="j" class="cell">{{ format(val) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  tensor: Array,
  title: String
});

function format(val) {
  if (typeof val === 'number') {
    return val.toFixed(3);
  }
  return val;
}
</script>

<style scoped>
.tensor-grid {
  display: inline-block;
  background: #fafafa;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #ddd;
}
.row {
  display: flex;
}
.cell {
  width: 50px;
  text-align: right;
  padding: 5px;
  margin: 1px;
  background: #fff;
  border: 1px solid #eee;
}
</style>