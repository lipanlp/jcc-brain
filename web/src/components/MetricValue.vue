<script setup lang="ts">
import { computed } from 'vue';
const props = defineProps<{
    value: string | number | null | undefined;
}>();
const display = computed(() => props.value ?? '—');
</script>
<template>
  <span class="metric-number numeric">
    <Transition name="metric" mode="out-in">
      <span :key="display">{{ display }}</span>
    </Transition>
  </span>
</template>
<style scoped>
.metric-number {
  display: inline-block;
  min-width: 1ch;
}
.metric-enter-active,.metric-leave-active {
  transition: opacity var(--duration-metric) var(--ease);
}
.metric-enter-from,.metric-leave-to {
  opacity: 0;
}
</style>
