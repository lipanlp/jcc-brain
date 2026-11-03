<script setup lang="ts">
withDefaults(defineProps<{
    tone?: 'gray' | 'blue' | 'green' | 'orange' | 'red' | 'yellow' | 'purple';
}>(), { tone: 'gray' });
</script>
<template>
  <span class="u-badge" :class="tone">
    <slot />
  </span>
</template>
<style scoped>
.u-badge {
  position: relative;
  isolation: isolate;
  display: inline-flex;
  align-items: center;
  padding: var(--space-half) var(--space-2);
  border-radius: var(--radius-pill);
  font-size: var(--font-xs);
  font-weight: 500;
  white-space: nowrap;
  color: var(--secondary);
}
.u-badge::before {
  content: '';
  inset: 0;
  position: absolute;
  border-radius: inherit;
  background: currentColor;
  opacity: .12;
  z-index: -1;
}
.blue {
  color: var(--blue-text);
}
.green {
  color: var(--green-text);
}
.orange {
  color: var(--orange-text);
}
.red {
  color: var(--red-text);
}
.yellow {
  color: var(--yellow-text);
}
.purple {
  color: var(--purple-text);
}
</style>
