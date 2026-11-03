<script setup lang="ts">
withDefaults(defineProps<{
    size?: number;
    label?: string;
}>(), { size: 16, label: '加载中' });
</script>
<template>
  <span class="u-spinner" role="status" :aria-label="label" :style="{ width: `${size}px`, height: `${size}px` }">
    <i v-for="n in 12" :key="n" :style="{ transform: `rotate(${n * 30}deg)`, animationDelay: `${(n - 12) / 12}s` }" />
  </span>
</template>
<style scoped>
.u-spinner {
  position: relative;
  display: inline-block;
  flex-shrink: 0;
  color: inherit;
}
i {
  position: absolute;
  left: 45%;
  top: 0;
  width: 10%;
  height: 28%;
  border-radius: var(--radius-pill);
  background: currentColor;
  transform-origin: 50% 179%;
  animation: spoke var(--duration-spin) linear infinite;
  opacity: .22;
}
@keyframes spoke {
  0% {
    opacity: 1;
  }
  100% {
    opacity: .22;
  }
}
</style>
