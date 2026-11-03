<script setup lang="ts">
import type { Component } from 'vue';
import USpinner from './USpinner.vue';
withDefaults(defineProps<{
    variant?: 'primary' | 'secondary' | 'plain' | 'destructive';
    size?: 'sm' | 'md';
    loading?: boolean;
    disabled?: boolean;
    icon?: Component;
}>(), { variant: 'secondary', size: 'md', loading: false, disabled: false });
defineEmits<{
    click: [
        event: MouseEvent
    ];
}>();
</script>
<template>
  <button type="button" class="u-button" :class="[variant, size]" :disabled="disabled || loading" :aria-busy="loading" @click="$emit('click', $event)">
    <USpinner v-if="loading" :size="13" />
    <component :is="icon" v-else-if="icon" :size="15" />
    <slot />
  </button>
</template>
<style scoped>
.u-button {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-1);
  height: var(--control-height);
  padding: 0 var(--space-3);
  border-radius: var(--radius-sm);
  white-space: nowrap;
  background: var(--surface-raised);
  box-shadow: var(--shadow-control);
  font-weight: 500;
  font-size: var(--font-sm);
  transition: transform var(--duration-fast) var(--ease), opacity var(--duration-fast) var(--ease);
}
.u-button::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: var(--pressed);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease);
  pointer-events: none;
}
.u-button:hover::before {
  opacity: .6;
}
.u-button:active:not(:disabled) {
  transform: scale(.97);
}
.u-button:disabled {
  opacity: .4;
}
.u-button:disabled::before {
  opacity: 0;
}
.primary {
  background: var(--accent);
  color: var(--white);
  box-shadow: none;
}
.destructive {
  background: var(--red);
  color: var(--white);
  box-shadow: none;
}
.plain {
  background: var(--transparent);
  box-shadow: none;
}
.sm {
  height: var(--control-sm);
  padding: 0 var(--space-2);
  font-size: var(--font-xs);
}
</style>
