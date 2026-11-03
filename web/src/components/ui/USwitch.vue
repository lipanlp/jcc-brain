<script setup lang="ts">
withDefaults(defineProps<{
    modelValue: boolean;
    disabled?: boolean;
    label?: string;
}>(), { disabled: false });
defineEmits<{
    'update:modelValue': [
        value: boolean
    ];
}>();
</script>
<template>
  <button type="button" class="u-switch" :class="{ on: modelValue }" role="switch" :aria-checked="modelValue" :aria-label="label" :disabled="disabled" @click="$emit('update:modelValue', !modelValue)">
    <span class="on-track" />
    <span class="thumb" />
  </button>
</template>
<style scoped>
.u-switch {
  position: relative;
  display: inline-block;
  flex-shrink: 0;
  width: 36px;
  height: 22px;
  padding: 0;
  border-radius: var(--radius-pill);
  background: var(--tertiary);
}
.on-track {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: var(--green);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease);
}
.thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 18px;
  height: 18px;
  border-radius: var(--radius-pill);
  background: var(--white);
  box-shadow: var(--shadow-control);
  transition: transform var(--duration-spring) var(--spring);
}
.on .on-track {
  opacity: 1;
}
.on .thumb {
  transform: translateX(14px);
}
.u-switch:disabled {
  opacity: .4;
}
</style>
