<script setup lang="ts">
import { ChevronsUpDown } from 'lucide-vue-next';
import type { SelectOption } from '../../types';
defineProps<{
    modelValue: string;
    options: SelectOption[];
    label?: string;
    disabled?: boolean;
}>();
defineEmits<{
    'update:modelValue': [
        value: string
    ];
}>();
</script>
<template>
  <div class="u-select">
    <select class="control" :value="modelValue" :aria-label="label" :disabled="disabled" @change="$emit('update:modelValue', ($event.target as HTMLSelectElement).value)">
      <option v-for="option in options" :key="option.value" :value="option.value">{{ option.label }}</option>
    </select>
    <ChevronsUpDown :size="12" />
  </div>
</template>
<style scoped>
.u-select {
  position: relative;
  display: inline-block;
  min-width: 100px;
}
select {
  appearance: none;
  width: 100%;
  padding: 0 28px 0 var(--space-2);
  font-size: var(--font-sm);
}
svg {
  position: absolute;
  right: var(--space-2);
  top: 9px;
  pointer-events: none;
  color: var(--secondary);
}
select:disabled {
  opacity: .4;
}
</style>
