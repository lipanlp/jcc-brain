<script setup lang="ts">
import { ChevronUp, ChevronDown } from 'lucide-vue-next';
import { ref, watch } from 'vue';
const props = withDefaults(defineProps<{
    modelValue: number;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
    label?: string;
}>(), { step: 1 });
const emit = defineEmits<{
    'update:modelValue': [
        value: number
    ];
}>();
const draft = ref(String(props.modelValue));
watch(() => props.modelValue, value => {
    draft.value = String(value);
});
function normalize(value: number) {
    return Math.min(props.max ?? Infinity, Math.max(props.min ?? -Infinity, Number(value.toFixed(10))));
}
function commit() {
    const value = Number(draft.value);
    if (!draft.value.trim() || !Number.isFinite(value) || value < (props.min ?? -Infinity) || value > (props.max ?? Infinity)) {
        draft.value = String(props.modelValue);
        return;
    }
    ;
    emit('update:modelValue', value);
    draft.value = String(value);
}
function change(direction: number) {
    const value = normalize(props.modelValue + direction * props.step);
    draft.value = String(value);
    emit('update:modelValue', value);
}
</script>
<template>
  <div class="u-number control">
    <input v-model="draft" inputmode="decimal" role="spinbutton" :aria-label="label" :aria-valuenow="modelValue" :aria-valuemin="min" :aria-valuemax="max" :disabled="disabled" @blur="commit" @keydown.enter="commit" @keydown.up.prevent="change(1)" @keydown.down.prevent="change(-1)" />
    <div class="steppers">
      <button type="button" aria-label="增加" :disabled="disabled || modelValue >= (max ?? Infinity)" @click="change(1)">
        <ChevronUp :size="11" />
      </button>
      <button type="button" aria-label="减少" :disabled="disabled || modelValue <= (min ?? -Infinity)" @click="change(-1)">
        <ChevronDown :size="11" />
      </button>
    </div>
  </div>
</template>
<style scoped>
.u-number {
  display: inline-flex;
  width: 110px;
  overflow: hidden;
}
.u-number:focus-within {
  box-shadow: var(--shadow-focus);
}
input {
  width: 100%;
  min-width: 0;
  border: 0;
  outline: 0;
  background: var(--transparent);
  padding: 0 var(--space-2);
  font-variant-numeric: tabular-nums;
}
input:focus-visible {
  box-shadow: none;
}
.steppers {
  border-left: 1px solid var(--separator);
  display: flex;
  flex-direction: column;
  width: 23px;
  flex-shrink: 0;
}
button {
  height: 50%;
  display: grid;
  place-items: center;
  padding: 0;
  background: var(--hover);
}
button + button {
  border-top: 1px solid var(--separator);
}
button:disabled {
  opacity: .3;
}
</style>
