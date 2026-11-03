<script setup lang="ts">
import { Search, XCircle } from 'lucide-vue-next';
import { onBeforeUnmount, ref, watch } from 'vue';
const props = withDefaults(defineProps<{
    modelValue: string;
    placeholder?: string;
    label?: string;
    disabled?: boolean;
}>(), { placeholder: '搜索' });
const emit = defineEmits<{
    'update:modelValue': [
        value: string
    ];
}>();
const draft = ref(props.modelValue);
let timer: ReturnType<typeof setTimeout> | undefined;
watch(() => props.modelValue, value => {
    clearTimeout(timer);
    draft.value = value;
});
function input(value: string) {
    draft.value = value;
    clearTimeout(timer);
    timer = setTimeout(() => emit('update:modelValue', value), 200);
}
function clear() {
    clearTimeout(timer);
    draft.value = '';
    emit('update:modelValue', '');
}
onBeforeUnmount(() => clearTimeout(timer));
</script>
<template>
  <div class="u-search control">
    <Search :size="14" />
    <input type="search" :value="draft" :placeholder="placeholder" :aria-label="label || placeholder" :disabled="disabled" @input="input(($event.target as HTMLInputElement).value)" />
    <button v-if="draft" type="button" aria-label="清除搜索" :disabled="disabled" @click="clear">
      <XCircle :size="13" />
    </button>
  </div>
</template>
<style scoped>
.u-search {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-2);
  color: var(--secondary);
}
.u-search:focus-within {
  box-shadow: var(--shadow-focus);
}
input {
  width: 100%;
  min-width: 0;
  background: var(--transparent);
  border: none;
  outline: none;
  height: 100%;
  font-size: var(--font-sm);
  color: var(--text);
}
input:focus-visible {
  box-shadow: none;
}
input::-webkit-search-cancel-button {
  display: none;
}
button {
  padding: 0;
  display: flex;
  background: var(--transparent);
  color: var(--secondary);
}
input::placeholder {
  color: var(--secondary);
}
</style>
