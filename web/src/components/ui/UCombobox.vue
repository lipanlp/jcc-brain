<script setup lang="ts">
import { computed, nextTick, ref, useId, watch } from 'vue';
import { ChevronsUpDown, Check } from 'lucide-vue-next';
import UPopover from './UPopover.vue';
import type { SelectOption } from '../../types';
const props = withDefaults(defineProps<{
    modelValue: string;
    options: SelectOption[];
    placeholder?: string;
    label?: string;
    disabled?: boolean;
}>(), { placeholder: '搜索并选择…' });
const emit = defineEmits<{
    'update:modelValue': [
        value: string
    ];
}>();
const open = ref(false);
const query = ref('');
const active = ref(0);
const input = ref<HTMLInputElement>();
const listId = useId();
const selectedLabel = computed(() => props.options.find(item => item.value === props.modelValue)?.label ?? props.modelValue);
const filtered = computed(() => props.options.filter(item => `${item.label} ${item.hint ?? ''}`.toLocaleLowerCase().includes(query.value.toLocaleLowerCase())));
const display = ref(selectedLabel.value);
watch(selectedLabel, value => {
    if (!open.value)
        display.value = value;
});
// Reset before processing the first typed character, including keyboard focus.
watch(open, value => {
    if (value) {
        query.value = '';
        active.value = Math.max(0, filtered.value.findIndex(option => option.value === props.modelValue));
    }
    else
        display.value = selectedLabel.value;
}, { flush: 'sync' });
watch(query, () => {
    active.value = 0;
}, { flush: 'sync' });
function select(option: SelectOption) {
    emit('update:modelValue', option.value);
    display.value = option.label;
    open.value = false;
    input.value?.focus();
}
async function keydown(event: KeyboardEvent) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        if (!open.value)
            open.value = true;
        else
            active.value = (active.value + (event.key === 'ArrowDown' ? 1 : -1) + filtered.value.length) % Math.max(1, filtered.value.length);
        await nextTick();
        document.getElementById(`${listId}-${active.value}`)?.scrollIntoView({ block: 'nearest' });
    }
    else if (event.key === 'Enter') {
        event.preventDefault();
        if (!open.value)
            open.value = true;
        else if (filtered.value[active.value])
            select(filtered.value[active.value]!);
    }
    else if (event.key === 'Tab')
        open.value = false;
}
function search(value: string) {
    if (!open.value)
        open.value = true;
    display.value = value;
    query.value = value;
}
</script>
<template>
  <UPopover v-model="open" match-width :auto-focus="false" :label="label || '选择选项'">
    <template #trigger="{ toggle }">
      <div class="u-combobox control">
        <input ref="input" :value="display" role="combobox" aria-autocomplete="list" :aria-expanded="open" :aria-controls="listId" :aria-activedescendant="open && filtered.length ? `${listId}-${active}` : undefined" :aria-label="label || placeholder" :placeholder="placeholder" :disabled="disabled" @input="search(($event.target as HTMLInputElement).value)" @keydown="keydown" @click="open = true" />
        <button type="button" tabindex="-1" aria-label="展开选项" :disabled="disabled" @click="toggle(); input?.focus()">
          <ChevronsUpDown :size="12" />
        </button>
      </div>
    </template>
    <div :id="listId" role="listbox" class="combo-list">
      <div v-for="(option, index) in filtered" :id="`${listId}-${index}`" :key="option.value" role="option" :aria-selected="option.value === modelValue" class="combo-option" :class="{ active: active === index }" @pointermove="active = index" @mousedown.prevent @click="select(option)">
        <span :style="{ color: option.color }">{{ option.label }}</span>
        <small>{{ option.hint }}</small>
        <Check v-if="option.value === modelValue" :size="13" />
      </div>
      <p v-if="!filtered.length" class="combo-empty">没有匹配的选项</p>
    </div>
  </UPopover>
</template>
<style scoped>
.u-combobox {
  display: flex;
  width: 100%;
  min-width: 180px;
}
.u-combobox:focus-within {
  box-shadow: var(--shadow-focus);
}
input {
  width: 100%;
  min-width: 0;
  padding: 0 var(--space-2);
  background: var(--transparent);
  border: 0;
  outline: none;
  font-size: var(--font-sm);
}
input:focus-visible {
  box-shadow: none;
}
button {
  display: grid;
  place-items: center;
  background: var(--transparent);
  color: var(--secondary);
  padding: 0 var(--space-2);
}
.combo-list {
  max-height: 256px;
  overflow: auto;
  margin: calc(var(--space-2) * -1);
}
.combo-option {
  min-height: 32px;
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
  font-size: var(--font-sm);
}
.combo-option.active {
  background: var(--accent-soft);
}
small {
  color: var(--secondary);
  margin-left: auto;
  font-size: var(--font-xs);
}
.combo-empty {
  color: var(--secondary);
  padding: var(--space-3);
  font-size: var(--font-sm);
}
</style>
