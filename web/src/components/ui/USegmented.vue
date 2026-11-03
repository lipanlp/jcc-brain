<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, nextTick, type Component } from 'vue';
interface Option {
    value: string;
    label: string;
    icon?: Component;
}
const props = defineProps<{
    modelValue: string | null;
    options: Option[];
    label?: string;
    disabled?: boolean;
}>();
const emit = defineEmits<{
    'update:modelValue': [
        value: string
    ];
}>();
const root = ref<HTMLElement>();
const block = ref({ width: '0px', transform: 'translateX(0px)', opacity: '0' });
const index = computed(() => props.options.findIndex(item => item.value === props.modelValue));
function measure() {
    const button = root.value?.querySelectorAll('button')[index.value];
    block.value = button ? { width: `${button.offsetWidth}px`, transform: `translateX(${button.offsetLeft - 3}px)`, opacity: '1' } : { ...block.value, opacity: '0' };
}
watch([index, () => props.options], async () => {
    await nextTick();
    measure();
}, { deep: true });
const observer = new ResizeObserver(measure);
onMounted(() => {
    if (root.value)
        observer.observe(root.value);
    measure();
});
onBeforeUnmount(() => observer.disconnect());
function move(event: KeyboardEvent, from: number) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
        return;
    event.preventDefault();
    const n = event.key === 'Home' ? 0 : event.key === 'End' ? props.options.length - 1 : (from + (event.key === 'ArrowRight' ? 1 : -1) + props.options.length) % props.options.length;
    emit('update:modelValue', props.options[n]!.value);
    root.value?.querySelectorAll('button')[n]?.focus();
}
</script>
<template>
  <div ref="root" class="u-segmented" role="group" :aria-label="label">
    <span class="selection" :style="block" />
    <button v-for="(option, i) in options" :key="option.value" type="button" :disabled="disabled" :aria-pressed="modelValue === option.value" @click="emit('update:modelValue', option.value)" @keydown="move($event, i)">
      <component :is="option.icon" v-if="option.icon" :size="14" />
      {{ option.label }}
    </button>
  </div>
</template>
<style scoped>
.u-segmented {
  position: relative;
  display: inline-flex;
  padding: var(--space-inset);
  border-radius: var(--radius-sm);
  background: var(--hover);
  isolation: isolate;
  max-width: 100%;
}
.selection {
  position: absolute;
  top: var(--space-inset);
  bottom: var(--space-inset);
  left: var(--space-inset);
  border-radius: var(--radius-sm);
  background: var(--surface-raised);
  box-shadow: var(--shadow-control);
  transition: transform var(--duration-normal) var(--ease), opacity var(--duration-fast) var(--ease);
  z-index: -1;
}
button {
  height: 25px;
  padding: 0 var(--space-3);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-1);
  background: var(--transparent);
  border-radius: var(--radius-sm);
  font-size: var(--font-xs);
  white-space: nowrap;
  color: var(--secondary);
  flex: 1;
}
button[aria-pressed="true"] {
  color: var(--text);
}
button:disabled {
  opacity: .4;
}
</style>
