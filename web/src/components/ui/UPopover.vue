<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import { useOverlay } from '../../composables/overlays';
import { useFloating } from '../../composables/floating';
const props = withDefaults(defineProps<{
    modelValue: boolean;
    width?: number;
    matchWidth?: boolean;
    autoFocus?: boolean;
    label?: string;
}>(), { width: 260, matchWidth: false, autoFocus: true, label: '弹出面板' });
const emit = defineEmits<{
    'update:modelValue': [
        value: boolean
    ];
}>();
const anchor = ref<HTMLElement>();
const panel = ref<HTMLElement>();
const open = computed(() => props.modelValue);
function close() {
    emit('update:modelValue', false);
}
useOverlay(open, panel, close, false, props.autoFocus);
const position = useFloating(open, anchor, panel, props.matchWidth);
function outside(event: PointerEvent) {
    if (open.value && !anchor.value?.contains(event.target as Node) && !panel.value?.contains(event.target as Node))
        close();
}
document.addEventListener('pointerdown', outside, true);
onBeforeUnmount(() => document.removeEventListener('pointerdown', outside, true));
</script>
<template>
  <span ref="anchor" class="popover-anchor">
    <slot name="trigger" :open="modelValue" :toggle="() => emit('update:modelValue', !modelValue)" />
  </span>
  <Teleport to="body">
    <Transition name="popover">
      <div v-if="modelValue" ref="panel" class="u-popover" role="dialog" :aria-label="label" tabindex="-1" :style="{ width: `${width}px`, ...position }">
        <slot :close="close" />
      </div>
    </Transition>
  </Teleport>
</template>
<style scoped>
.popover-anchor {
  display: inline-flex;
  min-width: 0;
}
.u-popover {
  position: fixed;
  z-index: var(--z-popover);
  max-width: calc(100vw - 20px);
  max-height: calc(100vh - 20px);
  overflow: auto;
  background: var(--sidebar);
  backdrop-filter: var(--material-blur);
  border: 1px solid var(--separator);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-float);
  padding: var(--space-3);
  transform-origin: top center;
  outline: none;
}
.popover-enter-active, .popover-leave-active {
  transition: opacity var(--duration-fast) var(--ease), transform var(--duration-normal) var(--ease);
}
.popover-enter-from, .popover-leave-to {
  opacity: 0;
  transform: translateY(-4px) scale(.97);
}
</style>
