<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { X } from 'lucide-vue-next';
import UButton from './ui/UButton.vue';
import { useOverlay } from '../composables/overlays';
const props = defineProps<{
    name: string;
    open: boolean;
    label?: string;
    defaultWidth?: number;
}>();
const emit = defineEmits<{
    'update:open': [
        value: boolean
    ];
}>();
const root = ref<HTMLElement>();
const panel = ref<HTMLElement>();
const width = ref(props.defaultWidth ?? 380);
const available = ref(1060);
const media = window.matchMedia('(max-width: 1099px)');
const narrow = ref(media.matches);
const dragging = ref(false);
const limit = computed(() => Math.max(320, Math.min(560, available.value - 280)));
const actualWidth = computed(() => Math.max(320, Math.min(limit.value, width.value)));
const overlayOpen = computed(() => narrow.value && props.open);
useOverlay(overlayOpen, panel, () => emit('update:open', false), true);
const observer = new ResizeObserver(entries => {
    available.value = entries[0]?.contentRect.width ?? 1060;
});
function changeMedia() {
    narrow.value = media.matches;
}
function persist() {
    try {
        localStorage.setItem(`jcc-split-${props.name}`, String(width.value));
    }
    catch { /* memory-only preference */
    }
}
let startX = 0;
let startWidth = width.value;
function begin(event: PointerEvent) {
    if (event.button !== 0)
        return;
    event.preventDefault();
    startX = event.clientX;
    startWidth = actualWidth.value;
    dragging.value = true;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}
function move(event: PointerEvent) {
    if (dragging.value)
        width.value = Math.max(320, Math.min(limit.value, startWidth + startX - event.clientX));
}
function end() {
    if (dragging.value) {
        dragging.value = false;
        persist();
    }
}
function keyboard(event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
        return;
    event.preventDefault();
    width.value = event.key === 'Home' ? 320 : event.key === 'End' ? limit.value :
        Math.max(320, Math.min(limit.value, actualWidth.value + (event.key === 'ArrowLeft' ? 16 : -16)));
    persist();
}
onMounted(() => {
    try {
        const saved = Number(localStorage.getItem(`jcc-split-${props.name}`));
        if (Number.isFinite(saved) && saved >= 320 && saved <= 560)
            width.value = saved;
    }
    catch { /* default width */
    }
    if (root.value)
        observer.observe(root.value);
    media.addEventListener('change', changeMedia);
});
onBeforeUnmount(() => {
    observer.disconnect();
    media.removeEventListener('change', changeMedia);
});
</script>
<template>
  <div ref="root" class="split-view" :class="{ narrow, dragging }" :style="{'--inspector-width': `${actualWidth}px`}">
    <div class="split-main">
      <slot />
    </div>
    <div v-if="!narrow" class="split-divider" role="separator" tabindex="0" aria-label="调整检查器宽度" aria-orientation="vertical" :aria-valuenow="actualWidth" :aria-valuemin="320" :aria-valuemax="limit" @pointerdown="begin" @pointermove="move" @pointerup="end" @pointercancel="end" @lostpointercapture="end" @keydown="keyboard" />
    <Transition name="inspector-mask">
      <div v-if="overlayOpen" class="split-mask" @click="emit('update:open',false)" />
    </Transition>
    <Transition name="inspector">
      <aside v-if="!narrow || open" ref="panel" class="split-inspector" :role="narrow ? 'dialog' : 'complementary'" :aria-modal="narrow ? true : undefined" :aria-label="label ?? '检查器'" tabindex="-1">
        <div v-if="narrow" class="inspector-toolbar">
          <span>{{ label ?? '检查器' }}</span>
          <UButton variant="plain" :icon="X" aria-label="关闭检查器" @click="emit('update:open',false)" />
        </div>
        <slot name="inspector" />
      </aside>
    </Transition>
  </div>
</template>
<style scoped>
.split-view {
  position: relative;
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.split-main {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: auto;
}
.split-inspector {
  flex: 0 0 var(--inspector-width);
  width: var(--inspector-width);
  min-height: 0;
  overflow: auto;
  background: var(--surface);
  border-left: 1px solid var(--separator);
}
.split-divider {
  width: 5px;
  margin-left: -2px;
  margin-right: -3px;
  z-index: 2;
  flex-shrink: 0;
  cursor: col-resize;
  touch-action: none;
}
.split-divider:hover,.split-divider:focus-visible,.dragging .split-divider {
  background: var(--focus);
}
.dragging {
  cursor: col-resize;
}
.split-mask {
  position: absolute;
  inset: 0;
  background: var(--mask);
  z-index: 3;
}
.narrow .split-inspector {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  z-index: 4;
  width: min(var(--inspector-width), calc(100% - 32px));
  box-shadow: var(--shadow-float);
}
.inspector-toolbar {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-4);
  background: var(--surface);
  border-bottom: 1px solid var(--separator);
  color: var(--secondary);
  font-size: var(--font-xs);
}
.inspector-enter-active,.inspector-leave-active {
  transition: transform var(--duration-normal) var(--ease),opacity var(--duration-normal) var(--ease);
}
.inspector-enter-from,.inspector-leave-to {
  transform: translateX(100%);
  opacity: 0;
}
.inspector-mask-enter-active,.inspector-mask-leave-active {
  transition: opacity var(--duration-normal) var(--ease);
}
.inspector-mask-enter-from,.inspector-mask-leave-to {
  opacity: 0;
}
</style>
