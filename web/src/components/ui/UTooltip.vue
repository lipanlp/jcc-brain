<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue';
import { useFloating } from '../../composables/floating';
defineProps<{
    text: string;
}>();
const open = ref(false);
const anchor = ref<HTMLElement>();
const panel = ref<HTMLElement>();
const position = useFloating(open, anchor, panel);
let timer: ReturnType<typeof setTimeout> | undefined;
function show() {
    clearTimeout(timer);
    timer = setTimeout(() => {
        open.value = true;
    }, 500);
}
function hide() {
    clearTimeout(timer);
    open.value = false;
}
onBeforeUnmount(() => clearTimeout(timer));
</script>
<template>
  <span ref="anchor" class="tooltip-anchor" @mouseenter="show" @mouseleave="hide" @focusin="show" @focusout="hide" @keydown.esc="hide">
    <slot />
  </span>
  <Teleport to="body">
    <Transition name="tooltip">
      <span v-if="open" ref="panel" class="u-tooltip" role="tooltip" :style="position">{{ text }}</span>
    </Transition>
  </Teleport>
</template>
<style scoped>
.tooltip-anchor {
  display: inline-flex;
  min-width: 0;
}
.u-tooltip {
  position: fixed;
  z-index: var(--z-tooltip);
  background: var(--tooltip);
  color: var(--tooltip-text);
  border-radius: var(--radius-sm);
  padding: var(--space-1) var(--space-2);
  font-size: var(--font-xs);
  box-shadow: var(--shadow-control);
  pointer-events: none;
  max-width: 240px;
  white-space: pre-line;
}
.tooltip-enter-active, .tooltip-leave-active {
  transition: opacity var(--duration-fast) var(--ease);
}
.tooltip-enter-from, .tooltip-leave-to {
  opacity: 0;
}
</style>
