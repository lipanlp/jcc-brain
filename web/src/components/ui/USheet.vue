<script setup lang="ts">
import { computed, ref, useId } from 'vue';
import { X } from 'lucide-vue-next';
import UButton from './UButton.vue';
import { useOverlay } from '../../composables/overlays';
const props = withDefaults(defineProps<{
    modelValue: boolean;
    title: string;
    width?: number | string;
    height?: number;
    flush?: boolean;
}>(), { width: 560 });
const emit = defineEmits<{
    'update:modelValue': [
        value: boolean
    ];
}>();
const open = computed(() => props.modelValue);
const panel = ref<HTMLElement>();
const titleId = useId();
function close() {
    emit('update:modelValue', false);
}
useOverlay(open, panel, close, true);
</script>
<template>
  <Teleport to="body">
    <Transition name="mask">
      <div v-if="modelValue" class="overlay-mask sheet-mask" @click.self="close">
        <Transition name="sheet" appear>
          <section v-if="modelValue" ref="panel" class="u-sheet" role="dialog" aria-modal="true" :aria-labelledby="titleId" tabindex="-1" :style="{ width: typeof width === 'number' ? `${width}px` : width, height: height ? `${height}px` : undefined }">
            <header>
              <h2 :id="titleId">{{ title }}</h2>
              <UButton variant="plain" :icon="X" aria-label="关闭面板" @click="close" />
            </header>
            <div class="sheet-body" :class="{flush}">
              <slot />
            </div>
            <footer v-if="$slots.footer">
              <slot name="footer" :close="close" />
            </footer>
          </section>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>
<style scoped>
.sheet-mask {
  z-index: var(--z-sheet);
  align-items: flex-start;
}
.u-sheet {
  max-width: calc(100vw - 48px);
  max-height: calc(100vh - 52px);
  margin-top: var(--toolbar-height);
  display: flex;
  flex-direction: column;
  border: 1px solid var(--separator);
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow-modal);
  outline: none;
}
header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--separator);
}
h2 {
  font-size: var(--font-lg);
  font-weight: 600;
}
.sheet-body.flush {
  padding: 0;
  flex: 1;
  min-height: 0;
}
.sheet-body {
  overflow: auto;
  padding: var(--space-6);
}
footer {
  flex-shrink: 0;
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
  padding: var(--space-4) var(--space-6);
  border-top: 1px solid var(--separator);
}
.sheet-enter-active, .sheet-leave-active {
  transition: transform var(--duration-spring) var(--ease), opacity var(--duration-normal) var(--ease);
}
.sheet-enter-from, .sheet-leave-to {
  transform: translateY(-32px);
  opacity: 0;
}
.mask-leave-active .u-sheet {
  transition: transform var(--duration-normal) var(--ease);
}
.mask-leave-to .u-sheet {
  transform: translateY(-32px);
}
</style>
