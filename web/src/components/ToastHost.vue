<script setup lang="ts">
import { toastItems, dismissToast } from '../composables/toast';
import UToast from './ui/UToast.vue';
</script>
<template>
  <Teleport to="body">
    <TransitionGroup name="toast" tag="div" class="toast-host" aria-label="通知">
      <UToast v-for="item in toastItems" :key="item.id" :item="item" @close="dismissToast(item.id)" />
    </TransitionGroup>
  </Teleport>
</template>
<style scoped>
.toast-host {
  position: fixed;
  top: calc(var(--toolbar-height) + var(--space-3));
  right: var(--space-4);
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  pointer-events: none;
}
.toast-enter-active, .toast-leave-active, .toast-move {
  transition: transform var(--duration-normal) var(--ease), opacity var(--duration-normal) var(--ease);
}
.toast-enter-from, .toast-leave-to {
  transform: translateX(24px);
  opacity: 0;
}
.toast-leave-active {
  position: absolute;
}
</style>
