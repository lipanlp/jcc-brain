<script setup lang="ts">
import { CheckCircle2, CircleAlert, Info, TriangleAlert, X } from 'lucide-vue-next';
import type { ToastItem } from '../../composables/toast';
defineProps<{
    item: ToastItem;
}>();
defineEmits<{
    close: [
    ];
}>();
const icons = { success: CheckCircle2, error: CircleAlert, info: Info, warning: TriangleAlert };
</script>
<template>
  <div class="u-toast" :class="item.tone" :role="item.tone === 'error' ? 'alert' : 'status'">
    <component :is="icons[item.tone]" class="toast-icon" :size="18" />
    <p>{{ item.text }}</p>
    <button type="button" aria-label="关闭通知" @click="$emit('close')">
      <X :size="13" />
    </button>
  </div>
</template>
<style scoped>
.u-toast {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
  width: 330px;
  max-width: calc(100vw - 32px);
  padding: var(--space-4);
  border: 1px solid var(--separator);
  border-radius: var(--radius-md);
  background: var(--sidebar);
  backdrop-filter: var(--material-blur);
  box-shadow: var(--shadow-float);
  font-size: var(--font-sm);
  pointer-events: auto;
}
.toast-icon {
  margin-top: 1px;
  color: var(--blue-text);
}
.success .toast-icon {
  color: var(--green-text);
}
.warning .toast-icon {
  color: var(--orange-text);
}
.error .toast-icon {
  color: var(--red-text);
}
p {
  flex: 1;
  overflow-wrap: anywhere;
}
button {
  display: grid;
  place-items: center;
  padding: var(--space-half);
  color: var(--secondary);
  background: var(--transparent);
  border-radius: var(--radius-sm);
}
</style>
