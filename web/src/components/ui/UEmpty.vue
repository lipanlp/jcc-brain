<script setup lang="ts">
import { Inbox } from 'lucide-vue-next';
import type { Component } from 'vue';
defineProps<{
    title: string;
    description?: string;
    icon?: Component;
}>();
</script>
<template>
  <div class="u-empty">
    <div class="empty-icon">
      <component :is="icon ?? Inbox" :size="30" />
    </div>
    <h2>{{ title }}</h2>
    <p v-if="description">{{ description }}</p>
    <div v-if="$slots.action" class="empty-action">
      <slot name="action" />
    </div>
  </div>
</template>
<style scoped>
.u-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: var(--space-10) var(--space-6);
}
.empty-icon {
  display: grid;
  place-items: center;
  width: 66px;
  height: 66px;
  border: 1px solid var(--separator);
  border-radius: var(--radius-lg);
  background: var(--surface);
  color: var(--secondary);
  box-shadow: var(--shadow-card);
  margin-bottom: var(--space-5);
}
h2 {
  font-size: var(--font-xl);
  font-weight: 600;
  letter-spacing: -.4px;
}
p {
  color: var(--secondary);
  font-size: var(--font-sm);
  margin-top: var(--space-2);
  max-width: 340px;
  line-height: 1.8;
}
.empty-action {
  margin-top: var(--space-5);
}
</style>
