<script setup lang="ts">
withDefaults(defineProps<{
    title?: string;
    subtitle?: string;
    padded?: boolean;
}>(), { padded: true });
</script>
<template>
  <section class="u-card">
    <header v-if="title || $slots.actions">
      <div class="card-heading">
        <slot name="title-leading" />
        <div>
          <h2>{{ title }}</h2>
          <p v-if="subtitle">{{ subtitle }}</p>
        </div>
      </div>
      <div class="card-actions">
        <slot name="actions" />
      </div>
    </header>
    <div class="card-content" :class="{ 'card-body': padded }">
      <slot />
    </div>
  </section>
</template>
<style scoped>
.u-card {
  min-width: 0;
  background: var(--surface);
  border: 1px solid var(--separator);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-card);
  overflow: hidden;
}
header {
  padding: var(--space-4) var(--space-5);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  border-bottom: 1px solid var(--separator);
}
.card-heading { display: flex; align-items: center; gap: var(--space-2); }
h2 {
  font-size: var(--font-md);
  font-weight: 600;
}
p {
  color: var(--secondary);
  font-size: var(--font-xs);
  margin-top: var(--space-1);
}
.card-body {
  padding: var(--space-5);
}
.card-actions {
  display: flex;
  gap: var(--space-2);
  align-items: center;
}
</style>
