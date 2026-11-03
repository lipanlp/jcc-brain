<script setup lang="ts">
import type { Advice } from '../types';
import UBadge from './ui/UBadge.vue';
defineProps<{ plans: Advice['item_plan'] }>();
</script>
<template>
  <div class="item-assignments">
    <p class="assignment-note">按当前上场棋子和装备栏分配；人口预览不改变本安排。只提供建议，不自动操作。</p>
    <p v-if="!plans.length" class="muted">暂无当前装备安排</p>
    <article v-for="plan in plans" :key="plan.key" :data-plan-key="plan.key" class="assignment">
      <div class="assignment-heading">
        <UBadge :tone="plan.ready ? 'green' : 'gray'">{{ plan.action }}</UBadge>
        <strong>{{ plan.item || '等待装备信息' }}</strong>
      </div>
      <div v-if="plan.item" class="assignment-destinations">
        <span>当前：<b>{{ plan.current_unit?.label || plan.existing_item?.holder || (plan.action==='备料' ? '装备栏备料' : '暂留装备栏') }}</b></span>
        <span>最终：<b>{{ plan.final_to || '待确认' }}</b></span>
      </div>
      <p>{{ plan.reason }}</p>
    </article>
  </div>
</template>
<style scoped>
.item-assignments { padding: var(--space-3) var(--space-4); max-height: 380px; overflow: auto; }
.assignment-note { color: var(--secondary); font-size: var(--font-xs); margin-bottom: var(--space-3); }
.assignment { padding: var(--space-3) 0; border-top: 1px solid var(--separator); }
.assignment-heading,.assignment-destinations { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-4); }
.assignment-heading { font-size: var(--font-sm); }
.assignment-destinations { margin-top: var(--space-2); color: var(--secondary); font-size: var(--font-xs); }
.assignment-destinations b { color: var(--text); font-weight: 500; }
.assignment p { margin-top: var(--space-2); font-size: var(--font-xs); color: var(--secondary); line-height: 1.6; overflow-wrap: anywhere; }
</style>
