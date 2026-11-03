<script setup lang="ts">
import type { Unit } from '../types';
import UTooltip from './ui/UTooltip.vue';
defineProps<{
    units: Unit[];
    bench?: boolean;
}>();
// Recognition names and coordinate jitter must not replace existing chip nodes.
const key = (unit: Unit, index: number) => `${unit.zone}-${unit.slot ?? index}`;
function tooltip(unit: Unit) {
    const items = (unit.items ?? []).map(item => item.name).filter(Boolean);
    return (unit.traits.join('、') || '暂无羁绊信息') + (items.length ? `\n装备：${items.join('、')}` : '');
}
</script>
<template>
  <div class="unit-chips">
    <UTooltip v-for="(unit,index) in units" :key="key(unit,index)" :text="tooltip(unit)">
      <span class="unit-chip" :class="{unknown:!unit.known}" :style="{color:unit.known && unit.cost ? `var(--cost-${unit.cost})` : undefined}">
        <small v-if="bench">{{ (unit.slot ?? index)+1 }}.</small>
        {{ '★'.repeat(unit.star) }}
        {{ unit.name }}
      </span>
    </UTooltip>
    <span v-if="!units.length" class="units-empty">暂无棋子</span>
  </div>
</template>
<style scoped>
.unit-chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}
.unit-chip {
  display: inline-flex;
  gap: var(--space-1);
  align-items: center;
  padding: var(--space-1) var(--space-2);
  font-size: var(--font-xs);
  background: var(--hover);
  border: 1px solid var(--separator);
  border-radius: var(--radius-pill);
  white-space: nowrap;
}
.unit-chip.unknown {
  color: var(--secondary);
  border-style: dashed;
}
.unit-chip small {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.units-empty {
  color: var(--secondary);
  font-size: var(--font-xs);
}
</style>
