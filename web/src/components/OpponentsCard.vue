<script setup lang="ts">
import { computed, watch } from 'vue';
import { useAppStore } from '../stores/app';
import { useScouting } from '../composables/scouting';
import UButton from './ui/UButton.vue';
import { iconImages, requestIcons } from '../composables/icons';
import type { ScoutOpponent, TableColumn, TableRow } from '../types';
import UCard from './ui/UCard.vue';
import UTable from './ui/UTable.vue';
import UBadge from './ui/UBadge.vue';
const store = useAppStore();
const { progress, busy, unavailable, scout } = useScouting();
const rows = computed(() => (store.advice?.opponents ?? []).map(row => ({ ...row })));
const columns: TableColumn[] = [
  { key: 'name', title: '玩家', width: '13%' }, { key: 'hp', title: '血量', width: 48, align: 'right' },
  { key: 'level', title: '等级', width: 48, align: 'right' }, { key: 'comp_guess', title: '阵容猜测', width: '27%', wrap: true },
  { key: 'carries', title: '主C', width: '20%', wrap: true }, { key: 'traits', title: '主要羁绊', wrap: true },
  { key: 'round', title: '侦察回合', width: 78 }
];
function opponent(row: TableRow) { return row as unknown as ScoutOpponent; }
function topTraits(row: TableRow) { return [...opponent(row).traits].sort((a, b) => (b.count ?? 0) - (a.count ?? 0)).slice(0, 3); }
watch(rows, value => requestIcons(value.flatMap(row => [
  ...row.carries.map(u => `hero:${u.name}`), ...topTraits(row).map(t => `trait:${t.name}`)
])), { immediate: true });
</script>
<template>
  <UCard title="对手" class="opponents-card" flush>
    <template #title-leading><slot name="title-leading" /></template>
    <template #actions>
      <small v-if="busy">{{ progress?.notice ? `${progress.notice.seconds}秒后自动侦察` : progress?.scanning ? `侦察中 ${progress.captured}/${progress.total}` : progress?.processing ? `后台识别 ${progress.processed}/${progress.total}` : '准备侦察' }}</small>
      <UButton v-if="busy" size="sm" @click="scout(true)">{{ progress?.notice ? '取消本次' : '停止' }}</UButton>
      <UButton v-else size="sm" :disabled="!!unavailable" :title="unavailable" @click="scout()">立即侦察</UButton>
    </template>
    <p v-if="progress?.reason" class="scout-reason">{{ progress.reason }}</p>
    <UTable :columns="columns" :rows="rows" row-key="key" height="310px" label="对手侦察数据" empty-title="尚未侦察（辅助大脑页可开启自动侦察，或点「立即侦察」）">
      <template #cell-name="{row}">{{ row.name || '未知对手' }}</template>
      <template #cell-comp_guess="{row}">
        <span>{{ opponent(row).comp_guess?.name ?? '尚无法判断' }}</span>
        <small v-if="opponent(row).comp_guess">{{ opponent(row).comp_guess?.by === 'traits' ? '按羁绊' : '按棋子' }}</small>
        <UBadge v-if="row.contested" tone="orange">撞阵</UBadge>
      </template>
      <template #cell-carries="{row}">
        <span v-if="!opponent(row).carries.length">—</span>
        <span v-for="(unit,index) in opponent(row).carries" :key="index" class="scout-chip">
          <img v-if="iconImages[`hero:${unit.name}`]" :src="iconImages[`hero:${unit.name}`]!" alt="" />
          {{ unit.name }} {{ '★'.repeat(unit.star) }}
        </span>
      </template>
      <template #cell-traits="{row}">
        <span v-for="trait in topTraits(row)" :key="trait.name" class="scout-chip">
          <img v-if="iconImages[`trait:${trait.name}`]" :src="iconImages[`trait:${trait.name}`]!" class="trait-icon" alt="" />
          {{ trait.name }} {{ trait.count ?? '—' }}
        </span>
      </template>
      <template #cell-round="{row}">
        <span :class="{stale: Number(row.stale_rounds)>=2}">{{ row.round ?? '未知' }}<small v-if="Number(row.stale_rounds)>=2">已过 {{ row.stale_rounds }} 回合</small></span>
      </template>
    </UTable>
  </UCard>
</template>
<style scoped>
.opponents-card { grid-column: 1 / -1; min-width: 0; }
small { display: block; font-size: 10px; color: var(--secondary); }
.scout-chip { display: inline-flex; align-items: center; gap: 3px; margin-right: 6px; white-space: nowrap; }
.scout-chip img { width: 20px; height: 20px; object-fit: contain; }
:root[data-theme="light"] .scout-chip .trait-icon { background: rgba(0,0,0,.55); border: 1px solid rgba(0,0,0,.25); border-radius: 50%; padding: 2px; box-sizing: border-box; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme]) .scout-chip .trait-icon { background: rgba(0,0,0,.55); border: 1px solid rgba(0,0,0,.25); border-radius: 50%; padding: 2px; box-sizing: border-box; }
}
.stale { opacity: .45; }
</style>
