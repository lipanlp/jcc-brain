<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { costColor } from '../composables/presentation';
import type { DatabaseKind, DatabaseEntry, TableColumn } from '../types';
import SplitView from '../components/SplitView.vue';
import DatabaseInspector from '../components/DatabaseInspector.vue';
import UTable from '../components/ui/UTable.vue';
import USegmented from '../components/ui/USegmented.vue';
import USearchField from '../components/ui/USearchField.vue';
import UBadge from '../components/ui/UBadge.vue';
import UEmpty from '../components/ui/UEmpty.vue';
import USpinner from '../components/ui/USpinner.vue';
const kind = ref<DatabaseKind>('heroes');
const query = ref('');
const rows = shallowRef<(DatabaseEntry & {
    key: string;
})[]>([]);
const selectedKey = ref<string | null>(null);
const loading = ref(true);
const inspectorOpen = ref(false);
let ticket = 0;
const kinds = [
    { value: 'heroes', label: '英雄' }, { value: 'traits', label: '羁绊' }, { value: 'items', label: '装备' },
    { value: 'augments', label: '海克斯' }, { value: 'mechanics', label: '赛季机制' },
    { value: 'encounters', label: '开局奇遇' }, { value: 'guides', label: '官方攻略' }
];
const columnsByKind: Record<DatabaseKind, TableColumn[]> = {
    heroes: [{ key: 'name', title: '英雄', width: '22%' }, { key: 'cost', title: '费', width: '9%', align: 'right' }, { key: 'traits', title: '羁绊', width: '32%' }, { key: 'attack_range', title: '射程', width: '10%', align: 'right' }, { key: 'skill_name', title: '技能名', width: '27%' }],
    traits: [{ key: 'name', title: '羁绊', width: '35%' }, { key: 'kind', title: '类型', width: '25%' }, { key: 'breakpoints', title: '档位', width: '40%', format: value => (value as number[]).join('/') || '—' }],
    items: [{ key: 'name', title: '装备', width: '30%' }, { key: 'kind', title: '类型', width: '18%' }, { key: 'recipe', title: '合成', width: '35%', format: value => (value as string[]).join(' + ') || '—' }, { key: 'trait_name', title: '关联羁绊', width: '17%' }],
    augments: [{ key: 'name', title: '海克斯', width: '75%' }, { key: 'tier_label', title: '等级', width: '25%' }],
    mechanics: [{ key: 'name', title: '名称', width: '30%' }, { key: 'category', title: '分类', width: '22%' }, { key: 'price', title: '价格', width: '13%', align: 'right' }, { key: 'rounds', title: '出现回合', width: '35%', format: value => roundsText(value as DatabaseEntry['rounds']) }],
    encounters: [{ key: 'name', title: '名称' }],
    guides: [{ key: 'name', title: '攻略名', width: '33%' }, { key: 'quality', title: '评级', width: '12%' }, { key: 'heroes', title: '成型阵容', width: '55%', format: value => {
                const names = value as string[];
                return names.slice(0, 5).join('、') + (names.length > 5 ? ` 等 ${names.length} 个` : '');
            } }]
};
const columns = computed(() => columnsByKind[kind.value]);
const tableRows = computed(() => rows.value.map(row => ({ ...row })));
const selected = computed(() => rows.value.find(row => row.key === selectedKey.value));
function select(id: string | number) {
    selectedKey.value = String(id);
    inspectorOpen.value = true;
}
function changeKind(value: string) {
    kind.value = value as DatabaseKind;
}
function roundsText(rounds: DatabaseEntry['rounds']) {
    return rounds?.map(([a, b]) => `${a}~${b ?? '结束'}`).join('、') || '—';
}
async function load() {
    const current = ++ticket;
    loading.value = true;
    try {
        const value = await call<DatabaseEntry[]>('db_list', kind.value, query.value);
        if (current !== ticket)
            return;
        const counts = new Map<string, number>();
        const totals = new Map<string, number>();
        value.forEach(row => totals.set(row.name, (totals.get(row.name) ?? 0) + 1));
        rows.value = value.map(row => {
            const index = (counts.get(row.name) ?? 0) + 1;
            counts.set(row.name, index);
            return { ...row, key: totals.get(row.name)! > 1 ? `${row.name}#${index}` : row.name };
        });
        if (!rows.value.some(row => row.key === selectedKey.value))
            selectedKey.value = rows.value[0]?.key ?? null;
    }
    catch { /* call presents the error */
    }
    finally {
        if (current === ticket)
            loading.value = false;
    }
}
watch(kind, () => {
    rows.value = [];
    selectedKey.value = null;
    inspectorOpen.value = false;
});
watch([kind, query], load, { immediate: true });
const off = bus.on('data_changed', load);
onBeforeUnmount(() => {
    off();
    ticket++;
});
</script>
<template>
  <div class="browser-page database-page">
    <div class="browser-toolbar db-toolbar">
      <USegmented :model-value="kind" :options="kinds" label="数据类别" @update:model-value="changeKind" />
      <div class="db-search-row">
        <USearchField v-model="query" placeholder="搜索名称、描述…" />
        <span class="entry-count numeric">{{ loading ? '读取中…' : `${rows.length} 条` }}</span>
      </div>
    </div>
    <SplitView v-model:open="inspectorOpen" name="db" label="条目检查器">
      <div v-if="loading && !rows.length" class="browser-loading">
        <USpinner />
        正在读取数据…
      </div>
      <UTable v-else class="browser-table database-table" :class="`kind-${kind}`" :columns="columns" :rows="tableRows" row-key="key" :selected="selectedKey" label="数据库条目" empty-title="没有匹配的条目" @update:selected="select">
        <template #cell-name="{row}">
          <span :style="kind==='heroes' ? {color:costColor(Number(row.cost))} : undefined" :title="String(row.name)">{{ row.name }}</span>
        </template>
        <template #cell-traits="{value}">
          <div class="inspector-chips">
            <UBadge v-for="name in (value as string[])" :key="name" tone="blue">{{ name }}</UBadge>
          </div>
        </template>
        <template #cell-tier_label="{row}">
          <UBadge :tone="row.tier===1 ? 'gray' : row.tier===2 ? 'yellow' : row.tier===3 ? 'purple' : 'gray'">{{ row.tier_label || '—' }}</UBadge>
        </template>
      </UTable>
      <template #inspector>
        <DatabaseInspector v-if="selected" :kind="kind" :entry="selected" />
        <UEmpty v-else title="选择一个条目查看详情" />
      </template>
    </SplitView>
  </div>
</template>
<style scoped>
.db-toolbar>.u-segmented {
  width: 100%;
}
.db-search-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  width: 100%;
}
.entry-count {
  color: var(--secondary);
  font-size: var(--font-xs);
  white-space: nowrap;
}
.database-table :deep(table) {
  min-width: 570px;
}
.database-table :deep(td) {
  height: 36px;
}
.database-table :deep(.inspector-chips) {
  white-space: nowrap;
  flex-wrap: nowrap;
}
.database-table.kind-encounters :deep(table),.database-table.kind-augments :deep(table) {
  min-width: 0;
}
@media (max-width:1199px) {
  .db-toolbar :deep(.u-segmented button) {
    padding: 0 var(--space-2);
  }
}
</style>
