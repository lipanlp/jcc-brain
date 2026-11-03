<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { LayoutGrid, Lock, RefreshCw } from 'lucide-vue-next';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { useAppStore } from '../stores/app';
import { useWatchComps } from '../composables/watchComps';
import { toast } from '../composables/toast';
import { costColor, tierTone, decimal, rate } from '../composables/presentation';
import type { CompDetail, CompRow, CompsResult, TableColumn } from '../types';
import SplitView from '../components/SplitView.vue';
import CompInspector from '../components/CompInspector.vue';
import UTable from '../components/ui/UTable.vue';
import USegmented from '../components/ui/USegmented.vue';
import USearchField from '../components/ui/USearchField.vue';
import UBadge from '../components/ui/UBadge.vue';
import UButton from '../components/ui/UButton.vue';
import UEmpty from '../components/ui/UEmpty.vue';
import USpinner from '../components/ui/USpinner.vue';
const store = useAppStore();
const { watching, watchIds, isWatched, toggleWatch } = useWatchComps();
const source = ref('all');
const query = ref('');
const result = shallowRef<CompsResult>();
const selectedId = ref<string | null>(null);
const detail = shallowRef<CompDetail | null>(null);
const loading = ref(true);
const detailLoading = ref(false);
const inspectorOpen = ref(false);
const locking = ref(false);
const version = ref(0);
let listTicket = 0;
let detailTicket = 0;
const sources = [{ value: 'all', label: '全部' }, { value: 'official', label: '官网推荐' }, { value: 'ingame', label: '微社区' }, { value: 'dataj', label: 'dataj 胜率' }];
const columns: TableColumn[] = [
    { key: 'tier', title: '评级', width: '7%' }, { key: 'name', title: '阵容', width: '31%' },
    { key: 'style_label', title: '打法', width: '9%' },
    { key: 'avg_placement', title: '均名', width: '9%', align: 'right', sortable: true, format: decimal },
    { key: 'top4_rate', title: '前四', width: '10%', align: 'right', sortable: true, format: rate },
    { key: 'top_rate', title: '登顶', width: '10%', align: 'right', format: rate },
    { key: 'core', title: '核心', width: '24%' }
];
const filtered = computed(() => {
    const needle = query.value.trim().toLocaleLowerCase();
    return (result.value?.rows ?? []).filter(row => !needle || [row.name, ...row.core.map(hero => hero.name)].some(value => value.toLocaleLowerCase().includes(needle))).map(row => ({ ...row }));
});
watch(filtered, rows => {
    if (!rows.some(row => row.id === selectedId.value))
        selectedId.value = rows[0]?.id ?? null;
});
function select(id: string | number) {
    selectedId.value = String(id);
    inspectorOpen.value = true;
}
async function load() {
    const ticket = ++listTicket;
    loading.value = true;
    try {
        const value = await call<CompsResult>('comps', source.value);
        if (ticket !== listTicket)
            return;
        result.value = value;
        version.value++;
    }
    catch { /* call presents the error */
    }
    finally {
        if (ticket === listTicket)
            loading.value = false;
    }
}
watch(source, load, { immediate: true });
watch([selectedId, version], async ([id]) => {
    const ticket = ++detailTicket;
    detail.value = null;
    detailLoading.value = !!id;
    if (!id)
        return;
    try {
        const value = await call<CompDetail>('comp_detail', id);
        if (ticket === detailTicket)
            detail.value = value;
    }
    catch { /* call presents the error */
    }
    finally {
        if (ticket === detailTicket)
            detailLoading.value = false;
    }
});
async function setLock(comp: CompRow | null, asDefault = false) {
    if (locking.value)
        return;
    locking.value = true;
    try {
        await call(asDefault ? 'set_default_lock_comp' : 'lock_comp_for_game', comp?.id ?? '');
        toast.success(comp ? `${asDefault ? '默认锁定' : '本局锁定'}：${comp.name}` : `已解除${asDefault ? '默认' : '本局'}锁定`);
    }
    catch { /* call presents the error */
    }
    finally {
        locking.value = false;
    }
}
async function sync() {
    try {
        await call('sync', true);
    }
    catch { /* call presents the error */
    }
}
const off = bus.on('data_changed', load);
onBeforeUnmount(() => {
    off();
    listTicket++;
    detailTicket++;
});
</script>
<template>
  <div class="browser-page comps-page">
    <p v-if="store.status?.comp_source_notice" role="status" class="source-notice">{{ store.status.comp_source_notice }}</p>
    <div class="browser-toolbar">
      <USegmented v-model="source" :options="sources" label="阵容来源" />
      <USearchField v-model="query" placeholder="搜索阵容或核心英雄…" />
      <div class="browser-meta">
        <span>库内共 {{ result?.info.catalog_count ?? 0 }} 条 · 官网 {{ result?.info.official_count ?? 0 }} / 微社区 {{ result?.info.ingame_count ?? 0 }} / dataj {{ result?.info.dataj_count ?? 0 }}（官网与微社区重叠 {{ result?.info.dual_channel_count ?? 0 }} 条）</span>
        <span>版本 {{ result?.info.game_version || '—' }} · 同步于 {{ result?.info.synced_at || '—' }}</span>
        <div v-if="store.status?.lock" class="target-lock">
          <UBadge tone="blue">🔒 {{ store.status.lock.source==='game' ? '本局锁定' : '默认锁定' }}：{{ store.status.lock.name }}</UBadge>
          <UButton v-if="store.status.lock.source==='game'" variant="plain" size="sm" :loading="locking" @click="setLock(null)">解除本局锁定</UButton>
        </div>
        <span v-if="watchIds.length">已保留 {{ watchIds.length }} 个关注{{ store.status?.lock ? ' · 锁定优先，关注暂停；解除后恢复与预选配置的交集比较' : ' · 按与预选配置的交集比较' }}</span>
      </div>
    </div>
    <p v-if="result?.info.source_notice && !store.status?.comp_source_notice" role="status" class="source-notice">{{ result.info.source_notice }}</p>
    <UEmpty v-if="!loading && store.status?.db_empty" class="comps-empty" :icon="LayoutGrid" title="暂无阵容" description="当前赛季没有数据">
      <template #action>
        <UButton variant="primary" :icon="RefreshCw" :disabled="store.status?.syncing" @click="sync">同步数据</UButton>
      </template>
    </UEmpty>
    <SplitView v-else v-model:open="inspectorOpen" name="comps" label="阵容检查器" :default-width="520">
      <div v-if="loading && !result" class="browser-loading">
        <USpinner />
        正在读取阵容…
      </div>
      <UTable v-else class="browser-table comps-table" :columns="columns" :rows="filtered" row-key="id" :selected="selectedId" label="阵容列表" empty-title="没有匹配的阵容" @update:selected="select">
        <template #cell-tier="{value}">
          <UBadge :tone="tierTone(value as string)">{{ value || '—' }}</UBadge>
        </template>
        <template #cell-name="{row}">
          <span class="comp-list-name" :title="String(row.name)">
            <Lock v-if="store.status?.lock?.id===row.id" :size="12" />
            <span>{{ row.name }}</span>
          </span>
          <span class="source-label">{{ row.source_label }}</span>
        </template>
        <template #cell-core="{value}">
          <span class="core-names">
            <span v-for="hero in (value as CompRow['core'])" :key="hero.name" :style="{color:costColor(store.status?.hero_costs[hero.name])}">{{ hero.carry ? '★ ' : '' }}{{ hero.name }}</span>
          </span>
        </template>
      </UTable>
      <template #inspector>
        <div v-if="detailLoading" class="inspector-loading">
          <USpinner />
          正在读取详情…
        </div>
        <CompInspector v-else-if="detail" :comp="detail" :locked="store.status?.lock?.source==='game' && store.status.lock.id===detail.id"
                       :default-locked="store.status?.default_lock?.id===detail.id" :locking="locking"
                       :watched="isWatched(detail.id)" :watching="watching" :watch-full="watchIds.length>=4" @watch="toggleWatch(detail.id)"
                       @lock="setLock(detail)" @unlock="setLock(null)" @default-lock="setLock(detail,true)" @default-unlock="setLock(null,true)" />
        <UEmpty v-else title="选择一个阵容查看详情" />
      </template>
    </SplitView>
  </div>
</template>
<style scoped>
.source-notice { padding: 12px; color: var(--orange-text); }
.comps-table :deep(table) {
  min-width: 0;
}
.comp-list-name {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  min-width: 0;
}
.comp-list-name>span {
  overflow: hidden;
  text-overflow: ellipsis;
}
.comp-list-name>svg {
  color: var(--blue-text);
}
.core-names>span {
  margin-right: var(--space-2);
}
.source-label {
  display: block;
  color: var(--secondary);
  font-size: var(--font-xs);
  white-space: normal;
}
.comps-table :deep(th), .comps-table :deep(td) {
  padding-left: var(--space-2);
  padding-right: var(--space-2);
}
.core-names {
  gap: var(--space-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
}
.comps-empty {
  flex: 1;
}
</style>
