<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { Check, ChevronDown, RefreshCw, Database } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { call } from '../bridge/api';
import { confirm } from '../composables/confirm';
import { toast } from '../composables/toast';
import { statsOpen } from '../composables/tools';
import type { CompSource, SourceFlags, SeasonOption } from '../types';
import UPopover from './ui/UPopover.vue';
import UButton from './ui/UButton.vue';
import USwitch from './ui/USwitch.vue';
import USpinner from './ui/USpinner.vue';
const app = useAppStore();
const open = ref(false), loading = ref(false), switching = ref(false), sourceBusy = ref(false), starting = ref(false);
const seasons = shallowRef<SeasonOption[]>([]);
const official = ref(true), ingame = ref(true), dataj = ref(true);
const hasIngame = computed(() => seasons.value.find(season => season.id === app.status?.season.id)?.has_ingame);
watch(() => app.status?.sources, flags => {
    if (flags) {
        official.value = flags.official;
        ingame.value = flags.ingame;
        dataj.value = flags.dataj;
    }
}, { immediate: true });
let ticket = 0;
async function load() {
    const current = ++ticket;
    loading.value = true;
    try {
        const catalog = await call<SeasonOption[]>('seasons');
        if (current !== ticket)
            return;
        seasons.value = catalog;
    }
    catch { /* call reports */
    }
    finally {
        if (current === ticket)
            loading.value = false;
    }
}
watch(open, value => {
    if (value)
        void load();
});
// 来源由 status 推送更新；不因每次来源切换重读目录而短暂禁用下一项开关。
watch(() => app.status?.syncing, (syncing, previous) => {
    if (previous && !syncing && open.value)
        void load();
});
onBeforeUnmount(() => {
    ticket++;
});
async function switchSeason(season: SeasonOption) {
    if (season.id === app.status?.season.id || switching.value || app.status?.syncing)
        return;
    switching.value = true;
    try {
        if (app.status?.mode && !await confirm({ title: '切换赛季', message: '切换赛季会停止正在运行的模式，继续？', confirmText: '切换赛季' }))
            return;
        const value = await call<{
            needs_sync: boolean;
        }>('switch_season', season.id);
        await load();
        if (value.needs_sync && await confirm({ title: '同步赛季数据', message: `${season.label} 还没有同步过数据，现在同步？`, confirmText: '同步数据' }))
            await sync(true);
    }
    catch { /* call reports */
    }
    finally {
        switching.value = false;
    }
}
async function source(which: CompSource, on: boolean) {
    sourceBusy.value = true;
    try {
        const value = await call<SourceFlags & { message: string }>('set_source', which, on);
        official.value = value.official;
        ingame.value = value.ingame;
        dataj.value = value.dataj;
        if (value.message)
            toast.warning(value.message);
    }
    catch { /* call reports */
    }
    finally {
        sourceBusy.value = false;
    }
}
async function sync(details: boolean, all = false) {
    if (starting.value || app.status?.syncing)
        return;
    starting.value = true;
    try {
        await call('sync', details, all);
    }
    catch { /* call reports */
    }
    finally {
        starting.value = false;
    }
}
function stats() {
    open.value = false;
    statsOpen.value = true;
}
</script>
<template>
  <UPopover v-model="open" :width="340" label="赛季">
    <template #trigger="{toggle}">
      <UButton class="season-button" variant="plain" size="sm" @click="toggle">
        <USpinner v-if="app.status?.syncing" :size="12" />
        {{ app.status?.season.label ?? '赛季' }}
        <ChevronDown :size="11" />
      </UButton>
    </template>
    <div class="season-control">
      <h2>赛季与数据</h2>
      <div v-if="loading&&!seasons.length" class="season-loading">
        <USpinner />
        读取中…
      </div>
      <div class="season-options">
        <button v-for="season in seasons" :key="season.id" :disabled="switching||app.status?.syncing" :data-season="season.id" class="season-option" :class="{active:season.id===app.status?.season.id}" @click="switchSeason(season)">
          <Check v-if="season.id===app.status?.season.id" :size="14" />
          <span v-else class="check-space" />
          <span>
            <strong>{{ season.label }}</strong>
            <small>{{ season.game_version ? `版本 ${season.game_version}` : '未同步' }}{{ season.has_dataj ? '' : ' · 无 dataj 统计' }}</small>
          </span>
        </button>
      </div>
      <div class="season-sources">
        <label>
          <span>官网推荐</span>
          <USwitch :model-value="official" :disabled="sourceBusy||loading" label="官网推荐" @update:model-value="source('official',$event)" />
        </label>
        <label>
          <span>微社区</span>
          <USwitch :model-value="ingame" :disabled="sourceBusy||loading" label="微社区" @update:model-value="source('ingame',$event)" />
        </label>
        <label>
          <span>dataj.cc（阵容统计）</span>
          <USwitch :model-value="dataj" :disabled="sourceBusy||loading" label="dataj.cc（阵容统计）" @update:model-value="source('dataj',$event)" />
        </label>
      </div>
      <p class="source-help">开关立即影响阵容浏览与推荐。官网与微社区共享基础资料同步。</p>
      <p v-if="hasIngame===false" class="source-help source-unsupported" role="status">当前赛季暂不支持微社区阵容，微社区仅支持自然之力。</p>
      <div class="sync-buttons">
        <UButton :icon="RefreshCw" :disabled="starting||app.status?.syncing" @click="sync(true)">同步最新数据</UButton>
        <UButton :disabled="starting||app.status?.syncing" @click="sync(true,true)">同步全部赛季</UButton>
        <UButton :disabled="starting||app.status?.syncing" @click="sync(false)">快速同步（仅排行）</UButton>
      </div>
      <p v-if="app.status?.syncing" class="sync-progress">
        <USpinner :size="12" />
        {{ app.syncMessage || '准备同步…' }}
      </p>
      <UButton variant="plain" size="sm" :icon="Database" @click="stats">数据库统计…</UButton>
    </div>
  </UPopover>
</template>
<style scoped>
h2 {
  font-size: var(--font-sm);
  font-weight: 600;
  padding: var(--space-1) var(--space-2) var(--space-2);
}
.season-options {
  max-height: 240px;
  overflow: auto;
}
.season-option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2);
  text-align: left;
  background: var(--transparent);
  width: 100%;
  border-radius: var(--radius-sm);
}
.season-option:hover {
  background: var(--hover);
}
.season-option.active {
  background: var(--accent-soft);
  color: var(--blue-text);
}
.season-option strong {
  font-size: var(--font-sm);
  font-weight: 500;
}
.season-option small {
  display: block;
  font-size: var(--font-xs);
  color: var(--secondary);
}
.check-space {
  width: 14px;
  flex-shrink: 0;
}
.season-sources {
  border-top: 1px solid var(--separator);
  border-bottom: 1px solid var(--separator);
  margin: var(--space-2) 0;
  padding: var(--space-2) 0;
}
.season-sources label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  padding: var(--space-2);
  font-size: var(--font-sm);
}
.sync-buttons {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin: var(--space-3) 0;
}
.sync-progress,.season-loading {
  display: flex;
  gap: var(--space-2);
  align-items: center;
  font-size: var(--font-xs);
  color: var(--secondary);
  margin: var(--space-2);
}
.source-help {
  font-size: var(--font-xs);
  color: var(--secondary);
  line-height: 1.6;
  margin: var(--space-2);
}
.source-unsupported { color: var(--orange-text); }
@media (max-width:1120px) {
  .season-button {
    max-width: 90px;
    overflow: hidden;
  }
}
</style>
