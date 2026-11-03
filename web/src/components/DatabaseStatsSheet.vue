<script setup lang="ts">
import { ref, shallowRef, watch } from 'vue';
import { statsOpen } from '../composables/tools';
import { call } from '../bridge/api';
import type { DatabaseStats } from '../types';
import USheet from './ui/USheet.vue';
import UButton from './ui/UButton.vue';
import USpinner from './ui/USpinner.vue';
const data = shallowRef<DatabaseStats>();
const loading = ref(false);
const labels: Record<string, string> = { season: '赛季', season_id: '赛季 ID', game_version: '游戏版本', data_source: '基础资料来源', synced_at: '同步时间', heroes: '英雄', traits: '羁绊', items: '装备', augments: '海克斯', comps: 'dataj 阵容', guides: '官方攻略（渠道去重）', mechanics: '赛季机制', encounters: '开局奇遇',
    web_guides: '官网推荐', ingame_guides: '微社区攻略', dual_channel_guides: '两渠道重叠',
    comps_catalog_total: '目录原始阵容总数', comps_total: '推荐阵容（关联去重）',
    guides_with_stats: '带热度的攻略', guide_stat_date: '热度统计日期', stats_version: 'dataj 统计版本',
    linked_guides: '已关联统计的攻略', linked_comps: '已关联攻略的统计阵容' };
let ticket = 0;
watch(statsOpen, async (open) => {
    const current = ++ticket;
    if (!open)
        return;
    loading.value = true;
    try {
        const result = await call<DatabaseStats>('db_stats');
        if (current === ticket)
            data.value = result;
    }
    catch { /* call reports */
    }
    finally {
        if (current === ticket)
            loading.value = false;
    }
});
</script>
<template>
  <USheet v-model="statsOpen" title="数据库统计" :width="480">
    <p class="stats-note">官网与微社区有重复记录，目录按原始 ID 计数；推荐另按关联去重。热度不代表胜率。</p>
    <div v-if="loading" class="stats-loading">
      <USpinner />
      正在读取统计…
    </div>
    <dl v-else-if="data" class="database-stats">
      <div v-for="(value,key) in data.stats" :key="key">
        <dt>{{ labels[key] ?? key }}</dt>
        <dd class="numeric selectable">{{ value ?? '—' }}</dd>
      </div>
      <div v-for="(count,cost) in data.pool_count_by_cost" :key="`cost-${cost}`">
        <dt>{{ cost }} 费英雄</dt>
        <dd class="numeric">{{ count }}</dd>
      </div>
    </dl>
    <template #footer>
      <UButton variant="primary" @click="statsOpen=false">好</UButton>
    </template>
  </USheet>
</template>
<style scoped>
.database-stats {
  margin: 0;
  font-size: var(--font-sm);
}
.database-stats>div {
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: var(--space-3);
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--separator);
}
.database-stats dt {
  color: var(--secondary);
}
.database-stats dd {
  margin: 0;
  text-align: right;
  overflow-wrap: anywhere;
}
.stats-loading {
  display: flex;
  justify-content: center;
  gap: var(--space-2);
  color: var(--secondary);
}
.stats-note { color: var(--secondary); font-size: var(--font-xs); line-height: 1.6; margin-bottom: var(--space-3); }
</style>
