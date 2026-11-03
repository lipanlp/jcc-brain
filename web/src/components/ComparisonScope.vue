<script setup lang="ts">
import type { AssistantSelection } from '../types';
defineProps<{ scope?: AssistantSelection }>();
</script>
<template>
  <div v-if="scope" class="comparison-scope" role="status" :data-revision="scope.revision">
    <p>辅助预选：<b>{{ scope.name }}</b> · 已选目录 {{ scope.catalog_count }} · 实际可比较 {{ scope.effective_count }} · 本局候选 {{ scope.candidate_count }}</p>
    <p v-if="scope.notice" class="scope-notice">{{ scope.notice }}</p>
    <p v-if="scope.watch_outside.length">不在当前有效预选内的关注（记录保留）：{{ scope.watch_outside.map(row => row.name).join('、') }}</p>
    <p class="scope-explanation">自动模式在回合评估时比较全部有效候选，界面收敛展示3/2/1套；回合内刷新已有候选。本局关注取交集，手动锁定优先。仅影响辅助建议，不改变阵容库、对手识别、自动D牌或托管范围。</p>
  </div>
</template>
<style scoped>
.comparison-scope { color: var(--secondary); font-size: var(--font-xs); line-height: 1.6; overflow-wrap: anywhere; }
.comparison-scope b { color: var(--text); }
.comparison-scope p + p { margin-top: var(--space-1); }
.scope-notice { color: var(--orange-text); }
.scope-explanation { color: var(--tertiary); }
</style>
