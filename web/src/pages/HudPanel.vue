<script setup lang="ts">
import { computed } from 'vue';
import { visibleBuys } from '../composables/buyProjection';
import { hudAdvice, hudFrame } from '../composables/hud';
import { targetsByCost } from '../composables/presentation';
import { vWindowDrag } from '../composables/windowDrag';
const buys = computed(() => visibleBuys(hudAdvice.value));
const targets = computed(() => targetsByCost(hudAdvice.value?.targets ?? []));
</script>
<template>
  <div class="hud-panel">
    <header v-window-drag="'hud_panel'" class="hud-panel-title">
      <span v-if="hudAdvice?.target">🎯 {{ hudAdvice.target.name }}{{ hudAdvice.target.tier ? `（${hudAdvice.target.tier}）` : '' }}</span>
      <span v-else>目标卡概率</span>
    </header>
    <div v-if="buys.length" class="hud-panel-econ" aria-label="买卡建议">
      <div v-for="buy in buys" :key="buy.slot" :data-buy-slot="buy.slot">买第 {{ buy.slot + 1 }} 格 {{ buy.name }} · {{ buy.reason }}</div>
    </div>
    <div class="hud-panel-content">
      <p v-if="!hudAdvice" class="hud-panel-empty">等待画面…</p>
      <template v-else>
        <table class="numeric">
          <thead>
            <tr>
              <th>英雄</th>
              <th>有/需</th>
              <th>剩</th>
              <th>单次</th>
              <th>50%</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="target in targets" :key="target.name">
              <td :style="{ color: `var(--cost-${target.cost ?? 0}, #ddd)` }" :title="target.name">{{ target.name }}</td>
              <td>{{ target.have }}/{{ target.have + target.need }}</td>
              <td>{{ target.remaining ?? '—' }}</td>
              <td>{{ target.per_roll === null ? '—' : `${Math.round(target.per_roll * 100)}%` }}</td>
              <td :class="{ done: target.need === 0 }" :title="target.need > 0 && target.gold_for_50 === null ? '未估算50%预算不代表不能刷到，请结合出现概率与卡池余量判断' : undefined">{{ target.need === 0 ? '✓' : target.gold_for_50 === null ? '—' : `${target.gold_for_50}金` }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="!targets.length" class="hud-panel-empty">暂无目标卡</p>
      </template>
    </div>
    <footer v-if="hudAdvice?.econ" class="hud-panel-econ">{{ hudAdvice.econ.reason }}</footer>
  </div>
</template>
<style scoped>
.hud-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  color: #fff;
  background: rgba(28,28,30,.78);
  backdrop-filter: blur(30px);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 12px;
  overflow: hidden;
  font-size: 12px;
}
.hud-panel-title {
  height: 28px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  padding: 0 10px;
  border-bottom: 1px solid rgba(255,255,255,.1);
  font-weight: 600;
}
.hud-panel-title span {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  pointer-events: none;
}
.hud-panel-content {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 4px 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
th {
  position: sticky;
  top: -4px;
  background: #28282b;
  color: #aaa;
  font-size: 11px;
  font-weight: 500;
}
th, td {
  padding: 3px 2px;
  text-align: right;
  white-space: nowrap;
}
th:first-child, td:first-child {
  width: 34%;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
}
td.done {
  color: #30d158;
}
.hud-panel-econ {
  flex-shrink: 0;
  padding: 5px 10px;
  color: #8ab4f8;
  font-size: 11px;
  line-height: 1.4;
  border-top: 1px solid rgba(255,255,255,.08);
  max-height: 55px;
  overflow: auto;
}
.hud-panel-empty {
  color: #aaa;
  padding: 12px 2px;
}
</style>
