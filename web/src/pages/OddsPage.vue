<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef } from 'vue';
import { Percent, Plus, Download, Trash2, X, Check, Dices } from 'lucide-vue-next';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { useAppStore } from '../stores/app';
import { useOddsStore } from '../stores/odds';
import { costColor } from '../composables/presentation';
import type { HeroOption, TableColumn } from '../types';
import UCard from '../components/ui/UCard.vue';
import UNumberField from '../components/ui/UNumberField.vue';
import USwitch from '../components/ui/USwitch.vue';
import UCombobox from '../components/ui/UCombobox.vue';
import UButton from '../components/ui/UButton.vue';
import UBadge from '../components/ui/UBadge.vue';
import UTable from '../components/ui/UTable.vue';
import UEmpty from '../components/ui/UEmpty.vue';
const odds = useOddsStore();
const app = useAppStore();
const heroes = shallowRef<HeroOption[]>([]);
const hero = ref(''), need = ref(3);
const options = computed(() => heroes.value.map(h => ({ value: h.name, label: h.name, hint: `${h.cost} 费`, color: costColor(h.cost) })));
const resultRows = computed(() => (odds.result?.rows ?? []).map(row => ({ ...row })));
const columns: TableColumn[] = [
    { key: 'name', title: '英雄', width: 95 }, { key: 'cost', title: '费', width: 30, align: 'right' },
    { key: 'need', title: '需要', width: 68, align: 'right' }, { key: 'have', title: '已有', width: 40, align: 'right' },
    { key: 'others', title: '别人拿走', width: 78, align: 'right' }, { key: 'remaining', title: '剩余', width: 42, align: 'right' },
    { key: 'slot_prob', title: '单格', width: 60, align: 'right', format: percent }, { key: 'per_roll', title: '单次刷出', width: 95, align: 'right' },
    ...['rolls_50', 'gold_50', 'rolls_80', 'gold_80'].map((key, i) => ({ key, title: ['50% 次数', '50% 金币', '80% 次数', '80% 金币'][i]!, width: 62, align: 'right' as const })),
    { key: 'remove', title: '', width: 28 }
];
const estimates = ['rolls_50', 'gold_50', 'rolls_80', 'gold_80'];
function percent(value: unknown) {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}
function estimate(value: unknown) {
    return typeof value === 'number' && Number.isFinite(value) ? Math.ceil(value) : '—';
}
function fraction(value: number) {
    return Math.max(0, Math.min(1, value));
}
async function loadHeroes() {
    try {
        heroes.value = await call<HeroOption[]>('hero_options');
    }
    catch { /* call reports */
    }
}
void loadHeroes();
const off = bus.on('data_changed', loadHeroes);
onBeforeUnmount(off);
</script>
<template>
  <div class="odds-page">
    <div class="odds-top">
      <UCard title="参数">
        <div class="odds-parameters">
          <label>
            <span>
              等级
              <UBadge v-if="odds.follow && app.state" tone="blue">实时</UBadge>
            </span>
            <UNumberField :model-value="odds.level" :min="1" :max="10" label="等级" @update:model-value="odds.setLevel" />
          </label>
          <label>
            <span>
              金币
              <UBadge v-if="odds.follow && app.state" tone="blue">实时</UBadge>
            </span>
            <UNumberField :model-value="odds.gold" :min="0" :max="200" label="金币" @update:model-value="odds.setGold" />
          </label>
          <label>
            保留金币
            <UNumberField v-model="odds.keep" :min="0" :max="100" label="保留金币" />
          </label>
        </div>
        <div class="follow-row">
          <USwitch v-model="odds.follow" label="跟随实时画面" />
          <span>跟随实时画面</span>
        </div>
      </UCard>
      <UCard title="目标英雄">
        <div class="odds-add">
          <UCombobox v-model="hero" :options="options" placeholder="选择英雄…" label="目标英雄" />
          <UNumberField v-model="need" :min="1" :max="9" label="需要张数" />
          <UButton variant="primary" :icon="Plus" :disabled="!hero" @click="odds.add(hero,need)">添加</UButton>
        </div>
        <div class="odds-import">
          <UButton :icon="Download" @click="odds.importTarget">从目标阵容导入</UButton>
          <UButton variant="plain" :icon="Trash2" :disabled="!odds.rows.length" @click="odds.clear">清空</UButton>
        </div>
      </UCard>
    </div>
    <UEmpty v-if="!odds.rows.length" :icon="Percent" title="添加目标英雄，或从目标阵容导入" description="根据当前牌库计算刷新概率，规划每一枚金币" />
    <template v-else>
      <UCard title="刷新概率" :padded="false">
        <UTable class="odds-table" :columns="columns" :rows="resultRows" row-key="name" label="牌库概率结果" :empty-title="odds.calculating ? '正在计算…' : '暂无结果'">
          <template #cell-name="{row}">
            <span :style="{color:costColor(Number(row.cost))}">{{ row.name }}</span>
          </template>
          <template #cell-need="{row}">
            <UNumberField :model-value="Number(row.need)" :min="1" :max="9" :label="`${row.name} 需要张数`" @update:model-value="odds.update(String(row.name),'need',$event)" />
          </template>
          <template #cell-others="{row}">
            <UNumberField :model-value="Number(row.others)" :min="0" :max="30" :label="`${row.name} 别人拿走`" @update:model-value="odds.update(String(row.name),'others',$event)" />
          </template>
          <template #cell-per_roll="{value}">
            <div class="odds-probability">
              <span>{{ percent(value) }}</span>
              <i>
                <b :style="{transform:`scaleX(${fraction(Number(value))})`}" />
              </i>
            </div>
          </template>
          <template v-for="key in estimates" :key="key" #[`cell-${key}`]="{row,value}">
            <Check v-if="row.done" class="odds-done" :size="14" />
            <span v-else>{{ estimate(value) }}</span>
          </template>
          <template #cell-remove="{row}">
            <UButton class="remove-hero" variant="plain" size="sm" :icon="X" :aria-label="`移除 ${row.name}`" @click="odds.remove(String(row.name))" />
          </template>
        </UTable>
        <p v-if="odds.result?.any_hit!=null" class="odds-summary">
          {{ odds.level }}
          级：一次刷新至少出现一张目标卡的概率
          <strong>{{ percent(odds.result.any_hit) }}</strong>
        </p>
      </UCard>
      <UCard title="D 牌模拟">
        <template #actions>
          <UButton variant="primary" :icon="Dices" :loading="odds.simulating" @click="odds.simulate">D 牌模拟（3000 次）</UButton>
        </template>
        <div v-if="odds.simulation" class="simulation-result" :class="{stale:odds.stale}">
          <div class="simulation-headline">
            全部凑齐
            <strong class="numeric">{{ percent(odds.simulation.all_hit_rate) }}</strong>
          </div>
          <div class="simulation-bars">
            <div v-for="(prob,name) in odds.simulation.hit_rate" :key="name">
              <span>{{ name }}</span>
              <i>
                <b :style="{transform:`scaleX(${fraction(prob)})`}" />
              </i>
              <span class="numeric">{{ percent(prob) }}</span>
            </div>
          </div>
          <div class="simulation-stats">
            <span>
              平均刷新
              <b class="numeric">{{ odds.simulation.avg_rolls.toFixed(1) }}</b>
              次
            </span>
            <span>
              平均剩余
              <b class="numeric">{{ odds.simulation.avg_gold_left.toFixed(1) }}</b>
              金
            </span>
          </div>
        </div>
        <p v-else class="simulation-hint">模拟预算内的刷新结果，估算全部凑齐的把握。</p>
        <p v-if="odds.stale" class="stale-note">参数已变化，重新模拟</p>
      </UCard>
    </template>
  </div>
</template>
<style scoped>
.odds-page {
  height: 100%;
  overflow: auto;
  padding: var(--space-5) var(--space-6);
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}
.odds-top {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: var(--space-4);
}
.odds-parameters,.odds-add,.odds-import,.follow-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}
.odds-parameters label {
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
  gap: var(--space-2);
  font-size: var(--font-xs);
  color: var(--secondary);
}
.odds-parameters .u-number {
  width: 100%;
}
.follow-row {
  font-size: var(--font-sm);
  margin-top: var(--space-4);
}
.odds-add :deep(.popover-anchor) {
  flex: 1;
  min-width: 100px;
}
.odds-add .u-number {
  width: 80px;
  flex-shrink: 0;
}
.odds-import {
  margin-top: var(--space-4);
}
.odds-page>.u-card,.odds-top {
  flex-shrink: 0;
}
.odds-table :deep(th),.odds-table :deep(td) {
  padding: var(--space-half) var(--space-1);
  font-size: var(--font-xs);
}
.odds-table :deep(.u-number) {
  width: 58px;
  height: 26px;
}
.odds-table :deep(.u-number input) {
  text-align: right;
  padding: 0 var(--space-1);
}
.odds-table :deep(.steppers) {
  width: 16px;
}
.odds-table :deep(table) {
  min-width: 900px;
}
.odds-table :deep(.cell-content) {
  max-height: 32px;
}
.odds-summary {
  padding: var(--space-3) var(--space-4);
  font-size: var(--font-sm);
  color: var(--secondary);
  border-top: 1px solid var(--separator);
}
.odds-summary strong {
  color: var(--blue-text);
}
.odds-probability {
  display: flex;
  flex-direction: column;
  gap: var(--space-half);
}
.odds-probability i,.simulation-bars i {
  display: block;
  overflow: hidden;
  height: 3px;
  background: var(--accent-soft);
  border-radius: var(--radius-pill);
}
.odds-probability b,.simulation-bars b {
  display: block;
  width: 100%;
  height: 100%;
  transform-origin: left;
  background: var(--accent);
}
.odds-done {
  color: var(--green-text);
  vertical-align: middle;
}
.remove-hero {
  opacity: 0;
}
.odds-table :deep(tr:hover) .remove-hero,.odds-table :deep(tr:focus-within) .remove-hero {
  opacity: 1;
}
.simulation-headline {
  font-size: var(--font-lg);
}
.simulation-headline strong {
  font-size: var(--font-display);
  margin-left: var(--space-2);
  color: var(--blue-text);
}
.simulation-bars {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  margin-top: var(--space-4);
}
.simulation-bars>div {
  display: grid;
  grid-template-columns: 100px 1fr 64px;
  gap: var(--space-4);
  align-items: center;
  font-size: var(--font-sm);
}
.simulation-bars i {
  height: 5px;
}
.simulation-bars .numeric {
  text-align: right;
}
.simulation-stats {
  display: flex;
  gap: var(--space-6);
  font-size: var(--font-sm);
  color: var(--secondary);
  margin-top: var(--space-5);
}
.simulation-hint {
  font-size: var(--font-sm);
  color: var(--secondary);
}
.simulation-result.stale {
  opacity: .45;
}
.stale-note {
  margin-top: var(--space-3);
  color: var(--orange-text);
  font-size: var(--font-xs);
}
@media (max-width:1099px) {
  .odds-top {
    grid-template-columns: 1fr;
  }
}
</style>
