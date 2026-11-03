<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { tierTone, targetsByCost } from '../composables/presentation';
import { Activity, Lightbulb, Target, Coins, ShoppingCart, Trash2, Sparkles, Sword, Bot, Check, RefreshCw, Play, Lock, LockOpen } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { startMode, startingMode } from '../composables/modes';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { visibleBuys } from '../composables/buyProjection';
import { headlineSummary, purchaseHeadline, visibleCompReasons, visibleAiText, visibleAiAugment } from '../composables/purchasePresentation';
import { usePopulationPreview } from '../composables/populationPreview';
import { useWatchComps } from '../composables/watchComps';
import type { CompRow, CompsResult, Phase, TableColumn, TableRow } from '../types';
import UCard from '../components/ui/UCard.vue';
import UBadge from '../components/ui/UBadge.vue';
import UEmpty from '../components/ui/UEmpty.vue';
import UButton from '../components/ui/UButton.vue';
import UTable from '../components/ui/UTable.vue';
import UTooltip from '../components/ui/UTooltip.vue';
import USelect from '../components/ui/USelect.vue';
import UCombobox from '../components/ui/UCombobox.vue';
import MetricValue from '../components/MetricValue.vue';
import UnitChips from '../components/UnitChips.vue';
import BoardMini from '../components/BoardMini.vue';
import ItemAssignments from '../components/ItemAssignments.vue';
import ComparisonScope from '../components/ComparisonScope.vue';
import TraitsView from '../components/TraitsView.vue';
import OpponentsCard from '../components/OpponentsCard.vue';
import LiveCardGrid from '../components/live/LiveCardGrid.vue';
import LiveCardTools from '../components/live/LiveCardTools.vue';
const store = useAppStore();
const { population, manual, loading: previewLoading, error: previewError, board: displayBoard, targets: displayTargets } = usePopulationPreview(store);
const populationOptions = [{ value: 'auto', label: '自动人口' }, ...Array.from({ length: 10 }, (_, i) => ({ value: String(i + 1), label: `${i + 1} 人口` }))];
const { watching, watchIds, isWatched, toggleWatch, comparisonSelection, isCompared, toggleCompared } = useWatchComps();
const catalog = shallowRef<CompRow[]>([]);
const watchOptions = computed(() => catalog.value.filter(comp => !isWatched(comp.id))
    .map(comp => ({ value: comp.id, label: comp.name, hint: comp.tier ?? '' })));
let catalogTicket = 0;
async function loadCatalog() {
    const ticket = ++catalogTicket;
    if (!store.status) return;
    try {
        const result = await call<CompsResult>('comps');
        if (ticket===catalogTicket) catalog.value = result.rows;
    }
    catch { /* call reports errors */ }
}
watch(() => store.status?.season.id, () => { catalog.value = []; void loadCatalog(); }, { immediate: true });
const offCatalog = bus.on('data_changed', () => { void loadCatalog(); });
onBeforeUnmount(() => { catalogTicket++; offCatalog(); });
const state = computed(() => store.state);
const advice = computed(() => store.advice);
const augmentsTaken = computed(() => store.status?.augments_taken ?? advice.value?.augments_taken ?? []);
const locking = ref(false);
const lockingId = ref('');
const selectedHero = ref<string | number | null>(null);
const selectedName = computed(() => selectedHero.value===null ? null : String(selectedHero.value));
function isLocked(id: unknown) {
    return store.status?.lock?.source === 'game' && (store.status.lock.canonical_id ?? store.status.lock.id) === id;
}
async function lockForGame(id: string) {
    if (locking.value) return;
    locking.value = true;
    lockingId.value = id;
    try {
        await call('lock_comp_for_game', id);
    }
    catch { /* call reports errors */ }
    finally { locking.value = false; }
}
const phaseLabels: Record<Phase, string> = { planning: '准备阶段', combat: '战斗中', augment: '强化选择', carousel: '选秀', ended: '已结束', unknown: '未知阶段' };
const comps = computed(() => (advice.value?.comps ?? []).map((comp, index) => ({ ...comp, why: visibleCompReasons(advice.value!, comp), current: index === 0 })));
const targets = computed(() => targetsByCost(displayTargets.value.map(target => ({ ...target }))));
const compColumns: TableColumn[] = [
    { key: 'watch', title: '关注', width: 52, align: 'center' },
    { key: 'name', title: '阵容', width: '19%' }, { key: 'fit', title: '契合', width: '8%', align: 'right' },
    { key: 'avg_placement', title: '均名', width: '8%', align: 'right', format: value => decimal(value) },
    { key: 'seen', title: '来牌', width: '17%', wrap: true },
    { key: 'why', title: '理由', wrap: true }, { key: 'missing', title: '还缺', width: '20%', wrap: true },
    { key: 'lock', title: '', width: 76, align: 'center' }
];
const targetColumns: TableColumn[] = [
    { key: 'name', title: '英雄', width: '28%' },
    { key: 'have', title: '已有/目标', width: '20%', align: 'right' },
    { key: 'need', title: '还差', width: '12%', align: 'right' },
    { key: 'per_roll', title: '单次', width: '20%', align: 'right' },
    { key: 'gold_for_50', title: '50%', width: '20%', align: 'right' }
];
const actions = computed(() => [
    ...visibleBuys(advice.value).map(buy => ({ key: `buy-${buy.slot}`, icon: ShoppingCart, tone: 'buy', text: `买第 ${buy.slot + 1} 格 ${buy.name} — ${buy.reason}`, best: false, aiBest: false, score: null })),
    ...(advice.value?.sells ?? []).map(sell => ({ key: `sell-${sell.slot}`, icon: Trash2, tone: 'sell', text: `卖第 ${sell.slot + 1} 格 ${sell.name} — ${sell.reason}`, best: false, aiBest: false, score: null })),
    ...(advice.value?.augments ?? []).map(augment => ({ key: `augment-${augment.index}`, icon: Sparkles, tone: 'augment', text: `${augment.name} — ${augment.reason}`, best: augment.best, aiBest: augment.name === visibleAiAugment(advice.value, state.value), score: augment.score })),
    ...(advice.value?.item_plan ?? []).map((plan, index) => ({ key: `plan-${index}`, icon: Sword, tone: 'item', text: plan.reason, best: false, aiBest: false, score: null }))
]);
const xpProgress = computed(() => state.value?.xp && state.value.xp[1] > 0 ? fraction(state.value.xp[0] / state.value.xp[1]) : 0);
const headline = computed(() => headlineSummary(advice.value));
const buyHeadline = computed(() => purchaseHeadline(advice.value));
const aiText = computed(() => visibleAiText(advice.value, state.value));
const compTones = { '探索': 'gray', '过渡': 'orange', '定型': 'green', '锁定': 'blue', '关注': 'purple' } as const;
function fraction(value: unknown) {
    return typeof value === 'number' && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}
function percent(value: unknown) {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
}
function decimal(value: unknown) {
    return typeof value === 'number' ? value.toFixed(2) : '—';
}
function costColor(name: unknown) {
    const cost = store.status?.hero_costs?.[String(name)];
    return cost ? `var(--cost-${cost})` : 'var(--secondary)';
}
function missing(row: TableRow) {
    return (row.missing as string[]).slice(0, 6);
}
async function sync() {
    try {
        await call('sync', true);
    }
    catch { /* call reports errors */
    }
}
</script>
<template>
  <div class="live-page">
    <p v-if="store.status?.comp_source_notice" class="source-notice" role="status">{{ store.status.comp_source_notice }}</p>
    <UEmpty v-if="!state" class="waiting" :icon="Activity" title="等待画面" :description="store.status?.db_empty ? '数据库为空，请先同步数据' : '启动辅助大脑后，这里会实时显示对局状态与建议'">
      <template #action>
        <UButton v-if="store.status?.db_empty" variant="primary" :icon="RefreshCw" :disabled="store.status?.syncing" @click="sync">同步数据</UButton>
        <UButton v-else variant="primary" :icon="Play" :loading="startingMode==='assistant'" :disabled="!store.status" @click="startMode('assistant')">启动辅助大脑</UButton>
      </template>
    </UEmpty>
    <template v-else>
      <section class="state-card" aria-label="对局状态">
        <div class="state-caption">
          <span>当前对局</span>
          <UBadge v-if="state.units_stale" tone="gray">棋子沿用准备阶段</UBadge>
        </div>
        <div class="state-metrics">
          <div class="state-metric">
            <label>阶段</label>
            <MetricValue :value="state.stage" />
            <UBadge :tone="state.phase==='planning' ? 'blue' : state.phase==='combat' ? 'orange' : 'gray'">{{ phaseLabels[state.phase] }}</UBadge>
          </div>
          <div class="state-metric" :class="{urgent:state.timer!==null && state.timer<=5}">
            <label>倒计时</label>
            <div>
              <MetricValue :value="state.timer" />
              <small v-if="state.timer!==null">秒</small>
            </div>
            <span class="metric-note">剩余时间</span>
          </div>
          <div class="state-metric">
            <label>金币</label>
            <MetricValue :value="state.gold" />
            <span class="metric-note">利息 {{ state.interest===null ? '—' : `+${state.interest}` }}</span>
          </div>
          <div class="state-metric">
            <label>等级</label>
            <MetricValue :value="state.level" />
            <div class="xp-row">
              <span class="progress">
                <i :style="{transform:`scaleX(${xpProgress})`}" />
              </span>
              <small class="numeric">{{ state.xp ? `${state.xp[0]}/${state.xp[1]}` : '—' }}</small>
            </div>
          </div>
          <div class="state-metric" :class="{danger:state.hp!==null && state.hp<30}">
            <label>血量</label>
            <MetricValue :value="state.hp" />
            <UBadge v-if="state.streak" :tone="state.streak.kind==='win' ? 'green' : 'orange'">{{ state.streak.kind==='win' ? '连胜' : '连败' }} {{ state.streak.n }}</UBadge>
            <span v-else class="metric-note">当前生命值</span>
          </div>
        </div>
      </section>
      <div class="headline-banner">
        <Lightbulb :size="21" />
        <div class="headline-text">
          <div class="headline-summary">
          <Transition name="headline">
            <p :key="headline" class="strategy-headline">{{ headline }}</p>
          </Transition>
          </div>
          <p v-if="buyHeadline" class="purchase-headline">{{ buyHeadline }}</p>
        </div>
      </div>
      <LiveCardGrid class="live-grid">
        <template #targets>
        <UCard title="阵容与概率" class="targets-card" :padded="false">
          <template #title-leading><LiveCardTools id="targets" /></template>
          <template #actions>
            <USelect v-model="population" :options="populationOptions" label="预览人口" />
          </template>
          <div class="preview-caption" aria-live="polite">
            <span>{{ manual ? `预览${population}人口 · 概率按${population}级规则估算，不影响实际推荐` : '自动跟随当前容量 · 概率沿用实际建议' }}</span>
            <span v-if="displayBoard" class="board-source">{{ displayBoard.source }} · {{ displayBoard.population }} 人口</span>
            <span v-if="previewLoading">{{ displayBoard ? '正在刷新预览…' : '正在加载预览…' }}</span>
            <span v-if="previewError" role="alert">预览加载失败：{{ previewError }}</span>
          </div>
          <TraitsView :view="advice?.traits_view" />
          <div class="board-odds-layout">
            <div class="board-pane">
              <BoardMini v-if="displayBoard" :plan="displayBoard" :selected="selectedName" :targets="displayTargets" @select="selectedHero=$event" />
              <p v-else class="no-board muted">暂无站位数据</p>
              <p v-if="displayBoard" class="board-legend"><span class="carry-key">C 主C</span><span class="tank-key">T 主T</span><span>灰色＝还缺</span><span class="craft-key">绿框＝有合成材料，分配看实况安排</span><span v-if="displayBoard.next_hint" class="next-hint">{{ displayBoard.next_hint }}</span></p>
            </div>
          <div class="odds-pane">
          <UTable v-model:selected="selectedHero" :native-titles="false" class="target-table" :columns="targetColumns" :rows="targets" row-key="name" empty-title="暂无目标卡">
            <template #cell-name="{row}">
              <UTooltip :text="`${row.name} · ${row.cost ?? '未知'}费 · 卡池剩余 ${row.remaining ?? '未知'} 张`">
                <button type="button" class="odds-hero" :style="{color:row.cost ? `var(--cost-${row.cost})` : costColor(row.name)}">{{ row.name }}</button>
              </UTooltip>
            </template>
            <template #cell-have="{row}"><span>{{ row.have }}/{{ row.target_star }}★</span></template>
            <template #cell-need="{value}">
              <span v-if="Number(value)>0" class="need numeric">{{ value }}</span>
              <Check v-else class="done" :size="14" />
            </template>
            <template #cell-per_roll="{value}">
              <div class="fit-cell numeric">
                <span>{{ percent(value) }}</span>
                <span class="progress">
                  <i :style="{transform:`scaleX(${fraction(value)})`}" />
                </span>
              </div>
            </template>
            <template #cell-gold_for_50="{value,row}">
              <Check v-if="Number(row.need)===0" class="done" :size="14" />
              <span v-else>{{ typeof value==='number' && Number.isFinite(value) ? `${Math.ceil(value)} 金` : '—' }}</span>
            </template>
          </UTable>
          </div>
          </div>
        </UCard>
        </template>
        <template #comps>
          <UCard title="推荐阵容" class="comps-card" :padded="false">
            <template #title-leading><LiveCardTools id="comps" /></template>
            <template #actions>
              <template v-if="store.status?.lock?.source==='game'">
                <UBadge tone="blue">🔒 本局锁定：{{ store.status.lock.name }}</UBadge>
                <UButton variant="plain" size="sm" :loading="locking" @click="lockForGame('')">解除</UButton>
              </template>
              <UBadge v-else-if="store.status?.lock?.source==='default'" tone="blue">默认锁定：{{ store.status.lock.name }}</UBadge>
              <UTooltip v-if="watchIds.length" :text="store.status?.lock ? '已锁定，关注暂停；解除锁定后继续比较' : '比较这些阵容的来牌，选定后可锁定其中一个'">
                <UBadge class="watch-badge" :tone="store.status?.lock ? 'gray' : 'purple'">👀 关注 {{ watchIds.length }} 个<span v-if="store.status?.lock"> · 已锁定，关注暂停</span></UBadge>
              </UTooltip>
              <UBadge v-else-if="advice?.comp_status" :tone="compTones[advice.comp_status]">{{ advice.comp_status }}</UBadge>
            </template>
            <div class="watch-toolbar">
              <UCombobox model-value="" :options="watchOptions" placeholder="添加关注" label="添加关注阵容" :disabled="watching || watchIds.length>=4" @update:model-value="toggleWatch" />
              <small>最多关注 4 个阵容，来牌次数从本局商店累计</small>
              <UButton v-for="entry in store.status?.watch" :key="entry.id" size="sm" variant="plain" :disabled="watching"
                       :aria-label="`移除关注${entry.name}`" @click="toggleWatch(entry.id)">{{ entry.name }}{{ entry.available===false ? '（来源暂不可用）' : '' }} ×</UButton>
            </div>
            <UTable :columns="compColumns" :rows="comps" row-key="id" empty-title="暂无阵容建议">
              <template #cell-watch="{row}">
                <input :key="comparisonSelection(String(row.id))?.id ?? 'unselected'" type="checkbox" class="watch-check" :checked="isCompared(String(row.id))" :aria-label="`关注${row.name}`"
                       :title="comparisonSelection(String(row.id)) ? `取消来源：${comparisonSelection(String(row.id))!.name}` : `关注${row.name}`"
                       :disabled="watching || (!isCompared(String(row.id)) && watchIds.length>=4)" @click.stop @change="toggleCompared(String(row.id))" />
              </template>
              <template #cell-seen="{row}">
                <div class="seen-stats"><span>{{ row.seen }} 次 <small>主C {{ row.seen_carry }} 次</small></span><small>已拿 {{ row.owned_copies }} 张</small></div>
              </template>
              <template #cell-name="{row}">
                <span class="comp-name" :title="String(row.name)">
                  <Target v-if="row.current" :size="12" />
                  <UBadge v-if="row.tier" :tone="tierTone(String(row.tier))">{{ row.tier }}</UBadge>
                  <span>{{ row.name }}</span>
                </span>
                <small v-if="comparisonSelection(String(row.id)) && comparisonSelection(String(row.id))!.id!==row.id" class="muted">选择来源：{{ comparisonSelection(String(row.id))!.name }}</small>
              </template>
              <template #cell-fit="{value}">
                <div class="fit-cell numeric">
                  <span>{{ percent(value) }}</span>
                  <span class="progress">
                    <i :style="{transform:`scaleX(${fraction(value)})`}" />
                  </span>
                </div>
              </template>
              <template #cell-why="{value}">
                <span class="wrapping selectable comp-reasons">{{ (value as string[]).join('；') || '—' }}</span>
              </template>
              <template #cell-missing="{row}">
                <span class="missing-heroes">
                  <span v-for="name in missing(row)" :key="name" class="missing-chip" :style="{color:costColor(name)}">{{ name }}</span>
                  <span v-if="!missing(row).length">—</span>
                </span>
              </template>
              <template #cell-lock="{row}">
                <UButton size="sm" variant="plain" :icon="isLocked(row.id) ? LockOpen : Lock"
                         :disabled="locking" :loading="locking && lockingId===row.id"
                         :aria-label="`${isLocked(row.id) ? '解除本局锁定' : '本局锁定'}：${row.name}`"
                         @click.stop="lockForGame(isLocked(row.id) ? '' : String(row.id))">{{ isLocked(row.id) ? '解除' : '锁定' }}</UButton>
              </template>
            </UTable>
            <p v-if="advice?.switch_reason && advice.stage===state.stage" class="switch-reason muted">刚换阵：{{ advice.switch_reason }}</p>
          </UCard>
        </template>
        <template #opponents>
          <OpponentsCard><template #title-leading><LiveCardTools id="opponents" /></template></OpponentsCard>
        </template>
        <template #suggestions>
          <UCard title="本回合建议" class="suggestions-card">
            <template #title-leading><LiveCardTools id="suggestions" /></template>
            <div v-if="advice?.econ" class="econ-row">
              <Coins :size="17" />
              <span>{{ advice.econ.reason }}</span>
            </div>
            <div class="action-list">
              <div v-for="action in actions" :key="action.key" class="advice-action" :class="[action.tone,{best:action.best}]">
                <component :is="action.icon" :size="15" />
                <span class="selectable">{{ action.text }}</span>
                <UBadge v-if="action.best" tone="blue">规则推荐</UBadge>
                <UBadge v-if="action.aiBest" tone="purple">AI 推荐</UBadge>
                <small v-if="action.score!==null" class="numeric">{{ action.score.toFixed(3) }}</small>
              </div>
              <p v-if="!actions.length" class="muted">暂无操作建议</p>
            </div>
            <div v-if="aiText" class="ai-advice">
              <h3>
                <Bot :size="16" />
                AI 建议
              </h3>
              <p class="selectable">{{ aiText }}</p>
            </div>
          </UCard>
        </template>
        <template #units>
          <UCard title="对局数据" class="units-card">
            <template #title-leading><LiveCardTools id="units" /></template>
            <section class="unit-section">
              <h3>棋盘</h3>
              <UnitChips :units="state.board" />
            </section>
            <section class="unit-section">
              <h3>备战席</h3>
              <UnitChips :units="state.bench" bench />
            </section>
            <section class="unit-section">
              <h3>散件</h3>
              <div class="trait-chips">
                <UBadge v-for="(item,index) in (state.item_bench ?? []).filter(item => item.name)" :key="index" tone="gray">{{ item.name }}</UBadge>
                <span v-if="!(state.item_bench ?? []).some(item => item.name)" class="muted">暂无散件</span>
              </div>
            </section>
            <section class="unit-section">
              <h3>羁绊</h3>
              <div class="trait-chips">
                <UBadge v-for="trait in state.traits" :key="trait.name" :tone="trait.active ? 'blue' : 'gray'">{{ trait.name }} {{ trait.count ?? '—' }}</UBadge>
                <span v-if="!state.traits.length" class="muted">暂无羁绊</span>
              </div>
            </section>
            <section class="unit-section">
              <h3>已选海克斯</h3>
              <div class="taken-augments">
                <span v-if="!augmentsTaken.length" class="muted">尚未识别到已选海克斯</span>
                <span v-for="name in augmentsTaken" :key="name" class="taken-augment">
                  {{ name }}
                </span>
              </div>
            </section>
            <section v-if="advice?.notes.length" class="unit-section">
              <h3>提示</h3>
              <ul class="advice-notes selectable">
                <li v-for="note in advice.notes" :key="note">{{ note }}</li>
              </ul>
            </section>
          </UCard>
        </template>
      </LiveCardGrid>
        <UCard title="辅助比较范围" class="comparison-scope-card">
          <ComparisonScope :scope="store.status?.assistant_selection" />
          <RouterLink to="/comp-presets">管理预选阵容配置</RouterLink>
        </UCard>
        <UCard title="当前装备安排（按实况）" class="item-assignments-card" :padded="false">
          <ItemAssignments :plans="advice?.item_plan ?? []" />
        </UCard>

    </template>
  </div>
</template>
<style scoped>
.live-page {
  height: 100%;
  overflow: auto;
  min-width: 0;
  padding: var(--space-5) var(--space-6) var(--space-6);
  overflow-anchor: none;
}
.waiting {
  height: 100%;
}
.switch-reason {
  margin: var(--space-2) var(--space-4);
  font-size: var(--font-xs);
}
.state-card {
  border: 1px solid var(--separator);
  border-radius: var(--radius-lg);
  background: var(--surface);
  box-shadow: var(--shadow-card);
  padding: var(--space-3) var(--space-5) var(--space-4);
}
.state-caption {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: var(--secondary);
  font-size: var(--font-xs);
  height: 23px;
  margin-bottom: var(--space-2);
}
.state-metrics {
  display: grid;
  grid-template-columns: 1.2fr .8fr .9fr 1.1fr 1fr;
}
.state-metric {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-1);
  padding-left: var(--space-5);
  border-left: 1px solid var(--separator);
  min-width: 0;
}
.state-metric:first-child {
  padding-left: 0;
  border: 0;
}
.state-metric label {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.state-metric :deep(.metric-number) {
  font-size: var(--font-display);
  font-weight: 600;
  line-height: 1.25;
  letter-spacing: -.5px;
}
.state-metric small {
  font-size: var(--font-xs);
  color: var(--secondary);
  margin-left: var(--space-1);
}
.metric-note {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.urgent :deep(.metric-number) {
  color: var(--orange-text);
}
.danger :deep(.metric-number) {
  color: var(--red-text);
}
.xp-row {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  width: 100%;
  padding-right: var(--space-4);
}
.progress {
  display: block;
  height: 3px;
  background: var(--accent-soft);
  border-radius: var(--radius-pill);
  overflow: hidden;
  width: 100%;
}
.progress i {
  display: block;
  height: 100%;
  width: 100%;
  background: var(--accent);
  transform-origin: left center;
}
.xp-row .progress {
  flex: 1;
}
.xp-row small {
  white-space: nowrap;
}
.headline-banner {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-4) var(--space-1);
  color: var(--text);
}
.headline-banner>svg {
  color: var(--orange-text);
}
.headline-text {
  position: relative;
  flex: 1;
  min-width: 0;
}
.headline-summary { display: grid; }
.strategy-headline { grid-area: 1/1; }
.headline-text p {
  font-size: var(--font-xl);
  font-weight: 600;
  letter-spacing: -.3px;
}
.headline-enter-active,.headline-leave-active {
  transition: opacity var(--duration-fast) var(--ease);
}
.headline-enter-from,.headline-leave-to {
  opacity: 0;
}
.live-grid :deep(.u-card header) {
  padding: var(--space-3) var(--space-4);
}
.live-grid :deep(.card-body) {
  padding: var(--space-4);
}
.live-grid :deep(th),.live-grid :deep(td) {
  padding: var(--space-2);
  font-size: var(--font-xs);
}
.comps-card :deep(td) {
  vertical-align: top;
}
.watch-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-3); padding: var(--space-2) var(--space-4); }
.watch-toolbar > .u-button { max-width: 100%; white-space: normal; overflow-wrap: anywhere; text-align: left; }
.watch-toolbar small,.seen-stats small { color: var(--secondary); font-size: var(--font-xs); }
.watch-check { accent-color: var(--accent); width: 14px; height: 14px; cursor: pointer; }
.seen-stats { display: flex; flex-direction: column; gap: 2px; }
.comp-name {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  min-width: 0;
}
.comp-name>span:last-child {
  overflow: hidden;
  text-overflow: ellipsis;
}
.comp-name>svg {
  color: var(--blue-text);
}
.comp-name :deep(.u-badge) {
  padding: 0 var(--space-1);
}
.wrapping {
  white-space: normal;
  line-height: 1.7;
}
.missing-heroes {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-2);
  white-space: normal;
}
.missing-chip {
  padding: var(--space-half) var(--space-1);
  background: var(--hover);
  border-radius: var(--radius-pill);
  white-space: nowrap;
}
.fit-cell {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  font-size: var(--font-xs);
  text-align: right;
}
.source-notice { padding: 12px; color: var(--orange-text); }
.preview-caption { display: flex; flex-wrap: wrap; gap: var(--space-2); padding: var(--space-2) var(--space-4); color: var(--secondary); font-size: var(--font-xs); }
.item-assignments-card,.comparison-scope-card { grid-column: 1 / -1; }
.targets-card {
  grid-column: 1 / -1;
}
.board-odds-layout {
  display: grid;
  grid-template-columns: minmax(0,48%) minmax(300px,1fr);
  gap: var(--space-3);
  padding: var(--space-3);
  align-items: stretch;
}
.board-pane {
  min-width: 0;
}
.odds-pane {
  position: relative;
  min-width: 0;
}
.target-table {
  position: absolute;
  inset: 0;
}
.target-table :deep(tr.selected td) { background: rgb(255 255 255 / 8%); color: var(--text); }
.target-table :deep(tr.selected td:first-child) { box-shadow: inset 3px 0 #fff; }
.target-table :deep(tr:focus-visible td) { box-shadow: inset 0 0 0 1px rgb(255 255 255 / 40%); }
.target-table :deep(tr.selected:focus-visible td:first-child) { box-shadow: inset 3px 0 #fff; }
.board-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 2px var(--space-2);
  margin-top: var(--space-1);
  color: var(--secondary);
  font-size: var(--font-xs);
}
.carry-key { color: var(--yellow-text); }
.tank-key { color: #8daac0; }
.craft-key { color: var(--green-text); }
.odds-hero {
  padding: 0;
  background: transparent;
  font-size: inherit;
  white-space: nowrap;
}
.board-source {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.next-hint {
  line-height: 1.5;
}
.no-board {
  padding: var(--space-5) 0;
}
@container live-card (max-width: 700px) {
  .board-odds-layout { grid-template-columns: minmax(0,1fr); }
  .odds-pane { height: 240px; }
}
.target-table :deep(table) {
  min-width: 0;
}
@media (max-width: 600px) {
  .target-table :deep(table) { min-width: 380px; }
}
.target-table :deep(th),.target-table :deep(td) {
  padding: 4px var(--space-2);
  height: 30px;
}
.target-table :deep(td>svg) {
  vertical-align: middle;
}
.need {
  color: var(--red-text);
}
.done {
  color: var(--green-text);
}
.econ-row {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  font-size: var(--font-sm);
  margin-bottom: var(--space-3);
}
.taken-augments { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); font-size: var(--font-sm); }
.taken-augment { display: inline-flex; align-items: center; max-width: 100%; overflow-wrap: anywhere; background: var(--accent-soft); border-radius: var(--radius-sm); padding: var(--space-half) var(--space-2); }
.econ-row>svg {
  color: var(--orange-text);
  margin-top: 2px;
}
.action-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}
.advice-action {
  display: flex;
  align-items: flex-start;
  gap: var(--space-2);
  font-size: var(--font-sm);
  padding: var(--space-1) 0;
}
.advice-action>svg {
  color: var(--secondary);
  margin-top: 2px;
}
.advice-action.buy>svg {
  color: var(--green-text);
}
.advice-action.sell>svg {
  color: var(--orange-text);
}
.advice-action.best {
  background: var(--accent-soft);
  border-radius: var(--radius-sm);
  padding: var(--space-2);
}
.advice-action small {
  margin-left: auto;
  color: var(--secondary);
}
.advice-action.augment :deep(.u-badge) { flex: 0 0 auto; }
.advice-action>span {
  flex: 1;
}
.muted {
  color: var(--secondary);
  font-size: var(--font-sm);
}
.ai-advice {
  background: var(--accent-soft);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  margin-top: var(--space-4);
}
.ai-advice h3 {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--font-sm);
  color: var(--blue-text);
  font-weight: 600;
}
.ai-advice p {
  white-space: pre-wrap;
  font-size: var(--font-sm);
  line-height: 1.8;
  margin-top: var(--space-2);
}
.unit-section+.unit-section {
  margin-top: var(--space-4);
}
.unit-section h3 {
  font-size: var(--font-xs);
  color: var(--secondary);
  font-weight: 500;
  margin-bottom: var(--space-2);
}
.trait-chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}
.advice-notes {
  padding-left: var(--space-4);
  margin: 0;
  font-size: var(--font-xs);
  color: var(--secondary);
  line-height: 1.8;
}
@media (max-width:1199px) {
  .state-metric {
    padding-left: var(--space-3);
  }
  .live-page {
    padding: var(--space-4);
  }
}
@media (max-width: 600px) {
  .state-metrics { grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px 0; }
  .state-metric { padding-left: 8px; }
  .state-metric:nth-child(4) { border-left: 0; padding-left: 0; }
  .state-card { padding-inline: 12px; }
  .live-page { padding: 12px; }
  .state-metric :deep(.metric-number) { font-size: 24px; }
  .headline-text p { font-size: var(--font-md); }
}
</style>
