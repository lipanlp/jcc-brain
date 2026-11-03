<script setup lang="ts">
import { computed, ref } from 'vue';
import { Play, Plus, X, RefreshCw } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { rollDraft, rollLoading, rollMetadataReady, unavailableRollTargets, invalidRollSelection, featureSaving, loadRollDraft, importRollTargets, rollOptions, saveRollDefaults } from '../composables/features';
import { startMode, startingMode, stoppingMode } from '../composables/modes';
import UButton from '../components/ui/UButton.vue';
import USegmented from '../components/ui/USegmented.vue';
import UCombobox from '../components/ui/UCombobox.vue';
import UNumberField from '../components/ui/UNumberField.vue';
import './feature.css';
const store = useAppStore();
void loadRollDraft();
const addHero = ref('');
const custom = ref(false);
const presets = [0,20,30,50];
const options = computed(() => (rollDraft.value?.all_heroes ?? []).map(hero => {
    const target = rollDraft.value?.heroes.find(h => h.name === hero.name);
    const state = target ? (target.checked ? ' · 已选择' : ' · 已添加，未勾选') : '';
    return { value: hero.name, label: hero.name, hint: `${hero.cost} 费${state}`, color: `var(--cost-${hero.cost})` };
}));
const candidate = computed(() => rollDraft.value?.all_heroes.find(h => h.name === addHero.value));
const existingTarget = computed(() => rollDraft.value?.heroes.find(h => h.name === candidate.value?.name));
const addLabel = computed(() => existingTarget.value ? (existingTarget.value.checked ? '已选择' : '选中目标') : '添加英雄');
function add() {
    const hero = candidate.value;
    if (!hero || !rollDraft.value) return;
    if (existingTarget.value) existingTarget.value.checked = true;
    else rollDraft.value.heroes.push({...hero,star:2,checked:true});
    addHero.value = '';
}
const selected = computed(() => rollDraft.value?.heroes.filter(h => h.checked).length ?? 0);
</script>
<template>
  <section class="feature-page" aria-label="自动 D 牌控制页"><div class="feature-card">
    <header class="feature-heading"><h2>自动 D 牌</h2><p>选好英雄目标和预算，完成一次 D 牌任务。</p></header>
    <template v-if="rollDraft">
      <div class="feature-row"><div><h3>目标英雄</h3><p>目标与星级仅用于本次任务。导航和推荐更新不会替换草稿。</p></div><div class="feature-controls">
        <div class="feature-inline"><span class="feature-hint">{{ rollDraft.comp_name ? `导入来源：${rollDraft.comp_name}` : '自选目标' }}</span><UButton variant="plain" :icon="RefreshCw" size="sm" :disabled="rollLoading" @click="importRollTargets">从当前阵容导入</UButton></div>
        <div class="target-list"><div v-for="hero in rollDraft.heroes" :key="hero.name" class="target-row"><label><input v-model="hero.checked" type="checkbox" :aria-label="`选择${hero.name}`" /><span class="target-name" :style="{color:`var(--cost-${hero.cost})`}">{{ hero.name }}<small v-if="unavailableRollTargets.includes(hero.name)" class="feature-warning">当前赛季不可用，请删除或重新导入</small></span></label><USegmented :model-value="String(hero.star)" :options="[1,2,3].map(n=>({value:String(n),label:`${n} 星`}))" :label="`${hero.name}目标星级`" @update:model-value="hero.star=Number($event)" /><UButton variant="plain" size="sm" :icon="X" :aria-label="`删除${hero.name}`" @click="rollDraft.heroes=rollDraft.heroes.filter(h=>h.name!==hero.name)" /></div><p v-if="!rollDraft.heroes.length" class="feature-hint">尚无目标，请导入阵容或手动添加英雄。</p></div>
        <div class="feature-inline"><UCombobox v-model="addHero" :options="options" label="添加目标英雄" placeholder="搜索英雄…" /><UButton :icon="Plus" :disabled="!candidate || !!existingTarget?.checked" @click="add">{{ addLabel }}</UButton></div>
      </div></div>
      <p v-if="!rollMetadataReady" class="feature-hint feature-warning">{{ rollLoading ? '英雄资料正在更新…' : '英雄资料刷新失败，请重试。' }}<UButton v-if="!rollLoading" size="sm" @click="loadRollDraft(true)">重新读取资料</UButton></p>
      <div class="feature-row"><div><h3>最低保留金币</h3><p>买卡累计支出与每次刷新共同遵守这条保留线。</p></div><div class="feature-controls"><div class="gold-options" role="group" aria-label="最低保留金币快选"><button v-for="value in presets" :key="value" :aria-pressed="!custom && rollDraft.keep===value" @click="custom=false;rollDraft.keep=value">{{ value }}</button><button :aria-pressed="custom || !presets.includes(rollDraft.keep)" @click="custom=true">自定义</button></div><label v-if="custom || !presets.includes(rollDraft.keep)" class="feature-inline">保留金币<UNumberField v-model="rollDraft.keep" :min="0" :max="100" label="最低保留金币" /></label></div></div>
      <div class="feature-row"><div><h3>最多刷新次数</h3><p>达到上限后处理最后一次刷新的商店，再结束任务。</p></div><div class="feature-inline"><UNumberField v-model="rollDraft.max_rolls" :min="1" :max="200" label="最多刷新次数" /><span>次</span></div></div>
      <div class="feature-notice">目标达成、买卡或刷新预算不足或出现既有异常停止条件时结束。{{ store.status?.mode?.key==='autoroll' ? '运行中修改仅用于下次任务。' : '启动时提交本次草稿。' }}</div>
      <div v-if="store.status?.autoroll_task" class="feature-notice" aria-label="正在执行的D牌快照"><strong>正在执行的任务</strong><p>{{ store.status.autoroll_task.targets.map(t=>`${t.name}${t.star}星`).join('、') }} · 保留 {{ store.status.autoroll_task.keep }} 金币 · 最多 {{ store.status.autoroll_task.max_rolls }} 次</p><p>已刷新 {{ store.status.autoroll_task.rolls }} 次 · 已买 {{ store.status.autoroll_task.bought }} 张；上方编辑为下次任务草稿。</p></div>
      <footer class="feature-footer"><p>下次草稿 {{ selected }} 个目标 · 保留 {{ rollDraft.keep }} 金币 · 最多 {{ rollDraft.max_rolls }} 次<br />保存只保留预算与刷新默认，不保存英雄目标，也不启动。</p><UButton :loading="featureSaving" @click="saveRollDefaults">保存设置</UButton><UButton variant="primary" :icon="Play" :disabled="!selected || invalidRollSelection || !rollMetadataReady || !store.status || !!startingMode || stoppingMode || store.status.stopping || store.status.mode?.key==='autoroll'" @click="startMode('autoroll',rollOptions())">{{ store.status?.mode?.key==='autoroll' ? 'D 牌运行中' : '开始 D 牌' }}</UButton></footer>
    </template>
    <div v-else class="feature-loading">{{ rollLoading ? '正在读取目标…' : '目标读取失败' }}<UButton v-if="!rollLoading" @click="loadRollDraft()">重试</UButton></div>
  </div></section>
</template>
<style scoped>
.target-list { max-height: 235px; overflow: auto; }
.target-row { display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) 0; border-bottom: 1px solid var(--separator); }
.target-row label { display:flex;align-items:center;gap:var(--space-2);flex:1;min-width:70px;font-size:var(--font-md); }
.target-name { min-width:0; overflow-wrap:anywhere; }
.target-name small { display:block; font-size:var(--font-xs); line-height:1.5; }
.target-row input { accent-color: var(--accent); }
.gold-options { display:flex;border:1px solid var(--separator);border-radius:var(--radius-sm);overflow:hidden; }
.gold-options button { flex:1;min-height:36px;background:var(--hover);font-size:var(--font-sm); }
.gold-options button + button { border-left:1px solid var(--separator); }
.gold-options button[aria-pressed=true] { background:var(--accent-soft);color:var(--blue-text); }
.feature-footer { margin-top: var(--space-5); }
</style>
