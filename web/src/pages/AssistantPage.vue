<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { Play, X } from 'lucide-vue-next';
import { call } from '../bridge/api';
import { useAppStore } from '../stores/app';
import { featureDraft, featureLoading, featureSaving, featureMetadataReady, defaultLockInvalid, defaultLockOptions, compOptions, scoutRounds, economyLabels, loadFeatures, saveFeatures } from '../composables/features';
import { startMode, startingMode, stoppingMode } from '../composables/modes';
import { useScouting } from '../composables/scouting';
import type { Status } from '../types';
import UButton from '../components/ui/UButton.vue';
import USegmented from '../components/ui/USegmented.vue';
import USelect from '../components/ui/USelect.vue';
import UCombobox from '../components/ui/UCombobox.vue';
import USwitch from '../components/ui/USwitch.vue';
import UNumberField from '../components/ui/UNumberField.vue';
import './feature.css';
const store = useAppStore();
const { progress, busy, unavailable, scout } = useScouting();
void loadFeatures();
const direction = ref('auto');
watch([() => store.status?.lock?.id, () => store.status?.assistant_selection?.lock_selection_id, () => store.status?.watch.map(c => c.id).join(',')], () => {
    direction.value = store.status?.lock || store.status?.assistant_selection?.lock_selection_id ? 'lock' : store.status?.watch.length ? 'watch' : 'auto';
}, { immediate: true, flush: 'sync' });
const pick = ref('');
const directionBusy = ref(false);
const availableComps = computed(() => compOptions.value.filter(c => c.value));
async function chooseDirection(value: string) {
    if (directionBusy.value) return;
    directionBusy.value = true;
    try {
        if (value !== 'lock') store.status = await call<Status>('lock_comp_for_game', '');
        if (value === 'auto') store.status = await call<Status>('set_watch_comps', []);
        direction.value = value;
    }
    catch { /* bridge reports */ }
    finally { directionBusy.value = false; }
}
async function lock(id: string) {
    if (!id) return;
    try { store.status = await call<Status>('lock_comp_for_game', id); }
    catch { /* bridge reports */ }
}
async function watchComp(id: string, remove = false) {
    const ids = (store.status?.watch ?? []).map(c => c.id).filter(cid => cid !== id);
    if (!remove && id) ids.push(id);
    try { store.status = await call<Status>('set_watch_comps', ids); pick.value = ''; }
    catch { /* bridge reports */ }
}
const hudLabels = { show_names: '棋子名字', show_traits: '羁绊', show_odds_panel: '概率面板', show_positions: '站位指引' } as const;
</script>
<template>
  <section class="feature-page" aria-label="辅助大脑控制页">
    <div class="feature-card">
      <header class="feature-heading"><h2>辅助大脑</h2><p>实时给出买牌、运营与阵容建议。侦察需单独开启，不会自动买卡。</p></header>
      <template v-if="featureDraft">
        <div class="feature-row">
          <div><h3>阵容方向</h3><p>本局立即生效。关注会比较候选，锁定只围绕一个阵容给建议。</p></div>
          <div class="feature-controls">
            <USegmented :model-value="direction" :options="[{value:'auto',label:'自动推荐'},{value:'watch',label:'关注阵容'},{value:'lock',label:'锁定阵容'}]" label="本局阵容方向" :disabled="directionBusy" @update:model-value="chooseDirection" />
            <UCombobox v-if="direction==='lock'" :model-value="store.status?.lock?.id ?? store.status?.assistant_selection?.lock_selection_id ?? ''" :options="availableComps" label="本局锁定阵容" placeholder="选择本局锁定阵容…" @update:model-value="lock" />
            <template v-if="direction==='watch'">
              <div class="feature-inline"><UCombobox v-model="pick" :options="availableComps.filter(c => !store.status?.watch.some(w => w.id === c.value))" label="本局关注阵容" placeholder="选择候选阵容…" /><UButton :disabled="!pick || (store.status?.watch.length ?? 0)>=4" @click="watchComp(pick)">关注</UButton></div>
              <div v-if="store.status?.watch.length" class="feature-inline"><UButton v-for="comp in store.status.watch" :key="comp.id" size="sm" :icon="X" :aria-label="`移除关注${comp.name}`" @click="watchComp(comp.id,true)">{{ comp.name }}{{ comp.available===false ? '（来源暂不可用）' : '' }}</UButton></div>
              <p class="feature-hint">本局最多关注 4 个阵容；锁定时保留关注，解除锁定后恢复与预选配置的交集比较。</p>
            </template>
            <p v-if="direction==='lock' && store.status?.watch.length" class="feature-hint">已保留关注：{{ store.status.watch.map(c => c.name).join('、') }}。{{ store.status.lock ? '锁定优先，关注暂停；解除锁定或选择关注阵容后恢复比较。' : '选定锁定阵容后暂停关注，解除后恢复比较。' }}</p>
            <p class="feature-hint">选择关注阵容会解除本局锁定；选择自动推荐会解除锁定并清空本局关注，按预选配置比较。</p>
            <p v-if="store.status?.assistant_selection?.notice" class="feature-hint feature-warning" role="status">{{ store.status.assistant_selection.notice }}</p>
            <p class="feature-hint">当前本局：{{ store.status?.lock ? `锁定 ${store.status.lock.name}` : store.status?.watch.length ? `关注 ${store.status.watch.length} 个阵容` : '自动推荐' }}</p>
            <details class="default-direction"><summary>长期默认 · 下局开始使用</summary><USelect v-model="featureDraft.assistant.default_lock_comp_id" :options="defaultLockOptions" label="长期默认阵容" /><p class="feature-hint">保存仅更新新局默认，不覆盖当前本局选择。</p><p v-if="defaultLockInvalid" class="feature-hint feature-warning" role="alert">所选长期默认阵容当前不可用，请重新选择或改为自动推荐。未保存的选择仍保留。</p></details>
          </div>
        </div>
        <div class="feature-row">
          <div><h3>运营偏好</h3><p>保血更早补强，经济减少非必要支出；关键升级与危急保命始终优先。</p></div>
          <div class="feature-controls"><USegmented v-model="featureDraft.assistant.economy_preference" :options="[{value:'adaptive',label:'自适应'},{value:'health',label:'保血优先'},{value:'economy',label:'经济优先'}]" label="运营偏好" :disabled="featureSaving" /><p class="feature-hint">保存后下一次决策生效，与托管共享。已保存：{{ economyLabels[store.status?.economy_preference ?? 'adaptive'] }}。</p></div>
        </div>
        <div class="feature-row">
          <div><h3>对手侦察</h3><p>开启自动侦察后，会短暂切换对手视角。</p></div>
          <div class="feature-controls">
            <div class="scout-options" role="group" aria-label="侦察方式"><button :aria-pressed="!featureDraft.scout.enabled" :disabled="featureSaving" @click="featureDraft.scout.enabled=false">手动</button><button :aria-pressed="featureDraft.scout.enabled && featureDraft.scout.schedule==='key_rounds'" :disabled="featureSaving" @click="featureDraft.scout.enabled=true;featureDraft.scout.schedule='key_rounds'">关键回合 · 推荐</button><button :aria-pressed="featureDraft.scout.enabled && featureDraft.scout.schedule==='every_round'" :disabled="featureSaving" @click="featureDraft.scout.enabled=true;featureDraft.scout.schedule='every_round'">每回合</button></div>
            <p class="feature-hint">{{ featureDraft.scout.enabled ? featureDraft.scout.schedule==='every_round' ? '每个有效准备回合按安全条件调度；' : `在 ${scoutRounds.join('、')} 准备阶段调度；` : '手动触发，不自动切换视角；' }}保存后下次调度生效。开始前由 HUD 提示 3 秒，可取消；剩余时间不足时跳过。</p>
            <div class="feature-notice"><div class="feature-inline"><span>{{ busy ? progress?.notice ? `${progress.notice.seconds} 秒后自动侦察` : '侦察任务处理中' : unavailable || '准备阶段，可立即侦察' }}</span><UButton v-if="busy" size="sm" @click="scout(true)">{{ progress?.notice ? '取消本次' : '停止侦察' }}</UButton><UButton v-else size="sm" :disabled="!!unavailable" @click="scout()">立即侦察</UButton></div><p v-if="progress?.reason">{{ progress.reason }}</p><p>使用顶部全局演练设置；自动侦察需要 HUD 可见并完成预告确认。</p></div>
          </div>
        </div>
        <details class="feature-details"><summary>显示偏好</summary><div class="feature-details-grid"><label>HUD 字号<UNumberField v-model="featureDraft.hud.font_size" :min="8" :max="32" label="HUD 字号" /></label><label>不透明度<UNumberField v-model="featureDraft.hud.opacity" :min=".1" :max="1" :step=".05" label="HUD 不透明度" /></label><label v-for="(label,key) in hudLabels" :key="key">{{ label }}<USwitch v-model="featureDraft.hud[key]" :label="label" /></label></div><p class="feature-hint">保存后立即更新 HUD。</p></details>
        <p v-if="!featureMetadataReady" class="feature-hint feature-warning">{{ featureLoading ? '阵容资料正在更新…' : '阵容资料刷新失败，请重试。' }}<UButton v-if="!featureLoading" size="sm" @click="loadFeatures(true)">重新读取资料</UButton></p>
        <footer class="feature-footer"><p>保存不会启动；启动使用已保存设置。<br />本局方向已即时应用。</p><UButton :loading="featureSaving" :disabled="defaultLockInvalid || !featureMetadataReady" @click="saveFeatures(['assistant','scout','hud'])">保存设置</UButton><UButton variant="primary" :icon="Play" :disabled="!store.status || !!startingMode || stoppingMode || store.status.stopping || store.status.mode?.key==='assistant'" @click="startMode('assistant')">{{ store.status?.mode?.key==='assistant' ? '辅助大脑运行中' : '启动辅助大脑' }}</UButton></footer>
      </template>
      <div v-else class="feature-loading">{{ featureLoading ? '正在读取设置…' : '设置读取失败' }}<UButton v-if="!featureLoading" @click="loadFeatures()">重试</UButton></div>
    </div>
  </section>
</template>
<style scoped>
.default-direction { color: var(--secondary); font-size: var(--font-sm); }
.default-direction summary { cursor: pointer; margin-bottom: var(--space-2); }
.default-direction .u-select { width: 100%; }
.scout-options { display: flex; border: 1px solid var(--separator); border-radius: var(--radius-sm); overflow: hidden; }
.scout-options button { flex: 1; min-height: 36px; background: var(--hover); font-size: var(--font-sm); }
.scout-options button + button { border-left: 1px solid var(--separator); }
.scout-options button[aria-pressed=true] { background: var(--accent-soft); color: var(--blue-text); }
.scout-options button:disabled { opacity: .4; }
@media (max-width: 780px) {
  .feature-page, .feature-card { padding: var(--space-3); }
  .feature-row { grid-template-columns: minmax(0, 1fr); }
  .feature-controls :deep(.u-segmented button) { padding: 0 var(--space-2); }
}
</style>
