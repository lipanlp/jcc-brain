<script setup lang="ts">
import { Play, ArrowRight } from 'lucide-vue-next';
import { useRouter } from 'vue-router';
import { useAppStore } from '../stores/app';
import { featureDraft, featureLoading, featureSaving, economyLabels, loadFeatures, saveFeatures } from '../composables/features';
import { startMode, setDryRun, changingDryRun, startingMode, stoppingMode, resumeAutopilot, resumingAutopilot } from '../composables/modes';
import UButton from '../components/ui/UButton.vue';
import UBadge from '../components/ui/UBadge.vue';
import USegmented from '../components/ui/USegmented.vue';
import './feature.css';
const store = useAppStore();
const router = useRouter();
void loadFeatures();
</script>
<template>
  <section class="feature-page" aria-label="全自动托管控制页"><div class="feature-card">
    <header class="feature-heading"><h2>全自动托管 <UBadge tone="orange">实验</UBadge></h2><p>按共享策略执行对局操作，随时可以停止。</p></header>
    <template v-if="featureDraft">
      <div class="feature-row"><div><h3>当前策略</h3><p>与辅助大脑共用本局阵容方向和运营规则。</p></div><div class="feature-controls"><div class="feature-inline"><strong>{{ store.status?.lock ? `锁定 ${store.status.lock.name}` : store.status?.watch.length ? `关注 ${store.status.watch.length} 个阵容` : '自动推荐' }} · {{ economyLabels[store.status?.economy_preference ?? 'adaptive'] }}</strong><UButton variant="plain" :icon="ArrowRight" @click="router.push('/assistant')">调整策略</UButton></div><p class="feature-hint">此处显示已保存的运营偏好；保存后下一次决策生效，未保存的草稿不会影响托管。</p></div></div>
      <div class="feature-row"><div><h3>执行方式</h3><p>演练只记录操作计划，不点击游戏。</p></div><div class="feature-controls"><USegmented :model-value="store.status?.dry_run ? 'dry' : 'real'" :options="[{value:'dry',label:'演练'},{value:'real',label:'真实操作'}]" label="全局执行方式" :disabled="!store.status || changingDryRun" @update:model-value="setDryRun($event==='dry')" /><p class="feature-hint">全局即时设置，同时影响自动 D 牌和辅助侦察。运行中切换需确认停止并应用，不会自动重启。</p></div></div>
      <div class="feature-row"><div><h3>海克斯选择</h3><p>保存为下次启动默认，运行中的方式不变。</p></div><div class="feature-controls"><USegmented v-model="featureDraft.autopilot.augment_selection" :options="[{value:'manual',label:'手动处理'},{value:'auto',label:'自动选择'}]" label="海克斯选择" /><p class="feature-hint">手动处理会暂停全部自动操作。选完后等待有效己方画面稳定，再点击“确认继续”；自动选择每次选项只选择一次。</p><p v-if="store.status?.autopilot" class="feature-hint">本次运行：{{ store.status.autopilot.augment_selection==='manual' ? '手动处理' : '自动选择' }} · 保存仅下次启动生效。</p></div></div>
      <div v-if="store.status?.autopilot?.paused" class="feature-notice" role="status" aria-label="海克斯手动暂停"><strong>托管已暂停 · 等待你处理海克斯</strong><p>{{ store.status.autopilot.reason }}</p><p>{{ store.status.autopilot.can_resume ? '画面已稳定，可确认继续。' : '尚不可继续：请完成选择并返回自己的可操作画面。' }}</p><UButton :disabled="!store.status.autopilot.can_resume || store.status.stopping" :loading="resumingAutopilot" @click="resumeAutopilot">确认继续</UButton></div>
      <div class="feature-row"><div><h3>托管范围</h3><p>当前实现的操作能力。</p></div><div class="feature-notice"><p>已覆盖：买卡、备战席腾位、升级、D 牌、补满人口。</p><p>尚未覆盖：装备合成、完整站位调整。</p></div></div>
      <div class="feature-notice">随时点击顶部“停止”，或按 ⌘. 中止；真实操作保留鼠标甩到左上角急停。</div>
      <footer class="feature-footer"><p>{{ store.status?.mode?.key==='autopilot' ? '托管运行中' : '未启动' }} · 保存不会启动功能<br />执行方式通过上方全局设置立即应用。</p><UButton :loading="featureSaving" @click="saveFeatures(['autopilot'])">保存设置</UButton><UButton variant="primary" :icon="Play" :disabled="!store.status || !!startingMode || stoppingMode || store.status.stopping || store.status.mode?.key==='autopilot'" @click="startMode('autopilot')">{{ store.status?.mode?.key==='autopilot' ? '托管运行中' : store.status?.dry_run ? '开始演练' : '开始真实托管' }}</UButton></footer>
    </template>
    <div v-else class="feature-loading">{{ featureLoading ? '正在读取设置…' : '设置读取失败' }}<UButton v-if="!featureLoading" @click="loadFeatures()">重试</UButton></div>
  </div></section>
</template>
