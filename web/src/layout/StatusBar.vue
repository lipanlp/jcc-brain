<script setup lang="ts">
import { ref } from 'vue';
import UPopover from '../components/ui/UPopover.vue';
import USwitch from '../components/ui/USwitch.vue';
import { setDryRun, changingDryRun, startingMode } from '../composables/modes';
import { useAppStore } from '../stores/app';
import USpinner from '../components/ui/USpinner.vue';
const store = useAppStore();
const operationOpen = ref(false);
</script>
<template>
  <footer class="status-bar" aria-label="应用状态">
    <span class="mode-status">
      <i v-if="store.status?.mode" class="running-dot" />
      {{ !store.connected ? (store.startupPhase==='failed' ? '连接失败' : '准备中…') : store.status?.mode ? `运行中：${store.status.mode.title}` : '空闲' }}
    </span>
    <span v-if="store.status?.syncing" class="sync-status">
      <USpinner :size="11" />
      {{ store.syncMessage || '正在同步…' }}
    </span>
    <span class="season-status">
      赛季：
      {{ store.status?.season.label ?? '读取中…' }}
      <span v-if="store.status?.season.rules_note" class="rules-note">{{ store.status.season.rules_note }}</span>
    </span>
    <span class="ai-status">AI：{{ store.status?.llm.enabled ? `${store.status.llm.preset_title}（${store.status.llm.model}）` : '关' }}</span>
    <span class="operation-status">
      <UPopover v-model="operationOpen" label="操作模式" :width="310">
        <template #trigger="{toggle}">
          <button type="button" class="operation-button" :class="{real:store.status && !store.status.dry_run}" :disabled="!store.status" @click="toggle">{{ store.status && !store.status.dry_run ? '⚠️ 真实' : '演练' }}</button>
        </template>
        <div class="operation-setting">
          <label>演练模式（不真实点击）</label>
          <USwitch :model-value="store.status?.dry_run ?? true" :disabled="changingDryRun || !!startingMode" label="演练模式（不真实点击）" @update:model-value="setDryRun" />
        </div>
        <p class="operation-note">全局演练影响自动 D 牌、托管和辅助侦察。运行中切换需确认停止并应用，不会自动重启。</p>
      </UPopover>
    </span>
    <span class="vision-status" :title="store.state?.summary ?? store.status?.vision_backend">视觉：{{ store.state?.summary ?? store.status?.vision_backend ?? '连接中…' }}</span>
  </footer>
</template>
<style scoped>
.status-bar {
  height: var(--status-height);
  display: flex;
  align-items: center;
  padding: 0 var(--space-4);
  color: var(--secondary);
  font-size: var(--font-xs);
  background: var(--toolbar);
  border-top: 1px solid var(--separator);
  flex-shrink: 0;
  min-width: 0;
  gap: var(--space-3);
}
.status-bar > span {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
  min-width: 0;
  white-space: nowrap;
}
.status-bar > span + span {
  border-left: 1px solid var(--separator);
  padding-left: var(--space-3);
}
.status-bar .vision-status {
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
  flex: 1;
}
.rules-note {
  color: var(--orange-text);
}
.real {
  color: var(--red-text);
  font-weight: 600;
}
.running-dot {
  width: 6px;
  height: 6px;
  flex-shrink: 0;
  border-radius: var(--radius-pill);
  background: var(--green);
  animation: breathe var(--duration-breathe) var(--ease) infinite alternate;
}
.sync-status {
  max-width: 210px;
  overflow: hidden;
}
@keyframes breathe {
  from {
    opacity: .45;
  }
  to {
    opacity: 1;
  }
}
.operation-button {
  background: var(--transparent);
  padding: 0 var(--space-1);
  border-radius: var(--radius-sm);
  font-size: inherit;
  color: inherit;
}
.operation-button:hover {
  background: var(--hover);
}
.operation-button.real {
  color: var(--red-text);
  font-weight: 600;
}
.operation-setting {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  font-size: var(--font-sm);
}
.operation-note {
  color: var(--secondary);
  font-size: var(--font-xs);
  line-height: 1.8;
  margin-top: var(--space-3);
}
</style>
