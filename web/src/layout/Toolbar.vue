<script setup lang="ts">
import { stopMode, stoppingMode, setDryRun, changingDryRun } from '../composables/modes';
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { Square, PanelTop, ChartNoAxesColumn } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { toggleHud, hudBusy, panelBusy } from '../composables/hud';
import { vWindowDrag } from '../composables/windowDrag';
import UButton from '../components/ui/UButton.vue';
import USwitch from '../components/ui/USwitch.vue';
import SeasonControl from '../components/SeasonControl.vue';
import AIControl from '../components/AIControl.vue';
import ToolsMenu from '../components/ToolsMenu.vue';
import UTooltip from '../components/ui/UTooltip.vue';
const store = useAppStore();
const route = useRoute();
const title = computed(() => String(route.meta.title ?? '实时决策'));
</script>
<template>
  <header v-window-drag="'main'" class="toolbar">
    <h1 class="toolbar-title">{{ title }}</h1>
    <div class="toolbar-space" aria-hidden="true" />
    <div class="toolbar-actions">
      <span class="running-status" role="status">{{ store.status?.stopping ? '停止中' : store.status?.autopilot?.paused ? '托管已暂停' : store.status?.mode?.title ?? '未运行' }} · {{ store.status?.dry_run === false ? '真实操作' : '演练' }}</span>
      <USwitch :model-value="store.status?.dry_run ?? true" label="全局演练" :disabled="changingDryRun || !store.status" @update:model-value="setDryRun" />
      <UButton variant="destructive" size="sm" :icon="Square" :disabled="!store.status?.mode" :loading="stoppingMode" @click="stopMode">停止</UButton>
      <span class="toolbar-separator" />
      <UTooltip text="HUD 浮层 ⌘H">
        <UButton class="icon-toggle" variant="plain" size="sm" :icon="PanelTop" aria-label="HUD 浮层" :aria-pressed="store.status?.hud.overlay ?? false" :disabled="hudBusy || !store.status" @click="toggleHud()" />
      </UTooltip>
      <UTooltip text="概率面板 ⌘J">
        <UButton class="icon-toggle" variant="plain" size="sm" :icon="ChartNoAxesColumn" aria-label="概率面板" :aria-pressed="store.status?.hud.panel ?? false" :disabled="panelBusy || !store.status" @click="toggleHud(true)" />
      </UTooltip>
      <span class="toolbar-separator" />
      <SeasonControl />
      <AIControl />
      <ToolsMenu />
    </div>
  </header>
</template>
<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  height: var(--toolbar-height);
  flex-shrink: 0;
  padding: 0 var(--space-5);
  background: var(--toolbar);
  backdrop-filter: var(--material-blur);
  border-bottom: 1px solid var(--separator);
}
.running-status { color: var(--secondary); white-space: nowrap; font-size: var(--font-sm); }
.toolbar-title {
  align-self: stretch;
  display: flex;
  align-items: center;
  font-size: var(--font-lg);
  font-weight: 600;
  white-space: nowrap;
  letter-spacing: -.3px;
}
.toolbar-space {
  flex: 1;
  height: 100%;
  min-width: var(--space-3);
}
.toolbar-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}
.toolbar-separator {
  height: 17px;
  width: 1px;
  background: var(--separator);
  margin: 0 var(--space-half);
}
.icon-toggle[aria-pressed="true"] {
  color: var(--blue-text);
  background: var(--accent-soft);
}
@media (max-width: 1120px) {
  .toolbar {
    gap: var(--space-1);
    padding: 0 var(--space-3);
  }
  .toolbar-actions {
    gap: var(--space-1);
  }
  .toolbar-actions :deep(.u-segmented button) {
    padding: 0 var(--space-2);
  }
  .season-button {
    max-width: 90px;
    overflow: hidden;
  }
  .toolbar-separator {
    margin: 0;
  }
}
</style>
