<script setup lang="ts">
import { ref, watch } from 'vue';
import { Sparkles, Plug, Settings } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { call } from '../bridge/api';
import { toast } from '../composables/toast';
import { openSettings } from '../composables/settings';
import { presets, presetBusy, loadPresets, switchPreset } from '../composables/ai';
import UPopover from './ui/UPopover.vue';
import UButton from './ui/UButton.vue';
import USwitch from './ui/USwitch.vue';
const app = useAppStore();
const open = ref(false), changing = ref(false), testing = ref(false);
watch(open, async (value) => {
    if (value)
        try {
            await loadPresets();
        }
        catch { /* call reports */
        }
});
async function enable(value: boolean) {
    changing.value = true;
    try {
        await call('set_llm_enabled', value);
    }
    catch { /* call reports */
    }
    finally {
        changing.value = false;
    }
}
async function test() {
    testing.value = true;
    try {
        toast.success(await call<string>('test_llm'));
    }
    catch { /* call reports */
    }
    finally {
        testing.value = false;
    }
}
function settings() {
    open.value = false;
    openSettings('llm');
}
</script>
<template>
  <UPopover v-model="open" :width="360" label="AI 设置">
    <template #trigger="{toggle}">
      <UButton class="ai-button" :class="{'ai-enabled':app.status?.llm.enabled}" variant="plain" size="sm" :icon="Sparkles" :aria-pressed="app.status?.llm.enabled ?? false" @click="toggle">AI</UButton>
    </template>
    <div class="ai-control">
      <label class="popover-switch">
        <span>启用大模型</span>
        <USwitch :model-value="app.status?.llm.enabled ?? false" :disabled="changing" label="启用大模型" @update:model-value="enable" />
      </label>
      <div class="preset-list">
        <button v-for="preset in presets" :key="preset.key" class="preset-option" :class="{active:app.status?.llm.preset===preset.key}" :disabled="presetBusy" :aria-pressed="app.status?.llm.preset===preset.key" @click="switchPreset(preset.key)">
          <span class="preset-radio" />
          <div>
            <strong>{{ preset.title }}</strong>
            <p>{{ preset.model }} · 平时 {{ preset.effort || '默认' }} · 海克斯 {{ preset.effort_augment || '默认' }}</p>
            <small>{{ preset.note }}</small>
          </div>
        </button>
      </div>
      <div class="popover-footer">
        <UButton :icon="Plug" :loading="testing" @click="test">测试连接</UButton>
        <UButton variant="plain" :icon="Settings" @click="settings">大模型设置…</UButton>
      </div>
    </div>
  </UPopover>
</template>
<style scoped>
.ai-enabled {
  color: var(--blue-text);
  background: var(--accent-soft);
}
.popover-switch {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-1);
  font-size: var(--font-sm);
}
.preset-list {
  border-top: 1px solid var(--separator);
  border-bottom: 1px solid var(--separator);
  margin: var(--space-2) 0;
  padding: var(--space-2) 0;
}
.preset-option {
  display: flex;
  gap: var(--space-3);
  width: 100%;
  padding: var(--space-3) var(--space-2);
  border-radius: var(--radius-sm);
  background: var(--transparent);
  text-align: left;
}
.preset-option:hover {
  background: var(--hover);
}
.preset-option.active {
  background: var(--accent-soft);
}
.preset-radio {
  width: 13px;
  height: 13px;
  border: 1px solid var(--tertiary);
  border-radius: var(--radius-pill);
  margin-top: 3px;
  flex-shrink: 0;
}
.active .preset-radio {
  border: 4px solid var(--accent);
}
.preset-option strong {
  font-size: var(--font-sm);
  font-weight: 600;
}
.preset-option p {
  font-size: var(--font-xs);
  color: var(--secondary);
  margin-top: var(--space-1);
}
.preset-option small {
  font-size: var(--font-xs);
  color: var(--tertiary);
  line-height: 1.5;
  display: block;
  margin-top: var(--space-1);
}
.popover-footer {
  display: flex;
  gap: var(--space-2);
  justify-content: space-between;
}
</style>
