<script setup lang="ts">
import { computed, ref, shallowRef, watch } from 'vue';
import { Eye, EyeOff, FolderOpen, Plug, Palette } from 'lucide-vue-next';
import { settingsOpen, settingsSection } from '../composables/settings';
import { useAppStore } from '../stores/app';
import { call } from '../bridge/api';
import { confirm } from '../composables/confirm';
import { toast } from '../composables/toast';
import { useTheme, type Theme } from '../composables/theme';
import type { SettingsSection, SettingsValues, SettingValue, SelectOption } from '../types';
import USheet from './ui/USheet.vue';
import USwitch from './ui/USwitch.vue';
import UNumberField from './ui/UNumberField.vue';
import USelect from './ui/USelect.vue';
import UCombobox from './ui/UCombobox.vue';
import USegmented from './ui/USegmented.vue';
import UButton from './ui/UButton.vue';
import USpinner from './ui/USpinner.vue';
const app = useAppStore();
const sections = shallowRef<SettingsSection[]>([]);
const values = ref<SettingsValues>({});
const baseline = shallowRef<SettingsValues>({});
const loading = ref(false), saving = ref(false), testing = ref(false), closing = ref(false);
const reveal = ref(false);
const theme = useTheme();
const themes = [{ value: 'system', label: '跟随系统' }, { value: 'light', label: '浅色' }, { value: 'dark', label: '深色' }];
const current = computed(() => sections.value.find(section => section.key === settingsSection.value));
const changes = computed(() => {
    const result: SettingsValues = {};
    for (const [section, fields] of Object.entries(values.value)) {
        for (const [key, value] of Object.entries(fields)) {
            if (Array.isArray(value) ? JSON.stringify(value) !== JSON.stringify(baseline.value[section]?.[key]) : value !== baseline.value[section]?.[key])
                (result[section] ??= {})[key] = value;
        }
    }
    return result;
});
const dirty = computed(() => Object.keys(changes.value).length > 0);
const sourceOnly = computed(() => Object.keys(changes.value).length === 1 && !!changes.value.data
    && Object.keys(changes.value.data).every(key => ['source_official', 'source_ingame', 'source_dataj'].includes(key)));
const requiresStop = computed(() => !!app.status?.mode && !sourceOnly.value);
let generation = 0;
watch(settingsOpen, async (open) => {
    const ticket = ++generation;
    if (!open)
        return;
    loading.value = true;
    reveal.value = false;
    sections.value = [];
    values.value = {};
    baseline.value = {};
    try {
        const schema = await call<SettingsSection[]>('settings_schema');
        if (ticket !== generation)
            return;
        sections.value = schema;
        const data = Object.fromEntries(schema.map(section => [section.key, Object.fromEntries(section.fields.map(field => [field.key, field.value]))]));
        baseline.value = data;
        values.value = JSON.parse(JSON.stringify(data)) as SettingsValues;
    }
    catch { /* call reports */
    }
    finally {
        if (ticket === generation)
            loading.value = false;
    }
});
function set(section: string, key: string, value: SettingValue) {
    (values.value[section] ??= {})[key] = value;
}
async function requestClose() {
    if (saving.value || closing.value)
        return;
    closing.value = true;
    try {
        if (!dirty.value || await confirm({ title: '放弃未保存的更改？', message: '未保存的设置将被放弃。', confirmText: '放弃更改', destructive: true }))
            settingsOpen.value = false;
    }
    finally {
        closing.value = false;
    }
}
async function save() {
    if (!dirty.value || saving.value)
        return;
    saving.value = true;
    try {
        if (requiresStop.value && !await confirm({ title: '保存设置', message: '应用公共配置需要停止当前模式，保存后不会自动重启。', confirmText: '停止并应用' }))
            return;
        await call('save_settings', JSON.parse(JSON.stringify(changes.value)), requiresStop.value);
        toast.success('设置已保存');
        settingsOpen.value = false;
    }
    catch { /* call reports */
    }
    finally {
        saving.value = false;
    }
}
async function browse(section: string, key: string) {
    try {
        const path = await call<string | null>('browse_file');
        if (path)
            set(section, key, path);
    }
    catch { /* call reports */
    }
}
async function testConnection() {
    testing.value = true;
    try {
        toast.success(await call<string>('test_llm', JSON.parse(JSON.stringify(values.value))));
    }
    catch { /* call reports */
    }
    finally {
        testing.value = false;
    }
}
</script>
<template>
  <USheet :model-value="settingsOpen" title="通用设置" :width="760" :height="560" flush @update:model-value="requestClose">
    <div v-if="loading" class="settings-loading">
      <USpinner />
      正在读取设置…
    </div>
    <div v-else class="settings-layout">
      <nav class="settings-sections" aria-label="设置分节">
        <button v-for="section in sections" :key="section.key" :class="{active:settingsSection===section.key}" @click="settingsSection=section.key">{{ section.title }}</button>
        <button :class="{active:settingsSection==='appearance'}" @click="settingsSection='appearance'">
          <Palette :size="14" />
          外观
        </button>
      </nav>
      <div class="settings-form">
        <template v-if="settingsSection==='appearance'">
          <h2>外观</h2>
          <p class="settings-note">主题会立即应用，并保存到这台设备。</p>
          <USegmented :model-value="theme" :options="themes" label="主题" @update:model-value="theme=$event as Theme" />
        </template>
        <template v-else-if="current">
          <h2>{{ current.title }}</h2>
          <fieldset :disabled="saving" class="settings-fields">
            <div v-for="field in current.fields" :key="`${current.key}.${field.key}`" class="settings-row" :data-setting="`${current.key}.${field.key}`">
              <label :for="`setting-${current.key}-${field.key}`">{{ field.label }}</label>
              <div class="settings-control">
                <input v-if="field.type==='rounds'" class="control settings-text" :aria-label="field.label" :value="Array.isArray(values[current.key]?.[field.key]) ? (values[current.key]?.[field.key] as string[]).join(', ') : values[current.key]?.[field.key]" @input="set(current.key,field.key,($event.target as HTMLInputElement).value)" />
                <USwitch v-else-if="field.type==='bool'" :model-value="Boolean(values[current.key]?.[field.key])" :label="field.label" @update:model-value="set(current.key,field.key,$event)" />
                <UNumberField v-else-if="field.type==='int'||field.type==='float'" :model-value="Number(values[current.key]?.[field.key])" :step="field.type==='float' ? 0.1 : 1" :label="field.label" @update:model-value="set(current.key,field.key,$event)" />
                <USelect v-else-if="field.type==='choice'" :model-value="String(values[current.key]?.[field.key] ?? '')" :options="(field.choices ?? []).filter((value): value is string => typeof value==='string').map(value=>({value,label:value || '默认'}))" :label="field.label" @update:model-value="set(current.key,field.key,$event)" />
                <UCombobox v-else-if="field.type==='comp'" :model-value="String(values[current.key]?.[field.key] ?? '')" :options="(field.choices ?? []).filter((value): value is SelectOption => typeof value!=='string')" :label="field.label" :disabled="saving" @update:model-value="set(current.key,field.key,$event)" />
                <template v-else>
                  <input :id="`setting-${current.key}-${field.key}`" class="control settings-text" :type="field.type==='secret'&&!reveal ? 'password' : 'text'" :value="values[current.key]?.[field.key]" :aria-label="field.label" autocomplete="off" spellcheck="false" @input="set(current.key,field.key,($event.target as HTMLInputElement).value)" />
                  <UButton v-if="field.type==='secret'" variant="plain" size="sm" :icon="reveal ? EyeOff : Eye" :aria-label="reveal ? '隐藏 API Key' : '显示 API Key'" @click="reveal=!reveal" />
                  <UButton v-if="field.type==='file'" size="sm" :icon="FolderOpen" @click="browse(current.key,field.key)">选择…</UButton>
                </template>
              </div>
            </div>
          </fieldset>
          <p v-if="current.note" class="settings-note">{{ current.note }}</p>
          <UButton v-if="current.key==='llm'" :icon="Plug" :loading="testing" :disabled="saving" @click="testConnection">测试连接</UButton>
        </template>
      </div>
    </div>
    <template #footer>
      <span v-if="dirty" class="dirty-indicator">有未保存的更改</span>
      <UButton :disabled="saving" @click="requestClose">取消</UButton>
      <UButton variant="primary" :loading="saving" :disabled="!dirty||loading" @click="save">保存</UButton>
    </template>
  </USheet>
</template>
<style scoped>
.settings-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: var(--space-2);
  height: 100%;
  color: var(--secondary);
}
.settings-layout {
  display: flex;
  height: 100%;
  min-height: 0;
}
.settings-sections {
  width: 150px;
  flex-shrink: 0;
  overflow: auto;
  padding: var(--space-3);
  background: var(--hover);
  border-right: 1px solid var(--separator);
}
.settings-sections button {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  width: 100%;
  text-align: left;
  border-radius: var(--radius-sm);
  padding: var(--space-2) var(--space-3);
  margin-bottom: var(--space-half);
  background: var(--transparent);
  font-size: var(--font-sm);
}
.settings-sections button:hover {
  background: var(--hover);
}
.settings-sections button.active {
  background: var(--accent);
  color: var(--white);
}
.settings-form {
  flex: 1;
  overflow: auto;
  min-width: 0;
  padding: var(--space-4) var(--space-5);
}
.settings-form h2 {
  font-size: var(--font-lg);
  font-weight: 600;
  margin-bottom: var(--space-3);
}
.settings-fields {
  margin: 0;
  padding: 0;
  border: 0;
  min-width: 0;
}
.settings-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-4);
  min-height: 43px;
  border-bottom: 1px solid var(--separator);
  padding: var(--space-2) 0;
}
.settings-row>label {
  font-size: var(--font-sm);
  flex: 1;
  min-width: 100px;
}
.settings-control {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-1);
  flex: 0 1 260px;
  min-width: 0;
}
.settings-control>.settings-text {
  width: 100%;
  min-width: 0;
  padding: 0 var(--space-2);
}
.settings-control>.u-select {
  width: 100%;
}
.settings-note {
  font-size: var(--font-xs);
  color: var(--secondary);
  white-space: pre-wrap;
  line-height: 1.7;
  margin: var(--space-4) 0;
  overflow-wrap: anywhere;
}
.dirty-indicator {
  margin-right: auto;
  align-self: center;
  color: var(--secondary);
  font-size: var(--font-xs);
}
</style>
