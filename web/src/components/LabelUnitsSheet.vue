<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { labelInfo } from '../composables/tools';
import { call } from '../bridge/api';
import { alert } from '../composables/confirm';
import USheet from './ui/USheet.vue';
import UCombobox from './ui/UCombobox.vue';
import UButton from './ui/UButton.vue';
const assignments = ref<Record<string, string>>({});
const saving = ref(false);
const options = computed(() => labelInfo.value?.heroes.map(name => ({ value: name, label: name })) ?? []);
watch(labelInfo, () => {
    assignments.value = {};
});
function close() {
    if (!saving.value)
        labelInfo.value = null;
}
async function save() {
    saving.value = true;
    try {
        const result = await call<string[]>('label_units', Object.entries(assignments.value).filter(([, name]) => name.trim()).map(([key, name]) => ({ key, name })));
        labelInfo.value = null;
        await alert({ title: '标注结果', message: result.length ? result.join('\n') : '没有可标注的单位（画面可能已变化）' });
    }
    catch { /* call reports */
    }
    finally {
        saving.value = false;
    }
}
</script>
<template>
  <USheet :model-value="!!labelInfo" title="标注单位" :width="680" @update:model-value="close">
    <p class="label-note">选择正确的英雄名称；留空的单位保持原样。</p>
    <div class="label-units">
      <div v-for="unit in labelInfo?.units" :key="unit.key" class="label-unit">
        <span :class="{unknown:!unit.known}">{{ unit.where }} · {{ '★'.repeat(unit.star) }} {{ unit.name }}</span>
        <UCombobox :model-value="assignments[unit.key] ?? ''" :options="options" :disabled="saving" :label="`${unit.where} 英雄`" placeholder="留空 = 不改" @update:model-value="assignments[unit.key]=$event" />
      </div>
      <p v-if="!labelInfo?.units.length" class="label-note">当前画面没有棋子。</p>
    </div>
    <template #footer>
      <UButton :disabled="saving" @click="close">取消</UButton>
      <UButton variant="primary" :loading="saving" @click="save">保存</UButton>
    </template>
  </USheet>
</template>
<style scoped>
.label-note {
  color: var(--secondary);
  font-size: var(--font-sm);
  margin-bottom: var(--space-4);
}
.label-unit {
  display: flex;
  gap: var(--space-4);
  align-items: center;
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--separator);
}
.label-unit>span {
  flex: 1;
  min-width: 0;
  font-size: var(--font-sm);
}
.label-unit .unknown {
  color: var(--tertiary);
}
.label-unit :deep(.popover-anchor) {
  width: 210px;
  flex-shrink: 0;
}
</style>
