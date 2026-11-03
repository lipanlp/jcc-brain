<script setup lang="ts">
import { ref } from 'vue';
import { Wrench, Tags, Crosshair, ScanEye, ShieldCheck, FolderOpen, Info } from 'lucide-vue-next';
import { openLabels, calibrationOpen, checkVision, checkExecutor, chooseReplay, about } from '../composables/tools';
import UPopover from './ui/UPopover.vue';
import UButton from './ui/UButton.vue';
const open = ref(false);
const items = [{ label: '标注单位…', icon: Tags, key: '⌘L', run: openLabels }, { label: '坐标标定…', icon: Crosshair, key: '', run: () => {
            calibrationOpen.value = true;
        } }, { label: '检查视觉模块', icon: ScanEye, key: '', run: checkVision }, { label: '检查操作权限', icon: ShieldCheck, key: '', run: checkExecutor }, { label: '打开回放文件…', icon: FolderOpen, key: '', run: chooseReplay }, { label: '关于金铲铲大脑', icon: Info, key: '', run: about }];
function run(item: typeof items[number]) {
    open.value = false;
    void item.run();
}
</script>
<template>
  <UPopover v-model="open" :width="245" label="工具菜单">
    <template #trigger="{toggle}">
      <UButton variant="plain" size="sm" :icon="Wrench" aria-label="工具" @click="toggle" />
    </template>
    <div class="tools-menu" role="menu">
      <button v-for="(item,index) in items" :key="item.label" role="menuitem" :class="{separated:index===5}" @click="run(item)">
        <component :is="item.icon" :size="15" />
        <span>{{ item.label }}</span>
        <kbd>{{ item.key }}</kbd>
      </button>
    </div>
  </UPopover>
</template>
<style scoped>
.tools-menu button {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2);
  text-align: left;
  background: var(--transparent);
  width: 100%;
  border-radius: var(--radius-sm);
  font-size: var(--font-sm);
}
.tools-menu button:hover,.tools-menu button:focus-visible {
  background: var(--accent-soft);
  color: var(--blue-text);
}
.tools-menu button.separated {
  margin-top: var(--space-2);
  border-top: 1px solid var(--separator);
  padding-top: var(--space-3);
  border-radius: 0;
}
.tools-menu kbd {
  font: inherit;
  font-size: var(--font-xs);
  color: var(--tertiary);
  margin-left: auto;
}
</style>
