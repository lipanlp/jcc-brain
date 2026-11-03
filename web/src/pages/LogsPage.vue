<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { Copy, Trash2, ScrollText } from 'lucide-vue-next';
import { useAppStore } from '../stores/app';
import { toast } from '../composables/toast';
import type { StoredLog } from '../types';
import VirtualList from '../components/VirtualList.vue';
import USegmented from '../components/ui/USegmented.vue';
import USelect from '../components/ui/USelect.vue';
import USearchField from '../components/ui/USearchField.vue';
import USwitch from '../components/ui/USwitch.vue';
import UButton from '../components/ui/UButton.vue';
import UEmpty from '../components/ui/UEmpty.vue';
const store = useAppStore();
const source = ref('all');
const level = ref('all');
const query = ref('');
const autoScroll = ref(true);
const copying = ref(false);
const sources = [{ value: 'all', label: '全部' }, { value: 'mode', label: '模式' }, { value: 'app', label: '程序' }];
const levels = [{ value: 'all', label: '全部级别' }, { value: 'warning', label: '警告及以上' }];
const filtered = computed(() => store.logs.filter(log => (source.value === 'all' || (source.value === 'mode' ? log.source === 'mode' : log.source !== 'mode')) &&
    (level.value === 'all' || log.level !== 'info') &&
    (!query.value.trim() || `${log.time} ${log.text}`.toLocaleLowerCase().includes(query.value.trim().toLocaleLowerCase()))));
const list = ref<{
    scrollToBottom(): void;
    position(): {
        top: number;
        distance: number;
    };
}>();
let following = false;
let ticket = 0;
watch([filtered, autoScroll], async () => {
    const current = ++ticket;
    if (!autoScroll.value) {
        following = false;
        return;
    }
    following = true;
    await nextTick();
    if (current !== ticket || !autoScroll.value) {
        following = false;
        return;
    }
    list.value?.scrollToBottom();
    requestAnimationFrame(() => {
        if (current === ticket)
            following = false;
    });
}, { immediate: true, flush: 'pre' });
function scrolled(position: {
    top: number;
    distance: number;
}) {
    if (following)
        return;
    if (position.distance > 40)
        autoScroll.value = false;
    else
        autoScroll.value = true;
}
async function copyLogs() {
    const text = filtered.value.map(log => `${log.time} ${log.text}`).join('\n');
    copying.value = true;
    try {
        try {
            if (!navigator.clipboard?.writeText)
                throw new Error('clipboard unavailable');
            await navigator.clipboard.writeText(text);
        }
        catch {
            const previous = document.activeElement as HTMLElement | null;
            const input = document.createElement('textarea');
            input.value = text;
            input.className = 'clipboard-copy selectable';
            document.body.append(input);
            input.select();
            try {
                if (!document.execCommand('copy'))
                    throw new Error('无法写入剪贴板');
            }
            finally {
                input.remove();
                previous?.focus();
            }
        }
        toast.success(`已复制 ${filtered.value.length} 条日志`);
    }
    catch {
        toast.error('复制失败，请重试');
    }
    finally {
        copying.value = false;
    }
}
</script>
<template>
  <div class="logs-page">
    <div class="logs-toolbar">
      <USegmented v-model="source" :options="sources" label="日志来源" />
      <USelect v-model="level" :options="levels" label="日志级别" />
      <USearchField v-model="query" placeholder="搜索日志…" />
      <label class="auto-scroll">
        自动滚动
        <USwitch v-model="autoScroll" label="自动滚动" />
      </label>
      <UButton :icon="Copy" :loading="copying" :disabled="!filtered.length" @click="copyLogs">复制</UButton>
      <UButton :icon="Trash2" :disabled="!store.logs.length" @click="store.clearLogs">清空</UButton>
    </div>
    <div class="logs-summary">
      <span>
        {{ filtered.length }}
        条记录
        <span v-if="filtered.length!==store.logs.length"> / 共 {{ store.logs.length }} 条</span>
      </span>
      <span>最多保留 3000 条</span>
    </div>
    <VirtualList ref="list" :items="filtered" :row-key="log=>log.id" :row-height="24" :overscan="10" label="运行日志" @scroll="scrolled">
      <template #default="{item}">
        <div class="log-row" :class="item.level" :data-log-id="item.id" :title="`${item.time} ${item.text}`">
          <time>{{ item.time }}</time>
          <i class="source-dot" :class="item.source==='mode' ? 'mode' : 'app'" :title="item.source==='mode' ? '模式' : '程序'" />
          <span class="log-text selectable">{{ item.text }}</span>
        </div>
      </template>
      <template #empty>
        <UEmpty :icon="ScrollText" :title="store.logs.length ? '没有匹配的日志' : '暂无运行记录'" description="启动模式后，日志会实时显示在这里" />
      </template>
    </VirtualList>
  </div>
</template>
<style scoped>
.logs-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: var(--space-5) var(--space-6);
  gap: var(--space-3);
}
.logs-toolbar {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  flex-wrap: wrap;
}
.logs-toolbar :deep(.u-search) {
  flex: 1;
  min-width: 140px;
}
.auto-scroll {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--font-xs);
  color: var(--secondary);
  white-space: nowrap;
}
.logs-summary {
  display: flex;
  justify-content: space-between;
  font-size: var(--font-xs);
  color: var(--secondary);
  font-variant-numeric: tabular-nums;
}
.log-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: 0 var(--space-2);
  height: 24px;
  font-size: var(--font-sm);
  border-radius: var(--radius-sm);
}
.log-row:hover {
  background: var(--hover);
}
.log-row time {
  font-size: var(--font-xs);
  font-variant-numeric: tabular-nums;
  color: var(--tertiary);
  flex-shrink: 0;
}
.source-dot {
  width: 5px;
  height: 5px;
  background: var(--tertiary);
  border-radius: var(--radius-pill);
  flex-shrink: 0;
}
.source-dot.mode {
  background: var(--accent);
}
.log-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.warning .log-text {
  color: var(--orange-text);
}
.error .log-text {
  color: var(--red-text);
}
</style>
<style>
.clipboard-copy {position:fixed;left:-10000px;top:0;opacity:0;}
</style>
