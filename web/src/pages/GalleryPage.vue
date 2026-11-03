<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { Plus, ArrowUpRight, MousePointer2, Layers, Search, Inbox } from 'lucide-vue-next';
import UButton from '../components/ui/UButton.vue';
import USegmented from '../components/ui/USegmented.vue';
import USwitch from '../components/ui/USwitch.vue';
import UCard from '../components/ui/UCard.vue';
import UTable from '../components/ui/UTable.vue';
import USearchField from '../components/ui/USearchField.vue';
import UNumberField from '../components/ui/UNumberField.vue';
import USelect from '../components/ui/USelect.vue';
import UCombobox from '../components/ui/UCombobox.vue';
import UPopover from '../components/ui/UPopover.vue';
import USheet from '../components/ui/USheet.vue';
import UBadge from '../components/ui/UBadge.vue';
import USpinner from '../components/ui/USpinner.vue';
import UEmpty from '../components/ui/UEmpty.vue';
import UTooltip from '../components/ui/UTooltip.vue';
import { toast } from '../composables/toast';
import { confirm } from '../composables/confirm';
import { useTheme, type Theme } from '../composables/theme';
import { call } from '../bridge/api';
import type { HeroOption, TableColumn, TableRow } from '../types';
const theme = useTheme();
const themeOptions = [{ value: 'system', label: '跟随系统' }, { value: 'light', label: '浅色' }, { value: 'dark', label: '深色' }];
const segment = ref('daily');
const enabled = ref(true);
const off = ref(false);
const search = ref('');
const number = ref(50);
const select = ref('one');
const hero = ref('');
const heroes = ref<HeroOption[]>([]);
const heroState = ref('正在读取英雄…');
const heroOptions = computed(() => heroes.value.map(item => ({ value: item.name, label: item.name, hint: `${item.cost} 费`, color: `var(--cost-${item.cost})` })));
onMounted(async () => {
    try {
        heroes.value = await call<HeroOption[]>('hero_options');
        heroState.value = heroes.value.length ? `来自当前赛季 · ${heroes.value.length} 位英雄` : '当前赛季暂无英雄数据';
    }
    catch {
        heroState.value = '读取失败，请检查后端连接';
    }
});
const tones = ['gray', 'blue', 'green', 'orange', 'red', 'yellow', 'purple'] as const;
const toneLabels = ['默认', '信息', '成功', '提醒', '错误', '关注', '精选'];
const popover = ref(false);
const sheet = ref(false);
const confirmResult = ref('');
async function ask() {
    confirmResult.value = await confirm({ title: '确认此操作？', message: '这是组件展示中的确认框。选择确认或取消，查看返回结果。', confirmText: '确认操作' }) ? '已确认' : '已取消';
}
const rows = ref<TableRow[]>(Array.from({ length: 30 }, (_, index) => ({ id: `demo-${index + 1}`, name: `示例条目 ${String(index + 1).padStart(2, '0')}`, cost: index % 5 + 1, count: 18 + index % 12, score: Number((70 + (index * 7) % 29 + .4).toFixed(1)) })));
const selected = ref<string | number | null>('demo-12');
const refreshing = ref(false);
const refreshCount = ref(0);
let timer: ReturnType<typeof setInterval> | undefined;
const columns: TableColumn[] = [{ key: 'name', title: '条目名称', sortable: true, width: '35%' }, { key: 'cost', title: '费用', sortable: true, align: 'right' }, { key: 'count', title: '数量', sortable: true, align: 'right' }, { key: 'score', title: '评分', sortable: true, align: 'right', format: value => Number(value).toFixed(1) }];
function toggleRefresh() {
    refreshing.value = !refreshing.value;
    clearInterval(timer);
    if (refreshing.value)
        timer = setInterval(() => {
            for (let i = 0; i < 3; i++) {
                const row = rows.value[Math.floor(Math.random() * rows.value.length)]!;
                row.count = Math.floor(10 + Math.random() * 40);
                row.score = Number((60 + Math.random() * 40).toFixed(1));
            }
            ;
            refreshCount.value++;
        }, 500);
}
onBeforeUnmount(() => clearInterval(timer));
</script>
<template>
  <div class="gallery-page">
    <div class="gallery-heading">
      <div>
        <div class="eyebrow">界面基础</div>
        <h1>组件展示</h1>
        <p>一致的细节，让每一次操作自然顺手。</p>
      </div>
      <USegmented :model-value="theme" :options="themeOptions" label="主题" @update:model-value="theme = $event as Theme" />
    </div>
    <div class="gallery-grid">
      <UCard title="按钮" subtitle="清晰的层级与操作反馈">
        <div class="control-row">
          <UButton variant="primary" :icon="Plus" @click="toast.success('操作已完成')">主要操作</UButton>
          <UButton>次要操作</UButton>
          <UButton variant="plain" :icon="ArrowUpRight">普通操作</UButton>
          <UButton variant="destructive">删除</UButton>
        </div>
        <div class="control-row spaced">
          <UButton size="sm">小尺寸</UButton>
          <UButton>标准尺寸</UButton>
          <UButton loading>加载中</UButton>
          <UButton disabled>不可用</UButton>
        </div>
      </UCard>
      <UCard title="选择与标签" subtitle="分段选择、开关与语义状态">
        <div class="control-row">
          <USegmented v-model="segment" :options="[{ value: 'daily', label: '日常', icon: Layers }, { value: 'focus', label: '专注', icon: MousePointer2 }]" label="展示分段" />
          <USwitch v-model="enabled" label="开启示例" />
          <USwitch v-model="off" label="关闭示例" />
          <USwitch :model-value="true" disabled label="禁用示例" />
        </div>
        <div class="control-row spaced">
          <UBadge v-for="(tone, index) in tones" :key="tone" :tone="tone">{{ toneLabels[index] }}</UBadge>
        </div>
      </UCard>
      <UCard title="输入与选择" subtitle="支持键盘操作，保留熟悉的输入方式">
        <div class="form-grid">
          <label class="field">
            <span>搜索</span>
            <USearchField v-model="search" placeholder="搜索组件…" />
            <small>当前值：{{ search || '—' }}</small>
          </label>
          <label class="field">
            <span>数值</span>
            <UNumberField v-model="number" :min="0" :max="100" :step="5" label="展示数值" />
            <small>0–100，步进 5</small>
          </label>
          <label class="field">
            <span>选择框</span>
            <USelect v-model="select" :options="[{ value: 'one', label: '选项一' }, { value: 'two', label: '选项二' }]" label="展示选择框" />
          </label>
          <div class="field">
            <span>英雄选择</span>
            <UCombobox v-model="hero" :options="heroOptions" placeholder="输入英雄名称…" label="英雄选择" />
            <small id="hero-hint">{{ heroState }}</small>
          </div>
        </div>
      </UCard>
      <UCard title="提示与浮层" subtitle="适时出现，保持当前操作的上下文">
        <div class="control-row">
          <UButton data-demo="toast-success" @click="toast.success('设置已保存')">成功</UButton>
          <UButton data-demo="toast-error" @click="toast.error('连接失败，请稍后重试')">错误</UButton>
          <UButton @click="toast.info('数据已是最新版本')">信息</UButton>
          <UButton @click="toast.warning('当前为演练模式')">提醒</UButton>
        </div>
        <div class="control-row spaced">
          <UButton data-demo="confirm" @click="ask">确认框</UButton>
          <UButton data-demo="sheet" @click="sheet = true">Sheet 面板</UButton>
          <UPopover v-model="popover" label="展示 Popover">
            <template #trigger="{ toggle }">
              <UButton data-demo="popover" @click="toggle">Popover</UButton>
            </template>
            <div class="popover-demo">
              <h3>轻量操作面板</h3>
              <p>跟随触发位置，自动避开窗口边缘。点击外部或按 Esc 关闭。</p>
              <UButton size="sm" @click="popover = false">完成</UButton>
            </div>
          </UPopover>
          <UTooltip text="悬停 500ms 后出现的提示">
            <UButton data-demo="tooltip" :icon="Search" variant="plain" aria-label="查看提示" />
          </UTooltip>
        </div>
        <div class="feedback-line">
          <USpinner :size="14" />
          <span>正在加载</span>
          <span v-if="confirmResult" class="confirm-result">{{ confirmResult }}</span>
        </div>
      </UCard>
      <UCard class="full-width" title="数据表格" subtitle="30 行示例 · 稳定行标识 · 实时更新">
        <template #actions>
          <UBadge :tone="refreshing ? 'green' : 'gray'">{{ refreshing ? '每 500ms 更新' : '已暂停' }}</UBadge>
          <UButton data-demo="refresh" size="sm" @click="toggleRefresh">{{ refreshing ? '暂停刷新' : '随机刷新数据' }}</UButton>
        </template>
        <div class="table-caption">
          <span>选中：{{ selected || '无' }}</span>
          <span class="numeric">已刷新 {{ refreshCount }} 次</span>
        </div>
        <UTable v-model:selected="selected" :columns="columns" :rows="rows" row-key="id" height="280px">
          <template #cell-cost="{ value }">
            <span class="cost-label" :style="{ color: `var(--cost-${value})` }">{{ value }} 费</span>
          </template>
        </UTable>
      </UCard>
      <UCard class="full-width" title="空状态" :padded="false">
        <UEmpty :icon="Inbox" title="这里还没有内容" description="内容准备好后，会出现在这里。">
          <template #action>
            <UButton size="sm" variant="primary" @click="toast.info('已收到你的操作')">示例操作</UButton>
          </template>
        </UEmpty>
      </UCard>
    </div>
    <USheet v-model="sheet" title="面板示例" :width="560">
      <div class="sheet-demo">
        <UBadge tone="blue">Sheet</UBadge>
        <h3>专注于眼前的操作</h3>
        <p>面板从顶部出现，支持 Esc 关闭，并将键盘焦点保留在面板内。</p>
        <USearchField v-model="search" placeholder="试试 Tab 切换焦点…" />
      </div>
      <template #footer>
        <UButton @click="sheet = false">取消</UButton>
        <UButton variant="primary" @click="sheet = false; toast.success('已完成面板操作')">完成</UButton>
      </template>
    </USheet>
  </div>
</template>
<style scoped>
.gallery-page {
  height: 100%;
  overflow: auto;
  padding: var(--space-6) var(--space-8) var(--space-10);
}
.gallery-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-5);
  margin-bottom: var(--space-6);
}
.eyebrow {
  font-size: var(--font-xs);
  color: var(--secondary);
  margin-bottom: var(--space-1);
}
h1 {
  font-size: var(--font-display);
  letter-spacing: -.7px;
  font-weight: 650;
}
.gallery-heading p {
  font-size: var(--font-sm);
  color: var(--secondary);
  margin-top: var(--space-1);
}
.gallery-grid {
  display: grid;
  grid-template-columns: minmax(0,1fr) minmax(0,1fr);
  gap: var(--space-4);
}
.control-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
}
.spaced {
  margin-top: var(--space-4);
}
.form-grid {
  display: grid;
  grid-template-columns: minmax(0,1.2fr) minmax(0,1fr);
  gap: var(--space-4);
}
.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  font-size: var(--font-sm);
  min-width: 0;
}
.field > span, .field > label {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.field small {
  color: var(--secondary);
  font-size: var(--font-xs);
}
.field :deep(.popover-anchor) {
  width: 100%;
}
.full-width {
  grid-column: 1 / -1;
}
.feedback-line {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--font-xs);
  color: var(--secondary);
  margin-top: var(--space-5);
}
.confirm-result {
  margin-left: auto;
  color: var(--blue-text);
}
.table-caption {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--space-3);
  color: var(--secondary);
  font-size: var(--font-xs);
}
.cost-label {
  font-weight: 500;
}
.popover-demo h3 {
  font-size: var(--font-md);
  margin-bottom: var(--space-2);
}
.popover-demo p {
  font-size: var(--font-sm);
  line-height: 1.8;
  color: var(--secondary);
  margin-bottom: var(--space-3);
}
.sheet-demo h3 {
  margin: var(--space-4) 0 var(--space-2);
  font-size: var(--font-xl);
}
.sheet-demo p {
  color: var(--secondary);
  margin-bottom: var(--space-5);
  line-height: 1.8;
}
@media (max-width: 1080px) {
  .gallery-page {
    padding: var(--space-5);
  }
  .gallery-grid {
    grid-template-columns: 1fr;
  }
  .gallery-heading {
    align-items: flex-start;
  }
  .form-grid {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
