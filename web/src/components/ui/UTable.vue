<script setup lang="ts">
import { computed, nextTick, ref, type DirectiveBinding } from 'vue';
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-vue-next';
import UEmpty from './UEmpty.vue';
import type { TableColumn, TableRow } from '../../types';
const props = withDefaults(defineProps<{
    columns: TableColumn[];
    rows: TableRow[];
    rowKey: string | ((row: TableRow) => string | number);
    selected?: string | number | null;
    height?: string;
    emptyTitle?: string;
    label?: string;
    nativeTitles?: boolean;
}>(), { emptyTitle: '暂无数据', nativeTitles: true });
const emit = defineEmits<{
    'update:selected': [
        value: string | number
    ];
    sort: [
        key: string | null,
        direction: 'asc' | 'desc' | null
    ];
}>();
const setTitle = (el: HTMLElement, binding: DirectiveBinding<boolean>) => {
    if (binding.value === false) {
        el.removeAttribute('title');
        return;
    }
    el.title = el.textContent?.replace(/\s+/g, ' ').trim() ?? '';
};
const vCellTitle = { mounted: setTitle, updated: setTitle };
const sortKey = ref<string | null>(null);
const direction = ref<'asc' | 'desc' | null>(null);
const root = ref<HTMLElement>();
const localSelected = ref<string | number | null>(null);
const selection = computed(() => props.selected === undefined ? localSelected.value : props.selected);
function select(row: TableRow) {
    localSelected.value = key(row);
    emit('update:selected', key(row));
}
async function move(event: KeyboardEvent) {
    if (!['ArrowUp', 'ArrowDown'].includes(event.key) || !displayed.value.length ||
        (event.target instanceof HTMLElement && event.target.matches('input,select,textarea')))
        return;
    event.preventDefault();
    const index = displayed.value.findIndex(row => key(row) === selection.value);
    const next = index < 0 ? (event.key === 'ArrowDown' ? 0 : displayed.value.length - 1) :
        Math.max(0, Math.min(displayed.value.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)));
    const row = displayed.value[next]!;
    select(row);
    await nextTick();
    const element = [...(root.value?.querySelectorAll<HTMLElement>('tbody tr') ?? [])].find(node => node.dataset.rowKey === String(key(row)));
    element?.focus({ preventScroll: true });
    element?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}
function key(row: TableRow) {
    return typeof props.rowKey === 'function' ? props.rowKey(row) : row[props.rowKey] as string | number;
}
function sort(column: TableColumn) {
    if (!column.sortable)
        return;
    if (sortKey.value !== column.key) {
        sortKey.value = column.key;
        direction.value = 'asc';
    }
    else if (direction.value === 'asc')
        direction.value = 'desc';
    else {
        sortKey.value = null;
        direction.value = null;
    }
    ;
    emit('sort', sortKey.value, direction.value);
}
const displayed = computed(() => {
    if (!sortKey.value)
        return props.rows;
    const field = sortKey.value;
    const multiplier = direction.value === 'asc' ? 1 : -1;
    return [...props.rows].sort((a, b) => {
        const x = a[field], y = b[field];
        if (x == null)
            return y == null ? 0 : 1;
        if (y == null)
            return -1;
        return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), 'zh-CN', { numeric: true })) * multiplier;
    });
});
function width(column: TableColumn) {
    return typeof column.width === 'number' ? `${column.width}px` : column.width;
}
</script>
<template>
  <div ref="root" class="u-table-scroll" :style="{ maxHeight: height }" tabindex="0" :aria-label="label ?? '数据表格'" @keydown="move">
    <table v-if="rows.length" class="u-table">
      <thead>
        <tr>
          <th v-for="column in columns" :key="column.key" :style="{ width: width(column), textAlign: column.align ?? 'left' }" :aria-sort="sortKey === column.key ? (direction === 'asc' ? 'ascending' : 'descending') : column.sortable ? 'none' : undefined">
            <button v-if="column.sortable" type="button" :style="{ justifyContent: column.align === 'right' ? 'flex-end' : column.align === 'center' ? 'center' : 'flex-start' }" @click="sort(column)">
              {{ column.title }}
              <component :is="sortKey === column.key ? direction === 'asc' ? ChevronUp : ChevronDown : ChevronsUpDown" :size="12" />
            </button>
            <template v-else>{{ column.title }}</template>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in displayed" :key="key(row)" :data-row-key="key(row)" :aria-selected="selection === key(row)" :class="{ selected: selection === key(row) }" tabindex="-1" @click="select(row)" @keydown.enter.prevent="select(row)" @keydown.space.prevent="select(row)">
          <td v-for="column in columns" :key="column.key" :style="{ textAlign: column.align ?? 'left' }" :class="{ wrap: column.wrap, numeric: column.align === 'right' || typeof row[column.key] === 'number' }">
            <div v-cell-title="nativeTitles" class="cell-content">
              <slot :name="`cell-${column.key}`" :value="row[column.key]" :row="row">{{ column.format ? column.format(row[column.key], row) : row[column.key] ?? '—' }}</slot>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
    <UEmpty v-else :title="emptyTitle" description="有数据时会显示在这里" />
  </div>
</template>
<style scoped>
.u-table-scroll {
  overflow: auto;
  min-width: 0;
  scrollbar-gutter: stable;
}
.u-table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: var(--font-sm);
  table-layout: fixed;
}
th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface-raised);
  color: var(--secondary);
  font-weight: 500;
  border-bottom: 1px solid var(--separator);
  height: 35px;
}
th, td {
  padding: var(--space-2) var(--space-4);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
th button {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  padding: 0;
  width: 100%;
  background: var(--transparent);
  color: inherit;
  font-size: var(--font-xs);
}
td {
  border-bottom: 1px solid var(--separator);
  height: 36px;
  padding-top: var(--space-half);
  padding-bottom: var(--space-half);
}
.cell-content {
  max-height: 32px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
td.wrap .cell-content {
  max-height: none;
  white-space: normal;
}
tr:nth-child(even) td {
  background: var(--stripe);
}
tr:hover td {
  background: var(--hover);
}
tr.selected td {
  background: var(--accent-soft);
  color: var(--blue-text);
}
tr:focus-visible {
  outline: none;
}
tr:focus-visible td {
  box-shadow: inset 0 0 0 1px var(--accent);
}
</style>
