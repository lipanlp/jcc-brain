<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { useAppStore } from '../stores/app';
import type { CompPresetsResult } from '../types';
import UCard from '../components/ui/UCard.vue';
import UButton from '../components/ui/UButton.vue';
import ComparisonScope from '../components/ComparisonScope.vue';
const store = useAppStore();
const data = shallowRef<CompPresetsResult | null>(null);
const chosen = ref('all'), name = ref(''), members = ref<string[]>([]), query = ref(''), source = ref('all');
const busy = ref(false), error = ref(''), message = ref('');
let ticket = 0;
const selected = computed(() => data.value?.configs.find(row => row.id === chosen.value));
const builtin = computed(() => chosen.value === 'all');
const dirty = computed(() => chosen.value === 'new' || !!selected.value && !builtin.value &&
    (name.value !== selected.value.name || JSON.stringify([...members.value].sort()) !== JSON.stringify([...selected.value.members].sort())));
const catalog = computed(() => data.value?.catalog ?? []);
const filtered = computed(() => catalog.value.filter(row =>
    (source.value === 'all' || (source.value === 'dataj' ? row.source === 'dataj' : row.source === 'official' && row.channels.includes(source.value === 'ingame' ? 'ingame' : 'web')))
    && (!query.value.trim() || [row.name, ...row.heroes].some(value => value.toLocaleLowerCase().includes(query.value.trim().toLocaleLowerCase())))));
const unavailable = computed(() => members.value.filter(id => !catalog.value.some(row => row.id === id))
    .map(id => ({ id, name: selected.value?.unavailable.find(row => row.id === id)?.name ?? id })));
function draft(id: string) {
    chosen.value = id;
    const row = data.value?.configs.find(row => row.id === id);
    name.value = row?.name ?? '';
    members.value = [...(row?.members ?? [])];
    message.value = ''; error.value = '';
}
async function load() {
    const epoch = ++ticket; busy.value = false;
    try {
        const next = await call<CompPresetsResult>('comp_presets', { silent: true });
        if (epoch !== ticket || store.status && next.season_id !== store.status.season.id) return;
        const preserve = data.value?.season_id === next.season_id && dirty.value;
        const selection = data.value?.season_id === next.season_id && next.configs.some(row => row.id === chosen.value)
            ? chosen.value : next.active_id === 'invalid' ? 'all' : next.active_id;
        data.value = next;
        if (!preserve) draft(selection);
    } catch (e) { if (epoch === ticket) error.value = e instanceof Error ? e.message : '读取失败'; }
}
async function mutate(method: string, args: unknown[]) {
    if (!data.value || busy.value) return;
    const season = data.value.season_id, epoch = ++ticket; busy.value = true; error.value = ''; message.value = '';
    try {
        const next = await call<CompPresetsResult>(method, season, ...args, { silent: true });
        if (epoch !== ticket || store.status && next.season_id !== store.status.season.id) return;
        data.value = next;
        draft(method === 'delete_comp_preset' ? next.active_id : next.saved_id ?? chosen.value);
        message.value = method === 'select_comp_preset' ? '已启用；下一帧有效辅助状态使用新范围' : method === 'delete_comp_preset' ? '配置已删除' : '已保存；启用状态见上方，未启用配置不会影响实况';
    } catch (e) { if (epoch === ticket) error.value = e instanceof Error ? e.message : '保存失败，当前配置未改变'; }
    finally { if (epoch === ticket) busy.value = false; }
}
function copyAll() { draft('new'); name.value = '全部阵容 副本'; members.value = catalog.value.map(row => row.id); }
function allVisible() { members.value = [...new Set([...members.value, ...filtered.value.map(row => row.id)])]; }
watch(() => store.status?.season.id, () => { data.value = null; draft('all'); void load(); }, { immediate: true });
const off = bus.on('data_changed', () => { void load(); });
onBeforeUnmount(() => { off(); ticket++; });
</script>
<template>
  <div class="preset-page">
    <UCard title="预选阵容配置">
      <ComparisonScope :scope="store.status?.assistant_selection ?? data?.scope" />
      <p class="muted">按赛季保存。全部阵容随启用来源和目录动态更新；自定义成员失效时保留记录，恢复来源后重新可用。评级仅作资料标签，不参与规则强度、同分排序或AI推荐。</p>
    </UCard>
    <p v-if="error || data?.error" role="alert" class="preset-error">{{ error || data?.error }}</p>
    <p v-if="message" role="status" class="preset-message">{{ message }}</p>
    <UCard title="管理配置">
      <div class="preset-controls">
        <label>查看配置<select aria-label="查看配置" :value="chosen" :disabled="busy" @change="draft(($event.target as HTMLSelectElement).value)">
          <option v-for="row in data?.configs ?? []" :key="row.id" :value="row.id">{{ row.name }}{{ row.id===data?.active_id ? '（已启用）' : '' }} · 已选{{ row.selected_count }} / 可比较{{ row.effective_count }}</option>
          <option v-if="chosen==='new'" value="new">新配置（未保存）</option>
        </select></label>
        <UButton :disabled="busy || !!data?.error" @click="draft('new')">新增配置</UButton>
        <UButton :disabled="busy || !data || !!data.error" @click="copyAll">复制全部为自定义</UButton>
      </div>
      <div class="preset-controls edit-controls">
        <label>配置名称<input v-model="name" aria-label="配置名称" maxlength="40" :disabled="builtin || busy" /></label>
        <UButton variant="primary" :disabled="builtin || busy || !dirty || !!data?.error" @click="mutate('save_comp_preset',[chosen==='new' ? null : chosen,name,members])">保存配置</UButton>
        <UButton :disabled="busy || dirty || chosen===data?.active_id || !!data?.error" @click="mutate('select_comp_preset',[chosen])">启用此配置</UButton>
        <UButton :disabled="builtin || busy || chosen==='new' || !!data?.error" @click="mutate('delete_comp_preset',[chosen])">删除配置</UButton>
      </div>
      <p class="draft-state" role="status">{{ builtin ? '内置动态全部：不能改名、删成员或删除，可复制为自定义。' : dirty ? '有未保存草稿，未生效；先保存再启用。' : '已保存' }} · 当前启用：{{ data?.configs.find(row => row.id===data?.active_id)?.name ?? '损坏配置（暂停比较）' }}</p>
      <p v-if="selected" class="saved-count">已选目录 {{ selected.selected_count }} · 实际可比较 {{ selected.effective_count }}（关联条目去重，来源关闭不可绕过）</p>
      <div class="preset-controls filters">
        <label>检索<input v-model="query" aria-label="搜索阵容或英雄" placeholder="搜索阵容或英雄" /></label>
        <label>来源<select v-model="source" aria-label="配置成员来源"><option value="all">全部启用来源</option><option value="official">官网</option><option value="ingame">微社区</option><option value="dataj">dataj</option></select></label>
        <UButton :disabled="builtin || busy" @click="allVisible">全选筛选结果</UButton>
        <UButton :disabled="builtin || busy" @click="members=[]">清空成员</UButton>
      </div>
      <p class="muted">草稿目录成员 {{ members.length }}；可保存空配置，启用后有效0，不会回退全部。</p>
      <div v-if="unavailable.length" class="unavailable-members"><p>已保存但当前不可用（保留或手动取消）：</p>
        <label v-for="row in unavailable" :key="row.id"><input v-model="members" type="checkbox" :value="row.id" :disabled="builtin || busy" />{{ row.name }} · {{ row.id }}（来源关闭、资料缺失或记录消失）</label>
      </div>
      <div class="preset-members" aria-label="配置成员">
        <label v-for="row in filtered" :key="row.id" class="preset-member" :data-comp-id="row.id"><input v-model="members" type="checkbox" :value="row.id" :disabled="builtin || busy" /><span><b>{{ row.name }}</b><small>{{ row.source_label }} · {{ row.id }}</small><small>{{ row.heroes.join('、') }}</small></span></label>
        <p v-if="!filtered.length" class="muted">没有可用成员。检查来源开关、当前赛季数据或检索条件。</p>
      </div>
    </UCard>
  </div>
</template>
<style scoped>
.preset-page { overflow: auto; padding: var(--space-4); display: flex; flex-direction: column; gap: var(--space-4); }
.preset-controls { display: flex; flex-wrap: wrap; align-items: end; gap: var(--space-3); }
.preset-controls label { display: flex; flex-direction: column; gap: var(--space-1); min-width: 0; font-size: var(--font-sm); }
.preset-controls input,.preset-controls select { border: 1px solid var(--separator); border-radius: var(--radius-sm); background: var(--input); color: var(--text); padding: 7px; max-width: 100%; min-width: 0; }
.preset-controls select { width: 320px; }
.edit-controls,.filters { margin-top: var(--space-4); }
.preset-page p { font-size: var(--font-xs); line-height: 1.6; margin-top: var(--space-2); overflow-wrap: anywhere; }
.preset-error { color: var(--red-text); }
.preset-message { color: var(--green-text); }
.preset-members { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(270px,100%),1fr)); gap: var(--space-2); margin-top: var(--space-3); }
.preset-member { display: flex; align-items: flex-start; gap: var(--space-2); border: 1px solid var(--separator); border-radius: var(--radius-sm); padding: var(--space-3); overflow-wrap: anywhere; }
.preset-member>span { min-width: 0; }
.preset-member small { display: block; font-size: var(--font-xs); color: var(--secondary); margin-top: var(--space-1); }
.unavailable-members { margin-top: var(--space-3); color: var(--orange-text); }
.unavailable-members label { display: flex; gap: var(--space-2); font-size: var(--font-sm); }
@media(max-width:700px) { .preset-controls label { width: 100%; } .preset-controls select,.preset-controls input { width: 100%; } }
</style>
