import { computed, ref } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { toast } from './toast';
import type { AutoRollDefaults, AutoRollOptions, FeatureSettings, FeatureSettingsResult, HeroOption, SelectOption, SettingsValues, Status } from '../types';

// 草稿跨路由共享；已保存值与选项刷新均不得替换用户的未保存编辑。
const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
export const featureDraft = ref<FeatureSettings | null>(null);
export const featureBaseline = ref<FeatureSettings | null>(null);
export const compOptions = ref<SelectOption[]>([]);
export const scoutRounds = ref<string[]>([]);
export const economyLabels = { adaptive: '自适应', health: '保血优先', economy: '经济优先' } as const;
export const featureLoading = ref(false);
export const featureSaving = ref(false);
export const featureMetadataReady = ref(false);
let loading: Promise<void> | undefined;
let refreshFeaturesAgain = false;
let metadataEpoch = 0;
let savedRevision = 0;
let savedValues: FeatureSettings | undefined;
const compLabels = new Map<string, string>();

type Section = keyof FeatureSettings;
let pendingSubmission: { snapshot?: FeatureSettings; sections: Section[]; changedFields: Set<string> } | undefined;
function mergeSaved(next: FeatureSettings, submitted?: FeatureSettings, sections: Section[] = [], preserveSections: Section[] = []) {
    if (!featureDraft.value || !featureBaseline.value) {
        featureDraft.value = clone(next);
        featureBaseline.value = clone(next);
        return;
    }
    for (const section of Object.keys(next) as Section[]) {
        const draft = featureDraft.value[section] as Record<string, unknown>;
        const baseline = featureBaseline.value[section] as Record<string, unknown>;
        const expected = (submitted && sections.includes(section) ? submitted[section] : baseline) as Record<string, unknown>;
        for (const [key, value] of Object.entries(next[section])) {
            if (!preserveSections.includes(section) && draft[key] === expected[key]) draft[key] = value;
            baseline[key] = value;
        }
    }
}
function acceptSaved(next: FeatureSettings, confirmed = false) {
    const previous = savedValues ?? featureBaseline.value;
    if (pendingSubmission && previous) {
        for (const section of Object.keys(next) as Section[]) {
            for (const [key, value] of Object.entries(next[section])) {
                if (value !== (previous[section] as Record<string, unknown>)[key]) pendingSubmission.changedFields.add(`${section}.${key}`);
            }
        }
    }
    savedValues = clone(next);
    savedRevision++;
    // 未打开功能页时只缓存保存值，不因此触发加载。
    // 普通 status 不是保存确认：待提交分组只更新基线，保留用户当前草稿。
    if (featureDraft.value) {
        if (confirmed) mergeSaved(next, pendingSubmission?.snapshot, pendingSubmission?.sections);
        else mergeSaved(next, undefined, [], pendingSubmission?.sections);
    }
}
function completeSubmission(result: FeatureSettings) {
    const pending = pendingSubmission;
    if (!pending) { acceptSaved(result); return; }
    const next = clone(result);
    // 按字段保留期间实际变化的共享保存值；重复的旧 status 不会遮住成功响应。
    if (savedValues) {
        for (const section of Object.keys(next) as Section[]) {
            for (const key of Object.keys(next[section])) {
                if (pending.changedFields.has(`${section}.${key}`)) (next[section] as Record<string, unknown>)[key] = (savedValues[section] as Record<string, unknown>)[key];
            }
        }
    }
    // 明确返回后，只更新仍与提交快照相同的草稿，保留等待期间的再次编辑。
    acceptSaved(next, true);
}
bus.on('status', payload => {
    const status = payload as Status;
    if (status.feature_settings) acceptSaved(status.feature_settings);
    else if (featureBaseline.value) {
        // 兼容只提供原状态字段的桥接端；新后端完整同步 HUD/侦察等保存值。
        const next = clone(savedValues ?? featureBaseline.value);
        next.assistant.default_lock_comp_id = status.default_lock?.id ?? '';
        if (status.economy_preference) next.assistant.economy_preference = status.economy_preference;
        if (status.augment_selection) next.autopilot.augment_selection = status.augment_selection;
        acceptSaved(next);
    }
});
export const defaultLockInvalid = computed(() => {
    const id = featureDraft.value?.assistant.default_lock_comp_id;
    return !!id && featureMetadataReady.value && !compOptions.value.some(c => c.value === id);
});
export const defaultLockOptions = computed(() => {
    const id = featureDraft.value?.assistant.default_lock_comp_id;
    return defaultLockInvalid.value && id ? [...compOptions.value, { value: id, label: `当前不可用：${compLabels.get(id) ?? id}` }] : compOptions.value;
});
export function loadFeatures(refresh = false): Promise<void> {
    if (loading) { if (refresh) refreshFeaturesAgain = true; return loading; }
    if (featureDraft.value && !refresh) return Promise.resolve();
    featureLoading.value = true;
    loading = (async () => {
        try {
            do {
                refreshFeaturesAgain = false;
                const epoch = metadataEpoch, revision = savedRevision;
                const result = await call<FeatureSettingsResult>('feature_settings');
                if (epoch !== metadataEpoch) { refreshFeaturesAgain = true; continue; }
                compOptions.value = result.comp_options;
                for (const c of result.comp_options) compLabels.set(c.value, c.label);
                scoutRounds.value = result.scout_rounds;
                featureMetadataReady.value = true;
                // 请求期间收到更新的 status 时，以共享已保存状态为准。
                if (revision === savedRevision) acceptSaved(result.values);
                if (!featureDraft.value) mergeSaved(savedValues ?? result.values);
            } while (refreshFeaturesAgain);
        }
        catch { /* bridge reports; keep the draft for retry */ }
        finally { featureLoading.value = false; loading = undefined; }
    })();
    return loading;
}
export async function saveFeatures(sections: Section[]) {
    if (!featureDraft.value || !featureBaseline.value || featureSaving.value) return;
    if (sections.includes('assistant') && (!featureMetadataReady.value || defaultLockInvalid.value)) {
        toast.error('请等待阵容资料更新，并重新选择有效的长期默认阵容或自动推荐');
        return;
    }
    const submitted = clone(featureDraft.value);
    const edits: SettingsValues = {};
    for (const section of sections) {
        for (const [key, value] of Object.entries(submitted[section])) {
            if (value !== (featureBaseline.value[section] as Record<string, unknown>)[key]) (edits[section] ??= {})[key] = value;
        }
    }
    featureSaving.value = true;
    pendingSubmission = { snapshot: submitted, sections, changedFields: new Set() };
    try {
        const result = await call<FeatureSettingsResult>('save_feature_settings', edits);
        completeSubmission(result.values);
        toast.success('设置已保存，不会自动启动');
    }
    catch { /* bridge reports; preserve all edits */ }
    finally { pendingSubmission = undefined; featureSaving.value = false; }
}

export const rollDraft = ref<AutoRollDefaults | null>(null);
export const rollLoading = ref(false);
export const rollMetadataReady = ref(false);
let rollLoad: Promise<boolean> | undefined;
let refreshRollAgain = false;
export const unavailableRollTargets = computed(() => {
    const names = new Set(rollDraft.value?.all_heroes.map(h => h.name));
    return rollDraft.value?.heroes.filter(h => !names.has(h.name)).map(h => h.name) ?? [];
});
export const invalidRollSelection = computed(() => rollDraft.value?.heroes.some(h => h.checked && unavailableRollTargets.value.includes(h.name)) ?? false);

function readRoll(importTargets = false, refresh = false): Promise<boolean> {
    if (rollLoad) { if (refresh) refreshRollAgain = true; return rollLoad; }
    if (rollDraft.value && !refresh && !importTargets) return Promise.resolve(false);
    rollLoading.value = true;
    rollLoad = (async () => {
        let imported = false, replace = importTargets;
        try {
            do {
                refreshRollAgain = false;
                const epoch = metadataEpoch;
                const needDefaults = !rollDraft.value || replace;
                const result = needDefaults ? await call<AutoRollDefaults>('autoroll_defaults') : await call<HeroOption[]>('hero_options');
                if (epoch !== metadataEpoch) {
                    // 显式导入期间环境已变化：丢弃旧目标，只刷新资料，等待用户重新导入。
                    replace = false;
                    refreshRollAgain = true;
                    continue;
                }
                if (!rollDraft.value) rollDraft.value = result as AutoRollDefaults;
                else {
                    rollDraft.value.all_heroes = needDefaults ? (result as AutoRollDefaults).all_heroes : result as HeroOption[];
                    if (replace) {
                        rollDraft.value.heroes = (result as AutoRollDefaults).heroes;
                        rollDraft.value.comp_name = (result as AutoRollDefaults).comp_name;
                        imported = true;
                    }
                }
                rollMetadataReady.value = true;
                replace = false;
            } while (refreshRollAgain);
        }
        catch { /* bridge reports; preserve the task draft */ }
        finally { rollLoading.value = false; rollLoad = undefined; }
        return imported;
    })();
    return rollLoad;
}
export function loadRollDraft(refresh = false) { return readRoll(false, refresh); }
export async function importRollTargets() {
    if (rollLoading.value) return;
    const epoch = metadataEpoch;
    if (await readRoll(true)) toast.info(rollDraft.value?.heroes.length ? '已从当前阵容导入目标' : '当前没有阵容目标，可手动添加英雄');
    else if (epoch !== metadataEpoch) toast.info('资料已变化，已保留原目标，请重新导入');
}
export function rollOptions(): AutoRollOptions | undefined {
    const draft = rollDraft.value;
    if (!draft || !rollMetadataReady.value || invalidRollSelection.value) return;
    return { targets: draft.heroes.filter(hero => hero.checked).map(({ name, star }) => ({ name, star })), keep: draft.keep, max_rolls: draft.max_rolls };
}
export async function saveRollDefaults() {
    if (!rollDraft.value || featureSaving.value) return;
    const values = { keep_gold: rollDraft.value.keep, max_rolls: rollDraft.value.max_rolls };
    const submitted = featureDraft.value ? clone(featureDraft.value) : undefined;
    featureSaving.value = true;
    pendingSubmission = { snapshot: submitted, sections: ['autoroll'], changedFields: new Set() };
    try {
        const result = await call<FeatureSettingsResult>('save_feature_settings', { autoroll: values });
        completeSubmission(result.values);
        toast.success('预算与刷新默认已保存；英雄目标仅用于本次任务');
    }
    catch { /* bridge reports */ }
    finally { pendingSubmission = undefined; featureSaving.value = false; }
}

bus.on('data_changed', () => {
    metadataEpoch++;
    if (featureDraft.value || loading) { featureMetadataReady.value = false; void loadFeatures(true); }
    if (rollDraft.value || rollLoad) { rollMetadataReady.value = false; void loadRollDraft(true); }
});
