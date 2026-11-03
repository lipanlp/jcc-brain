import { computed, ref, shallowRef, watch } from 'vue';
import { defineStore } from 'pinia';
import { useAppStore } from './app';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { toast } from '../composables/toast';
import type { OddsInput, OddsResult, SimulationResult } from '../types';
export const useOddsStore = defineStore('odds', () => {
    const app = useAppStore();
    const rows = ref<OddsInput[]>([]);
    const level = ref(8), gold = ref(50), keep = ref(0), follow = ref(true);
    const result = shallowRef<OddsResult | null>(null);
    const simulation = shallowRef<SimulationResult | null>(null);
    const calculating = ref(false), simulating = ref(false);
    const simulatedSignature = ref('');
    const dataRevision = ref(0);
    const signature = computed(() => JSON.stringify([dataRevision.value, app.status?.season.id, rows.value, level.value, gold.value, keep.value, follow.value, follow.value ? app.state : null]));
    const stale = computed(() => !!simulation.value && simulatedSignature.value !== signature.value);
    let initialized = false, ticket = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    function initialize() {
        if (initialized)
            return;
        initialized = true;
        bus.on('data_changed', () => {
            dataRevision.value++;
            schedule();
        });
        watch([() => app.state, follow], ([state]) => {
            if (!follow.value || !state)
                return;
            if (state.level != null)
                level.value = Math.max(1, Math.min(10, state.level));
            if (state.gold != null)
                gold.value = Math.max(0, Math.min(200, state.gold));
        }, { immediate: true, flush: 'sync' });
        watch([rows, level, follow, () => app.state], schedule, { deep: true, immediate: true, flush: 'sync' });
    }
    function schedule() {
        const current = ++ticket;
        clearTimeout(timer);
        if (!rows.value.length) {
            result.value = null;
            calculating.value = false;
            return;
        }
        calculating.value = true;
        timer = setTimeout(async () => {
            try {
                const value = await call<OddsResult>('odds_calc', rows.value.map(row => ({ ...row })), level.value, follow.value);
                if (current === ticket)
                    result.value = value;
            }
            catch { /* call presents errors */
            }
            finally {
                if (current === ticket)
                    calculating.value = false;
            }
        }, 150);
    }
    function setLevel(value: number) {
        follow.value = false;
        level.value = value;
    }
    function setGold(value: number) {
        follow.value = false;
        gold.value = value;
    }
    function add(name: string, need: number) {
        const existing = rows.value.find(row => row.name === name);
        if (existing)
            existing.need = need;
        else
            rows.value.push({ name, need, others: 0 });
    }
    function update(name: string, field: 'need' | 'others', value: number) {
        const row = rows.value.find(row => row.name === name);
        if (row)
            row[field] = value;
    }
    function remove(name: string) {
        rows.value = rows.value.filter(row => row.name !== name);
    }
    function clear() {
        rows.value = [];
        simulation.value = null;
    }
    async function importTarget() {
        try {
            const value = await call<{
                comp_name: string | null;
                rows: OddsInput[];
            }>('odds_import_target');
            rows.value = value.rows;
            if (value.comp_name)
                toast.success(`已导入阵容 ${value.comp_name}`);
            else
                toast.info('还没有可导入的目标阵容');
        }
        catch { /* call presents errors */
        }
    }
    async function simulate() {
        if (simulating.value)
            return;
        const before = signature.value;
        simulating.value = true;
        try {
            const value = await call<SimulationResult | null>('odds_simulate', rows.value.map(row => ({ ...row })), level.value, gold.value, keep.value, follow.value, 3000);
            simulation.value = value;
            simulatedSignature.value = before;
            if (!value)
                toast.info('没有还差的目标卡');
        }
        catch { /* call presents errors */
        }
        finally {
            simulating.value = false;
        }
    }
    return { rows, level, gold, keep, follow, result, simulation, calculating, simulating, stale, initialize, setLevel, setGold, add, update, remove, clear, importTarget, simulate };
});
