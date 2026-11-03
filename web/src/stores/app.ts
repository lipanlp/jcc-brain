import { computed, onScopeDispose, ref, shallowRef } from 'vue';
import { defineStore } from 'pinia';
import { bus } from '../bridge/events';
import { currentBuyState, currentFullAdvice } from '../composables/buyProjection';
import { toast } from '../composables/toast';
import { call, cancelWaiting } from '../bridge/api';
import type { Status, GameState, Advice, LogEntry, StoredLog } from '../types';
export const useAppStore = defineStore('app', () => {
    const status = shallowRef<Status | null>(null);
    const state = shallowRef<GameState | null>(null);
    const advice = shallowRef<Advice | null>(null);
    const syncMessage = ref('');
    const connected = ref(false);
    const startupPhase = ref<'loading' | 'retrying' | 'failed' | 'ready'>('loading');
    const startupSlow = ref(false);
    let attempts = 0;
    let inFlight = false;
    let generation = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let slowTimer: ReturnType<typeof setTimeout> | undefined;
    const subscriptions: (() => void)[] = [];
    function clearStartupTimers() {
        clearTimeout(retryTimer);
        clearTimeout(slowTimer);
        retryTimer = slowTimer = undefined;
    }
    function ready() {
        connected.value = true;
        startupPhase.value = 'ready';
        startupSlow.value = false;
        clearStartupTimers();
    }
    function acceptStatus(next: Status) {
        if (status.value && status.value.season.id !== next.season.id) {
            state.value = null;
            advice.value = null;
        }
        if (next.assistant_selection && advice.value?.selection?.revision
            && next.assistant_selection.revision !== advice.value.selection.revision) advice.value = null;
        status.value = next;
        ready();
    }
    async function requestInitialStatus() {
        if (!initialized || inFlight || connected.value) return;
        const ticket = generation;
        const before = statusRevision;
        inFlight = true;
        attempts++;
        startupPhase.value = attempts > 1 ? 'retrying' : 'loading';
        startupSlow.value = false;
        slowTimer = setTimeout(() => {
            if (initialized && ticket === generation && !connected.value) startupSlow.value = true;
        }, 1000);
        try {
            const value = await call<Status>('status', { silent: true });
            if (!initialized || ticket !== generation || connected.value) return;
            if (!value?.season?.id) throw new Error('本地数据状态无效');
            if (before === statusRevision || !status.value) acceptStatus(value);
        }
        catch {
            if (!initialized || ticket !== generation || connected.value) return;
            if (attempts < 3) {
                startupPhase.value = 'retrying';
                retryTimer = setTimeout(() => { retryTimer = undefined; void requestInitialStatus(); }, attempts * 500);
            }
            else startupPhase.value = 'failed';
        }
        finally {
            inFlight = false;
            if (ticket === generation) {
                clearTimeout(slowTimer);
                slowTimer = undefined;
            }
            else if (initialized && !connected.value) void requestInitialStatus();
        }
    }
    function retryStartup() {
        if (!initialized || inFlight || connected.value || startupPhase.value !== 'failed') return;
        attempts = 0;
        clearStartupTimers();
        void requestInitialStatus();
    }
    function dispose() {
        initialized = false;
        generation++;
        clearStartupTimers();
        cancelWaiting();
        for (const unsubscribe of subscriptions.splice(0)) unsubscribe();
        window.removeEventListener('pagehide', dispose);
    }
    onScopeDispose(dispose);
    if (import.meta.hot) import.meta.hot.dispose(dispose);
    const capacity = 3000;
    const ring = new Array<StoredLog>(capacity);
    let sequence = 0;
    let head = 0;
    let length = 0;
    const revision = ref(0);
    const logs = computed(() => {
        void revision.value;
        return Array.from({ length }, (_, i) => ring[(head + i) % capacity]!);
    });
    let initialized = false;
    let statusRevision = 0;
    function initialize() {
        if (initialized)
            return;
        initialized = true;
        attempts = 0;
        window.addEventListener('pagehide', dispose);
        function listen(type: string, handler: (payload: unknown) => void) { subscriptions.push(bus.on(type, handler)); }
        listen('status', payload => {
            const next = payload as Status;
            if (!next?.season?.id) return;
            statusRevision++;
            acceptStatus(next);
        });
        listen('lock_changed', payload => {
            const locks = payload as Pick<Status, 'lock' | 'default_lock'>;
            statusRevision++;
            if (status.value)
                status.value = { ...status.value, ...locks };
        });
        listen('watch_changed', payload => {
            statusRevision++;
            if (status.value)
                status.value = { ...status.value, ...(payload as Pick<Status, 'watch'>) };
        });
        listen('state', payload => {
            if (currentBuyState((payload as GameState).buy_source)) state.value = payload as GameState;
        });
        listen('advice', payload => {
            const next = payload as Advice;
            if (next.selection?.revision && status.value?.assistant_selection
                && next.selection.revision !== status.value.assistant_selection.revision) return;
            if (currentFullAdvice(next, advice.value)) advice.value = next;
        });
        listen('log', payload => {
            for (const entry of payload as LogEntry[]) {
                ring[(head + length) % capacity] = { ...entry, id: ++sequence };
                if (length < capacity)
                    length++;
                else
                    head = (head + 1) % capacity;
            }
            revision.value++;
        });
        listen('sync_progress', payload => {
            syncMessage.value = (payload as {
                message: string;
            }).message;
        });
        listen('sync_done', payload => {
            const value = payload as {
                ok: boolean;
                message: string;
            };
            (value.ok ? toast.success : toast.error)(value.message);
        });
        if (status.value) ready();
        else void requestInitialStatus();
    }
    function clearLogs() {
        ring.fill(undefined!);
        head = 0;
        length = 0;
        revision.value++;
    }
    return { status, state, advice, logs, syncMessage, connected, startupPhase, startupSlow, initialize, retryStartup, dispose, clearLogs };
});
