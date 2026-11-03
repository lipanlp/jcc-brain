import { ref, shallowRef } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { currentBuyHud, currentFullAdvice } from './buyProjection';
import { useAppStore } from '../stores/app';
import type { Advice, Hud, Status } from '../types';
export const hudFrame = shallowRef<Hud | null>(null);
export const hudAdvice = shallowRef<Advice | null>(null);
export const hudBusy = ref(false);
export const panelBusy = ref(false);
let initialized = false;
export function initializeHud() {
    if (initialized)
        return;
    initialized = true;
    bus.on('hud', payload => {
        if (!currentBuyHud(payload as Hud)) return;
        hudFrame.value = payload as Hud;
        if (hudFrame.value.mode === 'idle')
            hudAdvice.value = null;
    });
    bus.on('advice', payload => {
        if (currentFullAdvice(payload as Advice, hudAdvice.value)) hudAdvice.value = payload as Advice;
    });
}
export async function toggleHud(panel = false) {
    const busy = panel ? panelBusy : hudBusy;
    const store = useAppStore();
    if (busy.value || !store.status)
        return;
    busy.value = true;
    try {
        await call<Status>(panel ? 'hud_panel_toggle' : 'hud_toggle', !(panel ? store.status.hud.panel : store.status.hud.overlay));
    }
    catch { /* call displays the backend error; status remains authoritative. */
    }
    finally {
        busy.value = false;
    }
}
