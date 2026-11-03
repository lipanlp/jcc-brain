import { ref, shallowRef } from 'vue';
import { call } from '../bridge/api';
import { toast } from './toast';
import type { LLMPreset } from '../types';
export const presets = shallowRef<LLMPreset[]>([]);
export const presetBusy = ref(false);
export async function loadPresets() {
    presets.value = await call<LLMPreset[]>('llm_presets');
}
export async function switchPreset(key: string) {
    if (presetBusy.value)
        return;
    presetBusy.value = true;
    try {
        await call('set_llm_preset', key);
        const preset = presets.value.find(item => item.key === key);
        toast.success(`已切换到 ${preset?.title ?? key}`);
    }
    catch { /* call reports */
    }
    finally {
        presetBusy.value = false;
    }
}
export async function switchPresetIndex(index: number) {
    try {
        await loadPresets();
        const preset = presets.value[index];
        if (preset)
            await switchPreset(preset.key);
    }
    catch { /* call reports */
    }
}
