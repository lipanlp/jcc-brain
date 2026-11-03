import { ref, watch } from 'vue';
export type Theme = 'system' | 'light' | 'dark';
const theme = ref<Theme>('system');
export function initTheme() {
    try {
        const saved = localStorage.getItem('jcc-theme');
        if (saved === 'light' || saved === 'dark')
            theme.value = saved;
    }
    catch { /* Storage may be disabled in embedded browsers. */
    }
    apply(theme.value);
}
function apply(value: Theme) {
    if (value === 'system')
        delete document.documentElement.dataset.theme;
    else
        document.documentElement.dataset.theme = value;
}
watch(theme, value => {
    apply(value);
    try {
        localStorage.setItem('jcc-theme', value);
    }
    catch { /* The in-memory preference still works. */
    }
}, { flush: 'sync' });
export function useTheme() {
    return theme;
}
