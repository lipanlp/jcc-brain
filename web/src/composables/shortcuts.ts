import { openSettings } from './settings';
import { openLabels } from './tools';
import { switchPresetIndex } from './ai';
import { startMode, stopMode } from './modes';
import { toggleHud } from './hud';
import type { ModeKey } from '../types';
import type { Router } from 'vue-router';
export function registerShortcuts(router: Router): () => void {
    const paths = ['/live', '/odds', '/comps', '/db', '/logs'];
    const listener = (event: KeyboardEvent) => {
        if (event.isComposing || event.repeat)
            return;
        const editing = event.target instanceof Element ? event.target.closest('input, textarea, select, [contenteditable="true"]') : null;
        if (editing && /^[a-z]$/i.test(event.key) && !event.metaKey && !event.ctrlKey && !event.altKey)
            return;
        if (event.metaKey && event.shiftKey && !event.altKey && !event.ctrlKey && /^Digit[1-3]$/.test(event.code)) {
            event.preventDefault();
            void switchPresetIndex(Number(event.code.slice(-1)) - 1);
            return;
        }
        if (event.metaKey && !event.altKey && !event.ctrlKey && !event.shiftKey) {
            if (event.code === 'KeyH' || event.code === 'KeyJ') {
                event.preventDefault();
                void toggleHud(event.code === 'KeyJ');
                return;
            }
            if (event.code === 'Comma' || event.key === ',') {
                event.preventDefault();
                openSettings();
                return;
            }
            if (event.code === 'KeyL' || event.key.toLowerCase() === 'l') {
                event.preventDefault();
                void openLabels();
                return;
            }
            const modes: Record<string, ModeKey> = { Digit1: 'assistant', Digit2: 'autoroll', Digit3: 'autopilot' };
            const mode = modes[event.code] ?? ({ '1': 'assistant', '2': 'autoroll', '3': 'autopilot' } as Record<string, ModeKey>)[event.key];
            if (mode) {
                event.preventDefault();
                void startMode(mode);
                return;
            }
            if (event.key === '.' || event.code === 'Period') {
                event.preventDefault();
                void stopMode();
                return;
            }
        }
        // code remains Digit1 on macOS even when Option changes the printed character.
        const index = /^Digit[1-5]$/.test(event.code) ? Number(event.code.slice(-1)) - 1 : Number(event.key) - 1;
        if (event.metaKey && event.altKey && !event.ctrlKey && !event.shiftKey && index >= 0 && index < paths.length) {
            event.preventDefault();
            void router.push(paths[index]!);
        }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
}
