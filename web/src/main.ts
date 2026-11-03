import { createApp, nextTick } from 'vue';
import { createPinia } from 'pinia';
import { createRouter, createWebHashHistory } from 'vue-router';
import App from './App.vue';
import { initializeModes, setAutoRollNavigation } from './composables/modes';
import { initializeHud } from './composables/hud';
import { refreshWindowDragRegions } from './composables/windowDrag';
import HudOverlay from './pages/HudOverlay.vue';
import HudPanel from './pages/HudPanel.vue';
import HudToast from './pages/HudToast.vue';
import { useOddsStore } from './stores/odds';
import OddsPage from './pages/OddsPage.vue';
import AssistantPage from './pages/AssistantPage.vue';
import AutoRollPage from './pages/AutoRollPage.vue';
import AutopilotPage from './pages/AutopilotPage.vue';
import LivePage from './pages/LivePage.vue';
import LogsPage from './pages/LogsPage.vue';
import CompsPage from './pages/CompsPage.vue';
import CompPresetsPage from './pages/CompPresetsPage.vue';
import DatabasePage from './pages/DatabasePage.vue';
import GalleryPage from './pages/GalleryPage.vue';
import { bus, type BridgeEvent } from './bridge/events';
import { useAppStore } from './stores/app';
import { initTheme } from './composables/theme';
import { registerShortcuts } from './composables/shortcuts';
import { navigation } from './navigation';
import './styles/tokens.css';
import './styles/base.css';
import './styles/browser.css';
declare global {
    interface Window {
        __jcc_emit: (event: BridgeEvent) => void;
    }
}
// Register the bridge before creating stores or mounting any component.
let mounted = false;
const pending: BridgeEvent[] = [];
window.__jcc_emit = event => {
    if (mounted)
        bus.emit(event.type, event.payload);
    else
        pending.push(event);
};
initTheme();
const floating = /^#\/hud(?:-panel|-toast)?(?:\?|$)/.test(location.hash);
if (floating) {
    document.documentElement.classList.add('hud-window');
    document.documentElement.dataset.theme = 'dark';
}
const router = createRouter({ history: createWebHashHistory(), routes: [
        { path: '/', redirect: '/live' },
        ...navigation.map(item => ({ path: item.path, component: ({ '/live': LivePage, '/odds': OddsPage, '/logs': LogsPage, '/comps': CompsPage, '/comp-presets': CompPresetsPage, '/db': DatabasePage, '/assistant': AssistantPage, '/autoroll': AutoRollPage, '/autopilot': AutopilotPage } as const)[item.path as '/live' | '/odds' | '/logs' | '/comps' | '/comp-presets' | '/db' | '/assistant' | '/autoroll' | '/autopilot'], meta: { title: item.title } })),
        { path: '/gallery', component: GalleryPage, meta: { title: '组件展示' } },
        { path: '/hud', component: HudOverlay, meta: { floating: true } },
        { path: '/hud-toast', component: HudToast, meta: { floating: true } },
        { path: '/hud-panel', component: HudPanel, meta: { floating: true } },
        { path: '/:pathMatch(.*)*', redirect: '/live' },
    ] });
setAutoRollNavigation(() => { void router.push('/autoroll'); });
const app = createApp(App);
router.afterEach(() => { void nextTick().then(refreshWindowDragRegions); });
const pinia = createPinia();
app.use(pinia);
app.use(router);
if (floating)
    initializeHud();
else {
    useAppStore(pinia).initialize();
    useOddsStore(pinia).initialize();
    initializeModes();
}
const unregisterShortcuts = floating ? () => {
} : registerShortcuts(router);
if (import.meta.hot)
    import.meta.hot.dispose(unregisterShortcuts);
void router.isReady().then(async () => {
    app.mount('#app');
    await nextTick();
    mounted = true;
    for (const event of pending.splice(0))
        bus.emit(event.type, event.payload);
});
