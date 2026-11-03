import { ref } from 'vue';
import { call } from '../bridge/api';
import { bus } from '../bridge/events';
import { useAppStore } from '../stores/app';
import { confirm } from './confirm';
import { toast } from './toast';
import type { AutoRollOptions, ModeKey, Status } from '../types';
export const startingMode = ref<ModeKey | null>(null);
export const stoppingMode = ref(false);
export const resumingAutopilot = ref(false);
export const changingDryRun = ref(false);
let generation = 0;
const titles: Record<ModeKey, string> = { assistant: '辅助大脑', autoroll: '自动 D 牌', autopilot: '全自动托管' };
let openAutoRoll: (() => void) | undefined;
export function setAutoRollNavigation(open: () => void) { openAutoRoll = open; }
export async function startMode(key: ModeKey, options?: AutoRollOptions) {
    if (key === 'autoroll' && !options) { openAutoRoll?.(); return; }
    const store = useAppStore();
    if (store.status?.stopping || store.status?.mode?.key === key || startingMode.value || stoppingMode.value || changingDryRun.value)
        return;
    const attempt = ++generation;
    startingMode.value = key;
    try {
        if (!store.status)
            store.status = await call<Status>('status');
        const switching = !!store.status?.mode;
        if (switching && !await confirm({ title: `启动${titles[key]}`, message: `当前正在运行${store.status?.mode?.title}。继续将先停止当前模式，再启动${titles[key]}。`, confirmText: '停止并启动' })) return;
        const mode = store.status?.modes.find(item => item.key === key);
        if (!mode)
            throw new Error('运行模式尚未就绪');
        if (mode.acts && !store.status?.dry_run) {
            const accepted = await confirm({ title: mode.title,
                message: `即将真实操作游戏（${mode.title}）。随时可按「停止」、⌘. 或把鼠标甩到屏幕左上角中止。`,
                confirmText: '开始', destructive: true });
            if (!accepted)
                return;
        }
        if (attempt !== generation)
            return;
        // Backend status events own the displayed mode, including an immediate finish.
        await call<Status>('start_mode', key, options ?? null, switching);
    }
    catch {
        // call already reports backend failures. Keep UI event handlers rejection-free.
    }
    finally {
        startingMode.value = null;
    }
}
export async function stopMode() {
    if (stoppingMode.value)
        return;
    ++generation;
    stoppingMode.value = true;
    try {
        await call<Status>('stop_mode');
    }
    catch { /* call reports the error */
    }
    finally {
        stoppingMode.value = false;
    }
}
export async function setDryRun(on: boolean) {
    const store = useAppStore();
    if (!store.status || changingDryRun.value || startingMode.value || stoppingMode.value || on === store.status.dry_run)
        return;
    changingDryRun.value = true;
    try {
        const running = !!store.status.mode;
        if ((running || !on) && !await confirm({ title: running ? '切换全局演练' : '关闭演练模式',
            message: `${running ? '需要停止当前模式并重建执行器。' : ''}${on ? '将切换为只记录不点击。' : '自动 D 牌、托管和辅助侦察将真实点击游戏，请先完成标定并授予辅助功能权限。'}保存后不会自动重启。`,
            confirmText: running ? '停止并应用' : '关闭演练', destructive: !on })) return;
        await call<Status>('set_dry_run', on, running);
    }
    catch { /* call reports the error */
    }
    finally {
        changingDryRun.value = false;
    }
}
let subscribed = false;
export function initializeModes() {
    if (subscribed)
        return;
    subscribed = true;
    bus.on('mode_finished', payload => {
        const { mode, reason } = payload as {
            mode: ModeKey;
            reason: string;
        };
        const message = `${titles[mode] ?? mode} 结束：${reason}`;
        if (/^(异常|视觉模块错误)/.test(reason))
            toast.error(message);
        else
            toast.info(message);
    });
}

export async function resumeAutopilot() {
    const status = useAppStore().status;
    if (!status?.autopilot?.can_resume || !status.autopilot.token || status.stopping || resumingAutopilot.value) return;
    resumingAutopilot.value = true;
    try { await call<Status>('autopilot_resume', status.autopilot.token); }
    catch { /* bridge reports */ }
    finally { resumingAutopilot.value = false; }
}
