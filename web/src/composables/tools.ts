import { ref, shallowRef } from 'vue';
import { call } from '../bridge/api';
import { alert } from './confirm';
import { toast } from './toast';
import type { LabelUnitsInfo } from '../types';
export const labelInfo = shallowRef<LabelUnitsInfo | null>(null);
export const calibrationOpen = ref(false);
export const statsOpen = ref(false);
let labelTicket = 0;
export async function openLabels() {
    const ticket = ++labelTicket;
    try {
        const value = await call<LabelUnitsInfo>('label_units_info');
        if (ticket === labelTicket)
            labelInfo.value = value;
    }
    catch { /* call reports */
    }
}
export async function checkVision() {
    try {
        const value = await call<{
            summary: string;
            shop: (string | null)[];
        }>('check_vision');
        await alert({ title: '视觉模块', message: `读取成功：${value.summary}\n商店：${value.shop.map(name => name ?? '空').join('、')}` });
    }
    catch { /* call reports */
    }
}
export async function checkExecutor() {
    try {
        const value = await call<{
            window: string;
            calibrated: boolean;
            problems: string[];
        }>('check_executor');
        await alert({ title: '操作权限', message: [value.window, value.calibrated ? '棋盘已标定' : '棋盘未标定（工具 → 坐标标定）', value.problems.length ? value.problems.join('\n') : '辅助功能权限：已授权'].join('\n') });
    }
    catch { /* call reports */
    }
}
export async function chooseReplay() {
    try {
        const path = await call<string | null>('choose_replay');
        if (path)
            toast.success(`已打开回放：${path.split(/[\\/]/).pop()}`);
    }
    catch { /* call reports */
    }
}
export async function about() {
    try {
        const info = await call<{
            version: string;
        }>('app_info');
        await alert({ title: '金铲铲大脑', message: `金铲铲大脑 ${info.version}\n\n视觉：game-eye（本地 OCR + 单位识别）\n数据：dataj.cc（金铲铲大数据）\n决策：规则引擎 + 可选大模型\n\n急停：工具栏「停止」/ ⌘. / 鼠标甩到屏幕左上角` });
    }
    catch { /* call reports */
    }
}
