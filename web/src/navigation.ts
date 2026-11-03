import { Activity, Percent, LayoutGrid, Database, ScrollText, Brain, Play, Settings2 } from 'lucide-vue-next';
export const navigation = [
    { path: '/live', group: '对局', title: '实时决策', icon: Activity, description: '对局状态与决策建议将在这里呈现。', step: 4 },
    { path: '/odds', group: '对局', title: '牌库概率', icon: Percent, description: '查看目标英雄的卡池与刷新概率。', step: 6 },
    { path: '/assistant', group: '功能控制', title: '辅助大脑', icon: Brain, description: '配置阵容方向、建议与侦察。', step: 4 },
    { path: '/autoroll', group: '功能控制', title: '自动 D 牌', icon: Play, description: '配置本次目标与预算。', step: 4 },
    { path: '/autopilot', group: '功能控制', title: '全自动托管', icon: Settings2, description: '按共享策略执行对局操作。', step: 4 },
    { path: '/comps', group: '资料', title: '阵容库', icon: LayoutGrid, description: '浏览阵容、装备与强化推荐。', step: 5 },
    { path: '/comp-presets', group: '资料', title: '预选阵容配置', icon: Settings2, description: '管理辅助大脑自动比较的阵容集合。', step: 5 },
    { path: '/db', group: '资料', title: '数据库', icon: Database, description: '查询英雄、羁绊和游戏资料。', step: 5 },
    { path: '/logs', group: '资料', title: '日志', icon: ScrollText, description: '运行记录与同步进度将在这里显示。', step: 4 },
];
