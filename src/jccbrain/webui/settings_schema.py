"""设置表单元数据、类型转换与校验。"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import fields

from jccbrain.engine.llm import PRESETS, apply_preset
from jccbrain.data.db import enabled_comp_sources

LABELS = {
    'unit_learning_mode': ('棋子身份学习', ['off','record','auto']),
    'unit_learning_idle_s': ('学习前输入空闲秒数', None),
    'unit_learning_interval_s': ('自动查看最小间隔秒数', None),
    'unit_learning_min_timer': ('学习开始最少剩余秒数', None),
    "economy_preference": ("运营偏好", ["adaptive", "health", "economy"]),
    "schedule": ("侦察调度", ["key_rounds", "every_round"]),
    "augment_selection": ("海克斯选择", ["manual", "auto"]),
    "rounds": ("自动侦察回合（逗号分隔）", None),
    "min_timer": ("开始侦察最少剩余秒数", None), "max_targets": ("最多侦察人数", None),
    "cut_threshold": ("切换画面差异阈值", None), "stable_threshold": ("稳定画面差异阈值", None),
    "switch_timeout_ms": ("单个玩家切换超时 ms", None), "skip_recent_rounds": ("跳过最近N回合已侦察的对手（0=不跳过）", None),
    "min_dwell_ms": ("切换后最少等待 ms", None), "stable_frames": ("三区连续稳定帧数", None),
    "backend": ("视觉来源", ["pyeye", "replay"]),
    "poll_interval_ms": ("读取间隔 ms", None), "replay_file": ("回放（jsonl 或截图目录）", None),
    "replay_interval_ms": ("回放间隔 ms", None),
    "base_url": ("接口地址", None), "game_version": ("dataj 统计版本（空=最新）", None),
    "source_official": ("阵容来源：官网推荐", None), "source_ingame": ("阵容来源：微社区", None),
    "auto_sync": ("打开时自动同步缺失/有新版本的赛季", None), "source_dataj": ("阵容来源：dataj.cc（阵容统计）", None),
    "official_version": ("官方资料库版本（空=最新）", None),
    "fetch_comp_details": ("同步阵容详情", None), "request_interval_s": ("请求间隔 s", None),
    "enabled": ("启用", None), "preset": ("档位", ["fast", "balanced", "deep", "custom"]), "provider": ("协议", ["openai", "anthropic"]), "api_key": ("API Key", None),
    "model": ("模型", None), "timeout_s": ("超时 s", None), "min_interval_s": ("最小间隔 s", None),
    "max_tokens": ("最大输出 token（含思考）", None),
    "reasoning_effort": ("思考强度", ["", "low", "high", "max"]),
    "reasoning_effort_augment": ("海克斯回合思考强度", ["", "low", "high", "max"]),
    "json_mode": ("JSON 输出模式", None),
    "dry_run": ("演练模式（只记录不点击）", None), "window_owner": ("游戏窗口进程名", None),
    "click_delay_ms": ("操作间隔 ms", None), "drag_steps": ("拖动步数", None), "jitter_px": ("随机偏移 px", None),
    "max_actions_per_sec": ("每秒最多操作", None), "failsafe_corner": ("鼠标甩左上角急停", None),
    "keep_gold": ("保留金币", None), "target_star": ("默认目标星级", None),
    "max_rolls": ("最多刷新次数", None), "buy_pairs_only": ("只买能成对的", None),
    "read_interval_ms": ("D牌读画面间隔(ms)", None),
    "show_names": ("显示棋子名字", None), "show_traits": ("显示羁绊", None), "show_odds_panel": ("显示概率面板", None),
    "font_size": ("字号", None), "opacity": ("不透明度", None),
    "show_positions": ("显示站位指引", None),
    "default_lock_comp_id": ("默认锁定阵容（空=自动推荐）", None),
}

HIDDEN = {'season_id'}

SECTIONS = [('vision', '视觉'), ('data', '数据'), ('llm', '大模型'), ('actuator', '操作'), ('autoroll', '自动D牌'), ('hud', 'HUD'), ('assistant', '辅助大脑'), ('scout', '侦察'), ('autopilot', '全自动托管')]

NOTES = {
    'vision': '棋子身份学习：off关闭，record仅记录待分析，auto在空闲时查看并学习；仅辅助大脑运行时启用，继承顶部全局演练（或托管页的执行方式）。',
    'data': '来源开关即时控制阵容浏览和推荐。官网或微社区任一开启，同步都会保存两个渠道的完整官方快照，'
            '基础资料以官方为准；微社区热度不代表胜率。微社区目前仅支持自然之力。',
    'scout': '自动侦察默认关闭；立即侦察可手动触发。使用操作分组的演练开关，演练只记录点击计划。',
    'llm': 'OpenAI 兼容：base_url 填到 /v1（智谱填 https://open.bigmodel.cn/api/paas/v4）；\n'
           'Anthropic：base_url 填 https://api.anthropic.com',
    'actuator': '执行方式在顶部全局演练或托管页调整。真实操作前请先完成标定，并授予「辅助功能」权限',
}


FEATURE_FIELDS = {
    'assistant': {'default_lock_comp_id', 'economy_preference'},
    'scout': {'enabled', 'schedule'},
    'autoroll': {'keep_gold', 'max_rolls'},
    'hud': {'show_names', 'show_traits', 'show_odds_panel', 'font_size', 'opacity', 'show_positions'},
    'autopilot': {'augment_selection'},
}


def schema(settings, db=None, *, full=False) -> list[dict]:
    sections = []
    for key, title in SECTIONS:
        if not full and key in FEATURE_FIELDS:
            continue
        obj = getattr(settings, key)
        editors = []
        for field in fields(obj):
            name = field.name
            if name in HIDDEN or (not full and key == 'actuator' and name == 'dry_run'):
                continue
            value = getattr(obj, name)
            label, choices = LABELS.get(name, (name, None))
            kind = ('rounds' if name == 'rounds' else 'secret' if name == 'api_key' else 'file' if name == 'replay_file' else 'choice' if choices else
                    'bool' if isinstance(value, bool) else 'int' if isinstance(value, int) else
                    'float' if isinstance(value, float) else 'str')
            if name == 'default_lock_comp_id':
                kind = 'comp'
                choices = [{'value': '', 'label': '不锁定（自动推荐）'}] + [
                    {'value': comp.id, 'label': f'[{comp.tier or "?"}] {comp.name}'}
                    for comp in (db.comps_catalog(enabled_sources=enabled_comp_sources(settings.data)) if db is not None else [])]
            editors.append({'key': name, 'label': label, 'type': kind, 'choices': choices, 'value': value})
        sections.append({'key': key, 'title': title, 'note': NOTES.get(key, ''), 'fields': editors})
    return sections


def _convert(value, field):
    label, kind = field['label'], field['type']
    try:
        if kind == 'rounds':
            values = re.split(r'[,，\s]+', value.strip()) if isinstance(value, str) else value
            if not isinstance(values, list) or any(not isinstance(v, str) or not re.fullmatch(r'[1-9]\d?-[1-7]', v) for v in values if v):
                raise ValueError('回合格式应为3-2、4-1，用逗号分隔')
            return list(dict.fromkeys(v for v in values if v))
        if kind == 'bool':
            if isinstance(value, bool):
                return value
            if isinstance(value, (str, int)):
                text = str(value).lower()
                if text in ('true', '1', 'on'):
                    return True
                if text in ('false', '0', 'off'):
                    return False
            raise ValueError('必须是布尔值')
        if kind in ('int', 'float'):
            if isinstance(value, bool):
                raise ValueError('必须是数字')
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('必须是有限数字')
            if kind == 'int':
                if not number.is_integer():
                    raise ValueError('必须是整数')
                if not -1 <= number <= 100000:
                    raise ValueError('必须在 -1 到 100000 之间')
                return int(number)
            if not 0 <= number <= 10000:
                raise ValueError('必须在 0 到 10000 之间')
            return number
        if not isinstance(value, str):
            raise ValueError('必须是文本')
        if kind == 'choice' and value not in field['choices']:
            raise ValueError('必须选择：' + ' / '.join(field['choices']))
        if kind == 'comp' and value not in {choice['value'] for choice in field['choices']}:
            raise ValueError('找不到阵容')
        return value
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'「{label}」{exc}') from exc


def _prepare(settings, values, db=None):
    """先在副本上完整校验，错误时不更改设置也不写文件。"""
    staged = deepcopy(settings)
    lookup = {section['key']: {field['key']: field for field in section['fields']} for section in schema(settings, db, full=True)}
    if not isinstance(values, dict):
        raise ValueError('设置必须是分组对象')
    before = (settings.llm.preset, settings.llm.model, settings.llm.reasoning_effort, settings.llm.reasoning_effort_augment)
    for section, edits in values.items():
        if section not in lookup or not isinstance(edits, dict):
            raise ValueError(f'未知的设置分组：{section}')
        for key, value in edits.items():
            if key not in lookup[section]:
                raise ValueError(f'「{LABELS.get(key, (key,))[0]}」不是可编辑的设置项')
            field = lookup[section][key]
            # 来源和默认阵容可以同次保存，阵容有效性按完整新设置校验。
            if field['type'] == 'comp':
                field = {**field, 'type': 'str'}
            setattr(getattr(staged, section), key, _convert(value, field))
    for section, name, low, high in [('autoroll', 'keep_gold', 0, 100), ('autoroll', 'max_rolls', 1, 200),
                                      ('autoroll', 'target_star', 1, 3), ('hud', 'font_size', 8, 32),
                                      ('hud', 'opacity', .1, 1)]:
        if not low <= getattr(getattr(staged, section), name) <= high:
            raise ValueError(f'「{LABELS[name][0]}」必须在{low}到{high}之间')
    staged.vision.validate_unit_learning()
    scout = staged.scout
    for name, low, high in [('min_timer', 5, 120), ('max_targets', 1, 7), ('switch_timeout_ms', 50, 10000),
                            ('min_dwell_ms', 0, 10000), ('stable_frames', 1, 30),
                            ('skip_recent_rounds', 0, 50), ('cut_threshold', .01, 255), ('stable_threshold', .01, 255)]:
        if not low <= getattr(scout, name) <= high:
            raise ValueError(f'「{LABELS[name][0]}」必须在{low}到{high}之间')
    if scout.min_dwell_ms >= scout.switch_timeout_ms:
        raise ValueError('「切换后最少等待 ms」必须小于切换超时')
    default = staged.assistant.default_lock_comp_id
    if db is not None and default and db.comp(default, enabled_sources=enabled_comp_sources(staged.data)) is None:
        if default != settings.assistant.default_lock_comp_id:
            raise ValueError('找不到阵容（来源可能已关闭）')
    llm = staged.llm
    if llm.preset != before[0] and llm.preset in PRESETS:
        apply_preset(llm, llm.preset)
    elif (llm.model, llm.reasoning_effort, llm.reasoning_effort_augment) != before[1:]:
        llm.preset = 'custom'
    return staged


def apply(settings, values, db=None) -> None:
    staged = _prepare(settings, values, db)
    staged.save()
    for section, _ in SECTIONS:
        obj = getattr(settings, section)
        for field in fields(obj):
            setattr(obj, field.name, getattr(getattr(staged, section), field.name))
