"""辅助大脑预选集合：赛季隔离、严格写入、损坏时关闭自动比较。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from uuid import uuid4

ALL = 'all'


def _id(value, *, preset=False):
    pattern = r'p-[0-9a-f]{32}' if preset else r'[A-Za-z0-9_:.\-]{1,120}'
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError('配置id无效' if preset else '阵容id无效')
    return value


def _name(value):
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= 40
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('配置名称须为1到40个字符，不能全为空格')
    return value.strip()


def read(raw, season):
    """不存在的赛季默认全部；已存在但损坏的记录绝不扩大到全部。"""
    try:
        if not isinstance(raw, dict):
            raise ValueError('预选配置存储格式无效')
        if 'selected' in raw or 'configs' in raw:
            raise ValueError('预选配置缺少赛季分层')
        value = raw.get(season, {'selected': ALL, 'configs': []})
        if not isinstance(value, dict) or set(value) != {'selected', 'configs'} or not isinstance(value['configs'], list):
            raise ValueError('当前赛季预选配置格式无效')
        ids = set()
        for row in value['configs']:
            if not isinstance(row, dict) or set(row) != {'id', 'name', 'members'}:
                raise ValueError('自定义配置格式无效')
            cid = _id(row['id'], preset=True)
            if cid in ids or _name(row['name']) != row['name']:
                raise ValueError('配置id重复或名称无效')
            ids.add(cid)
            if not isinstance(row['members'], list):
                raise ValueError('成员必须为阵容id列表')
            for member in row['members']:
                _id(member)
            if len(set(row['members'])) != len(row['members']):
                raise ValueError('成员id重复')
        if value['selected'] != ALL and value['selected'] not in ids:
            raise ValueError('启用配置不存在')
        return deepcopy(value), ''
    except (TypeError, ValueError) as exc:
        return {'selected': 'invalid', 'configs': []}, f'预选配置损坏，已暂停自动比较：{exc}'


def edit(raw, season, action, *, config_id=None, name=None, members=None, known_ids=()):
    value, error = read(raw, season)
    if error:
        raise ValueError(error + '；请先修复设置文件，当前不覆盖损坏记录')
    row = next((r for r in value['configs'] if r['id'] == config_id), None)
    if action in ('update', 'delete', 'select') and config_id != ALL:
        _id(config_id, preset=True)
        if row is None:
            raise ValueError('找不到当前赛季配置')
    if action in ('create', 'update'):
        if config_id == ALL:
            raise ValueError('全部阵容为内置动态配置，不能修改')
        clean_name = _name(name)
        if not isinstance(members, list):
            raise ValueError('成员必须为阵容id列表')
        clean = list(dict.fromkeys(_id(m) for m in members))
        # 允许原保存后消失的成员继续保留；拒绝新加入的未知成员。
        allowed = set(known_ids) | set(row['members'] if row else [])
        if set(clean) - allowed:
            raise ValueError('包含未知阵容成员：' + '、'.join(sorted(set(clean) - allowed)))
        if action == 'create':
            config_id = 'p-' + uuid4().hex
            value['configs'].append({'id': config_id, 'name': clean_name, 'members': clean})
        else:
            row.update(name=clean_name, members=clean)
    elif action == 'delete':
        if config_id == ALL:
            raise ValueError('全部阵容不能删除')
        value['configs'].remove(row)
        if value['selected'] == config_id:
            value['selected'] = ALL
    elif action == 'select':
        value['selected'] = config_id
    else:
        raise ValueError('未知配置操作')
    result = deepcopy(raw)
    result[season] = value
    return result, config_id


def snapshot(ctx):
    """只影响辅助比较；目录、Scout、自动D和托管继续使用完整启用资料。"""
    value, error = read(ctx.settings.assistant.comp_presets, ctx.db.season_id)
    catalog = ctx.enabled_comp_catalog()
    active = next((r for r in value['configs'] if r['id'] == value['selected']), None)
    members = [c.id for c in catalog] if value['selected'] == ALL else active['members'] if active else []
    resolved = {c.id: c for cid in members if (c := ctx.enabled_comp(cid)) is not None and c.heroes}
    ids = sorted(resolved)
    watch = {c.id for cid in ctx.game_watch_ids if (c := ctx.enabled_comp(cid)) is not None}
    selected_lock = ctx.selected_lock_id()
    locked = ctx.enabled_comp(selected_lock or '')
    if locked and not locked.heroes:
        locked = None
    candidates = ([locked.id] if locked else [] if selected_lock else
                  sorted(set(ids) & watch) if ctx.game_watch_ids else ids)
    outside = []
    for cid in ctx.game_watch_ids:
        c = ctx.enabled_comp(cid)
        if c is None or c.id not in ids:
            raw_comp = ctx.db.comp(cid, raw=True)
            outside.append({'id': cid, 'name': raw_comp.name if raw_comp else cid})
    notice = error or ('手动锁定' + ctx.comp_identity(selected_lock)['name'] + '来源暂不可用，选择已保留，自动比较已暂停；恢复来源或解除锁定后继续'
                       if selected_lock and not locked else
                       '当前手动锁定' + ctx.comp_identity(selected_lock)['name'] + '（配置外），解除锁定后回到预选'
                       if locked and locked.id not in ids else
                       '本局关注与预选没有可用交集，自动比较已暂停' if ctx.game_watch_ids and not candidates else
                       '当前配置有效0，没有可比较阵容；不会自动回退全部' if not candidates else '')
    identity = [ctx.db.season_id, value['selected'], ids, candidates,
                locked.id if locked else None, [] if locked else sorted(watch & set(ids)),
                bool(ctx.game_watch_ids) and not locked]
    if selected_lock and not locked:
        identity.append('unavailable_lock')
    revision = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    return {'revision': revision, 'preset_id': value['selected'], 'name': active['name'] if active else
            '全部阵容' if value['selected'] == ALL else '损坏配置', 'season_id': ctx.db.season_id,
            'catalog_count': len(members), 'effective_count': len(ids), 'candidate_count': len(candidates),
            'preset_ids': ids, 'candidate_ids': candidates, 'watch_ids': sorted(watch & set(ids)),
            'allowed_comps': [{'id': cid, 'name': (locked if locked and cid == locked.id else resolved[cid]).name}
                              for cid in candidates],
            'lock_id': locked.id if locked else None, 'lock_outside': bool(locked and locked.id not in ids),
            'selection_map': {cid: ctx.comp_identity(cid)['canonical_id']
                              for cid in dict.fromkeys([*members, *ctx.game_watch_ids, *([selected_lock] if selected_lock else [])])},
            'lock_selection_id': selected_lock or None,
            'watch_outside': outside, 'error': error, 'notice': notice}
