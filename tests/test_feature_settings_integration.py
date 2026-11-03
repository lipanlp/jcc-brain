"""功能页共享已保存状态与原入口集成；仅隔离设置和测试DB。"""
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

from jccbrain.core.settings import Settings
from jccbrain.modes.autoroll import AutoRollMode, RollTarget
from jccbrain.webui import settings_schema
from test_webui_api import data, isolation, setup_api


def test_status_feature_values_are_complete_and_never_include_common_secrets(setup_api):
    api, ctx, _ = setup_api
    ctx.settings.llm.api_key = 'test-secret-never-exposed'
    status = data(api.status())
    assert status['feature_settings'] == data(api.feature_settings())['values']
    assert {s: set(v) for s, v in status['feature_settings'].items()} == settings_schema.FEATURE_FIELDS
    assert 'llm' not in status['feature_settings']
    assert 'test-secret-never-exposed' not in str(status)
    status['feature_settings']['hud']['font_size'] = 32
    assert ctx.settings.hud.font_size != 32


def test_legacy_default_hud_and_common_save_publish_authoritative_values(setup_api):
    api, ctx, bus = setup_api
    data(api.set_default_lock_comp('104'))
    assert bus.events[-1][1]['feature_settings']['assistant']['default_lock_comp_id'] == '104'
    # 旧JSON/内部HUD入口仍允许完整校验保存，普通界面仅保留公共分组。
    data(api.save_settings({'hud': {'font_size': 19, 'opacity': .75, 'show_names': False}}))
    values = bus.events[-1][1]['feature_settings']
    assert values['hud']['font_size'] == 19 and values['hud']['opacity'] == .75
    assert values['hud']['show_names'] is False
    assert data(api.feature_settings())['values'] == values
    assert {s['key'] for s in data(api.settings_schema())}.isdisjoint({'assistant','hud','scout','autoroll','autopilot'})


def test_sources_and_season_change_refresh_catalog_state_and_keep_started_roll_snapshot(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    data(api.set_default_lock_comp('104'))
    defaults = data(api.autoroll_defaults())
    targets = [RollTarget(defaults['heroes'][0]['name'], 3)]
    mode = AutoRollMode(ctx, api._sink, targets, 37, 9)
    frozen = deepcopy(mode.status)
    data(api.save_settings({'data': {'source_official': False, 'source_ingame': False, 'source_dataj': False}}))
    fresh = data(api.feature_settings())
    assert fresh['comp_options'] == [{'value': '', 'label': '不锁定（自动推荐）'}]
    assert fresh['values']['assistant']['default_lock_comp_id'] == '104'
    assert any(name == 'data_changed' for name, _ in bus.events)
    assert mode.status == frozen and data(api.hero_options())
    # 无运行线程；切空赛季只改本测试DB缓存和隔离JSON，不连接视觉/游戏。
    monkeypatch.setattr(ctx, 'reset_runtime', Mock())
    data(api.switch_season('17_S18'))
    assert data(api.hero_options()) == []
    assert data(api.autoroll_defaults())['heroes'] == []
    assert mode.status == frozen
    assert data(api.status())['feature_settings'] == data(api.feature_settings())['values']


def test_failed_feature_save_is_atomic_and_does_not_emit_new_saved_values(setup_api):
    api, ctx, bus = setup_api
    previous, events = asdict(ctx.settings), len(bus.events)
    assert not api.save_feature_settings({'assistant': {'default_lock_comp_id': 'missing'}, 'hud': {'font_size': 20}})['ok']
    assert asdict(ctx.settings) == previous and len(bus.events) == events
    assert Settings.load().assistant.default_lock_comp_id == ''
