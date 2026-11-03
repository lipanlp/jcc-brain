"""后端接口离线契约；每个测试均隔离 JCC_HOME，并阻止实际网络连接。"""
from copy import deepcopy
from dataclasses import asdict, replace
import inspect
import json
import logging
import os
from pathlib import Path
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain.actuator.actions import BuyShop
from jccbrain.context import AppContext
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.data.official import to_gamedata
from jccbrain.data.sync import convert_gamedata
from jccbrain.engine.llm import PRESETS
from jccbrain.webui import serialize
from jccbrain.webui.api import Api
from jccbrain.webui.events import WebLogHandler, WebSink


PUBLIC_METHODS = set('''app_info status icons start_mode stop_mode autoroll_defaults settings_schema save_settings feature_settings save_feature_settings
set_dry_run set_llm_enabled set_llm_preset llm_presets test_llm seasons switch_season set_source sync db_stats
comps comp_detail lock_comp_for_game set_default_lock_comp set_watch_comps db_list hero_options odds_calc odds_simulate odds_import_target live_basics population_preview
check_vision check_executor choose_replay browse_file label_units_info label_units calibrate_points
calibrate_capture calibrate_save set_window hud_toggle hud_panel_toggle
window_drag_regions scout_now scout_stop autopilot_resume'''.split())
PUBLIC_METHODS.update({'comp_presets','save_comp_preset','select_comp_preset','delete_comp_preset'})


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    def blocked(*args, **kwargs):
        raise AssertionError('WebUI tests must not connect to the network')
    monkeypatch.setattr(socket.socket, 'connect', blocked)


class Bus:
    def __init__(self):
        self.events = []
    def emit(self, name, payload):
        json.dumps(payload)
        self.events.append((name, deepcopy(payload)))


class Runner:
    def __init__(self, problems=None):
        self.running = False
        self.mode = None
        self.problems = problems or []
    def start(self, mode):
        if not self.problems:
            self.mode, self.running = mode, True
        return self.problems
    def stop(self):
        self.mode, self.running = None, False


@pytest.fixture
def setup_api(db):
    ctx, bus = AppContext(Settings()), Bus()
    ctx.db = db
    api = Api(ctx, bus)
    yield api, ctx, bus
    api._shutdown()
    if api._sync_thread:
        api._sync_thread.join(2)
        assert not api._sync_thread.is_alive()
    ctx.close()


def data(result):
    assert result['ok'], result
    json.dumps(result['data'])
    return result['data']


class FakeHud:
    def __init__(self):
        self.status = {'overlay': False, 'panel': False}
        self.closed = False
    def set_refresh(self, callback):
        self.refresh = callback
    def toggle_overlay(self, on, info=None):
        self.status['overlay'] = on
    def toggle_panel(self, on):
        self.status['panel'] = on
    def close(self):
        self.closed = True
    def _drag_window(self):
        return None


@pytest.fixture
def hud_api(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    api._hud = FakeHud()
    api._runner = Runner()
    monkeypatch.setattr(ctx.locator, 'find', lambda *args: SimpleNamespace(x=1, y=2, width=1000, height=700))
    return api, ctx, bus


def test_hud_missing_game_reports_owner_and_does_not_start(hud_api, monkeypatch):
    api, ctx, _ = hud_api
    monkeypatch.setattr(ctx.locator, 'find', lambda *args: None)
    result = api.hud_toggle(True)
    assert result == {'ok': False, 'error': f'找不到游戏窗口（进程名包含「{ctx.settings.actuator.window_owner}」）'}
    assert not api._runner.running and data(api.status())['hud'] == {'overlay': False, 'panel': False}


@pytest.mark.parametrize('method,field', [('hud_toggle', 'overlay'), ('hud_panel_toggle', 'panel')])
def test_hud_auto_starts_assistant_and_tracks_status(hud_api, method, field):
    api, _, _ = hud_api
    status = data(getattr(api, method)(True))
    assert status['mode']['key'] == 'assistant' and status['hud'][field]
    mode = api._runner.mode
    data(getattr(api, method)(True))
    assert api._runner.mode is mode
    assert not data(getattr(api, method)(False))['hud'][field]
    assert api._runner.running  # 关闭显示不会停止模式


def test_panel_opens_without_game_and_start_failure_rolls_back(hud_api, monkeypatch):
    api, ctx, _ = hud_api
    monkeypatch.setattr(ctx.locator, 'find', lambda *args: None)
    assert data(api.hud_panel_toggle(True))['hud']['panel']
    data(api.hud_panel_toggle(False))
    api._runner = Runner(['测试：视觉模块不可用'])
    assert not api.hud_panel_toggle(True)['ok']
    assert not api._hud.status['panel']


def test_panel_exposes_only_window_drag_methods(hud_api, monkeypatch):
    api, _, _ = hud_api
    panel = SimpleNamespace(expose=Mock())
    monkeypatch.setattr(api._hud, '_drag_window', lambda: panel)
    assert data(api.hud_panel_toggle(True))['hud']['panel']
    data(api.hud_panel_toggle(False))
    data(api.hud_panel_toggle(True))
    panel.expose.assert_called_once_with(api.window_drag_regions)


def test_hidden_hud_does_not_serialize_then_sends_state_and_advice(hud_api, monkeypatch):
    api, ctx, bus = hud_api
    from jccbrain.engine.advisor import HeuristicAdvisor
    st = GameState.from_dict(make_state())
    adv = HeuristicAdvisor(ctx.db, ctx.rules).advise(st)
    original = serialize.hud
    spy = Mock(wraps=original)
    monkeypatch.setattr(serialize, 'hud', spy)
    api._sink.on_state(st)
    api._sink.on_advice(adv)
    spy.assert_not_called()
    api._hud.status['panel'] = True
    api._sink.on_state(st)
    api._sink.on_advice(adv)
    assert spy.call_count == 2
    assert bus.events[-1] == ('hud', original(st, adv, ctx.db, ctx.layout, ctx.settings.hud))
    assert [name for name, _ in bus.events[-4:]] == ['state', 'hud', 'advice', 'hud']


def test_saving_hud_settings_immediately_refreshes_latest_frame(hud_api):
    api, _, bus = hud_api
    api._hud.status['overlay'] = True
    api._sink.on_state(GameState.from_dict(make_state(board=[{'name': '奥恩', 'x': .4, 'y': .5}])))
    data(api.save_settings({'hud': {'show_names': False, 'font_size': 16, 'opacity': .6}}))
    payload = [payload for name, payload in bus.events if name == 'hud'][-1]
    assert payload['units'] == [] and payload['font_size'] == 16 and payload['opacity'] == .6
    api._shutdown()
    assert api._hud.closed


def test_public_api_exact_list():
    actual = {name for name, _ in inspect.getmembers(Api, inspect.isfunction) if not name.startswith('_')}
    assert actual == PUBLIC_METHODS
    assert all(hasattr(getattr(Api, name), '__wrapped__') for name in actual)


def test_icons_api_returns_cached_data_and_validates_keys(setup_api, monkeypatch):
    api, _, _ = setup_api
    monkeypatch.setattr(api._icons, 'get', Mock(return_value={'hero:奥恩': 'data:image/png;base64,YQ=='}))
    assert data(api.icons(['hero:奥恩'])) == {'hero:奥恩': 'data:image/png;base64,YQ=='}
    api._icons.get.assert_called_once_with(['hero:奥恩'])


def test_window_drag_api_routes_without_status_or_logs(setup_api):
    api, _, bus = setup_api
    drag = api._window_drag = Mock()
    regions = [{'x': 0, 'y': 0, 'w': 1280, 'h': 52}]
    exclusions = [{'x': 400, 'y': 10, 'w': 100, 'h': 30}]
    assert data(api.window_drag_regions('main', regions, exclusions)) is None
    drag.update.assert_called_once_with('main', regions, exclusions)
    assert not bus.events


def test_empty_database_cannot_start():
    ctx, bus = AppContext(), Bus()
    api = Api(ctx, bus)
    try:
        result = api.start_mode('assistant')
        assert not result['ok'] and '数据库为空' in result['error']
    finally:
        ctx.db.close()
        ctx.close()


@pytest.mark.parametrize('key', ['assistant', 'autoroll', 'autopilot'])
def test_modes_start_without_real_vision_or_actions(setup_api, key):
    api, ctx, bus = setup_api
    api._runner = Runner()
    options = {'targets': [{'name': '奥恩', 'star': 3}], 'keep': 20, 'max_rolls': 9}
    status = data(api.start_mode(key, options))
    assert status['mode']['key'] == key
    assert any(name == 'status' for name, _ in bus.events)
    if key == 'autoroll':
        assert api._runner.mode.targets[0].star == 3
        assert api._runner.mode.keep_gold == 20 and api._runner.mode.max_rolls == 9


def test_mode_problems_and_invalid_options(setup_api):
    api, _, _ = setup_api
    assert '未知的模式' in api.start_mode('unknown')['error']
    assert '没有勾选目标卡' in api.start_mode('autoroll', {'targets': []})['error']
    api._runner = Runner(['缺少授权', '未标定'])
    error = api.start_mode('autopilot')['error']
    assert error == '无法启动：\n缺少授权\n未标定'


def test_stop_order_is_guard_then_runner(setup_api, monkeypatch):
    api, ctx, _ = setup_api
    order = []
    monkeypatch.setattr(ctx.guard, 'stop', lambda reason: order.append(('guard', reason)))
    monkeypatch.setattr(api._runner, 'stop', lambda: order.append(('runner', None)))
    assert data(api.stop_mode())['mode'] is None
    assert order == [('guard', '手动停止'), ('runner', None)]


def test_sources_can_all_be_disabled_and_sync_requires_one(setup_api):
    api, ctx, _ = setup_api
    assert data(api.set_source('official', False))['dataj']
    assert data(api.set_source('ingame', False))['dataj']
    result = data(api.set_source('dataj', False))
    assert result == {'official': False, 'ingame': False, 'dataj': False,
                      'message': '三个数据源都已关闭，没有可推荐的阵容'}
    assert not ctx.settings.data.source_official
    assert api.sync(False)['error'] == '至少要选一个数据源'
    assert api.sync(False, True)['error'] == '至少要选一个数据源'
    assert not api._syncing and api._sync_thread is None


def test_settings_schema_hides_season_and_marks_secret(setup_api):
    api, _, _ = setup_api
    sections = data(api.settings_schema())
    assert not any(field['key'] == 'season_id' for sec in sections for field in sec['fields'])
    llm = next(sec for sec in sections if sec['key'] == 'llm')
    assert next(f for f in llm['fields'] if f['key'] == 'api_key')['type'] == 'secret'
    assert 'Anthropic' in llm['note']
    vision = next(sec for sec in sections if sec['key'] == 'vision')
    assert next(f for f in vision['fields'] if f['key'] == 'replay_file')['type'] == 'file'


def test_settings_conversion_and_persistence(setup_api):
    api, ctx, _ = setup_api
    data(api.save_settings({'vision': {'poll_interval_ms': '380'}, 'actuator': {'dry_run': 'false'},
                            'hud': {'opacity': '0.81'}, 'data': {'auto_sync': '0'}}))
    assert ctx.settings.vision.poll_interval_ms == 380
    assert ctx.settings.actuator.dry_run is False
    assert ctx.settings.hud.opacity == .81
    assert Settings.load().hud.opacity == .81
    assert not ctx.settings.data.auto_sync


@pytest.mark.parametrize('section,key,value,label', [
    ('vision', 'poll_interval_ms', 'no', '读取间隔 ms'),
    ('vision', 'poll_interval_ms', 2.5, '读取间隔 ms'),
    ('vision', 'backend', 'invalid', '视觉来源'),
    ('hud', 'opacity', 'nan', '不透明度'),
    ('hud', 'opacity', -2, '不透明度'),
    ('actuator', 'dry_run', 'maybe', '演练模式'),
])
def test_invalid_settings_are_atomic(setup_api, section, key, value, label):
    api, ctx, _ = setup_api
    before = asdict(ctx.settings)
    result = api.save_settings({'vision': {'poll_interval_ms': 900}, section: {key: value}})
    assert not result['ok'] and label in result['error']
    assert asdict(ctx.settings) == before
    assert not (Path(os.environ['JCC_HOME']) / 'settings.json').exists()


def test_preset_change_and_custom_model_follow_qt(setup_api):
    api, ctx, _ = setup_api
    data(api.save_settings({'llm': {'preset': 'fast', 'model': 'ignored'}}))
    assert ctx.settings.llm.model == PRESETS['fast'].model
    assert ctx.settings.llm.timeout_s == PRESETS['fast'].timeout_s
    data(api.save_settings({'llm': {'model': 'custom-model'}}))
    assert ctx.settings.llm.preset == 'custom'
    data(api.set_llm_preset('deep'))
    assert ctx.settings.llm.reasoning_effort == PRESETS['deep'].effort
    assert sum(row['active'] for row in data(api.llm_presets())) == 1


def test_llm_toggle_needs_configuration(setup_api):
    api, ctx, _ = setup_api
    assert api.set_llm_enabled(True)['error'] == '还没填 API Key / 模型：设置 → 大模型'
    ctx.settings.llm.api_key, ctx.settings.llm.model = 'test-key', 'test-model'
    assert data(api.set_llm_enabled(True))['llm']['enabled']
    assert not data(api.set_llm_enabled(False))['llm']['enabled']


def test_llm_test_uses_copy_and_does_not_save(setup_api, monkeypatch):
    api, ctx, _ = setup_api
    before = asdict(ctx.settings)
    seen = []
    class Client:
        def __init__(self, cfg):
            seen.append(cfg)
            self._http = SimpleNamespace(close=Mock())
        def ping(self):
            return '收到'
    monkeypatch.setattr('jccbrain.webui.api.LLMClient', Client)
    assert data(api.test_llm({'llm': {'model': 'draft'}})) == '收到'
    assert seen[0].model == 'draft'
    assert asdict(ctx.settings) == before
    assert not (Path(os.environ['JCC_HOME']) / 'settings.json').exists()


def test_game_lock_and_status_track_lock_and_clear(setup_api):
    api, ctx, bus = setup_api
    comp = ctx.db.comps_ranked()[0]
    lock = {'id': comp.id, 'name': comp.name, 'source': 'game', 'canonical_id': comp.id, 'available': True}
    assert data(api.lock_comp_for_game(comp.id))['lock'] == lock
    assert ('lock_changed', {'lock': lock, 'default_lock': None}) in bus.events
    assert data(api.status())['lock'] == lock
    data(api.lock_comp_for_game(''))
    assert data(api.status())['lock'] is None and ctx.game_lock_id == ''
    assert not api.lock_comp_for_game('missing')['ok']


def test_seasons_stats_comps_and_database(setup_api, rules):
    api, ctx, _ = setup_api
    ctx.rules = replace(rules, borrowed_from=18, borrowed_from_season='18_S19')
    assert '未核实' in data(api.status())['season']['rules_note']
    assert sum(row['current'] for row in data(api.seasons())) == 1
    assert data(api.db_stats())['stats']['heroes'] > 0
    comps = data(api.comps())
    assert comps['info']['dataj_count'] == 1 and comps['rows']
    assert data(api.comp_detail(comps['rows'][0]['id']))['heroes']
    assert api.comp_detail('missing')['error'] == '找不到阵容'
    assert data(api.db_list('heroes', '奥恩'))[0]['name'] == '奥恩'
    assert not api.db_list('invalid')['ok']


def test_odds_calculation_uses_owned_and_manual_taken(setup_api):
    api, ctx, _ = setup_api
    api._sink.on_state(GameState.from_dict(make_state(bench=[{'name': '奥恩', 'star': 2, 'slot': 0}])))
    result = data(api.odds_calc([{'name': '奥恩', 'need': 9, 'others': 2},
                                 {'name': '不存在', 'need': 3, 'others': 0}], 7))
    assert result['owned'] == {'奥恩': 3}
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['have'] == 3 and not row['done']
    assert row['remaining'] == ctx.rules.pool_size[1] - 5
    assert 0 <= row['per_roll'] <= 1 and 0 <= result['any_hit'] <= 1
    assert ctx.extra_taken['奥恩'] == 2
    done = data(api.odds_calc([{'name': '奥恩', 'need': 3, 'others': 0}], 7))['rows'][0]
    assert done['done'] and done['gold_50'] is None
    assert not data(api.odds_calc([{'name': '奥恩', 'need': 3, 'others': 0}], 7, False))['owned']


def test_odds_simulation_returns_probabilities_and_none_when_done(setup_api):
    api, _, _ = setup_api
    rows = [{'name': '奥恩', 'need': 3, 'others': 1}]
    result = data(api.odds_simulate(rows, 5, 40, 10, sims=50))
    assert 0 <= result['all_hit_rate'] <= 1
    assert 0 <= result['hit_rate']['奥恩'] <= 1
    assert 0 <= result['avg_gold_left'] <= 40
    assert data(api.odds_simulate([], 5, 40, 10)) is None
    assert not api.odds_simulate(rows, 5, 40, 10, sims=0)['ok']


def test_autoroll_options_import_and_live_basics(setup_api):
    api, ctx, _ = setup_api
    options = data(api.hero_options())
    assert options == sorted(options, key=lambda row: (row['cost'], row['name']))
    empty = data(api.autoroll_defaults())
    assert empty['comp_name'] is None and empty['heroes'] == []
    data(api.lock_comp_for_game('104'))
    defaults = data(api.autoroll_defaults())
    assert defaults['comp_name'] and defaults['heroes'] and defaults['all_heroes'] == options
    assert any(row['checked'] and row['star'] == 3 for row in defaults['heroes'])
    assert data(api.odds_import_target())['rows'][0]['need'] == 9
    assert data(api.live_basics()) is None
    api._sink.on_state(GameState.from_dict(make_state(level=8, gold=32)))
    assert data(api.live_basics()) == {'level': 8, 'gold': 32}


def test_autoroll_catalog_includes_alune_and_keeps_season_pool_boundary(setup_api):
    api, ctx, _ = setup_api
    samples = json.loads((Path(__file__).parent / 'fixtures' / 'official_units.json').read_text())
    # Exercise official conversion, unit classification, the season-scoped DB and both API paths.
    with ctx.db.transaction() as conn:
        for season_id, sample in samples.items():
            tables = convert_gamedata(to_gamedata(sample))
            for table in ('traits', 'heroes', 'hero_stars'):
                ctx.db.replace_table(conn, table, tables[table], season_id=season_id)
    ctx.db.reload()
    alune = ctx.db.hero('拉露恩')
    assert (alune.id, alune.cost, alune.in_pool, alune.unit_kind) == ('5456', 5, True, 'champion')
    assert ctx.db.heroes['5461'].unit_kind == 'variant' and not ctx.db.heroes['5461'].in_pool
    assert ctx.db.heroes['8704'].unit_kind == 'summon' and not ctx.db.heroes['8704'].in_pool

    options = data(api.hero_options())
    assert options == data(api.autoroll_defaults())['all_heroes']
    assert options.count({'name': '拉露恩', 'cost': 5}) == 1
    assert len(options) == len({row['name'] for row in options}) == 65
    assert {'name': '拉克丝', 'cost': 5} in options  # Ten variants still yield one candidate.
    assert not {'悠米', '卡莎'} & {row['name'] for row in options}
    assert all(ctx.db.hero(row['name']).in_pool for row in options)

    ctx.db.use_season('16_S19')
    other_options = data(api.hero_options())
    assert other_options == data(api.autoroll_defaults())['all_heroes']
    assert {'name': '卡莎', 'cost': 4} in other_options
    assert not {'拉露恩', '加里奥', '提伯斯'} & {row['name'] for row in other_options}


def test_websink_state_advice_and_finish_order(setup_api):
    api, ctx, bus = setup_api
    model = GameState.from_dict(make_state())
    api._sink.on_state(model)
    assert bus.events[-1] == ('state', serialize.state(model, ctx.db))
    api._sink.on_advice(ctx.advisor.advise(model))
    assert bus.events[-1][0] == 'advice'
    api._runner = Runner()
    data(api.start_mode('assistant'))
    api._sink.on_finished('assistant', '完成')
    assert [name for name, _ in bus.events[-2:]] == ['mode_finished', 'status']
    assert bus.events[-1][1]['mode'] is None


def test_websink_action_and_structured_mode_log(setup_api):
    api, _, bus = setup_api
    api._sink.on_action(BuyShop(0), False)
    assert bus.events[-2] == ('action', {'text': '买商店第 1 格', 'real': False})
    row = bus.events[-1][1]
    assert set(row) == {'time', 'text', 'level', 'source'}
    assert row['source'] == 'mode' and row['level'] == 'info' and len(row['time']) == 8


def test_log_handler_levels_and_format():
    bus = Bus()
    handler = WebLogHandler(bus)
    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
        handler.emit(logging.LogRecord('test', level, '', 0, '日志 %s', ('消息',), None))
    assert [event[1]['level'] for event in bus.events] == ['info', 'warning', 'error']
    assert all(event[1]['source'] == 'logger' and '日志 消息' in event[1]['text'] for event in bus.events)


def test_sync_is_background_rejects_duplicates_and_switch(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    entered, release = threading.Event(), threading.Event()
    connections = []
    def sync_all(details, progress, only_needed, db):
        connections.append(db)
        entered.set()
        release.wait(2)
        progress('离线同步进度')
        return []
    monkeypatch.setattr(ctx, 'sync_all_seasons', sync_all)
    try:
        assert data(api.sync(True, True)) == {'started': True}
        assert entered.wait(1)
        assert connections[0] is not ctx.db
        assert connections[0].path == ctx.db.path
        assert api.sync(True)['error'] == '正在同步'
        assert api.switch_season('16_S19')['error'] == '正在同步数据，完成后再切换'
    finally:
        release.set()
        api._sync_thread.join(2)
    assert not data(api.status())['syncing']
    assert ('sync_progress', {'message': '离线同步进度'}) in bus.events
    assert any(name == 'sync_done' and payload['ok'] for name, payload in bus.events)
    assert any(name == 'data_changed' for name, _ in bus.events)
    with pytest.raises(Exception, match='closed'):
        connections[0].synced_seasons()


def test_sync_failure_clears_busy_and_reports(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    def fail(*args, **kwargs):
        raise RuntimeError('离线失败样例')
    monkeypatch.setattr(ctx, 'sync', fail)
    data(api.sync(False))
    api._sync_thread.join(2)
    assert ('sync_done', {'ok': False, 'message': '同步失败：离线失败样例'}) in bus.events
    assert not data(api.status())['syncing']
    assert not any(name == 'data_changed' for name, _ in bus.events)


def test_single_season_sync_and_switch_clear_old_live_state(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    monkeypatch.setattr(ctx, 'sync', lambda *args, **kwargs: ctx.db.stats())
    data(api.sync(True))
    api._sync_thread.join(2)
    assert any(name == 'sync_done' and payload['ok'] for name, payload in bus.events)
    api._sink.on_state(GameState.from_dict(make_state()))
    result = data(api.switch_season('16_S19'))
    assert result['needs_sync']
    assert data(api.live_basics()) is None
    assert data(api.status())['season']['id'] == '16_S19'


def test_label_units_mapping_and_unknown_names_are_skipped(setup_api):
    api, ctx, _ = setup_api
    assert '还没有画面' in api.label_units_info()['error']
    model = GameState.from_dict(make_state(bench=[{'slot': 0, 'name': '?1'}],
                                          board=[{'name': '?2', 'x': .3, 'y': .4}]))
    api._sink.on_state(model)
    ctx.vision = SimpleNamespace(label_units=Mock(return_value=['已标注']), stop=Mock())
    info = data(api.label_units_info())
    assert [row['key'] for row in info['units']] == ['bench:0', 'board:0']
    result = data(api.label_units([{'key': 'bench:0', 'name': '奥恩'}, {'key': 'board:0', 'name': '厄斐琉斯'},
                                   {'key': 'board:2', 'name': '不存在'}]))
    assert result == ['已标注']
    ctx.vision.label_units.assert_called_once_with({0: '奥恩'}, [{'x': .3, 'y': .4, 'name': '厄斐琉斯'}])


def test_vision_and_executor_checks_are_read_only(setup_api, monkeypatch):
    api, ctx, _ = setup_api
    model = GameState.from_dict(make_state())
    ctx.vision = SimpleNamespace(start=Mock(), read=lambda: model, stop=Mock())
    assert data(api.check_vision()) == {'summary': model.summary(), 'shop': [None] * 5}
    ctx.locator = SimpleNamespace(find=lambda _: None)
    monkeypatch.setattr('jccbrain.actuator.executor.MacExecutor', lambda *args: SimpleNamespace(preflight=lambda: ['缺少权限']))
    assert data(api.check_executor()) == {'window': '游戏窗口：未找到', 'calibrated': False, 'problems': ['缺少权限']}


def test_file_dialog_cancel_and_replay_start(setup_api, tmp_path):
    api, ctx, _ = setup_api
    assert api.browse_file()['error'] == '窗口尚未就绪'
    window = SimpleNamespace(create_file_dialog=Mock(return_value=None))
    assert api.set_window(window)['ok']
    assert data(api.browse_file()) is None and data(api.choose_replay()) is None
    path = str(tmp_path / 'game.jsonl')
    window.create_file_dialog.return_value = [path]
    assert data(api.browse_file()) == path
    api._runner = Runner()
    assert data(api.choose_replay()) == path
    assert ctx.settings.vision.backend == 'replay'
    assert ctx.settings.vision.replay_file == path
    assert data(api.status())['mode']['key'] == 'assistant'


def test_calibration_capture_save_and_validation(setup_api, monkeypatch):
    api, ctx, _ = setup_api
    points = data(api.calibrate_points())
    assert len(points) == 10 and points[0]['key'] == 'r1c1'
    ctx.locator = SimpleNamespace(find=lambda _: None)
    assert api.calibrate_capture()['error'] == '找不到游戏窗口'
    ctx.locator = SimpleNamespace(find=lambda _: SimpleNamespace(to_relative=lambda x, y: (.123456, .654321)))
    monkeypatch.setattr('jccbrain.actuator.window.mouse_location', lambda: (100, 200))
    assert data(api.calibrate_capture()) == {'x': .1235, 'y': .6543}
    assert not api.calibrate_save({})['ok']
    values = {row['key']: [.25, .5] for row in points}
    values['bench8'] = [.75, .5]
    assert data(api.calibrate_save(values)) is None
    assert ctx.layout.board_calibrated and ctx.layout.bench_step == .0625
    assert ctx.layout.refresh_button == (.25, .5)


def test_dry_run_stops_acting_mode(setup_api):
    api, ctx, _ = setup_api
    api._runner = Runner()
    data(api.start_mode('autopilot'))
    result = data(api.set_dry_run(False, True))
    assert result['mode'] is None and not result['dry_run']
    assert not ctx.settings.actuator.dry_run


@pytest.mark.parametrize("initial_failure", [False, True])
def test_app_lifecycle_initial_status_auto_sync_and_cleanup(setup_api, monkeypatch, caplog, initial_failure):
    from jccbrain.webui import app

    _, ctx, _ = setup_api
    order, timers, created = [], [], []
    startups = []
    class Startup:
        def __init__(self, window):
            self.window, self.shown_count, self.close_count = window, 0, 0
            startups.append(self)
        def shown(self):
            self.shown_count += 1
        def close(self):
            self.close_count += 1
    monkeypatch.setattr(app, 'MainWindowStartup', Startup)
    monkeypatch.setattr(ctx, 'sync_all_seasons', lambda details, progress, only_needed, db: [])
    monkeypatch.setattr(ctx, 'close', lambda: order.append('ctx.close'))
    monkeypatch.setattr(app, 'AppContext', lambda: ctx)
    if initial_failure:
        real_emit = Api._emit_status
        failed = False
        def emit_once_failure(self):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError('暂时无法读取初始状态')
            return real_emit(self)
        monkeypatch.setattr(Api, '_emit_status', emit_once_failure)
    class Signal:
        def __init__(self):
            self.callbacks = []
        def __iadd__(self, callback):
            self.callbacks.append(callback)
            return self
        def fire(self):
            for callback in self.callbacks:
                callback()
    class Index:
        def __init__(self, *args):
            pass
        def with_name(self, name):
            return self
        def __truediv__(self, name):
            return self
        def is_file(self):
            return True
        def __str__(self):
            return 'test-static/index.html'
    class FakeBus(Bus):
        def attach(self, evaluate):
            self.evaluate = evaluate
        def detach(self, key):
            assert key == 'main'
        def close(self):
            order.append('bus.close')
    class Timer:
        def __init__(self, seconds, callback):
            self.seconds, self.callback = seconds, callback
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            order.append('timer.cancel')
    bus = FakeBus()
    def create_window(*args, **kwargs):
        window = SimpleNamespace(events=SimpleNamespace(loaded=Signal(), shown=Signal(), closed=Signal()),
                                 evaluate_js=lambda script: None, api=kwargs['js_api'])
        created.append(window)
        return window
    def start(**kwargs):
        window = created[0]
        assert len(startups) == 1 and startups[0].window is window
        window.events.shown.fire()
        window.events.loaded.fire()
        assert len(timers) == 1 and timers[0].seconds == 1.5
        if initial_failure:
            assert '首次 status 推送失败' in caplog.text
        window.events.loaded.fire()  # 重复 loaded 不应叠加同步 timer。
        statuses = [payload for name, payload in bus.events if name == 'status']
        assert statuses[-1]['season']['id'] == ctx.settings.data.season_id
        assert len(timers) == 1
        timers[0].callback()
        window.api._sync_thread.join(2)
        window.events.closed.fire()
        assert startups[0].close_count == 1
        window.events.loaded.fire()  # 关闭后迟到的 loaded 不能重新安排任务。
        assert len(timers) == 1
        timers[0].callback()  # 被取消后迟到的 callback 也不能启动同步。
    monkeypatch.setattr(app, 'Path', Index)
    monkeypatch.setattr(app, 'EventBus', lambda: bus)
    monkeypatch.setattr(app.threading, 'Timer', Timer)
    monkeypatch.setattr(app.webview, 'create_window', create_window)
    monkeypatch.setattr(app.webview, 'start', start)
    monkeypatch.setattr('jccbrain.modes.base.ModeRunner.stop', lambda self: order.append('runner.stop'))
    with caplog.at_level(logging.DEBUG):
        assert app.run_webui() == 0
    assert order[-3:] == ['runner.stop', 'bus.close', 'ctx.close']
    assert startups[0].shown_count == 1
    assert startups[0].close_count == 2  # closed 回调及主 runloop 退出后的幂等清理。
    assert '后台同步开始' in caplog.text
    for stage in ('AppContext 创建完成', '窗口创建完成', 'loaded', '首次 status 已推送',
                  'auto_sync 开始', 'auto_sync 结束'):
        assert stage in caplog.text
    assert not any(isinstance(handler, WebLogHandler) for handler in logging.getLogger().handlers)
    assert not any(name == 'tick' for name, _ in bus.events)


def test_previous_mode_finish_does_not_mark_restarted_mode_idle(setup_api):
    api, _, _ = setup_api
    api._runner = Runner()
    data(api.start_mode('assistant'))
    old_sink = api._sink
    data(api.start_mode('assistant', None, True))
    old_sink.on_finished('assistant', '旧线程迟到')
    assert data(api.status())['mode']['key'] == 'assistant'
    api._sink.on_finished('assistant', '当前模式结束')
    assert data(api.status())['mode'] is None


def test_status_includes_current_season_hero_costs(setup_api):
    api, ctx, _ = setup_api
    costs = data(api.status())['hero_costs']
    assert costs == {hero.name: hero.cost for hero in ctx.db.pool_heroes()}
    assert costs['奥恩'] == 1
    assert costs['厄斐琉斯'] == 4
    assert '小型石甲虫' not in costs


def test_legacy_js_drag_api_is_removed():
    assert not hasattr(Api, 'hud_panel_drag_start')
    assert not hasattr(Api, 'hud_panel_drag_end')


def test_status_distinguishes_default_game_and_explicit_unlock(setup_api):
    api, ctx, _ = setup_api
    assert data(api.status())['lock'] is None
    default = data(api.set_default_lock_comp('104'))
    assert default['lock'] == {'id': '104', 'name': '野怪小红', 'source': 'default', 'canonical_id': '104', 'available': True}
    assert default['default_lock'] == {'id': '104', 'name': '野怪小红', 'canonical_id': '104', 'available': True}
    assert ctx.game_lock_id is None
    game = data(api.lock_comp_for_game('104'))
    assert game['lock']['source'] == 'game'
    unlocked = data(api.lock_comp_for_game(''))
    assert unlocked['lock'] is None and unlocked['default_lock'] == default['default_lock']


def test_game_lock_does_not_persist_or_write_existing_settings(setup_api):
    api, ctx, _ = setup_api
    path = Path(os.environ['JCC_HOME']) / 'settings.json'
    data(api.lock_comp_for_game('104'))
    assert not path.exists()
    data(api.set_default_lock_comp('104'))
    before = path.read_bytes()
    data(api.lock_comp_for_game(''))
    assert path.read_bytes() == before
    restored = AppContext(Settings.load())
    restored.db = ctx.db  # 有效性校验使用同一合成赛季库
    assert restored.game_lock_id is None and restored.effective_lock_id() == '104'


def test_default_lock_schema_choices_and_save_settings_path(setup_api):
    api, ctx, bus = setup_api
    result = data(api.feature_settings())
    assert result['comp_options'][0] == {'value': '', 'label': '不锁定（自动推荐）'}
    assert result['comp_options'][1] == {'value': '104', 'label': '[S] 野怪小红'}
    ctx.game_lock_id = ''
    data(api.save_feature_settings({'assistant': {'default_lock_comp_id': '104'}}))
    result = data(api.status())
    assert result['lock'] is None and result['default_lock']['id'] == '104'
    assert Settings.load().assistant.default_lock_comp_id == '104'
    assert any(name == 'lock_changed' for name, _ in bus.events)


@pytest.mark.parametrize('method', ['lock_comp_for_game', 'set_default_lock_comp'])
@pytest.mark.parametrize('invalid', ['missing', None, 17])
def test_invalid_lock_does_not_change_context_or_settings(setup_api, method, invalid):
    api, ctx, _ = setup_api
    before = asdict(ctx.settings)
    assert not getattr(api, method)(invalid)['ok']
    assert ctx.game_lock_id is None and asdict(ctx.settings) == before
    assert not (Path(os.environ['JCC_HOME']) / 'settings.json').exists()


def test_invalid_default_setting_is_atomic(setup_api):
    api, ctx, _ = setup_api
    before = asdict(ctx.settings)
    assert not api.save_settings({'vision': {'poll_interval_ms': 999},
                                  'assistant': {'default_lock_comp_id': 'missing'}})['ok']
    assert asdict(ctx.settings) == before
    assert not (Path(os.environ['JCC_HOME']) / 'settings.json').exists()


def test_default_lock_change_does_not_override_current_game(setup_api):
    api, ctx, bus = setup_api
    data(api.lock_comp_for_game('104'))
    data(api.set_default_lock_comp('104'))
    result = data(api.set_default_lock_comp(''))
    assert result['lock']['source'] == 'game' and ctx.game_lock_id == '104'
    assert result['default_lock'] is None and Settings.load().assistant.default_lock_comp_id == ''
    assert any(name == 'lock_changed' and payload['default_lock'] is None for name, payload in bus.events)


def test_autoroll_and_odds_defaults_follow_game_default_then_rank(setup_api):
    api, ctx, _ = setup_api
    base = ctx.db.comp('104')
    ctx.db.comps['second'] = replace(base, id='second', name='本局第二套')
    data(api.set_default_lock_comp('104'))
    data(api.lock_comp_for_game('second'))
    assert data(api.autoroll_defaults())['comp_name'] == '本局第二套'
    assert data(api.odds_import_target())['comp_name'] == '本局第二套'
    data(api.lock_comp_for_game(''))
    assert data(api.autoroll_defaults())['comp_name'] is None
    assert data(api.odds_import_target())['comp_name'] == base.name
    data(api.set_default_lock_comp(''))
    assert data(api.odds_import_target())['comp_name'] == ctx.db.comps_ranked()[0].name


def test_mode_lock_callback_pushes_latest_status(setup_api):
    api, ctx, bus = setup_api
    ctx.game_lock_id = '104'
    api._sink.on_lock_changed()
    assert bus.events[-1][0] == 'status'
    assert bus.events[-1][1]['lock'] == {'id': '104', 'name': '野怪小红', 'source': 'game', 'canonical_id': '104', 'available': True}


def test_sync_publishes_loaded_data_without_reloading_main_connection(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    main_db = ctx.db
    monkeypatch.setattr(main_db, 'reload', Mock(side_effect=AssertionError('重复载入会阻塞界面锁')))

    def sync_all(details, progress, only_needed, db):
        with db.transaction() as conn:
            conn.execute("UPDATE heroes SET skill_desc='同步后的新技能' WHERE name='绯红树怪'")
        db.reload()
        return [db.season_id]

    monkeypatch.setattr(ctx, 'sync_all_seasons', sync_all)
    assert data(api.sync(True, True)) == {'started': True}
    api._sync_thread.join(2)
    assert not api._sync_thread.is_alive()
    main_db.reload.assert_not_called()
    assert ctx.db is main_db
    assert main_db.hero('绯红树怪').skill_desc == '同步后的新技能'
    assert any(name == 'sync_done' and payload['ok'] for name, payload in bus.events)
    assert any(name == 'data_changed' for name, _ in bus.events)


def test_scout_api_queues_and_stops_without_blocking_main_loop(setup_api):
    from jccbrain.modes.assistant import AssistantMode
    api, ctx, bus = setup_api
    api._runner = Runner()
    assert not api.scout_now()['ok']
    mode = AssistantMode(ctx, api._sink)
    api._runner.start(mode)
    assert not api.scout_now()['ok']
    api._last_state = GameState.from_dict(make_state(timer=20))
    assert not api.scout_now()['ok']  # 未识别己方位置时不能排入点击任务。
    api._last_state = GameState.from_dict(make_state(timer=20, players=[
        {'name': '自己', 'hp': 80, 'self': True, 'slot': 0},
        {'name': '对手', 'hp': 70, 'slot': 1}]))
    assert data(api.scout_now())['pending']
    assert data(api.status())['scout']['pending']
    assert not api.scout_now()['ok']  # 不重入
    assert data(api.scout_stop())['scout']['pending'] is False
    assert mode.scout._cancel.is_set()


def test_scout_notice_production_sink_refreshes_hud_before_first_fake_click(setup_api):
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.modes.scout import ScoutRunner
    from jccbrain.actuator.executor import DryRunExecutor
    from jccbrain.actuator.safety import SafetyGuard
    api, ctx, bus = setup_api
    class VisibleHud(FakeHud):
        def notice_capable(self): return True
        def set_notice(self, notice): self.notice = notice
        def notice_ready(self, window=None): return window == 'toast' and self.status['overlay']
    api._hud = VisibleHud(); api._hud.status['overlay'] = True
    ctx.executor = DryRunExecutor(ctx.layout, SafetyGuard(999, False), 0)
    ctx.settings.scout.enabled = True
    api._runner = Runner()
    mode = AssistantMode(ctx, api._sink); api._runner.start(mode)
    clock = [0.0]
    mode.scout.clock = lambda: clock[0]
    from scout_fakes import ControlledInput
    from test_scout_runner import WINDOW
    mode.scout.activity.clock = mode.scout.clock
    mode.scout.activity.monitor = ControlledInput(mode.scout.clock, WINDOW)
    ctx.locator = SimpleNamespace(find=lambda _: WINDOW)
    state = GameState.from_dict(make_state(stage='4-1', timer=30, players=[
        {'name':'自己','slot':0,'self':True,'hp':80}, {'name':'对手','slot':1,'hp':70}]))
    api._sink.on_state(state)
    assert not mode.scout.maybe_run(state)
    notice = [p for kind, p in bus.events if kind == 'hud'][-1]['scout_notice']
    assert notice['seconds'] == 3 and not ctx.executor.history
    assert api._scout_notice_shown(notice['id'], 'toast')
    clock[0] = 2.99
    assert not mode.scout.maybe_run(state) and not ctx.executor.history
    clock[0] = 3
    assert mode.scout.maybe_run(state)
    kinds = [kind for kind, p in bus.events]
    assert kinds.index('hud') < kinds.index('action')
    assert mode.scout.status['notice'] is None
    assert not hasattr(api, 'scout_notice_shown')  # 主页面不能冒充 HUD 确认


def test_feature_schema_owns_fields_without_leaking_connection_secrets(setup_api):
    api, ctx, _ = setup_api
    ctx.settings.llm.api_key = 'private-test-secret'
    public = data(api.settings_schema())
    assert {section['key'] for section in public} == {'vision', 'data', 'llm', 'actuator'}
    assert not any(f['key'] == 'dry_run' for s in public for f in s['fields'])
    features = data(api.feature_settings())
    assert set(features['values']) == {'assistant', 'scout', 'autoroll', 'hud', 'autopilot'}
    assert 'private-test-secret' not in json.dumps(features)
    assert 'llm' not in features['values'] and 'actuator' not in features['values']
    assert features['values']['assistant']['economy_preference'] == 'adaptive'
    assert features['values']['scout'] == {'enabled': False, 'schedule': 'key_rounds'}
    assert features['values']['autopilot']['augment_selection'] == 'manual'
    assert 'stable_frames' not in features['values']['scout']


def test_feature_save_updates_only_owned_defaults_and_never_starts_or_stops(setup_api, monkeypatch):
    api, ctx, bus = setup_api
    api._runner = Runner()
    start, stop, reset = Mock(), Mock(), Mock()
    monkeypatch.setattr(api._runner, 'start', start)
    monkeypatch.setattr(api._runner, 'stop', stop)
    monkeypatch.setattr(ctx, 'reset_runtime', reset)
    before = asdict(ctx.settings)
    result = data(api.save_feature_settings({'autoroll': {'keep_gold': 50, 'max_rolls': 17},
                                            'hud': {'font_size': 16, 'opacity': .7},
                                            'scout': {'enabled': True}}))
    assert result['values']['autoroll'] == {'keep_gold': 50, 'max_rolls': 17}
    assert Settings.load().scout.enabled
    assert asdict(ctx.settings.vision) == before['vision']
    assert asdict(ctx.settings.llm) == before['llm']
    start.assert_not_called()
    stop.assert_not_called()
    reset.assert_not_called()
    assert data(api.status())['mode'] is None


@pytest.mark.parametrize('values', [
    {'assistant': {'economy_preference': 'aggressive'}},
    {'scout': {'schedule': 'always'}},
    {'autopilot': {'augment_selection': 'random'}},
    {'autoroll': {'keep_gold': -1}}, {'autoroll': {'keep_gold': 101}},
    {'autoroll': {'max_rolls': 0}}, {'autoroll': {'max_rolls': 201}},
    {'hud': {'font_size': 7}}, {'hud': {'opacity': 1.1}},
    {'llm': {'api_key': 'override'}}, {'actuator': {'dry_run': False}},
    {'scout': {'stable_frames': 1}},
])
def test_invalid_feature_save_is_atomic(setup_api, values):
    api, ctx, _ = setup_api
    before = asdict(ctx.settings)
    result = api.save_feature_settings(values)
    assert not result['ok']
    assert asdict(ctx.settings) == before
    assert not (Path(os.environ['JCC_HOME']) / 'settings.json').exists()


@pytest.mark.parametrize('key', ['assistant', 'autopilot'])
def test_global_dry_run_requires_explicit_stop_including_assistant_scout(setup_api, key, monkeypatch):
    api, ctx, _ = setup_api
    api._runner = Runner()
    data(api.start_mode(key))
    current = api._runner.mode
    stop = Mock(wraps=api._runner.stop)
    reset = Mock(wraps=ctx.reset_runtime)
    monkeypatch.setattr(api._runner, 'stop', stop)
    monkeypatch.setattr(ctx, 'reset_runtime', reset)
    data(api.set_dry_run(True))  # 相同值没有停止或重建副作用。
    stop.assert_not_called()
    reset.assert_not_called()
    denied = api.set_dry_run(False)
    assert not denied['ok'] and '停止并应用' in denied['error']
    assert api._runner.mode is current and api._runner.running
    assert ctx.settings.actuator.dry_run
    result = data(api.set_dry_run(False, True))
    assert result['mode'] is None and result['dry_run'] is False
    stop.assert_called_once()
    reset.assert_called_once()


def test_public_runtime_save_and_mode_switch_require_explicit_stop(setup_api):
    api, ctx, _ = setup_api
    api._runner = Runner()
    data(api.start_mode('assistant'))
    current = api._runner.mode
    assert not api.save_settings({'vision': {'poll_interval_ms': 900}})['ok']
    assert ctx.settings.vision.poll_interval_ms == 250
    assert not api.start_mode('autopilot')['ok']
    assert api._runner.mode is current
    data(api.save_settings({'vision': {'poll_interval_ms': 900}}, True))
    assert not api._runner.running and ctx.settings.vision.poll_interval_ms == 900
    data(api.start_mode('assistant'))
    assert data(api.start_mode('autopilot', None, True))['mode']['key'] == 'autopilot'


def test_feature_save_preserves_current_direction_and_invalidates_old_scout_notice(setup_api):
    api, ctx, _ = setup_api
    api._runner = Runner()
    data(api.start_mode('assistant'))
    current = api._runner.mode
    assert ctx.game_lock_id is None
    current.scout.request()
    data(api.save_feature_settings({'assistant': {'default_lock_comp_id': '104'}, 'scout': {'enabled': True}}))
    assert api._runner.mode is current and api._runner.running
    assert ctx.game_lock_id == '' and data(api.status())['lock'] is None
    assert data(api.status())['default_lock']['id'] == '104'
    assert not current.scout.status['pending'] and current.scout._cancel.is_set()
    assert Settings.load().assistant.default_lock_comp_id == '104'


def test_manual_game_lock_has_priority_and_does_not_erase_watch(setup_api):
    api, ctx, _ = setup_api
    data(api.lock_comp_for_game('104'))
    watching = data(api.set_watch_comps(['104']))
    assert watching['lock']['id'] == '104' and watching['watch'][0]['id'] == '104'
    locked = data(api.lock_comp_for_game('104'))
    assert locked['lock']['id'] == '104' and locked['watch'][0]['id'] == '104'
    data(api.lock_comp_for_game(''))
    data(api.set_watch_comps([]))
    assert data(api.status())['lock'] is None and data(api.status())['watch'] == []


def test_autoroll_import_prefers_game_lock_then_current_advice_without_catalog_fallback(setup_api):
    api, ctx, _ = setup_api
    base = ctx.db.comp('104')
    ctx.db.comps['recommended'] = replace(base, id='recommended', name='当前主推荐')
    model = GameState.from_dict(make_state())
    api._on_state(model)
    advice = ctx.advisor.advise(model)
    advice.comps[0] = replace(advice.target, comp=ctx.db.comps['recommended'])
    api._on_advice(advice)
    assert data(api.autoroll_defaults())['comp_name'] == '当前主推荐'
    api._on_state(GameState.from_dict(make_state(stage='3-3')))
    assert data(api.autoroll_defaults())['comp_name'] is None
    api._on_state(model)
    data(api.lock_comp_for_game('104'))
    assert data(api.autoroll_defaults())['comp_name'] == base.name
    data(api.lock_comp_for_game(''))
    assert data(api.autoroll_defaults())['comp_name'] == '当前主推荐'
    api._on_state(GameState.from_dict(make_state(phase='ended')))
    assert data(api.autoroll_defaults())['heroes'] == []
    api._last_advice = None
    assert data(api.autoroll_defaults())['comp_name'] is None
    assert data(api.autoroll_defaults())['all_heroes']


def test_stop_timeout_does_not_change_settings_or_rebuild_runtime(setup_api, monkeypatch):
    from test_mode_ownership import BlockedMode, BlockedVision
    from jccbrain.actuator.executor import DryRunExecutor
    from jccbrain.actuator.layout import Layout
    api, ctx, bus = setup_api
    ctx.vision = BlockedVision()
    ctx.executor = DryRunExecutor(Layout(), ctx.guard, delay_s=0)
    monkeypatch.setattr('jccbrain.webui.api._MODE_TYPES', (BlockedMode,))
    monkeypatch.setattr(ctx, 'reset_runtime', Mock())
    original_stop = api._runner.stop
    monkeypatch.setattr(api._runner, 'stop', lambda: original_stop(.01))
    try:
        data(api.start_mode('assistant'))
        assert ctx.vision.entered.wait(1)
        owner, thread = api._runner.mode, api._runner._thread
        assert '停止中' in api.stop_mode()['error']
        assert data(api.status())['stopping'] and data(api.status())['mode']['key'] == 'assistant'
        for result in [api.start_mode('assistant'), api.start_mode('assistant', None, True),
                       api.set_dry_run(False, True), api.save_settings({'vision': {'poll_interval_ms': 700}}, True),
                       api.switch_season(ctx.db.season_id), api.calibrate_save({})]:
            assert not result['ok'] and '停止中' in result['error']
        assert api._runner.mode is owner and api._runner._thread is thread
        assert ctx.settings.actuator.dry_run and ctx.vision.starts == 1
        assert ctx.guard.stopped and owner.stop_event.is_set()
        ctx.reset_runtime.assert_not_called()
        assert any(name == 'status' and payload['stopping'] for name, payload in bus.events)
    finally:
        ctx.vision.release.set()
        data(api.stop_mode())
    assert not data(api.status())['stopping'] and data(api.status())['mode'] is None
    assert not ctx.executor.history
    data(api.start_mode('assistant'))
    api._runner._thread.join(1)
    assert ctx.vision.starts == 2


def test_autopilot_resume_rejects_idle_stale_instance_and_applies_next_default_only(setup_api):
    from jccbrain.modes import AutopilotMode
    from test_autopilot_takeover import augment, frame
    api, ctx, _ = setup_api
    assert not api.autopilot_resume('old')['ok']
    api._runner = Runner()
    mode = AutopilotMode(ctx, api._sink)
    api._runner.start(mode)
    mode._observe_augments(augment())
    token = mode.status['token']
    assert not api.autopilot_resume(token)['ok']
    data(api.save_feature_settings({'autopilot': {'augment_selection': 'auto'}}))
    assert mode.augment_selection == 'manual' and mode.status['paused']
    assert data(api.status())['augment_selection'] == 'auto'
    mode._observe_augments(frame())
    mode._observe_augments(frame())
    assert data(api.status())['autopilot']['can_resume']
    assert not api.autopilot_resume('other-instance')['ok']
    assert not data(api.autopilot_resume(token))['autopilot']['paused']
    assert not api.autopilot_resume(token)['ok']
    assert AutopilotMode(ctx, api._sink).augment_selection == 'auto'


@pytest.fixture
def preview_api(setup_api, board_comp):
    api, ctx, bus = setup_api
    state = GameState.from_dict(make_state(odds=[100, 0, 0, 0, 0], board_count=[6, 6]))
    advice = ctx.advisor.advise(state, '104')
    api._on_state(state)
    api._on_advice(advice)
    return api, ctx, bus, state, advice


def preview_call(api, population, target='104', season=None):
    return api.population_preview(population, target, season or api._ctx.db.season_id)


def test_population_preview_snapshot_is_readonly_and_uses_rule_odds(preview_api):
    from jccbrain.modes.assistant import AssistantMode
    from jccbrain.engine.odds import hit_odds
    api, ctx, bus, state, advice = preview_api
    # 正式模式里真实 tracker 的缓存与选择状态也必须保持。
    mode = AssistantMode(ctx, api._sink)
    api._runner.mode = mode
    before = deepcopy((state, advice, vars(mode.tracker), asdict(ctx.settings),
                       ctx.extra_taken, ctx.game_lock_id, ctx.game_watch_ids, ctx.scout.latest()))
    hud = serialize.hud(state, advice, ctx.db, ctx.layout, ctx.settings.hud)
    events = deepcopy(bus.events)
    auto = data(preview_call(api, None))
    assert auto['board'] == serialize.advice(advice)['board']
    assert auto['targets'] == serialize.advice(advice)['targets']
    assert auto['population'] == 6
    for level in (8, 9, 1, 10, 8):
        result = data(preview_call(api, level))
        assert result['population'] == result['board']['population'] == level
        pool = ctx.advisor.pool(state, ctx.extra_taken)
        for row, original in zip(result['targets'], advice.targets):
            expected = hit_odds(pool, original.name, level, max(original.need, 1))
            assert row['per_roll'] == (expected.per_roll if expected else None)
            assert (row['have'], row['need'], row['target_star']) == (original.have, original.need, original.target_star)
        assert result == data(preview_call(api, level))
    assert data(preview_call(api, 8))['board']['source'] == '攻略 L8'
    assert data(preview_call(api, 9))['board']['source'] == '攻略 L8'
    assert auto['targets'] != data(preview_call(api, 8))['targets']  # OCR 一费100%与8级规则分离
    assert before == (state, advice, vars(mode.tracker), asdict(ctx.settings),
                      ctx.extra_taken, ctx.game_lock_id, ctx.game_watch_ids, ctx.scout.latest())
    assert hud == serialize.hud(state, advice, ctx.db, ctx.layout, ctx.settings.hud)
    assert events == bus.events


@pytest.mark.parametrize('value', [True, False, 0, -1, 11, 8.0, 8.5, '8', [], {}])
def test_population_preview_rejects_invalid_population(preview_api, value):
    api = preview_api[0]
    assert not preview_call(api, value)['ok']


@pytest.mark.parametrize('target,season', [(True, '18_S19'), ('104', 18), ('old-target', None), ('104', 'old-season')])
def test_population_preview_rejects_stale_context(preview_api, target, season):
    assert not preview_call(preview_api[0], 8, target, season)['ok']


def test_population_preview_auto_unknown_and_manual_known_level(preview_api):
    api, ctx, _, _, _ = preview_api
    state = GameState.from_dict(make_state(level=None, board_count=None))
    advice = ctx.advisor.advise(state, '104')
    api._on_state(state)
    api._on_advice(advice)
    auto = data(preview_call(api, None))
    assert auto['population'] is None and auto['board'] is None
    assert all(row['per_roll'] is None for row in auto['targets'])
    assert any(row['per_roll'] is not None for row in data(preview_call(api, 8))['targets'])


@pytest.mark.parametrize('kind', ['none', 'ended', 'no-hud', 'no-target', 'next-stage'])
def test_population_preview_clears_without_current_target(preview_api, kind):
    from jccbrain.engine.advisor import Advice
    from jccbrain.core.models import Phase
    api = preview_api[0]
    if kind == 'none': api._on_state(None)
    if kind == 'ended': api._on_state(GameState.from_dict(make_state(phase='ended')))
    if kind == 'no-hud': api._on_state(GameState())
    if kind == 'no-target': api._on_advice(Advice(stage='3-2', phase=Phase.PLANNING))
    if kind == 'next-stage': api._on_state(GameState.from_dict(make_state(stage='3-3')))
    assert data(preview_call(api, 8)) is None


def test_population_preview_rejects_response_if_advice_changes_mid_compute(preview_api, monkeypatch):
    import jccbrain.webui.api as module
    api, _, _, _, advice = preview_api
    entered, release = threading.Event(), threading.Event()
    real = module.population_preview
    def blocked(*args):
        entered.set()
        assert release.wait(2)
        return real(*args)
    monkeypatch.setattr(module, 'population_preview', blocked)
    result = []
    worker = threading.Thread(target=lambda: result.append(preview_call(api, 8)))
    worker.start()
    assert entered.wait(2)
    # 模拟下一次建议先更新回调，再等ctx锁执行图标预取。
    with api._state_lock: api._last_advice = deepcopy(advice)
    release.set()
    worker.join(2)
    assert not worker.is_alive() and not result[0]['ok']


def test_population_preview_capacity_and_paired_advice_snapshot(preview_api):
    from jccbrain.engine.odds import hit_odds
    api, ctx, _, _, _ = preview_api
    state = GameState.from_dict(make_state(level=6, board_count=[7, 7]))
    advice = ctx.advisor.advise(state, '104')
    api._on_state(state)
    api._on_advice(advice)
    assert data(preview_call(api, None))['population'] == 7  # 容量不等于等级
    # 下一帧尚未产生新建议；只读展示仍与已有建议的持有量/缺张配对。
    latest = GameState.from_dict(make_state(level=6, board=[{'name': '绯红树怪', 'star': 2}]))
    api._on_state(latest)
    result = data(preview_call(api, 8))
    row = next(row for row in result['targets'] if row['name'] == '绯红树怪')
    original = next(row for row in advice.targets if row.name == '绯红树怪')
    expected = hit_odds(ctx.advisor.pool(state), original.name, 8, max(original.need, 1))
    assert row['have'] == 0 and row['per_roll'] == expected.per_roll
    api._on_advice(ctx.advisor.advise(latest, '104'))
    assert next(row for row in data(preview_call(api, 8))['targets'] if row['name'] == '绯红树怪')['have'] == 3


def test_dedicated_toast_clears_even_with_both_general_huds_off(setup_api):
    api,ctx,bus=setup_api
    hud=FakeHud();hud.set_notice=lambda n:None
    api._hud=hud
    scout=SimpleNamespace(status={'notice':{'id':'test','stage':'4-1','seconds':3}})
    api._runner=Runner()
    api._runner.mode=SimpleNamespace(scout=scout)
    api._emit_hud()
    assert [p for k,p in bus.events if k=='hud'][-1]['scout_notice']['id']=='test'
    scout.status['notice']=None
    api._emit_hud()
    assert [p for k,p in bus.events if k=='hud'][-1]['scout_notice'] is None
    count=len(bus.events);api._emit_hud()
    assert len(bus.events)==count  # 一般HUD关闭且没有提示时保留原惰性行为。
