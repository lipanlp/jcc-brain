from copy import deepcopy
import json
from dataclasses import replace

import pytest

from jccbrain.context import AppContext
from jccbrain.core.settings import Settings
from jccbrain.core.models import GameState
from jccbrain.data.models import Guide
from jccbrain.engine.tracker import CompTracker
from jccbrain.modes.autoroll import AutoRollMode
from jccbrain.modes.base import ModeSink
from jccbrain.webui.api import Api
from jccbrain.webui import settings_schema


class Bus:
    def __init__(self): self.events = []
    def emit(self, name, payload):
        json.dumps(payload)
        self.events.append((name, deepcopy(payload)))


def data(result):
    assert result['ok'], result
    return result['data']


@pytest.fixture
def sources(db, monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    ctx = AppContext(Settings())
    ctx.db = db
    # 已关联dataj的官网推荐：旧comp()会把它直接变成dataj对象。
    official = replace(db.comp('104'), id='official:test', source='official', name='官网野怪',
                       avg_placement=None, guide_id='test')
    guide = Guide('test', '官网野怪', 'S', None)
    guide.comp = db.comp('104')
    official.guide = guide
    db.guides['test'] = guide
    db.official_comps[official.id] = official
    bus = Bus()
    api = Api(ctx, bus)
    yield ctx, api, bus
    api._shutdown()
    ctx.close()


@pytest.mark.parametrize('official,dataj,expected', [(True,False,{'official'}), (False,True,{'dataj'}), (False,False,set())])
def test_sources_filter_matcher_lists_defaults_and_validation(sources, official, dataj, expected):
    ctx, api, _ = sources
    data(api.save_settings({'data': {'source_official':official, 'source_ingame':False, 'source_dataj':dataj}}))
    assert {c.source for c in ctx.enabled_comps()} == expected
    assert {m.comp.source for m in ctx.advisor.advise(GameState(stage='2-1')).comps} == expected
    for source in ('all','dataj','official'):
        assert {c['source'] for c in data(api.comps(source))['rows']} <= expected
    for id, source in [('104','dataj'), ('official:test','official')]:
        assert api.lock_comp_for_game(id)['ok'] == (source in expected)
        assert api.set_watch_comps([id])['ok'] == (source in expected)
        assert api.set_default_lock_comp(id)['ok'] == (source in expected)
        assert api.comp_detail(id)['ok'] == (source in expected)
    if not expected:
        assert data(api.status())['comp_source_notice'] == '三个数据源都已关闭，没有可推荐的阵容'
        assert data(api.autoroll_defaults())['heroes'] == []
        assert data(api.odds_import_target()) == {'comp_name': None, 'rows': []}
        assert AutoRollMode(ctx, ModeSink()).resolve_targets(GameState(stage='2-1')) == []
    elif expected == {'official'}:
        data(api.lock_comp_for_game(''))  # #15关注不再隐式解除手动锁定；这里显式进入自动。
        # 本局已显式解除锁定，且没有当前有效建议；D牌不能借目录首项冒充当前阵容。
        assert data(api.autoroll_defaults())['comp_name'] is None
        data(api.lock_comp_for_game('official:test'))
        assert data(api.autoroll_defaults())['comp_name'] == '官网野怪'
        assert data(api.odds_import_target())['comp_name'] == '官网野怪'
        choices = data(api.feature_settings())['comp_options']
        assert {c['value'] for c in choices} == {'','official:test'}


def test_source_change_clears_invalid_locks_and_preserves_watch_without_fallback(sources):
    ctx, api, bus = sources
    advisor = ctx.advisor
    data(api.lock_comp_for_game('104'))
    # #15保留关注记录，锁定优先；来源关闭后无交集不得全库兜底。
    ctx.game_watch_ids = ['104']
    data(api.set_default_lock_comp('104'))
    api._sink.on_state(GameState(stage='2-1'))
    before = advisor.advise(GameState(stage='2-1'), tracker=CompTracker())
    assert before.target.comp.source == 'dataj'
    data(api.save_settings({'data': {'source_dataj':False}}))
    assert ctx.advisor is advisor
    assert ctx.game_lock_id == '104' and ctx.game_watch_ids == ['104'] and ctx.settings.assistant.default_lock_comp_id == '104'
    assert advisor.advise(GameState(stage='2-1')).target.comp.source == 'official'
    assert api._last_advice.target is None
    assert any(n=='data_changed' for n,_ in bus.events)
    logs=[v['text'] for n,v in bus.events if n=='log']
    assert any('本局锁定保留但暂不可用' in s for s in logs)
    assert any('记录保留但暂不可用' in s for s in logs)
    assert any('默认锁定保留但暂不可用' in s for s in logs)
    data(api.save_settings({'data': {'source_dataj':True}}))
    assert advisor.advise(GameState(stage='2-1')).target.comp.source == 'dataj'
    assert api._last_advice.target.comp.source == 'dataj'


def test_same_round_settled_tracker_loses_disabled_target(sources):
    ctx, api, _ = sources
    tracker = CompTracker()
    state = GameState(stage='4-1')
    assert ctx.advisor.advise(state, tracker=tracker).target.comp.source == 'dataj'
    assert tracker.settled
    data(api.save_settings({'data':{'source_dataj':False}}))
    assert ctx.advisor.advise(state, tracker=tracker).target.comp.source == 'official'
    data(api.save_settings({'data':{'source_official':False, 'source_ingame':False}}))
    assert ctx.advisor.advise(state, tracker=tracker).comps == []
    assert tracker.target_id is None


def test_official_link_never_reintroduces_disabled_dataj(sources):
    ctx, api, _ = sources
    data(api.save_settings({'data':{'source_dataj':False}}))
    assert ctx.enabled_comp('official:test').source == 'official'
    assert ctx.enabled_comp('104') is None
    assert ctx.enabled_comps()[0].name == '官网野怪'


def test_schema_save_checks_final_switches_regardless_field_order(sources):
    ctx, api, _ = sources
    data(api.save_settings({'data': {'source_dataj':False}}))
    data(api.save_settings({'assistant':{'default_lock_comp_id':'104'}, 'data': {'source_dataj':True}}))
    assert ctx.settings.assistant.default_lock_comp_id == '104'
    old = deepcopy(ctx.settings)
    assert not api.save_settings({'assistant':{'default_lock_comp_id':'official:test'}, 'data':{'source_official':False}})['ok']
    assert ctx.settings == old


def test_disabled_saved_default_not_used_when_starting(sources):
    ctx, _, _ = sources
    ctx.settings.data.source_dataj = False
    ctx.settings.assistant.default_lock_comp_id = '104'
    assert ctx.effective_lock_id() is None
    mode = AutoRollMode(ctx, ModeSink())
    mode.resolve_targets(GameState(stage='2-1'))
    assert ctx.game_lock_id == ''


def test_source_only_save_keeps_active_mode(sources):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from jccbrain.modes.assistant import AssistantMode
    ctx, api, _ = sources
    mode = AssistantMode(ctx, api._sink)
    stop = Mock()
    api._runner = SimpleNamespace(mode=mode, running=True, stop=stop)
    api._sink.on_state(GameState(stage='2-1'))
    mode.compute_advice(api._last_state)
    data(api.save_settings({'data': {'source_dataj': False}}))
    stop.assert_not_called()
    assert api._last_advice.target is None  # 等下一帧有效状态，不拿旧状态冒充新范围建议。
    api._sink.on_advice(mode.compute_advice(api._last_state))
    assert api._runner.mode is mode and api._last_advice.target.comp.source == 'official'
