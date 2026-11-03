"""三渠道消费契约：原始目录、关联推荐、即时刷新和共享官方快照。"""
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jccbrain.context import AppContext
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.data.db import enabled_comp_sources
from jccbrain.data.models import Guide, GuideHero
from jccbrain.data.sync import SyncError
from jccbrain.modes.assistant import AssistantMode
from jccbrain.webui.api import Api
from test_comp_sources import Bus, data
from test_ingame_sync import IngameOfficial, STAT
from test_seasons import FakeDataj, GAMEDATA, COMP, LINEUP


SWITCHES = list(product((False, True), repeat=3))


@pytest.fixture
def channels(db, monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    ctx = AppContext(Settings())
    ctx.db = db
    statistical = db.comp('104')
    for gid, membership, linked in [('web', ('web',), True), ('community', ('ingame',), True),
                                     ('shared', ('web', 'ingame'), False)]:
        guide = Guide(gid, f'{gid}原始攻略', 'S', None, early_info=f'{gid}专属过渡',
                      author=f'{gid}作者', channels=membership,
                      use_count=100 if 'ingame' in membership else None,
                      boards={'final': [GuideHero(h.hero_id, h.name, is_carry=h.is_carry, cost=h.cost)
                                        for h in statistical.heroes]})
        original = db._guide_to_comp(guide)
        original.guide, original.guides = guide, [guide]
        guide.comp = statistical if linked else original
        db.guides[gid], db.official_comps[original.id] = guide, original
        if linked:
            statistical.guides.append(guide)
    statistical.guide = statistical.guides[0]
    statistical.guide_id = statistical.guide.id
    api = Api(ctx, Bus())
    yield ctx, api
    api._shutdown()
    ctx.close()


@pytest.mark.parametrize('official,ingame,dataj', SWITCHES)
def test_all_source_combinations_keep_original_catalog_and_ranked_dedup(channels, official, ingame, dataj):
    ctx, api = channels
    flags = dict(official=official, ingame=ingame, dataj=dataj)
    data(api.save_settings({'data': {'source_' + key: on for key, on in flags.items()}}))
    assert enabled_comp_sources(ctx.settings.data) == {key for key, on in flags.items() if on}
    assert data(api.status())['sources'] == flags
    expected = ({'104'} if dataj else set()) | ({'official:web'} if official else set())
    expected |= {'official:community'} if ingame else set()
    expected |= {'official:shared'} if official or ingame else set()
    rows = data(api.comps())['rows']
    assert {row['id'] for row in rows} == expected and len(rows) == len(expected)
    assert {row['id'] for row in data(api.comps('official'))['rows']} == (
        {'official:web', 'official:shared'} if official else set())
    assert {row['id'] for row in data(api.comps('ingame'))['rows']} == (
        {'official:community', 'official:shared'} if ingame else set())
    assert {row['id'] for row in data(api.comps('dataj'))['rows']} == ({'104'} if dataj else set())
    ranked = expected - ({'official:web', 'official:community'} if dataj else set())
    assert {c.id for c in ctx.enabled_comps()} == ranked
    assert {match.comp.id for match in ctx.advisor.advise(GameState(stage='2-1')).comps} == ranked
    choices = data(api.feature_settings())['comp_options']
    assert {choice['value'] for choice in choices} == expected | {''}
    for cid in ['104', 'official:web', 'official:community', 'official:shared']:
        available = cid in expected
        assert api.comp_detail(cid)['ok'] == available
        assert api.lock_comp_for_game(cid)['ok'] == available
        assert api.set_watch_comps([cid])['ok'] == available
        assert api.set_default_lock_comp(cid)['ok'] == available
        if available and cid.startswith('official:'):
            detail = data(api.comp_detail(cid))
            gid = cid.split(':')[1]
            assert detail['id'] == cid and detail['name'] == f'{gid}原始攻略'
            assert detail['avg_placement'] is None and detail['top4_rate'] is None
            assert detail['guide']['id'] == gid and detail['guide']['text'] == f'【前期过渡】{gid}专属过渡'
            assert ctx.enabled_comp(cid).id == ('104' if dataj and gid != 'shared' else cid)
    if dataj:
        view = ctx.enabled_comp('104')
        expected_guides = (['web'] if official else []) + (['community'] if ingame else [])
        assert [g.id for g in view.guides] == expected_guides
        assert (view.guide.id if view.guide else None) == next(iter(expected_guides), None)
        assert (data(api.comp_detail('104'))['guide'] is None) == (not expected_guides)
    # 过滤不能改共享缓存或关联顺序，重新启用后仍可读取全部指南。
    assert [g.id for g in ctx.db.comps['104'].guides] == ['web', 'community']


def test_source_payloads_identify_ingame_only_and_channel_counts(channels):
    _, api = channels
    payload = data(api.comps())
    community = next(row for row in payload['rows'] if row['id'] == 'official:community')
    shared = next(row for row in payload['rows'] if row['id'] == 'official:shared')
    assert community['source'] == 'official' and community['channels'] == ['ingame']
    assert community['source_label'] == '微社区'
    assert shared['source_label'] == '官网 / 微社区'
    assert payload['info']['official_count'] == 2 and payload['info']['ingame_count'] == 2
    assert payload['info']['dual_channel_count'] == 1 and payload['info']['catalog_count'] == 4
    stats = data(api.db_stats())['stats']
    assert stats['web_guides'] == 2 and stats['ingame_guides'] == 2 and stats['dual_channel_guides'] == 1
    assert stats['comps_total'] == 2 and stats['comps_catalog_total'] == 4


def test_original_selection_status_stays_on_requested_record(channels):
    ctx, api = channels
    cid = 'official:community'
    assert ctx.enabled_comp(cid).id == '104'
    status = data(api.lock_comp_for_game(cid))
    assert status['lock'] == {'id': cid, 'name': 'community原始攻略', 'source': 'game', 'canonical_id': '104', 'available': True}
    status = data(api.set_default_lock_comp(cid))
    assert status['default_lock'] == {'id': cid, 'name': 'community原始攻略', 'canonical_id': '104', 'available': True}
    status = data(api.set_watch_comps([cid]))
    assert status['watch'] == [{'id': cid, 'name': 'community原始攻略', 'canonical_id': '104', 'available': True}]
    advice = ctx.advisor.advise(GameState(stage='2-1'), ctx.effective_lock_id())
    assert advice.target.comp.id == '104'
    assert data(api.set_watch_comps([]))['watch'] == []


@pytest.mark.parametrize('method', ['set_source', 'save_settings'])
def test_ingame_switch_invalidates_running_mode_but_retains_watch(channels, method):
    ctx, api = channels
    data(api.save_settings({'data': {'source_official': False, 'source_dataj': False}}))
    cid = 'official:community'
    data(api.lock_comp_for_game(cid))
    data(api.set_default_lock_comp(cid))
    data(api.set_watch_comps([cid]))
    mode = AssistantMode(ctx, api._sink)
    stop = Mock()
    api._runner = SimpleNamespace(mode=mode, running=True, stop=stop)
    state = GameState(stage='4-1')
    api._sink.on_state(state)
    mode.compute_advice(state)
    assert mode.tracker.target_id == cid
    mode.tracker.reset = Mock(wraps=mode.tracker.reset)
    if method == 'set_source':
        assert not data(api.set_source('ingame', False))['ingame']
    else:
        data(api.save_settings({'data': {'source_ingame': False}}))
    stop.assert_not_called()
    mode.tracker.reset.assert_called_once()
    assert api._runner.mode is mode and mode.tracker.target_id is None
    assert ctx.game_lock_id == cid and ctx.game_watch_ids == [cid]
    assert ctx.settings.assistant.default_lock_comp_id == cid
    assert api._last_advice.target is None and api._last_advice.comps == []
    assert {name for name, _ in api._bus.events} >= {'data_changed', 'lock_changed', 'watch_changed', 'status', 'advice'}
    assert data(api.status())['comp_source_notice'] == '三个数据源都已关闭，没有可推荐的阵容'
    assert not Settings.load().data.source_ingame
    data(api.set_source('ingame', True))
    assert api._last_advice.target is None
    api._sink.on_advice(mode.compute_advice(state))
    assert api._last_advice.target and data(api.comps('ingame'))['rows']


@pytest.mark.parametrize('direction', ['lock', 'watch'])
def test_dual_channel_lock_survives_one_channel_closing(channels, direction):
    ctx, api = channels
    cid = 'official:shared'
    data(api.set_default_lock_comp(cid))
    if direction == 'lock':
        data(api.lock_comp_for_game(cid))
    else:
        data(api.set_watch_comps([cid]))
    data(api.set_source('official', False))
    assert ctx.game_lock_id == (cid if direction == 'lock' else None)
    assert ctx.game_watch_ids == ([cid] if direction == 'watch' else [])
    assert ctx.settings.assistant.default_lock_comp_id == cid and api.comp_detail(cid)['ok']
    data(api.set_source('ingame', False))
    assert ctx.effective_lock_id() is None and ctx.game_watch_ids == ([cid] if direction == 'watch' else [])
    assert ctx.settings.assistant.default_lock_comp_id == cid and not api.comp_detail(cid)['ok']
    assert ctx.assistant_selection()['candidate_ids'] == []
    data(api.set_source('ingame', True))
    assert ctx.effective_lock_id() == cid


@pytest.mark.parametrize('legacy', [{}, {'source_official': False}, {'source_official': True},
                                  {'use_official': False}, {'use_official': True},
                                  {'source_official': False, 'source_ingame': True},
                                  {'source_official': True, 'source_ingame': False},
                                  {'use_official': True, 'source_official': False}])
def test_new_channel_settings_inherit_old_official_choice(tmp_path, legacy):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'data': legacy}))
    loaded = Settings.load(path)
    official = legacy.get('source_official', legacy.get('use_official', True))
    assert loaded.data.source_official == official
    assert loaded.data.source_ingame == legacy.get('source_ingame', official)
    loaded.save(path)
    assert Settings.load(path) == loaded
    assert Settings().data.source_ingame and Settings.load(tmp_path / 'missing.json').data.source_ingame


@pytest.mark.parametrize('official,ingame,dataj', SWITCHES)
def test_context_sync_shares_full_official_snapshot(tmp_path, monkeypatch, official, ingame, dataj):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    from jccbrain.data import official as off, dataj as dj

    class SnapshotOfficial(IngameOfficial):
        def lineups(self, entry, season_id=None):
            calls.append('web')
            web_only = deepcopy(LINEUP)
            web_only['id'] = 'web-only'
            return [deepcopy(LINEUP), web_only]

        def ingame_lineups(self, sid):
            calls.append('ingame')
            return super().ingame_lineups(sid)

        def lineup_stats(self, ids):
            assert set(ids) == {LINEUP['id'], 'web-only', 'new'}
            return {LINEUP['id']: dict(STAT)}

        def close(self):
            calls.append('closed')

    calls = []
    def create_official():
        calls.append('official-client')
        return SnapshotOfficial()
    def create_dataj(*args):
        calls.append('dataj-client')
        client = FakeDataj(GAMEDATA, [COMP])
        client.close = lambda: calls.append('dataj-closed')
        return client
    monkeypatch.setattr(off, 'OfficialClient', create_official)
    monkeypatch.setattr(dj, 'DatajClient', create_dataj)
    ctx = AppContext(Settings())
    ctx.settings.data.source_official, ctx.settings.data.source_ingame, ctx.settings.data.source_dataj = official, ingame, dataj
    try:
        if not (official or ingame or dataj):
            with pytest.raises(SyncError, match='至少要选一个数据源'):
                ctx.sync(False)
            with pytest.raises(SyncError, match='至少要选一个数据源'):
                ctx.sync_all_seasons(False)
            assert calls == []
            return
        result = ctx.sync(False)
        assert ('official-client' in calls) == (official or ingame)
        assert ('dataj-client' in calls) == dataj
        if official or ingame:
            assert calls.count('web') == 1 and calls.count('ingame') == 1 and calls.count('closed') == 1
            assert {g.id: g.channels for g in ctx.db.guides.values()} == {
                LINEUP['id']: ('web', 'ingame'), 'web-only': ('web',), 'new': ('ingame',)}
            assert result['data_source'] == 'official'
            assert all(c.avg_placement is None for c in ctx.enabled_comp_catalog() if c.source == 'official')
        else:
            assert not ctx.db.guides and result['data_source'] == 'dataj'
    finally:
        if 'db' in ctx.__dict__:
            ctx.db.close()
        ctx.close()


def test_only_ingame_checks_official_versions(tmp_path, monkeypatch):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    from jccbrain.data import official as off
    client = IngameOfficial()
    client.close = Mock()
    create = Mock(return_value=client)
    monkeypatch.setattr(off, 'OfficialClient', create)
    ctx = AppContext(Settings())
    ctx.settings.data.source_official = ctx.settings.data.source_dataj = False
    try:
        assert ctx.seasons_needing_sync()
        create.assert_called_once()
        client.close.assert_called_once()
    finally:
        ctx.db.close()
        ctx.close()


def test_unsupported_season_ingame_filter_is_empty_and_explicit(channels):
    ctx, api = channels
    data(api.switch_season('16_S19'))
    data(api.save_settings({'data': {'source_official': False, 'source_dataj': False}}))
    result = data(api.comps('ingame'))
    assert result['rows'] == [] and result['info']['source_notice'] == '当前赛季暂不支持微社区阵容'
    assert data(api.status())['comp_source_notice'] == '当前赛季暂不支持微社区阵容'
    assert not next(s for s in data(api.seasons()) if s['current'])['has_ingame']
