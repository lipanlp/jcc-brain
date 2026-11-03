"""来源选择身份与算法身份分开，别名不会丢失或重复参与评分。"""
from dataclasses import replace

import pytest

from jccbrain.core.models import GameState
from jccbrain.data.models import Guide
from jccbrain.modes.assistant import AssistantMode
from jccbrain.modes.base import ModeSink
from test_comp_presets import presets, create, enable
from test_comp_sources import data


@pytest.fixture
def aliases(presets):
    ctx, api = presets
    for gid in ('first', 'second'):
        guide = Guide(gid, '来源' + gid, 'D', None, channels=['web', 'ingame'])
        guide.comp = ctx.db.comps['A']
        ctx.db.guides[gid] = guide
        alias = replace(ctx.db.comps['A'], id='official:' + gid, source='official', name=guide.name,
                        guide=guide, guides=[guide], guide_id=gid)
        ctx.db.official_comps[alias.id] = alias
    ctx.settings.data.source_official = ctx.settings.data.source_ingame = ctx.settings.data.source_dataj = True
    ctx.advisor.reload()
    return ctx, api


def advice(ctx):
    return AssistantMode(ctx, ModeSink()).compute_advice(GameState(stage='3-2', level=7, gold=50))


def test_watch_status_retains_source_identity_and_exposes_canonical(aliases):
    ctx, api = aliases
    result = data(api.set_watch_comps(['official:first']))
    assert result['watch'] == [{'id': 'official:first', 'name': '来源first', 'canonical_id': 'A', 'available': True}]
    assert ctx.game_watch_ids == ['official:first']
    assert ctx.assistant_selection()['selection_map']['official:first'] == 'A'
    assert advice(ctx).target.comp.id == 'A'


@pytest.mark.parametrize('official,ingame,dataj', [(True, True, True), (False, True, False), (False, True, True), (True, True, False)])
def test_source_matrix_maps_to_current_algorithm_identity(aliases, official, ingame, dataj):
    ctx, api = aliases
    data(api.set_watch_comps(['official:first']))
    data(api.save_settings({'data': {'source_official': official, 'source_ingame': ingame, 'source_dataj': dataj}}))
    canonical = 'A' if dataj else 'official:first'
    assert data(api.status())['watch'][0]['canonical_id'] == canonical
    assert ctx.assistant_selection()['candidate_ids'] == [canonical]
    assert advice(ctx).target.comp.id == canonical
    assert ctx.game_watch_ids == ['official:first']


def test_aliases_count_as_user_choices_but_algorithm_only_once(aliases):
    ctx, api = aliases
    data(api.set_watch_comps(['official:first', 'official:second', 'A']))
    assert len(data(api.status())['watch']) == 3
    assert ctx.assistant_selection()['candidate_ids'] == ['A']
    assert len(advice(ctx).comps) == 1
    revision = ctx.assistant_selection()['revision']
    data(api.set_watch_comps(['official:second', 'A']))
    assert ctx.game_watch_ids == ['official:second', 'A']
    assert ctx.assistant_selection()['revision'] == revision
    assert len(advice(ctx).comps) == 1


@pytest.mark.parametrize('lock_kind', ['game', 'default'])
def test_lock_source_disabled_pauses_and_restores_original_choice(aliases, lock_kind):
    ctx, api = aliases
    method = api.lock_comp_for_game if lock_kind == 'game' else api.set_default_lock_comp
    data(method('official:first'))
    assert data(api.status())['lock']['id'] == 'official:first'
    assert data(api.status())['lock']['canonical_id'] == 'A'
    data(api.save_settings({'data': {'source_official': False, 'source_ingame': False}}))
    assert (ctx.game_lock_id if lock_kind == 'game' else ctx.settings.assistant.default_lock_comp_id) == 'official:first'
    scope = ctx.assistant_selection()
    assert scope['candidate_ids'] == [] and '暂停' in scope['notice']
    assert advice(ctx).target is None
    data(api.set_source('ingame', True))
    assert data(api.status())['lock']['id'] == 'official:first'
    assert advice(ctx).target.comp.id == 'A'


def test_watch_source_disabled_retained_and_restored(aliases):
    ctx, api = aliases
    data(api.set_watch_comps(['official:first']))
    data(api.save_settings({'data': {'source_official': False, 'source_ingame': False}}))
    result = data(api.status())['watch'][0]
    assert result == {'id': 'official:first', 'name': '来源first', 'canonical_id': None, 'available': False}
    assert ctx.assistant_selection()['candidate_ids'] == [] and advice(ctx).target is None
    data(api.set_source('ingame', True))
    assert ctx.game_watch_ids == ['official:first'] and advice(ctx).target.comp.id == 'A'


def test_deleted_linked_dataj_uses_available_original_guide(aliases):
    ctx, api = aliases
    data(api.set_watch_comps(['official:first']))
    del ctx.db.comps['A']
    ctx.advisor.reload()
    assert ctx.enabled_comp('official:first').id == 'official:first'
    assert ctx.assistant_selection()['candidate_ids'] == ['official:first']
    assert advice(ctx).target.comp.id == 'official:first'


def test_preset_alias_intersection_lock_priority_and_no_fallback(aliases):
    ctx, api = aliases
    preset = create(api, ['official:first'])
    enable(api, preset)
    data(api.set_watch_comps(['official:second']))
    assert ctx.assistant_selection()['candidate_ids'] == ['A']
    data(api.lock_comp_for_game('B'))
    assert ctx.assistant_selection()['lock_outside'] and advice(ctx).target.comp.id == 'B'
    data(api.lock_comp_for_game(''))
    assert advice(ctx).target.comp.id == 'A'
    data(api.set_watch_comps(['B']))
    assert ctx.assistant_selection()['candidate_ids'] == [] and advice(ctx).target is None


@pytest.mark.parametrize('new_game', [False, True])
def test_auxiliary_initializes_unavailable_default_without_erasing_it(aliases, new_game):
    ctx, api = aliases
    data(api.set_default_lock_comp('official:first'))
    data(api.save_settings({'data': {'source_official': False, 'source_ingame': False}}))
    ctx.game_lock_id = None
    mode = AssistantMode(ctx, ModeSink())
    state = GameState(stage='3-2', level=7, gold=50)
    assert mode.compute_advice(state, new_game=new_game).target is None
    assert ctx.game_lock_id == 'official:first'
    assert '来源暂不可用' in ctx.assistant_selection()['notice']
    data(api.set_source('ingame', True))
    assert mode.compute_advice(state).target.comp.id == 'A'


def test_existing_unavailable_default_can_be_preserved_but_new_invalid_rejected(aliases):
    ctx, api = aliases
    data(api.set_default_lock_comp('official:first'))
    data(api.save_settings({'data': {'source_official': False, 'source_ingame': False}}))
    data(api.save_feature_settings({'assistant': {'default_lock_comp_id': 'official:first'}}))
    assert not api.save_feature_settings({'assistant': {'default_lock_comp_id': 'official:second'}})['ok']
    assert ctx.settings.assistant.default_lock_comp_id == 'official:first'
    assert ctx.effective_lock_id() is None


def test_source_refresh_without_running_mode_uses_canonical_lock(aliases):
    ctx, api = aliases
    data(api.lock_comp_for_game('official:first'))
    api._sink.on_state(GameState(stage='3-2', level=7, gold=50))
    data(api.set_source('official', False))
    assert api._last_advice.target.comp.id == 'A' and api._last_advice.locked
    assert data(api.status())['lock']['id'] == 'official:first'
