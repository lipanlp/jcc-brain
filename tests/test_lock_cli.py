"""一次性 CLI 建议共用锁定语义，不访问真实视觉或网络。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import make_state
from jccbrain import cli
from jccbrain.context import AppContext
from jccbrain.core.models import GameState
from jccbrain.core.settings import Settings
from jccbrain.engine.advisor import HeuristicAdvisor
from jccbrain.engine.llm import LLMReply


def context(db, rules, game, default):
    ctx = AppContext(Settings())
    ctx.db = db
    ctx.game_lock_id = game
    ctx.settings.assistant.default_lock_comp_id = default
    ctx.vision = SimpleNamespace(read=lambda: GameState.from_dict(make_state()))
    ctx.advisor = SimpleNamespace(advise=Mock(wraps=HeuristicAdvisor(db, rules).advise))
    ctx.close = Mock()
    return ctx


@pytest.mark.parametrize('game,default,expected', [(None, '104', '104'), ('', '104', None), ('104', '', '104')])
def test_advise_once_uses_effective_lock(db, rules, monkeypatch, capsys, game, default, expected):
    ctx = context(db, rules, game, default)
    monkeypatch.setattr(cli, '_ctx', lambda args: ctx)
    assert cli.cmd_advise(SimpleNamespace(once=True)) == 0
    assert ctx.advisor.advise.call_args.args[1] == expected
    ctx.close.assert_called_once_with()
    assert '目标阵容' in capsys.readouterr().out


@pytest.mark.parametrize('locked', [True, False])
def test_llm_cli_only_prints_switch_when_unlocked(db, rules, monkeypatch, capsys, locked):
    ctx = context(db, rules, None, '104' if locked else '')
    ctx.settings.llm.api_key = 'unit-test-key'
    ctx.settings.llm.model = 'unit-test-model'
    monkeypatch.setattr(cli, '_ctx', lambda args: ctx)
    reply = LLMReply('运营建议', [], '理由', switch_comp='另一套阵容')
    monkeypatch.setattr('jccbrain.engine.llm.LLMClient', lambda cfg: SimpleNamespace(advise=lambda *args: reply))
    assert cli.cmd_llm(SimpleNamespace(action='ask')) == 0
    output = capsys.readouterr().out
    assert ('建议转：' in output) is (not locked)
    if not locked:
        assert '另一套阵容' in output
    ctx.close.assert_called_once_with()
