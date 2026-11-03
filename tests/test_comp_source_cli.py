"""命令行三源覆盖、渠道标签与默认关联去重，不访问网络。"""
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jccbrain import cli
from jccbrain.core.settings import Settings
from test_ingame_sources import channels


@pytest.mark.parametrize('picked', [None] + [','.join(group) for size in (1, 2, 3)
                                         for group in combinations(('official', 'ingame', 'dataj'), size)])
def test_explicit_cli_sources_override_all_three_flags(monkeypatch, picked):
    settings = Settings()
    settings.data.source_official = settings.data.source_dataj = False
    ctx = SimpleNamespace(settings=settings, sync=Mock())
    monkeypatch.setattr(cli, '_ctx', lambda _: ctx)
    args = cli.build_parser().parse_args(['db', 'sync'] + (['--sources', picked] if picked else []))
    assert args.func(args) == 0
    assert {key for key in ('official', 'ingame', 'dataj') if getattr(settings.data, 'source_' + key)} == (
        set(picked.split(',')) if picked else {'ingame'})
    ctx.sync.assert_called_once_with(details=True, progress=print, season_id=None)


@pytest.mark.parametrize('value', ['unknown', 'official,bad', '', ' , '])
def test_cli_rejects_unknown_or_empty_sources_before_sync(value):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(['db', 'sync', '--sources', value])
    assert exc.value.code == 2


def test_cli_ingame_channel_keeps_original_name_and_default_ranked_semantics(channels, monkeypatch, capsys):
    ctx, _ = channels
    monkeypatch.setattr(cli, '_ctx', lambda _: ctx)
    args = cli.build_parser().parse_args(['comps', '--source', 'ingame', '--top', '100'])
    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert 'community原始攻略' in output and '微社区' in output
    assert 'id=official:community' in output and '官网推荐' not in output.splitlines()[0]
    args = cli.build_parser().parse_args(['comps', '--top', '100'])
    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert 'community原始攻略' not in output and 'id=104' in output
    ctx.settings.data.source_ingame = False
    args = cli.build_parser().parse_args(['comps', '--source', 'ingame'])
    assert args.func(args) == 0
    assert capsys.readouterr().out == ''
