"""默认 GUI 入口和缺失构建的回归；不创建原生窗口。"""
from unittest.mock import Mock

import pytest

from jccbrain import cli
from jccbrain.webui import app


@pytest.mark.parametrize('argv', [[], ['gui']])
def test_gui_and_default_command_use_webui(monkeypatch, tmp_path, argv):
    monkeypatch.setenv('JCC_HOME', str(tmp_path))
    monkeypatch.setattr(cli, 'setup_logging', lambda *_: None)
    launch = Mock(return_value=17)
    monkeypatch.setattr(app, 'run_webui', launch)
    assert cli.main(argv) == 17
    launch.assert_called_once_with()


def test_help_only_offers_current_gui_entry(capsys):
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert '打开图形界面（默认）' in help_text and 'webgui' not in help_text
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['webgui'])
    assert exc.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_missing_static_fails_before_creating_context(monkeypatch, tmp_path):
    monkeypatch.delenv('JCC_WEB_DEV', raising=False)
    monkeypatch.setattr(app, 'Path', lambda _: tmp_path / 'app.py')
    context = Mock()
    monkeypatch.setattr(app, 'AppContext', context)
    with pytest.raises(RuntimeError, match='界面尚未构建：请先运行 make web（打包版不应出现此错误）'):
        app.run_webui()
    context.assert_not_called()
