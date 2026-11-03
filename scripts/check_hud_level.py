#!/usr/bin/env python3
"""只激活游戏的 HUD 层级实机检查；不发送鼠标/键盘事件或改变游戏窗口。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def on_main(action):
    """从检查线程同步读取 AppKit；等待有上限，异常传回报告。"""
    from PyObjCTools import AppHelper

    done = threading.Event()
    result = []

    def run():
        try:
            result.append((True, action()))
        except Exception as exc:
            result.append((False, exc))
        finally:
            done.set()

    AppHelper.callAfter(run)
    if not done.wait(10):
        raise TimeoutError('等待 macOS 主线程超时')
    ok, value = result[0]
    if not ok:
        raise value
    return value


def snapshot(hud, game):
    import AppKit
    import Quartz

    rows = list(Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID) or [])
    indexed = {int(row['kCGWindowNumber']): (index, row) for index, row in enumerate(rows)}
    game_entry = indexed.get(game.window_id)
    game_bounds = {key: float(value) for key, value in game_entry[1]['kCGWindowBounds'].items()} if game_entry else None
    front = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    game_app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(game.pid)
    result = {'game_id': game.window_id, 'game_pid': game.pid,
              'checker_pid': os.getpid(), 'game_is_active': bool(game_app and game_app.isActive()),
              'game_order': game_entry[0] if game_entry else None,
              'game_layer': int(game_entry[1]['kCGWindowLayer']) if game_entry else None,
              'game_bounds': game_bounds, 'frontmost_pid': int(front.processIdentifier()) if front else None,
              'windows': {}, 'checks': {}}
    result['checks']['game_on_screen'] = game_entry is not None
    result['checks']['game_keeps_focus'] = result['frontmost_pid'] == game.pid
    for key, expected_level in (('hud', 8), ('hud_panel', 9)):
        native = hud._windows[key].native
        number = int(native.windowNumber())
        entry = indexed.get(number)
        frame = native.frame()
        primary_height = AppKit.NSScreen.screens()[0].frame().size.height
        native_bounds = {'X': float(frame.origin.x), 'Y': float(primary_height - frame.origin.y - frame.size.height),
                         'Width': float(frame.size.width), 'Height': float(frame.size.height)}
        window = {'id': number, 'level': int(native.level()),
                  'collection_behavior': int(native.collectionBehavior()),
                  'ignores_mouse_events': bool(native.ignoresMouseEvents()),
                  'can_become_key': bool(native.canBecomeKeyWindow()),
                  'can_become_main': bool(native.canBecomeMainWindow()),
                  'visible': bool(native.isVisible()), 'native_bounds': native_bounds,
                  'order': entry[0] if entry else None,
                  'layer': int(entry[1]['kCGWindowLayer']) if entry else None,
                  'bounds': {k: float(v) for k, v in entry[1]['kCGWindowBounds'].items()} if entry else None}
        result['windows'][key] = window
        result['checks'][key + '_in_front'] = bool(entry and game_entry and entry[0] < game_entry[0])
        result['checks'][key + '_higher_layer'] = bool(entry and game_entry and window['layer'] > result['game_layer'])
        result['checks'][key + '_expected_level'] = window['level'] == expected_level
    overlay = result['windows']['hud']
    result['checks']['overlay_mouse_passthrough'] = overlay['ignores_mouse_events']
    result['checks']['overlay_cannot_become_key'] = not overlay['can_become_key']
    result['checks']['overlay_cannot_become_main'] = not overlay['can_become_main']
    result['checks']['overlay_frame_matches'] = bool(game_bounds and overlay['bounds'] and all(
        abs(overlay['bounds'][key] - game_bounds[key]) <= 2 for key in ('X', 'Y', 'Width', 'Height')))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', default='金铲铲', help='与 WindowLocator 相同的游戏进程名匹配')
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--out', type=Path, default=Path('var/hud-level.json'))
    args = parser.parse_args()
    if args.cycles < 5:
        parser.error('--cycles 最少为 5')
    report = {'owner': args.owner, 'cycles_requested': args.cycles, 'cycles': [], 'failures': [], 'exit_code': 1}

    def write_report():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    if sys.platform != 'darwin':
        report.update(exit_code=2, skipped='仅支持 macOS 实机检查')
        write_report()
        print(report['skipped'])
        return 2

    import AppKit
    from jccbrain.actuator.window import WindowLocator

    original_app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    locator = WindowLocator(args.owner)
    game = locator.find(0)
    if game is None:
        report.update(exit_code=2, skipped='未找到游戏窗口')
        write_report()
        print('未找到游戏窗口')
        return 2

    import webview
    from jccbrain.webui.hud import HudController
    from jccbrain.webui.macos import _activate_by_pid

    index = Path(__file__).resolve().parents[1] / 'src/jccbrain/webui/static/index.html'
    if not index.is_file():
        report['failures'].append('界面尚未构建，请先运行 make web')
        write_report()
        print(report['failures'][0])
        return 1
    bus = SimpleNamespace(attach=lambda *a, **kw: None, detach=lambda *a, **kw: None)
    hud = HudController(SimpleNamespace(locator=locator), bus, str(index))

    def check():
        try:
            deadline = time.monotonic() + 30
            while not {'hud', 'hud_panel'} <= hud._loaded:
                if time.monotonic() >= deadline:
                    raise TimeoutError('等待两个 HUD 页面加载超时')
                time.sleep(.05)
            for cycle in range(1, args.cycles + 1):
                current = locator.find(0)
                if current is None:
                    raise RuntimeError('检查过程中游戏窗口消失')

                def activate():
                    app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(current.pid)
                    if app is None:
                        raise RuntimeError('游戏进程已退出')
                    direct = bool(app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps))
                    helper = _activate_by_pid(current.pid) if not direct else False
                    return {'direct': direct, 'helper': helper, 'success': direct or helper}

                activated = activate()
                time.sleep(.6)
                row = on_main(lambda: snapshot(hud, current))
                row['cycle'] = cycle
                row['activation_return'] = activated['success']
                row['activation'] = activated
                report['cycles'].append(row)
                report['failures'].extend(f'cycle {cycle}: {key}' for key, ok in row['checks'].items() if not ok)
            report['exit_code'] = 1 if report['failures'] else 0
        except Exception as exc:
            report['exit_code'] = 1
            report['failures'].append(f'{type(exc).__name__}: {exc}')
        finally:
            try:
                hud.close()
            except Exception as exc:
                report['exit_code'] = 1
                report['failures'].append(f'关闭 HUD 失败: {exc}')

    try:
        hud.toggle_overlay(True, game)
        hud.toggle_panel(True)
        webview.start(check, gui='cocoa')
    except Exception as exc:
        report['exit_code'] = 1
        report['failures'].append(f'{type(exc).__name__}: {exc}')
    finally:
        # 最后两个 HUD 销毁后 Cocoa 事件循环会退出；在主线程恢复原前台。
        try:
            hud.close()
            if original_app is not None:
                direct = bool(original_app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps))
                report['restore_used_helper'] = not direct
                report['restored_front_app'] = direct or _activate_by_pid(int(original_app.processIdentifier()))
                if not report['restored_front_app']:
                    raise RuntimeError('恢复原前台应用的请求失败')
        except Exception as exc:
            report['exit_code'] = 1
            report['failures'].append(f'清理/恢复前台失败: {exc}')
        write_report()
    for failure in report['failures']:
        print(failure)
    print(f"HUD 检查：{len(report['cycles'])}/{args.cycles} 次，退出码 {report['exit_code']}；报告 {args.out}")
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
