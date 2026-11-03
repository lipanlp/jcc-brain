"""视觉后端接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from jccbrain.core.models import GameState

if TYPE_CHECKING:
    from jccbrain.core.settings import VisionSettings


class VisionError(RuntimeError):
    pass


class VisionProvider(ABC):
    """所有视觉后端的共同接口。实现需保证 read() 线程安全。"""

    name = "base"
    interval_override_s: float | None = None

    @contextmanager
    def fast_frames(self, interval_s: float):
        """临时覆盖读帧间隔，退出或异常时恢复，支持嵌套。"""
        previous = self.interval_override_s
        self.interval_override_s = interval_s
        try:
            yield
        finally:
            self.interval_override_s = previous

    def start(self) -> None:
        """启动后端（拉起子进程等）。重复调用无副作用。"""

    def stop(self) -> None:
        """释放资源。"""

    @abstractmethod
    def read(self, fresh: bool = True) -> GameState:
        """读取当前状态。fresh=True 要求是调用之后的新画面（操作后确认结果时用）。"""

    def screenshot(self, width: int = 1024) -> bytes | None:
        """当前画面 JPEG（可选能力，交给多模态模型时用）。"""
        return None

    def label_units(self, bench: dict[int, str] | None = None,
                    board: list[dict[str, Any]] | None = None) -> list[str]:
        """手动告诉视觉后端某个单位是谁（可选能力）。"""
        return []

    def status(self) -> dict[str, Any]:
        return {"backend": self.name}

    def __enter__(self) -> "VisionProvider":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def create_provider(cfg: "VisionSettings", db=None, window_owner: str = "金铲铲") -> VisionProvider:
    """按设置创建视觉后端。db 用于 OCR 纠错词典（英雄名/羁绊名）。"""
    if cfg.backend == "replay":
        from pathlib import Path
        if Path(cfg.replay_file).is_dir():
            from jccbrain.vision.pyeye import ImageReplayProvider
            return ImageReplayProvider(cfg.replay_file, cfg.replay_interval_ms / 1000, db=db, ocr=cfg.ocr)
        from jccbrain.vision.replay import ReplayProvider
        return ReplayProvider(cfg.replay_file, cfg.replay_interval_ms / 1000)
    from jccbrain.vision.pyeye import PyEyeProvider
    return PyEyeProvider(window_owner, cfg.poll_interval_ms / 1000, db=db, ocr=cfg.ocr)
