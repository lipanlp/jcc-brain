"""用户设置：dataclass 定义默认值，JSON 文件持久化（var/settings.json）。

新增设置项只需在对应 dataclass 里加字段，旧的设置文件会自动补上默认值。
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from jccbrain.core import paths

log = logging.getLogger(__name__)


@dataclass
class VisionSettings:
    backend: str = "pyeye"               # pyeye（实时画面）| replay（回放）
    poll_interval_ms: int = 250           # 两次读取画面的最小间隔
    replay_file: str = ""                 # backend=replay 时读取：jsonl 状态文件，或截图目录
    replay_interval_ms: int = 800
    ocr: str = "rapid"                    # 文字识别后端：rapid（RapidOCR，默认）| vision（Apple Vision）
    unit_learning_mode: str = 'off'       # off / record / auto；只在辅助大脑运行
    unit_learning_idle_s: float = 2.0
    unit_learning_interval_s: float = 15.0
    unit_learning_min_timer: int = 15

    def validate_unit_learning(self):
        if self.unit_learning_mode not in ('off', 'record', 'auto'):
            raise ValueError('棋子身份学习必须选择off / record / auto')
        for name, low, high in (('unit_learning_idle_s',2,60), ('unit_learning_interval_s',10,300),
                                ('unit_learning_min_timer',10,120)):
            value = getattr(self,name)
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                    or not low <= value <= high or (name=='unit_learning_min_timer' and int(value)!=value)):
                raise ValueError(f'{name}必须在{low}到{high}之间' + ('且为整数' if name=='unit_learning_min_timer' else ''))


@dataclass
class DataSettings:
    season_id: str = "18_S19"             # 当前赛季（官方 id，18_S19 = 自然之力），见 core/seasons.py；在菜单「数据 → 赛季」切换
    # 三个阵容来源可多选；官网或微社区任一开启，均同步完整官方快照，基础数据以官方为准。
    # dataj 补阵容统计、搜索别名和仙灵出现回合；只选 dataj 时基础数据也从 dataj 拿。
    source_official: bool = True
    source_ingame: bool = True
    source_dataj: bool = True
    official_version: str = ""            # 官方资料库版本（如 18.18.5），留空 = 最新
    auto_sync: bool = True                # 打开界面时在后台检查：没同步过或官方有新版本的赛季自动同步
    base_url: str = "https://www.dataj.cc/api/web"
    game_version: str = ""                # dataj 统计版本，留空 = 自动取最新版本
    fetch_comp_details: bool = True       # 同步时逐个拉取阵容详情（站位/海克斯），较慢
    request_interval_s: float = 0.3       # 礼貌限速


@dataclass
class LLMSettings:
    enabled: bool = False
    preset: str = "balanced"              # fast / balanced / deep / custom，见 engine/llm.py 的 PRESETS
    provider: str = "openai"              # openai（兼容 OpenAI 协议的都可以）| anthropic
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""                     # 也可用环境变量 JCC_LLM_API_KEY
    model: str = ""
    timeout_s: float = 40.0
    min_interval_s: float = 6.0           # 同一回合内不重复请求
    max_tokens: int = 3000                # 含思考 token：思考模型（如 GLM-5.3）太小会只剩思考没有正文
    reasoning_effort: str = ""            # 思考强度 low/high/max；空 = 不传（模型默认）
    reasoning_effort_augment: str = ""    # 海克斯回合的思考强度（影响大、时间够，可以更高）
    json_mode: bool = True                # 请求 response_format=json_object


@dataclass
class ActuatorSettings:
    dry_run: bool = True                  # 默认只记录不执行，确认标定无误后再关闭
    window_owner: str = "金铲铲"           # 游戏窗口所属进程名（包含匹配）
    click_delay_ms: int = 60
    drag_steps: int = 12
    jitter_px: float = 2.0
    max_actions_per_sec: float = 6.0
    failsafe_corner: bool = True          # 鼠标甩到屏幕左上角立即停止


@dataclass
class AutoRollSettings:
    keep_gold: int = 0                    # D 牌时保留的金币下限
    target_star: int = 2
    max_rolls: int = 30
    buy_pairs_only: bool = False          # 只买能凑对的目标卡
    read_interval_ms: int = 100           # D 牌期间两次读画面的最小间隔


@dataclass
class HudSettings:
    show_names: bool = True
    show_traits: bool = True
    show_odds_panel: bool = True
    font_size: int = 12
    opacity: float = 0.92
    show_positions: bool = True


@dataclass
class AssistantSettings:
    default_lock_comp_id: str = ""
    economy_preference: str = "adaptive"  # adaptive / health / economy
    comp_presets: dict = field(default_factory=dict)  # 按赛季保存，缺省动态全部；由comp_presets严格校验。


@dataclass
class AutopilotSettings:
    augment_selection: str = "manual"     # manual / auto；接管行为在后续步骤启用


@dataclass
class ScoutSettings:
    enabled: bool = False
    schedule: str = "key_rounds"          # key_rounds / every_round；enabled=false 为手动
    rounds: list[str] = field(default_factory=lambda: ["3-2", "4-1", "4-5", "5-1", "5-5", "6-1"])
    min_timer: int = 15
    max_targets: int = 7
    cut_threshold: float = 25
    stable_threshold: float = 6
    switch_timeout_ms: int = 1200
    skip_recent_rounds: int = 0
    min_dwell_ms: int = 350
    stable_frames: int = 3


@dataclass
class Settings:
    vision: VisionSettings = field(default_factory=VisionSettings)
    data: DataSettings = field(default_factory=DataSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    actuator: ActuatorSettings = field(default_factory=ActuatorSettings)
    autoroll: AutoRollSettings = field(default_factory=AutoRollSettings)
    hud: HudSettings = field(default_factory=HudSettings)
    assistant: AssistantSettings = field(default_factory=AssistantSettings)

    scout: ScoutSettings = field(default_factory=ScoutSettings)
    autopilot: AutopilotSettings = field(default_factory=AutopilotSettings)

    # ---- 持久化 ----
    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or paths.settings_path()
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError('设置顶层必须为对象')
        except (OSError, ValueError) as e:
            log.warning("设置文件读取失败，使用默认值：%s", e)
            s = cls()
            s.assistant.comp_presets = None  # 整个设置损坏也不能默默扩大自动比较范围。
            return s
        if 'assistant' in raw and not isinstance(raw['assistant'], dict):
            raw['assistant'] = {'comp_presets': None}
        s = _from_dict(cls, raw)
        try:
            s.vision.validate_unit_learning()
        except ValueError as exc:
            log.warning('棋子身份学习设置无效，已关闭：%s', exc)
            defaults = VisionSettings()
            for name in ('unit_learning_mode','unit_learning_idle_s','unit_learning_interval_s','unit_learning_min_timer'):
                setattr(s.vision,name,getattr(defaults,name))
        if "target_comp_id" in raw:
            log.info("旧的永久锁定阵容已取消，可在偏好设置里设默认锁定阵容")
        _upgrade_legacy(s, raw)
        return s

    def save(self, path: Path | None = None) -> None:
        path = path or paths.settings_path()
        payload = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        # 同目录原子替换，写入失败保留原文件；调用方落盘成功后才更新内存。
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
                name = f.name
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            if name and Path(name).exists():
                Path(name).unlink()


def _upgrade_legacy(s: Settings, raw: dict[str, Any]) -> None:
    """旧设置：data.set_id（dataj setId）→ data.season_id（官方 id）；data.use_official → source_official。"""
    from jccbrain.core.seasons import DEFAULT_SEASON_ID, LEGACY_SET_IDS

    # 新枚举的旧文件缺省由 dataclass 补齐；损坏值回落明确默认，避免状态误报。
    for obj, key, allowed, default in [
        (s.assistant, "economy_preference", ("adaptive", "health", "economy"), "adaptive"),
        (s.scout, "schedule", ("key_rounds", "every_round"), "key_rounds"),
        (s.autopilot, "augment_selection", ("manual", "auto"), "manual"),
    ]:
        if getattr(obj, key) not in allowed:
            setattr(obj, key, default)
    data = raw.get("data") or {}
    if "season_id" not in data and "set_id" in data:
        try:
            s.data.season_id = LEGACY_SET_IDS.get(int(data["set_id"]), DEFAULT_SEASON_ID)
        except (TypeError, ValueError):
            pass
    if "source_official" not in data and "use_official" in data:
        s.data.source_official = bool(data["use_official"])
    if "source_ingame" not in data:
        # 旧用户关闭官方时，不因新增渠道重新引入官方阵容。
        s.data.source_ingame = s.data.source_official


def _from_dict(cls: type, raw: dict[str, Any]) -> Any:
    """按 dataclass 字段递归构造，未知字段忽略、缺失字段取默认。"""
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        v = raw[f.name]
        ftype = f.default_factory if f.default_factory is not MISSING else None
        if ftype is not None and is_dataclass(ftype) and isinstance(v, dict):
            kwargs[f.name] = _from_dict(ftype, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)
