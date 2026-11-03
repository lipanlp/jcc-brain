"""官方单位展示标记与变体归组；不推测特殊单位的卡池数量。"""
from __future__ import annotations

import json
from collections import defaultdict


def _flag(value) -> bool:
    return value is True or str(value) == "1"


def classify_units(rows: list[dict]) -> None:
    """写入 unit_kind / variant_of / in_pool。

    依据：官方 18.18.5-S19 与 17.17.8b-S19 chess.js。buyPrice 在这两个版本
    全为 0，不能用来判定商店；showHeroTag 才区分主展示形态与隐藏形态。
    同名的正常英雄形态按费用归组，先选 showHeroTag=1 的记录，再以 id 稳定排序。
    heroType=1 明确是召唤单位；heroType=7（加里奥）等特殊类型仍需确认获取机制，
    暂归 other，不能因为 showHeroTag=1 就加入普通池。dataj 没有展示标记时才退回
    isVisible（来源：dataj /gamedata）；这个降级规则不覆盖官方明确给出的 0。
    """
    groups: dict[tuple[str, int], list[tuple[dict, bool]]] = defaultdict(list)
    for row in rows:
        src = json.loads(row["raw"] or "{}")
        kind = int(src.get("heroType") or 0)
        trait_ids = [i for i in json.loads(row["trait_ids"]) if str(i) not in ("0", "-1", "")]
        row.update(unit_kind="other", variant_of=None, in_pool=0)
        if kind == 1 or (kind == 0 and (not trait_ids or row["cost"] == 0)):
            row["unit_kind"] = "summon"
        elif kind == 0 and 1 <= row["cost"] <= 5:
            visible = _flag(src["showHeroTag"]) if "showHeroTag" in src else _flag(src.get("isVisible"))
            # 没有一星记录的单位不推断为商店单位。
            visible = visible and str(src.get("id", "")).startswith("1")
            groups[(row["name"], row["cost"])].append((row, visible))
    for group in groups.values():
        visible = [r for r, show in group if show]
        if not visible:
            continue
        canonical = min(visible, key=lambda r: r["id"])
        canonical.update(unit_kind="champion", in_pool=1)
        for row, _ in group:
            if row is not canonical:
                row.update(unit_kind="variant", variant_of=canonical["id"])
