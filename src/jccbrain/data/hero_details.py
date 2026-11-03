"""官方 chess.js 的逐星属性和技能数值转换（不联网、不补算）。"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)
NUMBER = r"[-+]?\d+(?:\.\d+)?"
SERIES = re.compile(rf"(?<![\d.])({NUMBER}%?(?:\s*/\s*{NUMBER}%?)+)\s*$")
STATS = {"hp": "initHp", "attack_damage": "initAttackDamage", "attack_speed": "attackSpeed",
         "armor": "armor", "magic_resist": "magicResist", "init_mana": "initMp",
         "max_mana": "maxMp", "crit_chance": "criticalStrikeChance"}


def number(value) -> float | None:
    return float(value) if value not in (None, "") else None


def hero_key(h: dict) -> tuple[str, int] | None:
    hid = str(h["id"])
    return (hid[1:], int(hid[0])) if len(hid) == 5 and hid[0] in "1234" else None


def star_row(h: dict) -> dict:
    hid, star = hero_key(h)
    return {"hero_id": hid, "star": star, **{k: number(h.get(v)) for k, v in STATS.items()}}


def parse_skill_values(description: str | None, brief: str | None = None) -> list[dict]:
    """解析中英文冒号、百分比、小数与多加成。公式没有星级表时保留原式。

    skillBriefValue 有时比描述多出隐藏参数，不能盲目按序配对；仅在两边等长时补数值。
    完整 brief 另外存储，避免丢掉这些无标签的参数。
    """
    parts = (description or "").split("|")
    briefs = (brief or "").split("|")
    if (description or "").strip() == "0":  # 木桩假人用 0 表示无技能；原文另存 raw。
        return []
    out = []
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        label = re.match(r"\s*([^:：]+)[:：]\s*(.*)", part, re.S)
        if not label:
            # 英雄联盟传奇有「永久生命值 18/20/50」这样的无冒号标签。
            label = re.fullmatch(rf"\s*([^\d:：]+?)\s+({NUMBER}%?(?:\s*/\s*{NUMBER}%?)+)\s*", part)
        if not label:
            log.warning("技能数值片段无法解析：%s", part)
            out.append({"name": "", "scaling": "", "values": [], "raw": part, "formula": part,
                        "unit": "", "parse_error": True})
            continue
        name, formula = label.groups()
        series = SERIES.search(formula)
        table = series.group(1) if series else ""
        if not table and len(parts) == len(briefs) and re.fullmatch(rf"{NUMBER}%?(?:/{NUMBER}%?)+", briefs[i]):
            table = briefs[i]
        scaling = "、".join(dict.fromkeys(re.findall(r"【([^】]+)】", formula)))
        out.append({"name": name.strip(), "scaling": scaling,
                    "values": [float(x) for x in re.findall(NUMBER, table)],
                    "unit": "%" if "%" in table else "", "formula": formula[:series.start()].strip() if series else formula,
                    "raw": part})
    return out
