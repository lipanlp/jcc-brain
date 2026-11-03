"""官方文本的结构化视图；保留原文，不把展示数字当成可执行公式。"""
import re

# 官方 18.18.5-S19/equip.js 的 41805 用「约德尔人和朋友」，同版本 trait.js
# checkId=456 名称为「约德尔人」，两者描述均指定毛茸茸大朋友骑乘机制。
TRAIT_NAME_ALIASES = {"约德尔人和朋友": "约德尔人"}


def trait_levels(breakpoints: list[int], descriptions: list[str]) -> list[dict]:
    # | 也会分隔同一档的两个数值（例如神谕），宿敌还会在一段里重复写 (1)/(2)。
    # 优先用显式档位标记分组，不能简单按 | 与档位 zip。
    joined = "|".join(descriptions).strip()
    markers = [m for m in re.finditer(r"[（(](\d+)[）)]", joined) if int(m[1]) in breakpoints]
    chunks: list[tuple[int | None, str]] = []
    if markers:
        grouped: dict[int, list[str]] = {}
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(joined)
            text = joined[m.end():end].strip(" |")
            if text:
                grouped.setdefault(int(m[1]), []).append(text)
        preamble = joined[:markers[0].start()].strip(" |")
        if preamble:
            grouped.setdefault(int(markers[0][1]), []).insert(0, preamble)
        chunks = [(count, " ".join(grouped.get(count, []))) for count in breakpoints if count in grouped]
    elif len(breakpoints) == 1:
        chunks = [(breakpoints[0], joined)]
    else:
        chunks = [(breakpoints[i] if len(descriptions) == len(breakpoints) else None, text)
                  for i, text in enumerate(descriptions)]
    out = []
    for count, text in chunks:
        if not text:
            continue
        out.append({"count": count, "text": text,
                    "values": [float(x) for x in re.findall(r"(?<!\{)[-+]?\d+(?:\.\d+)?(?![\d}])", text)]})
    return out


def activation(breakpoints: list[int]) -> str:
    return "unique" if breakpoints == [1] else "special" if 0 in breakpoints else "standard"


def item_kind(raw: str | None, name: str) -> str | None:
    if str(raw) != "-1":
        return raw
    return "补剂" if name.endswith("补剂") else "赐福" if name.endswith("赐福") else "特殊消耗品"


def linked_trait(raw_id, name: str, kind: str | None, traits: dict[str, dict]) -> str | None:
    # 官方字段优先；只在字段为空时，由官方纹章名称精确对应官方羁绊，不做模糊匹配。
    if str(raw_id or "") not in ("", "0", "-1"):
        return str(raw_id) if str(raw_id) in traits else None
    if kind == "转职纹章":
        stem = name.removesuffix("纹章")
        stem = TRAIT_NAME_ALIASES.get(stem, stem)
        return next((tid for tid, t in traits.items() if t["name"] == stem), None)
    return None


def relation_ids(value) -> list[str]:
    return [s for s in re.split(r"[|,]", str(value or "")) if s not in ("", "0", "-1")]


def flag(value) -> bool | None:
    return None if value is None else value is True or str(value) == "1"
