"""OCR 文字后处理：词典纠错、抽数字。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CIRCLED = str.maketrans("⓪①②③④⑤⑥⑦⑧⑨", "0123456789")


def is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿"


def is_word_char(c: str) -> bool:
    return c.isalnum()


def has_words(s: str) -> bool:
    return any(is_cjk(c) or (c.isascii() and c.isalpha()) for c in s)


def norm_digits(s: str) -> str:
    """圈号数字（①②…）统一成普通数字，OCR 经常把带底框的数字认成圈号。"""
    return s.translate(CIRCLED)


def ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"[0-9]+", norm_digits(s))]


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def find_words(text: str, words: list[str]) -> list[str]:
    """在 OCR 文字里找词典词，按出现顺序返回。先精确，再对 4 字以上的词容 1 个错字
    （3 字词容错会把「魔战士」认成「狂战士」）。"""
    t = [c for c in text if is_cjk(c) or c.isalpha()]
    used = [False] * len(t)
    hits: list[tuple[int, str]] = []
    ordered = sorted(words, key=len, reverse=True)
    for fuzzy in (False, True):
        for w in ordered:
            if any(h[1] == w for h in hits) or not w or len(w) > len(t):
                continue
            tol = (1 if len(w) >= 4 else 0) if fuzzy else 0
            if fuzzy and tol == 0:
                continue
            best = None
            for s in range(len(t) - len(w) + 1):
                if any(used[s:s + len(w)]):
                    continue
                d = edit_distance("".join(t[s:s + len(w)]), w)
                if d <= tol and (best is None or d < best[1]):
                    best = (s, d)
                if d == 0:
                    break
            if best:
                for k in range(best[0], best[0] + len(w)):
                    used[k] = True
                hits.append((best[0], w))
    return [w for _, w in sorted(hits)]


# 数据库（dataj.cc）里的羁绊名 → 游戏内显示的名字（显示名更长，OCR 读到的是它）
TRAIT_DISPLAY = {"约德尔人": "约德尔人和朋友"}
DICT_FILE = Path(__file__).with_name("dict.json")


def _loose(s: str) -> str:
    """海克斯名比较用：去空格，罗马数字 I/l 和 1 视为同一个（OCR 常把「治疗法球 I」读成「治疗法球1」）。"""
    return s.replace(" ", "").replace("Ⅰ", "I").replace("Ⅱ", "II").replace("l", "I").replace("1", "I")


@dataclass
class Dictionary:
    """英雄名 → 费用 / 羁绊，羁绊名，海克斯名。运行时用游戏数据库（完整）；没有数据库时用 dict.json，
    它由 scripts/make_vision_dict.py 从数据库生成（不再手写）。羁绊一律是游戏内显示名。"""

    champions: dict[str, int] = field(default_factory=dict)
    traits: list[str] = field(default_factory=list)
    champion_traits: dict[str, list[str]] = field(default_factory=dict)
    augments: list[str] = field(default_factory=list)

    @classmethod
    def from_db(cls, db) -> "Dictionary":
        heroes = db.pool_heroes()
        return cls({h.name: h.cost for h in heroes},
                   sorted({TRAIT_DISPLAY.get(t, t) for t in db.traits}),
                   {h.name: [TRAIT_DISPLAY.get(t, t) for t in h.traits] for h in heroes},
                   sorted({a.name for a in db.augments.values()}))

    @classmethod
    def from_file(cls, path: Path | None = None) -> "Dictionary":
        d = json.loads((path or DICT_FILE).read_text(encoding="utf-8"))
        return cls(dict(d.get("champions", {})), list(d.get("traits", [])),
                   {k: list(v) for k, v in d.get("champion_traits", {}).items()}, list(d.get("augments", [])))

    def to_json(self) -> dict:
        return {"champions": self.champions, "traits": self.traits, "champion_traits": self.champion_traits,
                "augments": self.augments}

    def match_augment(self, text: str) -> str:
        """海克斯名纠错：完全一致（宽松比较）或 4 字以上容 1 个错字；对不上原样返回。"""
        t = text.strip()
        if not t or not self.augments:
            return t
        lt = _loose(t)
        for a in self.augments:
            if _loose(a) == lt:
                return a
        best = min(self.augments, key=lambda a: edit_distance(_loose(a), lt))
        return best if len(lt) >= 4 and edit_distance(_loose(best), lt) <= 1 else t

    def match_champion(self, text: str) -> tuple[str, int | None, bool] | None:
        """卡名纠错：(名字, 费用, 是否在词典里)。没有把握时原样返回并标记不确定。"""
        core = "".join(c for c in text if is_cjk(c) or c.isalpha() or c == "·")
        if not core:
            return None
        if core in self.champions:
            return core, self.champions[core], True
        names = list(self.champions)
        hit = find_words(core, names)
        if hit:
            return hit[0], self.champions[hit[0]], True
        if names:
            best = min(names, key=lambda n: edit_distance(n, core))
            if edit_distance(best, core) <= (1 if len(best) >= 4 else 0):
                return best, self.champions[best], True
        return core, None, False
