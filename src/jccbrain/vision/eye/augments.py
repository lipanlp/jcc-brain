"""已选强化符文：当前赛季官方图标，本人确认后保留至新局，不从候选猜选择。"""
from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from jccbrain.core import paths
from jccbrain.vision.eye.items import ItemCatalog, icon_vector, season_key

log = logging.getLogger(__name__)
UI = Path(__file__).with_name('augment_ui.npz')
# 1774×1136：左侧展示物上方，1/2/3个图标居中排列。位置与尺寸均按窗口比例。
CENTERS = tuple(x / 1774 for x in (480, 505, 530, 555, 580))
# 1774×1136实拍（旧22帧/新选择与常驻）：正确名最高相关最低.7471、异名最高.8735，
# 分布重叠（白银命运三个版本完全同图）；.82只保留清晰图，另要求异名领先差.07。
# 拒绝模糊/同图变体，分布与漏检另见docs；不把重叠分布宣称为可完全分离。
NAME_SCORE, NAME_MARGIN = .82, .07
# 1774×1136旧22/新22已标帧：异名最高.8735但异名领先差为0，
# 没有异名通过领先差.10；本人单图标.8735/.1048可单帧确认。
# 普通.82/.07候选必须连续两帧，已有位置的改名无论分数都必须连续两帧。
MEMORY_SCORE, MEMORY_MARGIN = .87, .10


class AugmentCatalog(ItemCatalog):
    @classmethod
    def from_db(cls, db=None, directory=None):
        sid = season_key(getattr(db, 'season_id', 'unknown'))
        entries = []
        # GameDB的Augment视图尚未暴露icon；独立只读连接，绝不迁移/修改数据库。
        if db is not None and getattr(db, 'path', None):
            try:
                with sqlite3.connect(Path(db.path).resolve().as_uri() + '?mode=ro', uri=True) as conn:
                    entries = [dict(id=i, name=n, tier=tier, picture=url) for i, n, tier, url in conn.execute(
                        'SELECT id,name,tier,icon FROM augments WHERE season_id=?', (sid,))]
            except sqlite3.Error as exc:
                log.warning('无法读取当前赛季强化符文目录：%s', exc)
        return cls(sid, entries, directory or paths.var_dir() / 'augment_icons')


@lru_cache(maxsize=1)
def eye_template():
    with np.load(UI, allow_pickle=False) as data:
        return data['spectate_eye']


def spectate_score(image):
    """右侧玩家栏的白色眼睛；纵向搜索以覆盖八个玩家位置。"""
    h, w = image.shape[:2]
    strip = image[round(.09*h):round(.69*h), round(.963*w):round(.983*w)]
    if not strip.size:
        return -1.
    strip = cv2.resize(strip, (35, 682), interpolation=cv2.INTER_AREA)
    return float(cv2.matchTemplate(strip, eye_template(), cv2.TM_CCOEFF_NORMED).max())


def icon_patches(image, center):
    h, w = image.shape[:2]
    # CLI缩放/JPEG与透明官方图标的留白相差1~4点；小范围配准，不扩大到棋盘单位。
    for dx in (-1, 0, 1):
        for cy in (307, 308, 309, 310, 311, 312, 313):
            for size in (44, 46, 48, 50):
                patch = image[round((cy-size/2)/1136*h):round((cy+size/2)/1136*h),
                              round((center+(dx-size/2)/1774)*w):round((center+(dx+size/2)/1774)*w)]
                if patch.size:
                    yield patch


class AugmentReader:
    def __init__(self, catalog):
        self.catalog = catalog
        urls = {}
        for row in catalog.entries:
            if row.get('picture'):
                urls.setdefault(row['picture'], set()).add(row['name'])
        self.ambiguous_names = {name for names in urls.values() if len(names) > 1 for name in names}
        self.reset()

    def reset(self):
        self.confirmed: dict[int, str] = {}
        self.pending: dict[int, tuple[str, int]] = {}

    def remembered(self):
        return list(dict.fromkeys(self.confirmed[i] for i in sorted(self.confirmed)))

    def candidates(self, image):
        if not self.catalog.loaded:
            return []
        found = []
        for x in CENTERS:
            # 保留BGR三通道；同图案的银/金/彩等级不可转成灰度后匹配。
            vectors = np.stack([icon_vector(p, 0) for p in icon_patches(image, x)])
            scores = (self.catalog.matrix @ vectors.T).max(axis=1)
            # 同名多id合并；同图不同名仍参与领先差，不能随意选一个版本。
            names = {}
            for entry, score in zip(self.catalog.loaded, scores):
                name = entry['name']
                names[name] = max(names.get(name, -1.), float(score))
            ranked = sorted(names.items(), key=lambda pair: pair[1], reverse=True)
            name, score = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else -1.
            found.append((x, name, score, second))
        return found

    def read(self, image, *, own_view: bool):
        # 玩家栏被面板遮挡、观战时不推断归属、不更新，只输出此前本人已确认的记忆。
        # 1774×1136：普通观战5帧最低.6808，已核对本人帧最高.5161；取两者之间.60。
        # 选秀重叠帧没有可见眼睛，图标区域不匹配；#5c单独处理阶段优先级。
        if not own_view or spectate_score(image) >= .60:
            self.pending.clear()
            return self.remembered()
        found = [r for r in self.candidates(image) if r[1] not in self.ambiguous_names
                 and r[2] >= NAME_SCORE and r[2]-r[3] >= NAME_MARGIN]
        # 只接受实拍的一/二/三图标布局，避免偏移候选重复认到相邻图案。
        layouts = ((2,), (1, 3), (0, 2, 4))
        indexed = {CENTERS.index(r[0]): r for r in found}
        layout = max(layouts, key=lambda ids: sum(indexed[i][2] for i in ids if i in indexed))
        pending = {}
        for slot, index in enumerate(layout):
            if index not in indexed:
                continue
            _, name, score, second = indexed[index]
            previous = self.confirmed.get(slot)
            if previous == name:
                continue
            if name in self.confirmed.values():
                continue  # 部分图标可见导致布局不确定时，不能把同一名字移到另一选择位置。
            old_name, count = self.pending.get(slot, (None, 0))
            count = count + 1 if old_name == name else 1
            pending[slot] = name, count
            high = score >= MEMORY_SCORE and score - second >= MEMORY_MARGIN
            if count >= 2 or (previous is None and high):
                if previous is not None:
                    log.info('已选海克斯位置%d连续确认变更：%s → %s', slot + 1, previous, name)
                self.confirmed[slot] = name
                pending.pop(slot)
        self.pending = pending
        return self.remembered()
