-- 金铲铲大脑 游戏数据库。由 `jcc db sync` 从官方资料库与 dataj.cc 同步。
-- 版本号见 meta.schema_version，结构变化时递增并在 db.py 里补迁移。
-- 每个赛季（season_id，官方编号如 "18_S19" = 自然之力，见 core/seasons.py）的牌库、羁绊、装备、海克斯、阵容各自独立，
-- 所有游戏数据表都以 season_id 开头作主键的一部分；同步某个赛季只替换该赛季的行。

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- 官网当前的赛季列表（同步时刷新），菜单按它显示
CREATE TABLE IF NOT EXISTS season_catalog (
    season_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    mode        TEXT NOT NULL,
    season      TEXT NOT NULL,
    sort        INTEGER NOT NULL,
    updated_at  TEXT
);

-- 已同步的赛季
CREATE TABLE IF NOT EXISTS seasons (
    season_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,      -- 赛季名，如「自然之力」
    game_version  TEXT,               -- 基础数据的游戏版本：官方资料库如 18.18.5（官方取不到时为 dataj 版本）
    stats_version TEXT,               -- 阵容统计（dataj）的版本，如 18.4
    data_source   TEXT,               -- 基础数据来源：official | dataj
    synced_at     TEXT
);

-- 英雄（棋子）。id 为官方基础 id（去掉星级前缀，如 "1500"）
CREATE TABLE IF NOT EXISTS heroes (
    season_id     TEXT NOT NULL,
    id            TEXT NOT NULL,
    name          TEXT NOT NULL,
    cost          INTEGER NOT NULL,
    traits        TEXT NOT NULL,      -- JSON: ["永恒之森", "护卫"]
    trait_ids     TEXT NOT NULL,      -- JSON: ["450", "341"]
    in_pool       INTEGER NOT NULL,   -- 赛季内静态的普通卡池候选，不含当前对局解锁状态
    role_tag      TEXT,               -- AD / AP / NULL
    attack_range  INTEGER,
    hp            INTEGER,
    armor         INTEGER,
    magic_resist  INTEGER,
    attack_damage INTEGER,
    attack_speed  REAL,
    init_mana     INTEGER,
    max_mana      INTEGER,
    skill_name    TEXT,
    skill_desc    TEXT,
    picture       TEXT,
    raw           TEXT,
    skill_values  TEXT NOT NULL DEFAULT '[]',
    skill_brief_values TEXT NOT NULL DEFAULT '[]',
    sell_price    INTEGER,
    skill_icon    TEXT,
    variant_of    TEXT,
    unit_kind     TEXT NOT NULL DEFAULT 'other', -- champion / variant / summon / other
    PRIMARY KEY (season_id, id)
);
CREATE INDEX IF NOT EXISTS idx_heroes_name ON heroes(season_id, name);

-- 官方 mission.js：静态解锁条件，不代表当前对局已完成，不提供卡池张数。
CREATE TABLE IF NOT EXISTS unlock_conditions (
    season_id TEXT NOT NULL,
    hero_id TEXT NOT NULL,
    id TEXT NOT NULL,
    description TEXT,
    task_tips TEXT,
    condition_id TEXT,
    visible INTEGER,
    raw TEXT,
    PRIMARY KEY (season_id, hero_id, id)
);

-- 官方逐星属性（召唤物也保留，缺失星级不推算）
CREATE TABLE IF NOT EXISTS hero_stars (
    season_id     TEXT NOT NULL,
    hero_id       TEXT NOT NULL,
    star          INTEGER NOT NULL CHECK (star BETWEEN 1 AND 4),
    hp            REAL,
    attack_damage REAL,
    attack_speed  REAL,
    armor         REAL,
    magic_resist  REAL,
    init_mana     REAL,
    max_mana      REAL,
    crit_chance   REAL,
    PRIMARY KEY (season_id, hero_id, star)
);

-- 羁绊。id 为 checkId（英雄 raceId/jobId 引用的就是它）
CREATE TABLE IF NOT EXISTS traits (
    season_id   TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,        -- origin（种族/特质）| class（职业）
    breakpoints TEXT NOT NULL,        -- JSON: [2, 4, 6]
    description TEXT,
    level_desc  TEXT,                 -- JSON: ["(2)…", "(4)…"]
    levels      TEXT NOT NULL DEFAULT '[]', -- JSON: [{count, text, values}]
    activation  TEXT NOT NULL DEFAULT 'standard',
    picture     TEXT,
    PRIMARY KEY (season_id, id)
);
CREATE INDEX IF NOT EXISTS idx_traits_name ON traits(season_id, name);

-- 装备
CREATE TABLE IF NOT EXISTS items (
    season_id   TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT,                 -- 基础装备 / 成型装备 / 光明武器 / 神器装备 / 转职纹章 / 特殊装备
    description TEXT,
    recipe      TEXT,                 -- JSON: ["1001", "1002"]，基础装备为空
    fetter_id   TEXT,
    trait_id    TEXT,
    effect_type TEXT,
    raw_kind    TEXT,
    picture     TEXT,
    PRIMARY KEY (season_id, id)
);

-- 海克斯强化
CREATE TABLE IF NOT EXISTS augments (
    season_id   TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT NOT NULL,
    tier        INTEGER,              -- 1 白银 / 2 黄金 / 3 棱彩
    description TEXT,
    icon        TEXT,
    fetter_id   TEXT,
    fetter_type TEXT,
    hero_enhancement_type TEXT,
    is_legend   INTEGER,
    PRIMARY KEY (season_id, id)
);
CREATE INDEX IF NOT EXISTS idx_augments_name ON augments(season_id, name);

-- 赛季机制（S18 的自然仙灵：每回合商店里的事件，其他赛季可能为空）
CREATE TABLE IF NOT EXISTS mechanics (
    season_id   TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT NOT NULL,
    category    TEXT,
    price       TEXT,
    description TEXT,
    rounds      TEXT,                 -- JSON
    upgrades    TEXT,                 -- JSON
    PRIMARY KEY (season_id, id)
);

-- 搜索别名（玩家俗称 → 正式名）
CREATE TABLE IF NOT EXISTS aliases (
    season_id   TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    name        TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    PRIMARY KEY (season_id, entity_type, keyword)
);

-- 推荐阵容
CREATE TABLE IF NOT EXISTS comps (
    season_id      TEXT NOT NULL,
    id             TEXT NOT NULL,
    name           TEXT NOT NULL,
    tier           TEXT,
    avg_placement  REAL,
    top4_rate      REAL,
    top_rate       REAL,
    pick_rate      REAL,
    sample_count   INTEGER,
    game_version   TEXT,
    game_code      TEXT,
    traits         TEXT,              -- JSON: ["峡谷野怪", …]
    conditions     TEXT,              -- JSON: 阵容成立条件（指定装备/海克斯）
    placements     TEXT,              -- JSON: 名次分布
    updated_at     TEXT,
    PRIMARY KEY (season_id, id)
);

CREATE TABLE IF NOT EXISTS comp_heroes (
    season_id   TEXT NOT NULL,
    comp_id     TEXT NOT NULL,
    hero_id     TEXT NOT NULL,
    hero_name   TEXT NOT NULL,
    cost        INTEGER,
    is_carry    INTEGER,
    is_subcarry INTEGER,
    is_core     INTEGER,
    star_target INTEGER,              -- 需要追的星级（3 = 追三星），可能为空
    position    TEXT,                 -- "行,列"，来自阵容详情
    items       TEXT,                 -- JSON: 推荐装备名
    sort        INTEGER,
    PRIMARY KEY (season_id, comp_id, hero_id)
);

CREATE TABLE IF NOT EXISTS comp_augments (
    season_id       TEXT NOT NULL,
    comp_id         TEXT NOT NULL,
    augment_id      TEXT NOT NULL,
    name            TEXT,
    tier            INTEGER,
    rating          TEXT,             -- S/A/B/C/D
    play_rate       REAL,
    avg_placement   REAL,
    top4_rate       REAL,
    top_rate        REAL,
    placement_delta REAL,             -- 相对阵容平均名次的变化，负数 = 更好
    rounds          TEXT,             -- JSON
    PRIMARY KEY (season_id, comp_id, augment_id)
);

-- 不同人口下的过渡站位（battle-builds）
CREATE TABLE IF NOT EXISTS comp_builds (
    season_id    TEXT NOT NULL,
    comp_id      TEXT NOT NULL,
    people       INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    win_rate     REAL,
    sample_count INTEGER,
    heroes       TEXT,                -- JSON: [{hero_id, name, star, position}]
    PRIMARY KEY (season_id, comp_id, people, rank)
);

-- 开局奇遇（官方资料库 galaxy：每局开始时全场生效的规则，如「金金金之旅」）
CREATE TABLE IF NOT EXISTS encounters (
    season_id   TEXT NOT NULL,
    id          TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    picture     TEXT,
    PRIMARY KEY (season_id, id)
);

-- 官方推荐阵容（官网 jcc.qq.com/#/lineup 的数据，lineup_detail_total.json），含攻略文字；
-- 载入时也会转成 Comp（source = official），没有 dataj 统计的赛季靠它做阵容推荐
CREATE TABLE IF NOT EXISTS guides (
    season_id       TEXT NOT NULL,
    id              TEXT NOT NULL,
    name            TEXT NOT NULL,    -- 如「【牛牛提莫】3神谕3约德尔人3永恒之森」
    quality         TEXT,             -- 官方评级 S/A/…
    sort            INTEGER,          -- 官方排序值，越大越靠前
    game_version    TEXT,             -- 攻略对应的游戏版本
    released_at     TEXT,
    category        TEXT,             -- 官网分类：新手推荐 / 高手进阶 / 趣味娱乐
    author          TEXT,             -- 作者（如「掌盟阵容推荐小助手」）
    early_info      TEXT,             -- 前期过渡
    d_time          TEXT,             -- 什么时候 D 牌 / 升级节奏
    location_info   TEXT,             -- 站位
    enemy_info      TEXT,             -- 克制与应对
    hex_info        TEXT,             -- 海克斯思路
    equipment_info  TEXT,             -- 装备思路
    adventure_info  TEXT,             -- 自然仙灵思路
    hexes           TEXT,             -- JSON: {"recommend": [海克斯 id], "replace": [...]}
    equipment_order TEXT,             -- JSON: 散件优先顺序 [装备 id]
    carry_items     TEXT,             -- JSON: 主 C 装备 {"main": [...], "backup": [...]}
    three_star      TEXT,             -- JSON: 需要追三星的英雄 [基础 id]
    hero_replace    TEXT,             -- JSON: [{hero_id, replace: [基础 id]}]
    adventures      TEXT,             -- JSON: 每阶段推荐的自然仙灵 {"1": [名称], …}
    channels        TEXT NOT NULL DEFAULT '["web"]', -- JSON: 官方来源渠道 web / ingame
    ingame_sort     INTEGER,          -- 微社区排序值，越大越靠前
    use_count       INTEGER,          -- 微社区累计使用次数
    week_use        INTEGER,          -- 近七天使用次数
    collect_count   INTEGER,          -- 收藏数
    comment_count   INTEGER,          -- 评论数
    stat_date       TEXT,             -- 热度统计日期 YYYYMMDD
    PRIMARY KEY (season_id, id)
);

-- 攻略里的站位：final 成型 / early 前期 / mid 中期 / L<人口> 各人口过渡
CREATE TABLE IF NOT EXISTS guide_heroes (
    season_id TEXT NOT NULL,
    guide_id  TEXT NOT NULL,
    phase     TEXT NOT NULL,
    sort      INTEGER NOT NULL,
    hero_id   TEXT NOT NULL,          -- 基础 id（与 heroes.id 一致）
    star      INTEGER,
    unit_type TEXT,                   -- hero | pet（召唤物等）
    position  TEXT,                   -- 官方原样 "行,列"
    is_carry  INTEGER,
    items     TEXT,                   -- JSON: [装备 id]
    PRIMARY KEY (season_id, guide_id, phase, sort)
);
