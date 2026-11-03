import pytest
from types import SimpleNamespace

from jccbrain import cli
from jccbrain.data.db import GameDB, SCHEMA_VERSION


INGAME_COLUMNS = {"channels", "ingame_sort", "use_count", "week_use", "collect_count", "comment_count", "stat_date"}


@pytest.fixture
def empty_db(tmp_path):
    db = GameDB(tmp_path / "game.db", "18_S19")
    yield db
    db.close()


def test_new_db_has_ingame_columns_and_version(empty_db):
    columns = {r[1] for r in empty_db._conn.execute("PRAGMA table_info(guides)")}
    assert INGAME_COLUMNS <= columns
    assert empty_db.meta("schema_version") == str(SCHEMA_VERSION)
    assert empty_db.stats()["ingame_guides"] == 0
    assert empty_db.stats()["guides_with_stats"] == 0
    assert empty_db.stats()["guide_stat_date"] is None


def test_v8_migration_keeps_existing_guide(tmp_path):
    path = tmp_path / "old.db"
    db = GameDB(path, "18_S19")
    with db.transaction() as conn:
        for column in sorted(INGAME_COLUMNS):
            conn.execute(f"ALTER TABLE guides DROP COLUMN {column}")
        conn.execute("UPDATE meta SET value = '8' WHERE key = 'schema_version'")
        conn.execute("INSERT INTO guides (season_id, id, name, sort) VALUES ('18_S19', 'old', '旧攻略', 20)")
    db.close()

    db = GameDB(path, "18_S19")
    try:
        assert INGAME_COLUMNS <= {r[1] for r in db._conn.execute("PRAGMA table_info(guides)")}
        assert db.meta("schema_version") == str(SCHEMA_VERSION)
        guide = db.guides["old"]
        assert guide.name == "旧攻略"
        assert guide.channels == ("web",)
        assert all(getattr(guide, key) is None for key in INGAME_COLUMNS - {"channels"})
        assert db._conn.execute("SELECT sort FROM guides").fetchone()[0] == 20
    finally:
        db.close()


def test_ingame_guide_loads_fields_and_stats(empty_db):
    with empty_db.transaction() as conn:
        conn.execute("INSERT INTO guides (season_id, id, name, sort, channels, ingame_sort, use_count, "
                     "week_use, collect_count, comment_count, stat_date) VALUES "
                     "('18_S19', 'both', '双渠道攻略', 1, '[\"web\",\"ingame\"]', 100, 10, 3, 2, 1, '20261022')")
        conn.execute("INSERT INTO guides (season_id, id, name, sort) VALUES ('18_S19', 'web', '官网攻略', 9)")
    empty_db.reload()
    guide = empty_db.guides["both"]
    assert guide.channels == ("web", "ingame")
    assert (guide.ingame_sort, guide.use_count, guide.week_use) == (100, 10, 3)
    assert (guide.collect_count, guide.comment_count, guide.stat_date) == (2, 1, "20261022")
    assert list(empty_db.guides) == ["web", "both"]  # 仍按官网 sort 排序
    assert empty_db.guides["web"].channels == ("web",)
    stats = empty_db.stats()
    assert stats["ingame_guides"] == 1
    assert stats["guides_with_stats"] == 1
    assert stats["guide_stat_date"] == "20261022"


@pytest.mark.parametrize(("ingame", "date"), [(False, None), (True, "20261022"), (True, None)])
def test_cli_db_stats_ingame_summary(empty_db, monkeypatch, capsys, ingame, date):
    if ingame:
        with empty_db.transaction() as conn:
            conn.execute("INSERT INTO guides (season_id, id, name, channels, use_count, stat_date) "
                         "VALUES ('18_S19', 'one', '微社区攻略', '[\"ingame\"]', 10, ?)", (date,))
        empty_db.reload()
    monkeypatch.setattr(cli, "_ctx", lambda args: SimpleNamespace(db=empty_db))
    assert cli.cmd_db_stats(SimpleNamespace()) == 0
    output = capsys.readouterr().out
    assert '"ingame_guides":' in output
    assert "每个费用的英雄数：" in output
    if ingame:
        summary = f"微社区：阵容 1 套，带热度 1 套，统计日期 {date or '—'}"
        assert summary in output
        assert output.index("}\n") < output.index(summary) < output.index("每个费用的英雄数：")
    else:
        assert "微社区：" not in output
