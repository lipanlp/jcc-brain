"""后台加载结果的发布必须保留对象身份、连接与完整可查询数据。"""
from unittest.mock import Mock

import pytest

from jccbrain.context import AppContext
from jccbrain.core.settings import Settings
from jccbrain.data.db import GameDB


def test_publish_loaded_data_preserves_db_and_advisor_references(db, monkeypatch, tmp_path):
    monkeypatch.setenv('JCC_HOME', str(tmp_path / 'runtime'))
    ctx = AppContext(Settings())
    ctx.db = db
    advisor = ctx.advisor
    connection = db._conn
    before = db.hero('绯红树怪').skill_desc
    loaded = GameDB(db.path, db.season_id)
    try:
        with loaded.transaction() as conn:
            conn.execute("UPDATE heroes SET skill_desc='后台更新技能' WHERE name='绯红树怪'")
            conn.execute("UPDATE comps SET name='后台更新阵容'")
        loaded.reload()
        assert db.hero('绯红树怪').skill_desc == before
        expected = loaded.stats()
        monkeypatch.setattr(db, 'reload', Mock(side_effect=AssertionError('不能在发布时重新解析')))
        ctx.reload_data(loaded)
        assert ctx.db is advisor.db is db
        assert db._conn is connection
        assert db.hero('绯红树怪').skill_desc == '后台更新技能'
        assert db.stats() == expected
        assert advisor.matcher.comps[0].name == '后台更新阵容'
    finally:
        loaded.close()
        ctx.close()
    # 发布源的 SQLite 连接关闭后，界面自己的连接与内存查询仍正常。
    assert db.catalog()
    assert db.comp('104').name == '后台更新阵容'


@pytest.mark.parametrize('different_file', [False, True])
def test_rejects_unrelated_loaded_data(db, tmp_path, different_file):
    loaded = GameDB(tmp_path / 'other.sqlite3' if different_file else db.path,
                    db.season_id if different_file else '16_S19')
    before = db.heroes
    try:
        with pytest.raises(ValueError, match='同一数据库、同一赛季'):
            db.reload_from(loaded)
        assert db.heroes is before
    finally:
        loaded.close()
