"""启动读库不应争抢无必要的写锁，初始化异常必须释放资源。"""
import hashlib
import sqlite3

import pytest

from jccbrain.data.db import GameDB, SCHEMA_VERSION


def test_current_schema_opens_while_another_connection_has_reserved_writer(db):
    before = hashlib.sha256(db.path.read_bytes()).digest()
    writer = sqlite3.connect(db.path)
    writer.execute('BEGIN IMMEDIATE')
    try:
        reader = GameDB(db.path, db.season_id)
        try:
            assert reader.hero('奥恩').name == '奥恩'
            assert reader.comp('104') is not None
            assert reader._conn.total_changes == 0
            assert reader.meta('schema_version') == str(SCHEMA_VERSION)
        finally:
            reader.close()
    finally:
        writer.rollback()
        writer.close()
    assert hashlib.sha256(db.path.read_bytes()).digest() == before


def test_current_version_still_repairs_missing_columns_and_closes_failed_connection(db, monkeypatch):
    # 版本相同也必须校验结构，不能用版本号跳过实际缺列。
    with db.transaction() as conn:
        conn.execute('ALTER TABLE heroes DROP COLUMN skill_icon')
    real_connect = sqlite3.connect
    connections = []
    class TrackedConnection(sqlite3.Connection):
        closed = False
        def close(self):
            self.closed = True
            super().close()
    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs, factory=TrackedConnection)
        connections.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    writer = real_connect(db.path)
    writer.execute('BEGIN IMMEDIATE')
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            GameDB(db.path, db.season_id)
        assert len(connections) == 1 and connections[0].closed
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connections[0].execute('SELECT 1')
    finally:
        writer.rollback()
        writer.close()
    retry = GameDB(db.path, db.season_id)
    try:
        assert 'skill_icon' in {row['name'] for row in retry._conn.execute('PRAGMA table_info(heroes)')}
        assert retry.hero('奥恩') and retry.meta('schema_version') == str(SCHEMA_VERSION)
    finally:
        retry.close()


def test_initialization_closes_connection_when_cache_loading_fails(tmp_path, monkeypatch):
    path = tmp_path / 'db.sqlite3'
    GameDB(path).close()
    real_connect = sqlite3.connect
    connections = []
    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        connections.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    def broken(self):
        raise ValueError('损坏的缓存资料')
    monkeypatch.setattr(GameDB, 'reload', broken)
    with pytest.raises(ValueError, match='损坏'):
        GameDB(path)
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        connections[0].execute('SELECT 1')
    # 不仅连接被关闭，也没有遗留阻止后续迁移/同步的事务。
    writer = real_connect(path)
    try:
        writer.execute('BEGIN EXCLUSIVE')
    finally:
        writer.rollback()
        writer.close()


def test_old_schema_version_is_upgraded_without_changing_current_data(db):
    with db.transaction() as conn:
        conn.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
    upgraded = GameDB(db.path, db.season_id)
    try:
        assert upgraded.meta('schema_version') == str(SCHEMA_VERSION)
        assert upgraded.hero('奥恩') and upgraded.comp('104')
    finally:
        upgraded.close()
