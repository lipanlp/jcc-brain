from jccbrain.core.models import GameState, Phase
from jccbrain.core.settings import Settings

from conftest import make_state


def test_parse_all_fixture_frames(fixture_states):
    for d in fixture_states:
        st = GameState.from_dict(d)
        assert len(st.shop) == 5
        assert st.phase in Phase


def test_state_fields_and_derived():
    st = GameState.from_dict(make_state(
        stage="4-1", xp=[10, 48],
        shop={"open": True, "slots": [{"name": "奥恩", "cost": 1, "traits": ["永恒之森"]}, None, None, None, None]},
        bench=[{"slot": 0, "star": 2, "name": "奥恩"}, {"slot": 1, "star": 1, "name": "?3"}],
        board=[{"x": 0.5, "y": 0.3, "star": 1, "name": "奥恩"}],
        board_count=[5, 7], streak={"kind": "win", "n": 3},
    ))
    assert st.stage_tuple == (4, 1)
    assert st.xp == (10, 48)
    assert st.board_count == (5, 7)
    assert st.owned_copies()["奥恩"] == 4          # 2 星 = 3 张 + 1 星 1 张
    assert "?3" not in st.owned_copies()
    assert st.bench_free == 7
    assert st.shop_names == ["奥恩", None, None, None, None]
    assert st.can_act


def test_bad_stage_and_unknown_phase():
    st = GameState.from_dict(make_state(stage="??", phase="weird"))
    assert st.stage_tuple is None
    assert st.phase == Phase.UNKNOWN


def test_rules(rules):
    assert rules.odds(6) == [0.30, 0.40, 0.25, 0.05, 0.0]
    assert rules.odds(99) == rules.odds(10)
    assert rules.interest(37) == 3 and rules.interest(120) == 5
    assert rules.streak_bonus(4) == 2
    assert rules.round_kind("2-4") == "carousel" and rules.round_kind("2-7") == "pve" and rules.round_kind("3-2") == "pvp"
    assert rules.target_level((3, 2)) == 6
    assert rules.target_level((4, 1), "fast8") == 8
    assert rules.next_augment_round((2, 5)) == "3-2"


def test_settings_roundtrip(tmp_path):
    s = Settings()
    s.llm.model = "deepseek-chat"
    s.actuator.dry_run = False
    s.assistant.default_lock_comp_id = "104"
    p = tmp_path / "s.json"
    s.save(p)
    s2 = Settings.load(p)
    assert s2.llm.model == "deepseek-chat" and s2.actuator.dry_run is False and s2.assistant.default_lock_comp_id == "104"


def test_settings_tolerates_unknown_and_missing(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"llm": {"model": "x", "unknown": 1}, "bogus": true}', encoding="utf-8")
    s = Settings.load(p)
    assert s.llm.model == "x" and s.vision.backend == "pyeye"


def test_legacy_permanent_lock_is_discarded_with_migration_log(tmp_path, caplog):
    import logging
    from dataclasses import asdict
    path = tmp_path / 'legacy.json'
    path.write_text('{"target_comp_id":"104","llm":{"model":"kept"}}')
    with caplog.at_level(logging.INFO, logger='jccbrain.core.settings'):
        settings = Settings.load(path)
    assert settings.assistant.default_lock_comp_id == ''
    assert 'target_comp_id' not in asdict(settings) and not hasattr(settings, 'target_comp_id')
    assert settings.llm.model == 'kept'
    assert '旧的永久锁定阵容已取消，可在偏好设置里设默认锁定阵容' in caplog.text


def test_default_lock_is_empty_for_new_settings():
    assert Settings().assistant.default_lock_comp_id == ''


def test_feature_settings_missing_fields_keep_legacy_behavior(tmp_path):
    import json
    path = tmp_path / 'old-settings.json'
    path.write_text(json.dumps({'scout': {'enabled': True, 'rounds': ['3-2']},
                                'autoroll': {'keep_gold': 20, 'read_interval_ms': 175}}))
    settings = Settings.load(path)
    assert settings.scout.enabled and settings.scout.schedule == 'key_rounds'
    assert settings.scout.rounds == ['3-2'] and settings.autoroll.read_interval_ms == 175
    assert settings.assistant.economy_preference == 'adaptive'
    assert settings.autopilot.augment_selection == 'manual'
    settings.save(path)
    assert Settings.load(path) == settings


def test_invalid_persisted_feature_enums_fall_back_to_explicit_defaults(tmp_path):
    import json
    path = tmp_path / 'bad-settings.json'
    path.write_text(json.dumps({'assistant': {'economy_preference': 'bad'}, 'scout': {'schedule': None},
                                'autopilot': {'augment_selection': 5}}))
    settings = Settings.load(path)
    assert settings.assistant.economy_preference == 'adaptive'
    assert settings.scout.schedule == 'key_rounds'
    assert settings.autopilot.augment_selection == 'manual'
