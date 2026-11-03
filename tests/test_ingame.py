import json
from pathlib import Path

import httpx
import pytest

from jccbrain.data import official as off


@pytest.fixture
def mock_client(monkeypatch):
    clients = []
    monkeypatch.setattr(off.time, "sleep", lambda _: None)

    def make(handler):
        client = off.OfficialClient(interval_s=0)
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize("way", ["7", 7])
def test_ingame_way_id_and_url_use_cached_home_config(mock_client, way):
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url.copy_with(query=None)) == off.INGAME_HOME
        assert request.url.params["t"].isdigit()
        config = {"lineup_way_id": way, "other": {"nested": True}}
        return httpx.Response(200, text=f'<script>ZMProjectConfig = {json.dumps(config)};</script>')

    client = mock_client(handler)
    assert client.ingame_way_id() == "7"
    assert client.ingame_way_id() == "7"
    assert client.ingame_lineup_url("18_S19") == off.INGAME_LINEUP.format(
        season_num="19", way="7", mode="18")
    assert len(requests) == 1


def test_ingame_way_id_accepts_script_end_followed_by_css(mock_client, caplog):
    html = ('<script>window.ZMProjectConfig = {"project_name":"ingame","lineup_way_id":"5",'
            '"ZMSERVICE":"https://mlol.qt.qq.com/"}</script>\n'
            '    <style>body{margin:0;padding:0}.a{width:1px;}</style>')
    client = mock_client(lambda request: httpx.Response(200, text=html))
    assert client.ingame_way_id() == "5"
    assert "微社区配置读取失败" not in caplog.text


@pytest.mark.parametrize(("status", "html", "attempts"), [
    (500, "server error", 3),
    (200, "<html>no config</html>", 1),
    (200, 'ZMProjectConfig = {"lineup_way_id":};', 1),
    (200, 'ZMProjectConfig = {"other": "value"};', 1),
    (200, 'ZMProjectConfig = [];', 1),
    (200, 'ZMProjectConfig = null;', 1),
])
def test_ingame_way_id_falls_back_and_caches(mock_client, caplog, status, html, attempts):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text=html)

    client = mock_client(handler)
    assert client.ingame_way_id() == "2"
    assert client.ingame_way_id() == "2"
    assert len(requests) == attempts
    assert "微社区配置读取失败" in caplog.text


def test_unsupported_ingame_season_makes_no_requests(mock_client):
    client = mock_client(lambda request: pytest.fail(f"不应发出请求：{request.url}"))
    assert client.ingame_lineup_url("17_S18") is None
    assert client.ingame_lineups("17_S18") == []


def test_ingame_fixture_normalizes_and_converts_guides(mock_client):
    fixture = Path(__file__).parent / "fixtures" / "ingame_lineups_18s19.json"
    body = json.loads(fixture.read_text(encoding="utf-8"))
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.params["t"].isdigit()
        if request.url.path == "/ingame/":
            return httpx.Response(200, text='ZMProjectConfig = {"lineup_way_id":"2"};')
        assert request.url.path.endswith("/m19/2/18/lineup_detail_total_new.json")
        return httpx.Response(200, json=body)

    lineups = mock_client(handler).ingame_lineups("18_S19")
    assert len(requests) == 2
    assert len(lineups) == 2
    assert all(isinstance(ln["detail"], str) for ln in lineups)
    assert all(isinstance(ln["lineupauthor_data"], dict) for ln in lineups)
    guides, heroes = off.convert_guides(lineups)
    assert len(guides) == 2
    assert all(guide["name"] and guide["author"] for guide in guides)
    assert {hero["guide_id"] for hero in heroes} == {guide["id"] for guide in guides}


def test_normalize_official_lineup_preserves_detail_and_decodes_author():
    lineup = {"id": "1", "detail": '{ "line_name": "测试阵容" }',
              "lineupauthor_data": '{"name":"测试作者"}'}
    normalized = off.normalize_lineup(lineup)
    assert normalized == {**lineup, "lineupauthor_data": {"name": "测试作者"}}
    assert normalized is not lineup
    assert isinstance(lineup["lineupauthor_data"], str)


def test_normalize_ingame_lineup_preserves_source_and_unicode():
    lineup = {"detail": {"line_name": "测试阵容"}, "lineupauthor_data": {"name": "测试作者"}}
    normalized = off.normalize_lineup(lineup)
    assert json.loads(normalized["detail"]) == lineup["detail"]
    assert "测试阵容" in normalized["detail"]
    assert isinstance(lineup["detail"], dict)
    assert normalized["lineupauthor_data"] is lineup["lineupauthor_data"]


@pytest.mark.parametrize("author", ["invalid JSON", '[]', '"name"', "null", None])
def test_normalize_unusable_author_becomes_none(author):
    assert off.normalize_lineup({"detail": "{}", "lineupauthor_data": author}) == {
        "detail": "{}", "lineupauthor_data": None}


@pytest.mark.parametrize(("status", "body"), [(500, "server error"), (200, "invalid JSON")])
def test_ingame_lineup_errors_propagate(mock_client, status, body):
    def handler(request):
        if request.url.path == "/ingame/":
            return httpx.Response(200, text='ZMProjectConfig = {"lineup_way_id":"2"};')
        return httpx.Response(status, text=body)

    with pytest.raises(off.OfficialError):
        mock_client(handler).ingame_lineups("18_S19")


def test_lineup_stats_normalizes_ids_and_values(mock_client):
    requests = []
    rows = [
        {"lineup_id": "1", "use": "100", "week_use_num": "20", "collect": 3,
         "comment": "4", "stat_date": "20261022"},
        {"lineup_id": "2", "use": 10, "week_use_num": -1, "collect": "bad",
         "comment": -2, "stat_date": "20261022"},
        {"lineup_id": "3", "use": 0, "week_use_num": -1, "collect": 0,
         "comment": 0, "stat_date": ""},
    ]

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == off.INGAME_STAT_URL
        assert json.loads(request.content) == {"mode": 6, "lineup_ids": ["1", "2", "3"]}
        return httpx.Response(200, json={"result": "0", "data": rows})

    client = mock_client(handler)
    result = client.lineup_stats(iter([1, "2", "", "1", 3]))
    assert len(requests) == 1
    assert result == {
        "1": {"use": 100, "week_use": 20, "collect": 3, "comment": 4, "stat_date": "20261022"},
        "2": {"use": 10, "week_use": None, "collect": None, "comment": None, "stat_date": "20261022"},
    }
    assert client.raw_responses == {}
    assert len(client.raw_posts) == 1
    post = client.raw_posts[0]
    assert post["url"] == off.INGAME_STAT_URL
    assert post["request"] == {"mode": 6, "lineup_ids": ["1", "2", "3"]}
    assert json.loads(post["body"]) == {"result": "0", "data": rows}


def test_lineup_stats_batches_and_keeps_all_raw_posts(mock_client, monkeypatch):
    monkeypatch.setattr(off, "INGAME_STAT_BATCH", 2)
    batches = []

    def handler(request):
        ids = json.loads(request.content)["lineup_ids"]
        batches.append(ids)
        return httpx.Response(200, json={"result": 0, "data": [
            {"lineup_id": i, "use": 10, "week_use_num": 1, "collect": 0,
             "comment": 0, "stat_date": "20261022"} for i in ids]})

    client = mock_client(handler)
    assert set(client.lineup_stats(range(5))) == {"0", "1", "2", "3", "4"}
    assert batches == [["0", "1"], ["2", "3"], ["4"]]
    assert len(client.raw_posts) == 3
    for post, ids in zip(client.raw_posts, batches):
        assert set(post) == {"url", "request", "body"}
        assert post["url"] == off.INGAME_STAT_URL
        assert post["request"] == {"mode": 6, "lineup_ids": ids}
        assert [r["lineup_id"] for r in json.loads(post["body"])["data"]] == ids


@pytest.mark.parametrize("ids", [[], ["", ""], iter(())])
def test_lineup_stats_empty_makes_no_requests(mock_client, ids):
    client = mock_client(lambda request: pytest.fail("空 id 不应发请求"))
    assert client.lineup_stats(ids) == {}
    assert client.raw_posts == []


@pytest.mark.parametrize("body", [
    {"result": 1000, "msg": "请求失败"},
    {"result": "bad", "err_msg": "请求失败"},
    {"msg": "请求失败"},
])
def test_lineup_stats_api_errors_raise(mock_client, body):
    client = mock_client(lambda request: httpx.Response(200, json=body))
    with pytest.raises(off.OfficialError, match="请求失败"):
        client.lineup_stats(["1"])


@pytest.mark.parametrize(("status", "content"), [(500, "server error"), (200, "not JSON")])
def test_lineup_stats_http_and_json_errors_retry(mock_client, status, content):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text=content)

    with pytest.raises(off.OfficialError):
        mock_client(handler).lineup_stats(["1"])
    assert len(requests) == 3


def test_lineup_stats_later_batch_failure_raises(mock_client, monkeypatch):
    monkeypatch.setattr(off, "INGAME_STAT_BATCH", 1)

    def handler(request):
        if json.loads(request.content)["lineup_ids"] == ["1"]:
            return httpx.Response(200, json={"result": 0, "data": [
                {"lineup_id": "1", "use": 10, "stat_date": "20261022"}]})
        return httpx.Response(200, json={"result": 1000, "msg": "第二批失败"})

    with pytest.raises(off.OfficialError, match="第二批失败"):
        mock_client(handler).lineup_stats(["1", "2"])
