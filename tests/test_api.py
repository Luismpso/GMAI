"""Integration tests for the inference API."""

import chess
import pytest
import torch
from fastapi.testclient import TestClient

from gmai import api
from gmai.agent import DQNAgent


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    agent.save(path)
    return path


@pytest.fixture
def client(checkpoint):
    api.load_agent(checkpoint)
    with TestClient(api.app) as c:
        yield c
    api.state.agent, api.state.checkpoint = None, None


@pytest.fixture
def empty_client():
    api.state.agent, api.state.checkpoint = None, None
    with TestClient(api.app) as c:
        yield c


KQVK = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
KRVK = "4k3/8/8/8/8/8/R7/4K3 w - - 0 1"


class TestHealth:
    def test_reports_ok_with_a_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "KQvK" in body["supported_endgames"]

    def test_reports_missing_model(self, empty_client):
        body = empty_client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["status"] == "no model"

    def test_includes_version(self, client):
        assert client.get("/health").json()["version"]


class TestMove:
    def test_returns_a_legal_move(self, client):
        body = client.post("/move", json={"fen": KQVK}).json()
        board = chess.Board(KQVK)
        assert chess.Move.from_uci(body["uci"]) in board.legal_moves
        assert body["san"]
        assert body["legal_moves"] == board.legal_moves.count()

    def test_reports_latency_and_q_value(self, client):
        body = client.post("/move", json={"fen": KQVK}).json()
        assert body["inference_ms"] > 0
        assert isinstance(body["q_value"], float)

    def test_rejects_malformed_fen(self, client):
        assert client.post("/move", json={"fen": "not a fen"}).status_code == 422

    def test_rejects_illegal_position(self, client):
        # Adjacent kings: parses as a FEN, but is not a reachable position.
        r = client.post("/move", json={"fen": "4kK2/8/8/8/8/8/8/8 w - - 0 1"})
        assert r.status_code == 422

    def test_rejects_finished_game(self, client):
        mate = "6k1/6Q1/6K1/8/8/8/8/8 b - - 0 1"  # verified checkmate
        assert client.post("/move", json={"fen": mate}).status_code == 409

    def test_missing_field_is_rejected(self, client):
        assert client.post("/move", json={}).status_code == 422

    def test_503_without_a_model(self, empty_client):
        assert empty_client.post("/move", json={"fen": KQVK}).status_code == 503


class TestScopeReporting:
    """The honest bit: the API says when it is out of its depth."""

    @pytest.mark.parametrize("fen,kind", [(KQVK, "KQvK"), (KRVK, "KRvK")])
    def test_trained_endgames_are_in_scope(self, client, fen, kind):
        body = client.post("/move", json={"fen": fen}).json()
        assert body["in_scope"] is True
        assert body["scope_detail"] == kind

    def test_start_position_is_out_of_scope(self, client):
        body = client.post("/move", json={"fen": chess.STARTING_FEN}).json()
        assert body["in_scope"] is False
        assert "material" in body["scope_detail"]

    def test_untrained_endgame_is_out_of_scope(self, client):
        # K+P vs K: playable and not drawn by material, but never trained on.
        body = client.post("/move", json={"fen": "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"})
        assert body.status_code == 200
        assert body.json()["in_scope"] is False
        assert "pawn" in body.json()["scope_detail"]

    def test_out_of_scope_still_returns_a_legal_move(self, client):
        fen = chess.STARTING_FEN
        body = client.post("/move", json={"fen": fen}).json()
        assert chess.Move.from_uci(body["uci"]) in chess.Board(fen).legal_moves


class TestScopeClassifier:
    @pytest.mark.parametrize(
        "fen,expected",
        [
            (KQVK, True),
            (KRVK, True),
            ("4k3/8/8/8/8/8/RR6/4K3 w - - 0 1", True),  # KRRvK
            ("4k3/8/8/8/8/8/N7/4K3 w - - 0 1", False),  # KNvK
            ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", False),  # KPvK
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", False),  # K vs K
            ("4k3/7q/8/8/8/8/Q7/4K3 w - - 0 1", False),  # both have material
        ],
    )
    def test_classification(self, fen, expected):
        assert api.classify_scope(chess.Board(fen))[0] is expected

    def test_detail_is_never_empty(self):
        for fen in (KQVK, chess.STARTING_FEN, "4k3/8/8/8/8/8/8/4K3 w - - 0 1"):
            assert api.classify_scope(chess.Board(fen))[1]


class TestMetrics:
    def test_exposes_prometheus_format(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "gmai_move_requests_total" in r.text

    def test_counts_requests(self, client):
        client.post("/move", json={"fen": KQVK})
        assert 'gmai_move_requests_total{outcome="ok"}' in client.get("/metrics").text

    def test_tracks_out_of_scope(self, client):
        client.post("/move", json={"fen": chess.STARTING_FEN})
        assert "gmai_out_of_scope_total" in client.get("/metrics").text

    def test_records_latency(self, client):
        client.post("/move", json={"fen": KQVK})
        assert "gmai_inference_seconds" in client.get("/metrics").text
