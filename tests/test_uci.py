import chess
import pytest
import torch

from gmai.agent import DQNAgent
from gmai.uci import _best_move, _parse_position, uci_loop


@pytest.fixture(scope="module")
def tiny_agent():
    torch.manual_seed(0)
    agent = DQNAgent(channels=8, n_blocks=2, hidden=32, device="cpu", seed=0)
    agent.epsilon = 0.0
    agent.online.eval()
    return agent


class TestParsePosition:
    def test_startpos(self):
        board = _parse_position(chess.Board(), ["startpos"])
        assert board.fen() == chess.STARTING_FEN

    def test_startpos_with_moves(self):
        board = _parse_position(chess.Board(), ["startpos", "moves", "e2e4", "e7e5"])
        assert board.fullmove_number == 2
        assert board.turn == chess.WHITE
        assert board.piece_at(chess.E4).piece_type == chess.PAWN

    def test_fen_position(self):
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        board = _parse_position(chess.Board(), ["fen"] + fen.split())
        assert board.fen() == fen

    def test_fen_with_moves(self):
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        board = _parse_position(chess.Board(), ["fen"] + fen.split() + ["moves", "g1f3"])
        assert board.piece_at(chess.F3).piece_type == chess.KNIGHT

    def test_malformed_move_is_ignored(self):
        board = _parse_position(chess.Board(), ["startpos", "moves", "e2e4", "zzzz"])
        assert len(board.move_stack) == 1


class TestBestMove:
    def test_returns_legal_move(self, tiny_agent):
        board = chess.Board()
        move = _best_move(tiny_agent, board)
        assert move in board.legal_moves

    def test_returns_none_when_game_over(self, tiny_agent):
        board = chess.Board()
        for san in ["f3", "e5", "g4", "Qh4#"]:
            board.push_san(san)
        assert _best_move(tiny_agent, board) is None


class TestUciLoop:
    def _run(self, tiny_agent, commands, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", iter(c + "\n" for c in commands))
        uci_loop(tiny_agent)
        return capsys.readouterr().out.splitlines()

    def test_handshake(self, tiny_agent, monkeypatch, capsys):
        out = self._run(tiny_agent, ["uci", "quit"], monkeypatch, capsys)
        assert any(line.startswith("id name") for line in out)
        assert "uciok" in out

    def test_isready(self, tiny_agent, monkeypatch, capsys):
        out = self._run(tiny_agent, ["isready", "quit"], monkeypatch, capsys)
        assert "readyok" in out

    def test_go_returns_legal_bestmove(self, tiny_agent, monkeypatch, capsys):
        out = self._run(
            tiny_agent,
            ["ucinewgame", "position startpos moves e2e4", "go", "quit"],
            monkeypatch,
            capsys,
        )
        bestmove = [ln for ln in out if ln.startswith("bestmove")][0]
        uci = bestmove.split()[1]
        board = chess.Board()
        board.push_san("e4")
        assert chess.Move.from_uci(uci) in board.legal_moves

    def test_go_in_finished_game_returns_null_move(
        self, tiny_agent, monkeypatch, capsys
    ):
        mate = "position startpos moves f2f3 e7e5 g2g4 d8h4"
        out = self._run(tiny_agent, [mate, "go", "quit"], monkeypatch, capsys)
        assert "bestmove 0000" in out

    def test_blank_lines_are_ignored(self, tiny_agent, monkeypatch, capsys):
        out = self._run(tiny_agent, ["", "  ", "isready", "quit"], monkeypatch, capsys)
        assert out == ["readyok"]
