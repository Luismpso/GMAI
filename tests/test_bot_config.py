"""The lichess-bot template must render to the settings ChessNet needs.

lichess-bot aborts a game if it sends a UCI option the engine does not
declare, and turns ``engine_options`` into command-line flags for the engine,
so both are pinned here.
"""

import re
from pathlib import Path

import pytest

from chessnet.play import build_parser

yaml = pytest.importorskip("yaml")

BOT = Path(__file__).resolve().parent.parent / "bot"
TEMPLATE = BOT / "lichess-bot.template.yml"


def render() -> dict:
    text = TEMPLATE.read_text(encoding="ascii")
    for key, value in (
        ("ENGINE_DIR", "/x/bot"),
        ("PYTHON", "/x/python"),
        ("CHECKPOINT", "/x/best.pt"),
    ):
        text = text.replace("{{" + key + "}}", value)
    return yaml.safe_load(text)


def test_files_are_plain_ascii():
    # Windows PowerShell 5.1 reads files without a BOM in the ANSI code page.
    for name in ("lichess-bot.template.yml", "run_bot.ps1"):
        (BOT / name).read_bytes().decode("ascii")


def test_only_known_placeholders():
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", TEMPLATE.read_text()))
    assert placeholders == {"ENGINE_DIR", "PYTHON", "CHECKPOINT"}


def test_no_token_in_the_repository():
    assert render()["token"] == ""
    assert "lip_" not in TEMPLATE.read_text()


def test_engine_settings():
    engine = render()["engine"]
    assert engine["protocol"] == "uci"
    assert (BOT / engine["name"]).is_file()
    assert engine["ponder"] is False
    assert engine["uci_options"] == {}, "ChessNet declares no UCI options"
    assert engine["draw_or_resign"] == {
        "resign_enabled": False,
        "offer_draw_enabled": False,
    }


def test_engine_options_are_flags_chessnet_play_accepts():
    options = render()["engine"]["engine_options"]
    # lichess-bot's rule: --key=value, or a bare --key when the value is empty.
    argv = [f"--{k}" if v is None else f"--{k}={v}" for k, v in options.items()]
    args = build_parser().parse_args(argv)
    assert args.uci and args.checkpoint == "/x/best.pt"
    assert args.nodes >= 100_000, "the clock, not a node cap, should limit search"


def test_challenges_are_safe_on_the_clock():
    config = render()
    challenge, matchmaking = config["challenge"], config["matchmaking"]
    assert "bullet" not in challenge["time_controls"]
    assert challenge["min_increment"] >= 1 and challenge["min_base"] >= 180
    assert challenge["concurrency"] == 1
    assert all(i >= 1 for i in matchmaking["challenge_increment"])
    assert matchmaking["opponent_rating_difference"] == 300
    assert config["move_overhead"] >= 1000
    assert config["quit_after_all_games_finish"] is True, "Ctrl+C must not forfeit a game"
