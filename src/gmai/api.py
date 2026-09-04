"""HTTP inference service.

Three endpoints:

``POST /move``
    Given a FEN, return the agent's move. The response carries **``in_scope``**,
    which is false whenever the position is outside the endgames the model was
    trained on. A model that quietly answers questions it was never trained for
    is worse than one that says so; the move is still returned, but the caller
    knows what it is worth.

``GET /health``
    Liveness plus which checkpoint is loaded.

``GET /metrics``
    Prometheus exposition: request counts, inference latency, Q-value
    distribution, and the rate of out-of-scope positions.

Run with::

    GMAI_CHECKPOINT=runs/<run>/final.pt uvicorn gmai.api:app --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from . import __version__
from .encoding import action_to_move
from .endgames import ENDGAME_SPECS

# --------------------------------------------------------------------- metrics
MOVES = Counter("gmai_move_requests_total", "Move requests", ["outcome"])
LATENCY = Histogram(
    "gmai_inference_seconds",
    "Model inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
OUT_OF_SCOPE = Counter(
    "gmai_out_of_scope_total", "Positions outside the trained endgames"
)
Q_VALUE = Histogram(
    "gmai_selected_q_value",
    "Q-value of the selected move",
    buckets=(-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0),
)
MODEL_LOADED = Gauge("gmai_model_loaded", "1 if a checkpoint is loaded")


# ---------------------------------------------------------------------- models
class MoveRequest(BaseModel):
    fen: str = Field(..., description="Position in Forsyth-Edwards Notation")


class MoveResponse(BaseModel):
    uci: str = Field(..., description="Selected move in UCI notation")
    san: str = Field(..., description="Same move in algebraic notation")
    q_value: float = Field(..., description="Q-value the model assigns to it")
    inference_ms: float
    in_scope: bool = Field(
        ...,
        description="Whether the position matches a trained endgame "
        "(KQ vs K, KR vs K, KRR vs K). Moves for out-of-scope positions are "
        "returned but are not supported by training.",
    )
    scope_detail: str
    legal_moves: int


class HealthResponse(BaseModel):
    status: str
    version: str
    checkpoint: str | None
    model_loaded: bool
    supported_endgames: list[str]


# ------------------------------------------------------------------- app state
class _State:
    agent: Any = None
    checkpoint: str | None = None


state = _State()


def load_agent(checkpoint: str | os.PathLike | None = None) -> None:
    """Load a checkpoint into the module-level agent."""
    checkpoint = checkpoint or os.environ.get("GMAI_CHECKPOINT")
    if not checkpoint:
        state.agent, state.checkpoint = None, None
        MODEL_LOADED.set(0)
        return

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    from .agent import DQNAgent  # imported lazily: keeps torch off the import path

    agent = DQNAgent.from_checkpoint(path, device=os.environ.get("GMAI_DEVICE"))
    agent.epsilon = 0.0
    agent.online.eval()
    state.agent, state.checkpoint = agent, str(path)
    MODEL_LOADED.set(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only load from the environment if nothing has been loaded already —
    # tests and embedders call load_agent() directly before starting the app,
    # and startup must not undo that.
    if state.agent is None:
        try:
            load_agent()
        except FileNotFoundError:
            MODEL_LOADED.set(0)  # start anyway: /health reports the problem
    yield


app = FastAPI(
    title="GMAI",
    description="Search-free deep RL chess agent for forced-mate endgames",
    version=__version__,
    lifespan=lifespan,
)


# -------------------------------------------------------------------- scope
def classify_scope(board: chess.Board) -> tuple[bool, str]:
    """Is this position one of the endgames the model was trained on?"""
    pieces = board.piece_map().values()
    by_color: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
    for piece in pieces:
        if piece.piece_type != chess.KING:
            by_color[piece.color].append(piece.piece_type)

    if len(board.piece_map()) - sum(len(v) for v in by_color.values()) != 2:
        return False, "both kings must be present"

    strong = [c for c in (chess.WHITE, chess.BLACK) if by_color[c]]
    if not strong:
        return False, "K vs K: drawn, nothing to play for"
    if len(strong) == 2:
        return False, "both sides have material: out of scope"

    material = sorted(by_color[strong[0]])
    known = {
        (chess.QUEEN,): "KQvK",
        (chess.ROOK,): "KRvK",
        (chess.ROOK, chess.ROOK): "KRRvK",
    }
    kind = known.get(tuple(material))
    if kind is None or kind not in ENDGAME_SPECS:
        names = ", ".join(chess.piece_name(p) for p in material)
        return False, f"K+{names} vs K is not a trained endgame"
    return True, kind


# ----------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.agent is not None else "no model",
        version=__version__,
        checkpoint=state.checkpoint,
        model_loaded=state.agent is not None,
        supported_endgames=list(ENDGAME_SPECS),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/move", response_model=MoveResponse)
def move(request: MoveRequest) -> MoveResponse:
    if state.agent is None:
        MOVES.labels(outcome="no_model").inc()
        raise HTTPException(503, "no checkpoint loaded; set GMAI_CHECKPOINT")

    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        MOVES.labels(outcome="bad_fen").inc()
        raise HTTPException(422, f"invalid FEN: {exc}") from exc

    if not board.is_valid():
        MOVES.labels(outcome="bad_fen").inc()
        raise HTTPException(422, "FEN describes an illegal position")
    if board.is_game_over(claim_draw=True):
        MOVES.labels(outcome="game_over").inc()
        raise HTTPException(409, "game is already over in this position")

    in_scope, detail = classify_scope(board)
    if not in_scope:
        OUT_OF_SCOPE.inc()

    import torch  # local import keeps module import light

    from .encoding import encode_board, legal_action_mask
    from .model import masked_q_values

    started = time.perf_counter()
    mask = legal_action_mask(board)
    with torch.no_grad():
        tensor = torch.from_numpy(encode_board(board)).unsqueeze(0).to(state.agent.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(state.agent.device)
        q = masked_q_values(state.agent.online(tensor, mask_t), mask_t)
        action = int(q.argmax(dim=1).item())
        q_value = float(q[0, action].item())
    elapsed = time.perf_counter() - started

    LATENCY.observe(elapsed)
    Q_VALUE.observe(q_value)
    MOVES.labels(outcome="ok").inc()

    chosen = action_to_move(action, board)
    return MoveResponse(
        uci=chosen.uci(),
        san=board.san(chosen),
        q_value=round(q_value, 4),
        inference_ms=round(elapsed * 1000, 3),
        in_scope=in_scope,
        scope_detail=detail,
        legal_moves=int(mask.sum()),
    )
