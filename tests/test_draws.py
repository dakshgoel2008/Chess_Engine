"""Tests for stalemate, threefold repetition, 50-move rule, insufficient material, and outcome()."""

from castled.core.board import Board
from castled.core.types import Color


def test_stalemate_detection():
    b = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert not b.is_check()
    assert b.is_stalemate()
    assert b.is_game_over()
    out = b.outcome()
    assert out is not None
    assert out.termination == "stalemate"
    assert out.winner is None
    assert out.result == "1/2-1/2"


def test_checkmate_outcome_white_wins():
    b = Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
    b.push(b.parse_uci("e1e8"))
    assert b.is_checkmate()
    out = b.outcome()
    assert out is not None
    assert out.termination == "checkmate"
    assert out.winner == Color.WHITE
    assert out.result == "1-0"


def test_threefold_repetition():
    b = Board()
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in cycle:
        b.push(b.parse_uci(uci))
    assert b.is_repetition(2)
    assert not b.is_repetition(3)
    assert not b.is_draw()

    for uci in cycle:
        b.push(b.parse_uci(uci))
    assert b.is_repetition(3)
    assert b.is_draw()
    assert b.is_game_over()
    out = b.outcome()
    assert out is not None
    assert out.termination == "repetition"
    assert out.result == "1/2-1/2"


def test_fifty_move_rule_boundary():
    b99 = Board("8/8/8/8/8/3k4/3r4/3K4 w - - 99 60")
    assert not b99.is_fifty_moves()
    assert not b99.is_draw()

    b100 = Board("8/8/8/8/8/3k4/3r4/3K4 w - - 100 60")
    assert b100.is_fifty_moves()
    assert b100.is_draw()
    assert b100.is_game_over()
    assert b100.outcome().termination == "fifty_moves"


def test_insufficient_material_cases():
    # K vs K
    assert Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1").is_insufficient_material()
    # K+N vs K
    assert Board("8/8/8/4k3/8/5N2/4K3/8 w - - 0 1").is_insufficient_material()
    # K+B vs K
    assert Board("8/8/8/4k3/8/5B2/4K3/8 w - - 0 1").is_insufficient_material()
    # K+B vs K+B on same color squares (c1 is dark, f8 is dark)
    assert Board("5b2/8/4k3/8/8/8/4K3/2B5 w - - 0 1").is_insufficient_material()
    # K+B vs K+B on opposite color squares (c1 is dark, c8 is light) -> sufficient
    assert not Board("2b5/8/4k3/8/8/8/4K3/2B5 w - - 0 1").is_insufficient_material()
    # K+P vs K -> sufficient
    assert not Board("8/8/8/4k3/4P3/8/4K3/8 w - - 0 1").is_insufficient_material()
    # K+N+N vs K -> not automatic insufficient material
    assert not Board("8/8/8/4k3/8/3NN3/4K3/8 w - - 0 1").is_insufficient_material()
