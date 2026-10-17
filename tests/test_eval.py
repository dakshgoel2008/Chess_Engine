"""Tests for classical evaluation: symmetry, positional features, and C-vs-Python parity."""

import random
import pytest
from castled.core.board import Board, HAS_CFFI
from castled.eval.classical import evaluate, evaluate_python


def _mirror_fen(fen: str) -> str:
    """Color-flip a FEN string (swap White/Black pieces, flip ranks 1-8, flip turn)."""
    parts = fen.split()
    rows = parts[0].split("/")
    flipped_rows = [row.swapcase() for row in reversed(rows)]
    turn = "b" if parts[1] == "w" else "w"

    castling = ""
    if "k" in parts[2]:
        castling += "K"
    if "q" in parts[2]:
        castling += "Q"
    if "K" in parts[2]:
        castling += "k"
    if "Q" in parts[2]:
        castling += "q"
    castling = castling or "-"

    ep = parts[3]
    if ep != "-":
        file_ch = ep[0]
        rank_num = 9 - int(ep[1])
        ep = f"{file_ch}{rank_num}"

    return f"{'/'.join(flipped_rows)} {turn} {castling} {ep} {parts[4]} {parts[5]}"


def test_starting_position_is_zero():
    b = Board()
    assert evaluate(b) == 0
    assert evaluate_python(b) == 0


def test_symmetry_across_random_positions():
    # A color-flipped position with flipped side-to-move must have the exact same stm score,
    # and the same position with flipped side-to-move must negate the score.
    rng = random.Random(2026)
    for _ in range(25):
        b = Board()
        for _ in range(rng.randint(1, 30)):
            legal = b.legal_moves
            if not legal:
                break
            b.push(rng.choice(legal))

        fen = b.fen
        mirrored = Board(_mirror_fen(fen))
        assert evaluate(b) == evaluate(mirrored)
        assert evaluate_python(b) == evaluate_python(mirrored)


@pytest.mark.skipif(not HAS_CFFI, reason="C extension not built")
def test_c_and_python_eval_exact_parity():
    rng = random.Random(999)
    for _ in range(25):
        b = Board(_use_c=True)
        for _ in range(30):
            assert evaluate(b) == evaluate_python(b)
            legal = b.legal_moves
            if not legal:
                break
            b.push(rng.choice(legal))


def test_material_advantage_ordering():
    equal = Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    up_pawn = Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    up_knight = Board("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
    up_rook = Board("4k3/8/8/8/4R3/8/8/4K3 w - - 0 1")
    up_queen = Board("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1")

    assert evaluate(equal) < evaluate(up_pawn) < evaluate(up_knight) < evaluate(up_rook) < evaluate(up_queen)


def test_side_to_move_perspective_flips_sign():
    w_turn = Board("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1")
    b_turn = Board("4k3/8/8/8/4Q3/8/8/4K3 b - - 0 1")
    assert evaluate(w_turn) == -evaluate(b_turn)
    assert evaluate(w_turn) > 800


def test_passed_pawn_and_advancement_bonus():
    # White passed pawn on e6 is worth much more than a blocked pawn on e3.
    passed_e6 = Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")
    blocked_e3 = Board("4k3/8/8/8/4p3/4P3/8/4K3 w - - 0 1")
    assert evaluate(passed_e6) > evaluate(blocked_e3) + 150


def test_doubled_and_isolated_pawn_penalties():
    # Connected pawns on d2+e2 vs doubled isolated pawns on d2+d3.
    healthy = Board("4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1")
    doubled_isolated = Board("4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1")
    assert evaluate(healthy) > evaluate(doubled_isolated)


def test_bishop_pair_bonus():
    # Two bishops vs bishop + knight with identical squares/material baseline.
    two_bishops = Board("4k3/8/8/8/8/8/8/2B2B1K w - - 0 1")
    one_bishop_Only = Board("4k3/8/8/8/8/8/8/2B4K w - - 0 1")
    diff_second_bishop = evaluate(two_bishops) - evaluate(one_bishop_Only)
    # Second bishop adds base bishop value + bishop pair bonus.
    assert diff_second_bishop > 330


def test_rook_open_vs_closed_file():
    open_file = Board("4k3/p7/8/8/8/8/P7/3RK3 w - - 0 1")
    closed_file = Board("4k3/3p4/8/8/8/8/3P4/3RK3 w - - 0 1")
    assert evaluate(open_file) > evaluate(closed_file)


def test_king_pawn_shield_bonus_in_middlegame():
    shielded = Board("r1bq1rk1/ppp2ppp/2n2n2/8/8/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1")
    unshielded = Board("r1bq1rk1/ppp2ppp/2n2n2/8/8/2N2N2/PPP5/R1BQ1RK1 w - - 0 1")
    assert evaluate(shielded) > evaluate(unshielded)

