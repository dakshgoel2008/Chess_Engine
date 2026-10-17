"""Tests for Standard Algebraic Notation (SAN) and Portable Game Notation (PGN)."""

import pytest
from castled.core.board import Board
from castled.core.pgn import parse_pgn, dump_pgn


def test_san_basic_opening_moves():
    b = Board()
    m1 = b.parse_san("e4")
    assert b.san(m1) == "e4"
    b.push(m1)

    m2 = b.parse_san("Nf6")
    assert b.san(m2) == "Nf6"
    b.push(m2)


def test_san_disambiguation_by_file_rank_and_full_square():
    # Two white rooks on a1 and h1 can both reach d1 -> disambiguate by file (Rad1 / Rhd1).
    b_file = Board("4k3/8/8/8/8/8/4K3/R6R w - - 0 1")
    m_rad1 = b_file.parse_san("Rad1")
    assert b_file.san(m_rad1) == "Rad1"

    # Two white rooks on a1 and a8 can both reach a4 -> disambiguate by rank (R1a4 / R8a4).
    b_rank = Board("R7/4k3/8/8/8/8/4K3/R7 w - - 0 1")
    m_r1a4 = b_rank.parse_san("R1a4")
    assert b_rank.san(m_r1a4) == "R1a4"

    # Three white queens on a1, d1, a4 can all reach d4 -> a1 queen needs full square Qa1d4.
    b_full = Board("8/7k/8/8/Q7/8/7K/Q2Q4 w - - 0 1")
    m_full = b_full.parse_san("Qa1d4")
    assert b_full.san(m_full) == "Qa1d4"


def test_san_castling_promotions_check_and_mate():
    # Castling
    b_castle = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    ks = b_castle.parse_san("O-O")
    qs = b_castle.parse_san("O-O-O")
    assert b_castle.san(ks) == "O-O"
    assert b_castle.san(qs) == "O-O-O"

    # Promotion with capture and check
    b_promo = Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    m_promo = b_promo.parse_san("axb8=Q+")
    assert b_promo.san(m_promo) == "axb8=Q+"

    # Fool's mate checkmate suffix '#'
    b_mate = Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    m_mate = b_mate.parse_san("Qh4#")
    assert b_mate.san(m_mate) == "Qh4#"


def test_san_illegal_move_raises():
    b = Board()
    with pytest.raises(ValueError):
        b.parse_san("Ke2")


def test_pgn_parse_and_roundtrip():
    pgn_text = """[Event "Test Match"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. f3 {bad opening} e5 2. g4 (2. e4) Qh4# 0-1
"""
    game = parse_pgn(pgn_text)
    assert game.headers["Event"] == "Test Match"
    assert len(game.moves) == 4
    final_board = game.board()
    assert final_board.is_checkmate()

    exported = dump_pgn(final_board, {"Event": "Test Match"})
    reparsed = parse_pgn(exported)
    assert reparsed.moves == game.moves
    assert reparsed.board().fen == final_board.fen
