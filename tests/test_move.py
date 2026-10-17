"""Tests for Move encoding and UCI parsing."""

import sys
sys.path.insert(0, '.')

from castled.core.move import Move
from castled.core.types import MoveFlag, PieceType, Square


def test_make_and_unpack():
    m = Move.make(Square.E2, Square.E4, MoveFlag.DOUBLE_PAWN)
    assert m.from_square == Square.E2
    assert m.to_square == Square.E4
    assert m.flags == MoveFlag.DOUBLE_PAWN

def test_capture_flag():
    m = Move.make(0, 9, MoveFlag.CAPTURE)
    assert m.is_capture()
    assert not m.is_promotion()

def test_promotion_flag():
    m = Move.make(48, 56, MoveFlag.PROMO_Q)
    assert m.is_promotion()
    assert not m.is_capture()
    assert m.promotion_piece() == PieceType.QUEEN

def test_capture_promotion():
    m = Move.make(49, 56, MoveFlag.CAPTURE_PROMO_N)
    assert m.is_capture()
    assert m.is_promotion()
    assert m.promotion_piece() == PieceType.KNIGHT

def test_castling():
    m = Move.make(4, 6, MoveFlag.KING_CASTLE)
    assert m.is_castling()
    assert not m.is_capture()

def test_en_passant():
    m = Move.make(33, 40, MoveFlag.EP_CAPTURE)
    assert m.is_en_passant()
    assert m.is_capture()  # ep is a capture

def test_to_uci():
    m = Move.make(Square.E2, Square.E4, MoveFlag.QUIET)
    assert m.to_uci() == "e2e4"

def test_to_uci_promotion():
    m = Move.make(Square.E7, Square.E8, MoveFlag.PROMO_Q)
    assert m.to_uci() == "e7e8q"

def test_from_uci():
    m = Move.from_uci("e2e4")
    assert m.from_square == Square.E2
    assert m.to_square == Square.E4

def test_from_uci_promotion():
    m = Move.from_uci("a7a8n")
    assert m.is_promotion()
    assert m.promotion_piece() == PieceType.KNIGHT

def test_null_move():
    assert Move.NULL._data == 0
    assert Move.NULL.to_uci() == "0000"

def test_equality():
    a = Move.make(0, 8, MoveFlag.QUIET)
    b = Move.make(0, 8, MoveFlag.QUIET)
    c = Move.make(0, 16, MoveFlag.QUIET)
    assert a == b
    assert a != c

def test_hash():
    a = Move.make(0, 8)
    b = Move.make(0, 8)
    assert hash(a) == hash(b)
    assert a in {b}  # works in sets

