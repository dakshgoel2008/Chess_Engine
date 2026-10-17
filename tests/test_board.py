import pytest
from castled.core.board import Board

def test_board_init():
    b = Board()
    assert len(b.legal_moves) == 20

def test_fen_roundtrip():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b = Board(fen)
    assert b.fen == fen
    
def test_make_unmake():
    b = Board()
    fen_before = b.fen
    m = b.legal_moves[0]
    b.push(m)
    b.pop()
    assert b.fen == fen_before

def test_checkmate():
    b = Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    assert b.is_game_over()
