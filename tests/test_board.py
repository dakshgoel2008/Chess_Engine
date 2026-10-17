import pytest
from castled.core.board import Board, HAS_CFFI

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
    undone = b.pop()
    assert undone == m
    assert b.fen == fen_before

def test_checkmate():
    b = Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    assert b.is_game_over()

@pytest.mark.skipif(not HAS_CFFI, reason="C extension not built")
def test_c_and_python_parity():
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    for fen in fens:
        bc = Board(fen, _use_c=True)
        bp = Board(fen, _use_c=False)
        assert bc.fen == bp.fen
        assert bc.zobrist_hash == bp.zobrist_hash
        assert set(bc.legal_moves) == set(bp.legal_moves)
        assert bc.perft(2) == bp.perft(2)

