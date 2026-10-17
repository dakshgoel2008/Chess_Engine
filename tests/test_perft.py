import pytest
from castled.core.board import Board

def test_perft_startpos():
    b = Board()
    assert b.perft(1) == 20
    assert b.perft(2) == 400
    assert b.perft(3) == 8902

def test_perft_kiwipete():
    b = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert b.perft(1) == 48
    assert b.perft(2) == 2039
    assert b.perft(3) == 97862

def test_perft_pos3():
    b = Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    assert b.perft(1) == 14
    assert b.perft(2) == 191

def test_perft_promo():
    b = Board("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1")
    assert b.perft(1) == 24
