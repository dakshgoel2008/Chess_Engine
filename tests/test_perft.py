"""Perft correctness tests across standard chess benchmark positions."""

from castled.core.board import Board


def test_perft_startpos():
    b = Board()
    assert b.perft(0) == 1
    assert b.perft(1) == 20
    assert b.perft(2) == 400
    assert b.perft(3) == 8902
    assert b.perft(4) == 197281


def test_perft_kiwipete():
    b = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert b.perft(1) == 48
    assert b.perft(2) == 2039
    assert b.perft(3) == 97862
    if b._use_c:
        assert b.perft(4) == 4085603


def test_perft_pos3_en_passant_pins():
    b = Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    assert b.perft(1) == 14
    assert b.perft(2) == 191
    assert b.perft(3) == 2812
    assert b.perft(4) == 43238
    if b._use_c:
        assert b.perft(5) == 674624


def test_perft_pos4_promotions_and_castling():
    b = Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    assert b.perft(1) == 6
    assert b.perft(2) == 264
    assert b.perft(3) == 9467
    if b._use_c:
        assert b.perft(4) == 422333


def test_perft_pos5_talkchess():
    b = Board("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
    assert b.perft(1) == 44
    assert b.perft(2) == 1486
    assert b.perft(3) == 62379


def test_perft_pos6_alternative():
    b = Board("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
    assert b.perft(1) == 46
    assert b.perft(2) == 2079
    if b._use_c:
        assert b.perft(3) == 89890


def test_perft_promo():
    b = Board("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1")
    assert b.perft(1) == 24
    assert b.perft(2) == 496
    assert b.perft(3) == 9483

