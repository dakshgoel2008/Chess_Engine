"""Comprehensive tests for transposition table, alpha-beta search, quiescence, and edge cases."""

import pytest

from castled.core.board import Board
from castled.core.move import Move
from castled.search import (
    SCORE_MATE,
    SCORE_MATE_THRESHOLD,
    Searcher,
    SearchInfo,
    SearchLimits,
    TTBound,
    TranspositionTable,
    scale_soft_time,
    search,
    see,
)


def test_tt_store_and_probe_bounds():
    for use_c in (True, False):
        tt = TranspositionTable(size_mb=1, _use_c=use_c)
        key = 0x123456789ABCDEF0
        move = Move.from_uci("e2e4")

        # Exact bound returns cutoff for any window at sufficient depth.
        tt.store(key, depth=4, score=150, bound=TTBound.EXACT, best_move=move, ply=0)
        hit, bm, sc = tt.probe(key, depth=4, alpha=-100, beta=200, ply=0)
        assert hit is True
        assert bm == move
        assert sc == 150

        # Deeper request does not cutoff, but still returns best_move for ordering.
        hit, bm, _ = tt.probe(key, depth=5, alpha=-100, beta=200, ply=0)
        assert hit is False
        assert bm == move

        # Lower bound cuts off only when score >= beta.
        tt.store(key, depth=4, score=250, bound=TTBound.LOWER, best_move=move, ply=0)
        hit_high, _, sc_high = tt.probe(key, depth=4, alpha=0, beta=200, ply=0)
        hit_low, bm_low, _ = tt.probe(key, depth=4, alpha=0, beta=300, ply=0)
        assert hit_high is True and sc_high == 250
        assert hit_low is False and bm_low == move

        # Upper bound cuts off only when score <= alpha.
        tt.store(key, depth=4, score=-50, bound=TTBound.UPPER, best_move=Move.NULL, ply=0)
        hit_up, bm_up, sc_up = tt.probe(key, depth=4, alpha=0, beta=100, ply=0)
        hit_miss, _, _ = tt.probe(key, depth=4, alpha=-100, beta=100, ply=0)
        assert hit_up is True and sc_up == -50
        # Preserves previous best_move when storing Move.NULL on fail-low for same key.
        assert bm_up == move
        assert hit_miss is False


def test_tt_mate_score_ply_adjustment():
    for use_c in (True, False):
        tt = TranspositionTable(size_mb=1, _use_c=use_c)
        key_win = 0xABCDEF0123456789
        key_loss = 0x9876543210FEDCBA
        move = Move.from_uci("d1h5")

        # Mate in 2 plies from a node at ply=4 has root-relative score SCORE_MATE - 6.
        tt.store(key_win, depth=5, score=SCORE_MATE - 6, bound=TTBound.EXACT, best_move=move, ply=4)
        # When probed at ply=1, it is still mate in 2 plies from that node -> SCORE_MATE - 3.
        hit, _, sc = tt.probe(key_win, depth=3, alpha=-SCORE_MATE, beta=SCORE_MATE, ply=1)
        assert hit is True
        assert sc == SCORE_MATE - 3

        # Being mated in 2 plies from ply=4 has root-relative score -SCORE_MATE + 6.
        tt.store(key_loss, depth=5, score=-SCORE_MATE + 6, bound=TTBound.EXACT, best_move=move, ply=4)
        hit, _, sc = tt.probe(key_loss, depth=3, alpha=-SCORE_MATE, beta=SCORE_MATE, ply=2)
        assert hit is True
        assert sc == -SCORE_MATE + 4


def test_mate_in_1():
    # Back-rank mate: Re8#
    fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=2)
        assert info.best_move.to_uci() == "e1e8"
        assert info.score == SCORE_MATE - 1


def test_mate_in_2():
    # Opera Game finish: 1. Qb8+ Nxb8 2. Rd8#
    fen = "4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=4)
        assert info.best_move.to_uci() == "b3b8"
        assert info.score == SCORE_MATE - 3


def test_mate_in_3():
    # Forced corridor mate in 3 with two rook interpositions:
    # 1. Rh8+ Rd8 2. Rxd8+ Re8 3. Rxe8#
    fen = "k7/p1K5/8/8/3r4/4r3/8/7R w - - 0 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=8, _use_c=use_c)
        info = searcher.search(board, depth=5)
        assert info.best_move.to_uci() == "h1h8"
        assert info.score == SCORE_MATE - 5


def test_quiescence_avoids_poisoned_capture():
    # White queen on d4 can capture pawn on d7 (defended by Black queen on d8)
    # or capture undefended rook on a4. Without quiescence Qxd7 looks playable,
    # while with quiescence Qxd7 loses the queen and Qxa4 wins a full rook.
    fen = "3q2k1/3p1ppp/8/8/r2Q4/8/5PPP/4R1K1 w - - 0 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=1)
        assert info.best_move.to_uci() == "d4a4"
        assert info.score > 300


def test_stalemate_avoidance_when_winning():
    # Black king trapped on a8; Qc7 would be stalemate, while Qa6# or Qc8# is checkmate.
    fen = "k7/8/1K6/8/2Q5/8/8/8 w - - 0 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=2)
        assert info.best_move.to_uci() != "c4c7"
        assert info.score > SCORE_MATE_THRESHOLD


def test_repetition_avoidance_when_ahead():
    # White is up a rook (+500). Play a back-and-forth sequence so the prior position is in history.
    fen = "6k1/8/8/8/8/8/R7/6K1 w - - 0 1"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        board.push(board.parse_uci("g1f1"))
        board.push(board.parse_uci("g8f8"))
        # Returning to g1 after f8g8 would repeat the starting position (score 0).
        board.push(board.parse_uci("f1g1"))
        # Now Black plays f8g8, reaching the initial position a second time.
        # If White plays g1f1 again, that repeats a position already visited.
        board.push(board.parse_uci("f8g8"))

        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=3)
        assert info.best_move.to_uci() != "g1f1"
        assert info.score > 300


def test_terminal_and_single_move_positions():
    # Already checkmated position returns Move.NULL and -SCORE_MATE.
    mated_fen = "6k1/5ppp/8/8/8/8/5PPP/3r2K1 w - - 0 1"
    for use_c in (True, False):
        board = Board(mated_fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=2, _use_c=use_c)
        info = searcher.search(board, depth=3)
        assert info.best_move == Move.NULL
        assert info.score == -SCORE_MATE

    # Already stalemated position returns Move.NULL and 0.
    stalemate_fen = "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"
    for use_c in (True, False):
        board = Board(stalemate_fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=2, _use_c=use_c)
        info = searcher.search(board, depth=3)
        assert info.best_move == Move.NULL
        assert info.score == 0

    # Single legal move (king forced to capture checking rook).
    only_move_fen = "6k1/5ppp/8/8/8/8/5PPP/5rK1 w - - 0 1"
    for use_c in (True, False):
        board = Board(only_move_fen, _use_c=use_c)
        searcher = Searcher(hash_size_mb=2, _use_c=use_c)
        info = searcher.search(board, depth=3)
        assert info.best_move.to_uci() == "g1f1"


def test_limits_callback_and_board_immutability():
    for use_c in (True, False):
        board = Board(_use_c=use_c)
        orig_fen = board.to_fen()
        orig_hash = board.zobrist_hash

        recorded: list[SearchInfo] = []
        searcher = Searcher(hash_size_mb=4, _use_c=use_c)
        info = searcher.search(board, depth=3, on_info=recorded.append)

        assert len(recorded) == 3
        assert [r.depth for r in recorded] == [1, 2, 3]
        assert info.best_move in board.legal_moves
        assert len(info.pv) >= 1
        assert info.pv[0] == info.best_move
        assert board.to_fen() == orig_fen
        assert board.zobrist_hash == orig_hash

        # Even with tiny node limit (nodes=1), depth 1 finishes and returns a legal move.
        info_tiny = searcher.search(board, limits=SearchLimits(depth=10, nodes=1))
        assert info_tiny.depth == 1
        assert info_tiny.best_move in board.legal_moves


def test_c_and_python_search_parity():
    fens = [
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]
    for fen in fens:
        b_c = Board(fen, _use_c=True)
        b_py = Board(fen, _use_c=False)
        s_c = Searcher(hash_size_mb=8, _use_c=True)
        s_py = Searcher(hash_size_mb=8, _use_c=False)

        res_c = s_c.search(b_c, depth=3)
        res_py = s_py.search(b_py, depth=3)

        assert res_c.score == res_py.score
        assert res_c.best_move == res_py.best_move
        assert res_c.pv == res_py.pv


def test_startpos_reaches_depth_8_under_1s():
    board = Board(_use_c=True)
    if not board._use_c:
        pytest.skip("C backend required for 1-second depth-8 benchmark")
    searcher = Searcher(hash_size_mb=64, _use_c=True)
    info = searcher.search(board, movetime_ms=1000)
    assert info.depth >= 10
    assert info.best_move in board.legal_moves


def test_null_move_make_unmake_c_and_python_parity():
    fen = "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3"
    for use_c in (True, False):
        board = Board(fen, _use_c=use_c)
        orig_fen = board.to_fen()
        orig_hash = board.zobrist_hash
        assert board.has_non_pawn_material() is True

        board.push_null()
        assert int(board.turn) == 1
        assert board.ep_square == 64
        assert board.zobrist_hash != orig_hash

        board.pop_null()
        assert board.to_fen() == orig_fen
        assert board.zobrist_hash == orig_hash


def test_static_exchange_evaluation_see_cases():
    # 1. Poisoned pawn: Queen on d4 captures pawn on d7 defended by Queen on d8 -> -800
    b1_c = Board("3q2k1/3p1ppp/8/8/r2Q4/8/5PPP/4R1K1 w - - 0 1", _use_c=True)
    b1_py = Board("3q2k1/3p1ppp/8/8/r2Q4/8/5PPP/4R1K1 w - - 0 1", _use_c=False)
    for b in (b1_c, b1_py):
        m_poison = b.parse_uci("d4d7")
        m_free = b.parse_uci("d4a4")
        assert see(b, m_poison) == -800
        assert see(b, m_free) == 500

    # 2. X-ray battery: Two White rooks (e1, e2) vs one Black rook (e8) on e5 pawn -> +100
    xray_fen = "4r1k1/8/8/4p3/8/8/4R3/4R1K1 w - - 0 1"
    for use_c in (True, False):
        b = Board(xray_fen, _use_c=use_c)
        assert see(b, b.parse_uci("e2e5")) == 100

    # 3. Exact C and Python SEE parity across all legal moves in Kiwipete
    kiwi = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    bc = Board(kiwi, _use_c=True)
    bpy = Board(kiwi, _use_c=False)
    for m in bc.legal_moves:
        assert see(bc, m) == see(bpy, m)


def test_aspiration_windows_and_zugzwang_guard():
    # Depth 5 exercises aspiration windows, LMR, NMP, RFP, FP, killers, and history.
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    bc = Board(fen, _use_c=True)
    bpy = Board(fen, _use_c=False)
    sc = Searcher(hash_size_mb=8, _use_c=True)
    spy = Searcher(hash_size_mb=8, _use_c=False)

    rc = sc.search(bc, depth=5)
    rpy = spy.search(bpy, depth=5)
    assert rc.score == rpy.score
    assert rc.best_move == rpy.best_move

    # Pure King + Pawn endgame has has_non_pawn_material() == False (zugzwang guard active).
    kp_fen = "8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1"
    for use_c in (True, False):
        b_kp = Board(kp_fen, _use_c=use_c)
        assert b_kp.has_non_pawn_material() is False
        s_kp = Searcher(hash_size_mb=4, _use_c=use_c)
        res_kp = s_kp.search(b_kp, depth=5)
        assert res_kp.best_move in b_kp.legal_moves


def test_time_manager_v2_scale_soft_time():
    assert scale_soft_time(100, 300, best_move_changes=0, score_drop_cp=0) == 100
    assert scale_soft_time(100, 300, best_move_changes=1, score_drop_cp=0) == 135
    assert scale_soft_time(100, 300, best_move_changes=2, score_drop_cp=40) == 205
    assert scale_soft_time(100, 150, best_move_changes=3, score_drop_cp=100) == 150


