"""Tests for Phase 13: High-level Engine API, MultiPV analysis, streaming, and Polyglot opening book."""

import io
import os
import tempfile
import pytest

from castled import (
    DEFAULT_BOOK_PATH,
    AnalysisResult,
    Board,
    Engine,
    Move,
    PolyglotBook,
    UCIEngine,
    build_default_book,
)
from castled.book.polyglot import decode_polyglot_move, encode_polyglot_move, polyglot_key
from castled.core.types import STARTING_FEN


def test_polyglot_key_and_move_encoding_roundtrip():
    b1 = Board(STARTING_FEN)
    b2 = Board(STARTING_FEN)
    assert polyglot_key(b1) == polyglot_key(b2)

    # En passant key is only hashed when an enemy pawn can actually capture en passant.
    b_ep_able = Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    b_ep_none = Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    assert polyglot_key(b_ep_able) != polyglot_key(b_ep_none)

    # Test move encoding/decoding across quiet, capture, castling, and promotion moves.
    test_cases = [
        (STARTING_FEN, "e2e4"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8"),
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8n"),
    ]
    for fen, uci_str in test_cases:
        board = Board(fen)
        move = board.parse_uci(uci_str)
        raw = encode_polyglot_move(move)
        decoded = decode_polyglot_move(board, raw)
        assert decoded == move


def test_polyglot_book_save_load_and_probe():
    assert os.path.isfile(DEFAULT_BOOK_PATH)
    book = PolyglotBook(DEFAULT_BOOK_PATH)
    assert len(book) > 20

    start_board = Board(STARTING_FEN)
    entries = book.get_entries(start_board)
    assert len(entries) >= 3
    # Highest weight startpos moves include e2e4 and d2d4
    top_move = book.probe(start_board, weighted=False)
    assert top_move is not None
    assert top_move.to_uci() in ("e2e4", "d2d4", "c2c4")

    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = os.path.join(tmpdir, "test_book.bin")
        custom_book = PolyglotBook()
        m_e4 = start_board.parse_uci("e2e4")
        m_d4 = start_board.parse_uci("d2d4")
        custom_book.add_entry(start_board, m_e4, weight=50)
        custom_book.add_entry(start_board, m_d4, weight=200)
        custom_book.save(custom_path)

        reloaded = PolyglotBook(custom_path)
        assert len(reloaded) == 2
        best = reloaded.probe(start_board, weighted=False)
        assert best == m_d4

        bad_path = os.path.join(tmpdir, "corrupt.bin")
        with open(bad_path, "wb") as fp:
            fp.write(b"12345")
        with pytest.raises(ValueError):
            PolyglotBook(bad_path)


def test_engine_api_context_manager_and_best_move():
    mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    board = Board(mate_fen)

    with Engine(hash_mb=16, threads=1, nnue=True, use_book=False) as engine:
        move = engine.best_move(board, depth=3)
        assert move.to_uci() == "h5f7"

        # Time-based search (in seconds)
        start_board = Board(STARTING_FEN)
        t_move = engine.best_move(start_board, time=0.05)
        assert t_move in start_board.legal_moves

    assert engine._closed is True

    # Test with opening book enabled
    with Engine(hash_mb=8, nnue=True, use_book=True) as book_engine:
        b = Board(STARTING_FEN)
        bm = book_engine.best_move(b, time=0.01)
        assert bm.to_uci() in ("e2e4", "d2d4", "c2c4")


def test_engine_multipv_analysis_and_streaming():
    board = Board(STARTING_FEN)
    with Engine(hash_mb=16, nnue=True) as engine:
        res = engine.analyse(board, depth=3, multipv=3)
        assert isinstance(res, AnalysisResult)
        assert len(res) == 3
        assert res.best_move == res[0].best_move
        assert res.score == res[0].score
        assert res.depth == 3
        assert res.nodes > 0
        assert len(res.pv) >= 1

        # Distinct root moves sorted descending by score
        root_moves = [line.best_move for line in res]
        assert len(set(root_moves)) == 3
        assert res[0].multipv == 1
        assert res[1].multipv == 2
        assert res[2].multipv == 3
        assert res[0].score >= res[1].score >= res[2].score

        # Streaming iterator yields depths 1..3
        streamed = list(engine.stream_analyse(board, depth=3))
        assert [info.depth for info in streamed] == [1, 2, 3]
        assert all(info.best_move != Move.NULL for info in streamed)


def test_uci_bookfile_option_integration():
    out = io.StringIO()
    uci = UCIEngine(out_stream=out)
    uci.handle_command(f"setoption name BookFile value {DEFAULT_BOOK_PATH}")
    assert uci.book is not None
    uci.handle_command("position startpos")
    uci.handle_command("go movetime 50")
    uci.stop_search()
    output = out.getvalue()
    assert any(f"bestmove {m}" in output for m in ("e2e4", "d2d4", "c2c4"))

