"""Tests for Phase 11: NNUE inference (C incremental accumulator + NumPy reference)."""

import io
import os
import random
import tempfile
import pytest
import numpy as np

from castled.core.board import Board, HAS_CFFI
from castled.core.types import STARTING_FEN
from castled.eval.nnue_ref import (
    DEFAULT_NNUE_PATH,
    NNUE_HIDDEN_SIZE,
    NNUE_INPUT_SIZE,
    NNUEWeights,
    build_default_weights,
    compute_accumulator_numpy,
    evaluate_nnue,
    evaluate_nnue_numpy,
    feature_idx_black,
    feature_idx_white,
    get_active_weights,
    get_c_accumulator,
    load_nnue_file,
    reset_default_weights,
    save_nnue_file,
    set_active_weights,
)
from castled.protocol.uci import UCIEngine
from castled.search.search import Searcher

TEST_FENS = [
    STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1",
]


@pytest.fixture(autouse=True)
def _restore_default_nnue():
    reset_default_weights()
    yield
    reset_default_weights()


def test_feature_index_bounds_and_perspective_symmetry():
    for pc in range(12):
        for sq in range(64):
            w_idx = feature_idx_white(pc, sq)
            b_idx = feature_idx_black(pc, sq)
            assert 0 <= w_idx < NNUE_INPUT_SIZE
            assert 0 <= b_idx < NNUE_INPUT_SIZE

    # White pawn on e2 (pc=0, sq=12) from White's view equals Black pawn on e7 (pc=6, sq=52) from Black's view.
    assert feature_idx_white(0, 12) == feature_idx_black(6, 52)
    # White king on g1 (pc=5, sq=6) from White's view equals Black king on g8 (pc=11, sq=62) from Black's view.
    assert feature_idx_white(5, 6) == feature_idx_black(11, 62)


def test_c_and_numpy_default_weights_match_across_fens():
    for fen in TEST_FENS:
        b = Board(fen)
        c_eval = evaluate_nnue(b, recompute=False)
        c_recompute = evaluate_nnue(b, recompute=True)
        np_eval = evaluate_nnue_numpy(b)
        assert c_eval == c_recompute == np_eval

        acc_w_c, acc_b_c = get_c_accumulator(b)
        acc_w_np, acc_b_np = compute_accumulator_numpy(b)
        np.testing.assert_array_equal(acc_w_c, acc_w_np)
        np.testing.assert_array_equal(acc_b_c, acc_b_np)


def test_c_and_numpy_match_with_random_int16_weights():
    rng = np.random.default_rng(42)
    random_weights = NNUEWeights(
        w1=rng.integers(-64, 65, size=(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE), dtype=np.int16),
        b1=rng.integers(-64, 65, size=(NNUE_HIDDEN_SIZE,), dtype=np.int16),
        w2=rng.integers(-64, 65, size=(NNUE_HIDDEN_SIZE * 2,), dtype=np.int16),
        b2=int(rng.integers(-200, 201)),
    )
    set_active_weights(random_weights)

    loaded = get_active_weights()
    np.testing.assert_array_equal(loaded.w1, random_weights.w1)
    np.testing.assert_array_equal(loaded.b1, random_weights.b1)
    np.testing.assert_array_equal(loaded.w2, random_weights.w2)
    assert loaded.b2 == random_weights.b2

    for fen in TEST_FENS:
        b = Board(fen)
        assert evaluate_nnue(b, recompute=False) == evaluate_nnue_numpy(b)
        assert evaluate_nnue(b, recompute=True) == evaluate_nnue_numpy(b)


def test_incremental_accumulator_matches_scratch_on_special_moves():
    # Position with castling, en passant, capture, and promotion opportunities.
    special_fens = [
        # En passant capture
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e5d6"),
        # White kingside castling
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"),
        # White queenside castling
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"),
        # Black kingside castling
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8"),
        # Quiet promotion and capture promotion
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8n"),
    ]

    rng = np.random.default_rng(99)
    set_active_weights(
        NNUEWeights(
            w1=rng.integers(-50, 51, size=(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE), dtype=np.int16),
            b1=rng.integers(-50, 51, size=(NNUE_HIDDEN_SIZE,), dtype=np.int16),
            w2=rng.integers(-50, 51, size=(NNUE_HIDDEN_SIZE * 2,), dtype=np.int16),
            b2=17,
        )
    )

    for fen, uci_str in special_fens:
        b = Board(fen)
        initial_eval = evaluate_nnue(b)
        assert initial_eval == evaluate_nnue_numpy(b)

        move = b.parse_uci(uci_str)
        b.push(move)

        acc_w_c, acc_b_c = get_c_accumulator(b)
        acc_w_np, acc_b_np = compute_accumulator_numpy(b)
        np.testing.assert_array_equal(acc_w_c, acc_w_np)
        np.testing.assert_array_equal(acc_b_c, acc_b_np)
        assert evaluate_nnue(b, recompute=False) == evaluate_nnue(b, recompute=True) == evaluate_nnue_numpy(b)

        b.pop()
        acc_w_c, acc_b_c = get_c_accumulator(b)
        acc_w_np, acc_b_np = compute_accumulator_numpy(b)
        np.testing.assert_array_equal(acc_w_c, acc_w_np)
        np.testing.assert_array_equal(acc_b_c, acc_b_np)
        assert evaluate_nnue(b, recompute=False) == initial_eval


def test_incremental_accumulator_random_walk_and_unmake():
    rng = random.Random(1337)
    b = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    initial_eval = evaluate_nnue(b)
    history_evals = [initial_eval]

    steps = 80
    for _ in range(steps):
        legal = b.legal_moves
        if not legal:
            break
        m = rng.choice(legal)
        b.push(m)
        inc_score = evaluate_nnue(b, recompute=False)
        rec_score = evaluate_nnue(b, recompute=True)
        np_score = evaluate_nnue_numpy(b)
        assert inc_score == rec_score == np_score
        history_evals.append(inc_score)

    while len(history_evals) > 1:
        b.pop()
        history_evals.pop()
        assert evaluate_nnue(b, recompute=False) == history_evals[-1]
        assert evaluate_nnue(b, recompute=False) == evaluate_nnue_numpy(b)


def test_nnue_file_save_load_and_validation():
    assert os.path.isfile(DEFAULT_NNUE_PATH)
    loaded_default = load_nnue_file(DEFAULT_NNUE_PATH, activate=True)
    active_default = get_active_weights()
    np.testing.assert_array_equal(loaded_default.w1, active_default.w1)
    np.testing.assert_array_equal(loaded_default.b1, active_default.b1)
    np.testing.assert_array_equal(loaded_default.w2, active_default.w2)
    assert loaded_default.b2 == active_default.b2

    built_default = build_default_weights()
    assert built_default.w1.shape == (NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)

    rng = np.random.default_rng(7)
    custom = NNUEWeights(
        w1=rng.integers(-40, 41, size=(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE), dtype=np.int16),
        b1=rng.integers(-40, 41, size=(NNUE_HIDDEN_SIZE,), dtype=np.int16),
        w2=rng.integers(-40, 41, size=(NNUE_HIDDEN_SIZE * 2,), dtype=np.int16),
        b2=-42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "custom.nnue")
        assert save_nnue_file(path, custom)
        rt = load_nnue_file(path, activate=True)
        np.testing.assert_array_equal(rt.w1, custom.w1)
        np.testing.assert_array_equal(rt.b1, custom.b1)
        np.testing.assert_array_equal(rt.w2, custom.w2)
        assert rt.b2 == custom.b2

        b = Board(STARTING_FEN)
        assert evaluate_nnue(b) == evaluate_nnue_numpy(b, custom)

        bad_path = os.path.join(tmpdir, "corrupt.nnue")
        with open(bad_path, "wb") as fp:
            fp.write(b"BAD_HEADER_BYTES_1234")
        with pytest.raises(ValueError):
            load_nnue_file(bad_path)


def test_searcher_and_uci_with_nnue_enabled():
    # Mate in 1 position: White plays Qf7#
    mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    b = Board(mate_fen)
    searcher = Searcher(hash_size_mb=4, use_nnue=True)
    info = searcher.search(b, depth=3)
    assert info.best_move.to_uci() == "h5f7"

    # Also verify pure-Python search with use_nnue=True
    b_py = Board(mate_fen, _use_c=False)
    searcher_py = Searcher(hash_size_mb=4, use_nnue=True, _use_c=False)
    info_py = searcher_py.search(b_py, depth=2)
    assert info_py.best_move.to_uci() == "h5f7"

    # Verify UCI UseNNUE and EvalFile options
    out = io.StringIO()
    uci = UCIEngine(out_stream=out)
    uci.handle_command("setoption name UseNNUE value true")
    uci.handle_command(f"setoption name EvalFile value {DEFAULT_NNUE_PATH}")
    assert uci.searcher.use_nnue is True
    assert uci.searcher.eval_file == DEFAULT_NNUE_PATH
    uci.handle_command(f"position fen {mate_fen}")
    uci.handle_command("go depth 3")
    uci.stop_search()
    assert "bestmove h5f7" in out.getvalue()


@pytest.mark.skipif(not HAS_CFFI, reason="C backend benchmark gate requires CFFI")
def test_nnue_c_inference_speed_gate():
    b_classical = Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        _use_c=True,
    )
    b_nnue = Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        _use_c=True,
    )
    s_classical = Searcher(hash_size_mb=16, use_nnue=False, _use_c=True)
    s_nnue = Searcher(hash_size_mb=16, use_nnue=True, eval_file="", _use_c=True)
    reset_default_weights()

    s_classical.clear()
    info_cls = s_classical.search(b_classical, movetime_ms=200)
    s_nnue.clear()
    info_nnue = s_nnue.search(b_nnue, movetime_ms=200)

    assert info_nnue.nodes > 0
    assert info_nnue.nps >= info_cls.nps * 0.50


