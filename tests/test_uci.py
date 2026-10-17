"""Comprehensive tests for UCI protocol, engine options, time management, and CLI."""

import io
import subprocess
import sys
import time

from castled.core.board import Board
from castled.core.types import Color
from castled.protocol.options import EngineOptions
from castled.protocol.uci import UCIEngine, format_uci_score
from castled.search.time_manager import allocate_time
from castled.search.transposition import SCORE_MATE


def test_engine_options_parsing_and_clamping():
    opts = EngineOptions()
    lines = opts.format_uci_options()
    assert len(lines) == 6
    assert any("option name Hash type spin" in ln for ln in lines)
    assert any("option name Threads type spin" in ln for ln in lines)
    assert any("option name MoveOverhead type spin" in ln for ln in lines)
    assert any("option name UseNNUE type check" in ln for ln in lines)
    assert any("option name EvalFile type string" in ln for ln in lines)
    assert any("option name BookFile type string" in ln for ln in lines)

    opts.set_option("hash", "128")
    assert opts.hash_size_mb == 128
    opts.set_option("Hash", "99999")
    assert opts.hash_size_mb == 1024
    opts.set_option("Hash", "-5")
    assert opts.hash_size_mb == 1

    opts.set_option("Threads", "8")
    assert opts.threads == 8
    opts.set_option("MoveOverhead", "50")
    assert opts.move_overhead_ms == 50
    opts.set_option("UseNNUE", "true")
    assert opts.use_nnue is True
    opts.set_option("UseNNUE", "false")
    assert opts.use_nnue is False

    opts.set_option("EvalFile", "weights/custom.nnue")
    assert opts.eval_file == "weights/custom.nnue"
    opts.set_option("EvalFile", "<empty>")
    assert opts.eval_file == ""
    opts.set_option("BookFile", "book/opening.bin")
    assert opts.book_file == "book/opening.bin"


def test_time_manager_allocations_and_edge_cases():
    # Single legal move replies instantly under tournament time control.
    inst = allocate_time(wtime_ms=10000, btime_ms=10000, turn=Color.WHITE, legal_move_count=1)
    assert inst.instant is True
    assert inst.soft_ms == 1
    assert inst.hard_ms == 1

    # Standard 10s + 0.1s time control never allocates more than half of remaining time.
    alloc_w = allocate_time(
        wtime_ms=10000, btime_ms=10000, winc_ms=100, binc_ms=100,
        turn=Color.WHITE, move_overhead_ms=10, legal_move_count=20,
    )
    assert alloc_w.instant is False
    assert 100 <= alloc_w.soft_ms <= alloc_w.hard_ms
    assert alloc_w.hard_ms <= 5000

    # Black turn uses btime and binc.
    alloc_b = allocate_time(
        wtime_ms=50000, btime_ms=2000, winc_ms=1000, binc_ms=50,
        turn=Color.BLACK, move_overhead_ms=10, legal_move_count=25,
    )
    assert alloc_b.hard_ms <= 1000
    assert alloc_b.soft_ms <= alloc_b.hard_ms

    # Explicit movetime subtracts overhead safely even when movetime is tiny.
    fixed = allocate_time(movetime_ms=500, move_overhead_ms=10, legal_move_count=20)
    assert fixed.soft_ms == 490
    assert fixed.hard_ms == 490

    tiny = allocate_time(movetime_ms=5, move_overhead_ms=10, legal_move_count=20)
    assert tiny.soft_ms == 5
    assert tiny.hard_ms == 5


def test_format_uci_score():
    assert format_uci_score(0) == "cp 0"
    assert format_uci_score(135) == "cp 135"
    assert format_uci_score(-42) == "cp -42"
    assert format_uci_score(SCORE_MATE - 1) == "mate 1"
    assert format_uci_score(SCORE_MATE - 3) == "mate 2"
    assert format_uci_score(SCORE_MATE - 5) == "mate 3"
    assert format_uci_score(-SCORE_MATE + 2) == "mate -1"
    assert format_uci_score(-SCORE_MATE + 4) == "mate -2"


def test_uci_engine_commands_and_mate_in_1():
    for use_c in (True, False):
        out = io.StringIO()
        eng = UCIEngine(out_stream=out, _use_c=use_c)

        eng.handle_command("uci")
        eng.handle_command("isready")
        eng.handle_command("setoption name Hash value 8")
        eng.handle_command("ucinewgame")
        eng.handle_command("position startpos moves e2e4 e7e5 g1f3")
        assert eng.board.turn == Color.BLACK

        # Mate in 1 position: Re8#
        eng.handle_command("position fen 6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        eng.handle_command("go depth 2")
        eng.handle_command("quit")

        lines = [ln.strip() for ln in out.getvalue().splitlines() if ln.strip()]
        assert "uciok" in lines
        assert "readyok" in lines
        assert any("score mate 1" in ln for ln in lines)
        assert "bestmove e1e8" in lines


def test_uci_infinite_search_and_immediate_stop():
    out = io.StringIO()
    eng = UCIEngine(out_stream=out, _use_c=True)
    eng.handle_command("position startpos")
    eng.handle_command("go infinite")

    # While search runs on worker thread, 'isready' is answered immediately.
    time.sleep(0.03)
    eng.handle_command("isready")
    assert "readyok" in out.getvalue()

    t0 = time.perf_counter()
    eng.handle_command("stop")
    stop_elapsed = time.perf_counter() - t0
    assert stop_elapsed < 0.25

    lines = [ln.strip() for ln in out.getvalue().splitlines() if ln.strip()]
    bestmove_lines = [ln for ln in lines if ln.startswith("bestmove ")]
    assert len(bestmove_lines) == 1
    move_uci = bestmove_lines[0].split()[1]
    assert eng.board.parse_uci(move_uci) in eng.board.legal_moves


def test_uci_single_legal_move_and_checkmated_position():
    out = io.StringIO()
    eng = UCIEngine(out_stream=out)

    # Single legal move under clock replies instantly.
    eng.handle_command("position fen 6k1/5ppp/8/8/8/8/5PPP/5rK1 w - - 0 1")
    t0 = time.perf_counter()
    eng.handle_command("go wtime 10000 btime 10000 winc 100 binc 100")
    eng.stop_search()
    assert (time.perf_counter() - t0) < 0.1
    assert "bestmove g1f1" in out.getvalue()

    # Already checkmated position outputs bestmove 0000.
    out2 = io.StringIO()
    eng2 = UCIEngine(out_stream=out2)
    eng2.handle_command("position fen 6k1/5ppp/8/8/8/8/5PPP/3r2K1 w - - 0 1")
    eng2.handle_command("go depth 2")
    eng2.handle_command("quit")
    assert "bestmove 0000" in out2.getvalue()


def test_subprocess_uci_and_cli_commands():
    # Test piping UCI session into 'python -m castled'.
    uci_input = (
        "uci\n"
        "isready\n"
        "setoption name Hash value 8\n"
        "ucinewgame\n"
        "position startpos moves e2e4 e7e5\n"
        "go depth 3\n"
        "quit\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "castled"],
        input=uci_input,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    stdout = proc.stdout
    assert "id name CastledEngine" in stdout
    assert "uciok" in stdout
    assert "readyok" in stdout
    assert "info depth 3" in stdout
    assert "bestmove " in stdout

    # Verify 'castled perft 3' and 'castled bench 2' CLI subcommands.
    perft_proc = subprocess.run(
        [sys.executable, "-m", "castled", "perft", "3"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "8902 nodes" in perft_proc.stdout

    bench_proc = subprocess.run(
        [sys.executable, "-m", "castled", "bench", "2"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "Nodes searched" in bench_proc.stdout
    assert "Nodes/second" in bench_proc.stdout

