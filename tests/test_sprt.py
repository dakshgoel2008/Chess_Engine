"""Tests for SPRT math, opening suite validation, cutechess-cli command generation, and match runner."""

import subprocess
import sys

from castled.core.board import Board
from tools.sprt import (
    DEFAULT_OPENINGS_PATH,
    build_cutechess_command,
    compute_sprt_stats,
    elo_from_score,
    format_sprt_report,
    load_openings,
    parse_cutechess_output,
    run_internal_sprt,
    score_from_elo,
)


def test_openings_epd_all_positions_valid_and_playable():
    openings = load_openings(DEFAULT_OPENINGS_PATH)
    assert len(openings) >= 40

    seen = set()
    for fen in openings:
        board = Board(fen)
        assert not board.is_game_over()
        assert len(board.legal_moves) >= 15
        seen.add(board.zobrist_hash)

    assert len(seen) == len(openings)


def test_elo_and_score_roundtrip():
    assert abs(elo_from_score(0.5)) < 1e-6
    assert abs(score_from_elo(0.0) - 0.5) < 1e-6

    for target_elo in (-150.0, -50.0, 25.0, 100.0, 240.0):
        expected_score = score_from_elo(target_elo)
        recovered_elo = elo_from_score(expected_score)
        assert abs(recovered_elo - target_elo) < 1e-4

    # Extreme 100% and 0% scores are clamped without ZeroDivisionError or math domain error.
    assert elo_from_score(1.0) > 1000.0
    assert elo_from_score(0.0) < -1000.0


def test_sprt_verdicts_and_edge_cases():
    # Zero games returns INCONCLUSIVE safely.
    zero = compute_sprt_stats(0, 0, 0)
    assert zero.total_games == 0
    assert zero.verdict == "INCONCLUSIVE"

    # Strong positive record triggers H1_ACCEPTED.
    h1 = compute_sprt_stats(wins=55, losses=5, draws=20, elo0=0.0, elo1=10.0)
    assert h1.elo > 100.0
    assert h1.llr >= h1.upper_bound
    assert h1.verdict == "H1_ACCEPTED"

    # Strong negative record triggers H0_ACCEPTED.
    h0 = compute_sprt_stats(wins=5, losses=55, draws=20, elo0=0.0, elo1=10.0)
    assert h0.elo < -100.0
    assert h0.llr <= h0.lower_bound
    assert h0.verdict == "H0_ACCEPTED"

    # Balanced small sample remains INCONCLUSIVE.
    inc = compute_sprt_stats(wins=2, losses=2, draws=4, elo0=0.0, elo1=10.0)
    assert inc.verdict == "INCONCLUSIVE"
    report = format_sprt_report(inc)
    assert "Verdict: INCONCLUSIVE" in report


def test_cutechess_command_and_output_parser():
    cmd = build_cutechess_command(
        new_cmd="castled_new",
        base_cmd="castled_base",
        tc="10+0.1",
        games=100,
        elo0=0.0,
        elo1=10.0,
    )
    assert cmd[0] == "cutechess-cli"
    assert "-sprt" in cmd
    assert "tc=10+0.1" in cmd

    sample_out = (
        "Score of New vs Base: 12 - 3 - 5  [0.725] 20\n"
        "Score of New vs Base: 55 - 8 - 17  [0.794] 80\n"
    )
    stats = parse_cutechess_output(sample_out, elo0=0.0, elo1=10.0)
    assert stats.wins == 55
    assert stats.losses == 8
    assert stats.draws == 17
    assert stats.verdict == "H1_ACCEPTED"


def test_internal_sprt_runner_and_cli_execution():
    # Deeper search (depth 2) vs shallow search (depth 1) over 2 games.
    stats = run_internal_sprt(games=2, new_depth=2, base_depth=1, max_plies=16)
    assert stats.total_games == 2

    # Verify CLI invocation of tools/sprt.py.
    proc = subprocess.run(
        [
            sys.executable, "tools/sprt.py",
            "--games", "2",
            "--new-depth", "2",
            "--base-depth", "1",
            "--max-plies", "16",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "Games:" in proc.stdout
    assert "Elo difference:" in proc.stdout
    assert "Verdict:" in proc.stdout
