"""SPRT (Sequential Probability Ratio Test) runner and cutechess-cli wrapper."""

import argparse
from dataclasses import dataclass
import math
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from castled.core.board import Board
from castled.core.move import Move
from castled.core.types import Color, STARTING_FEN
from castled.search.search import Searcher, SearchLimits

DEFAULT_OPENINGS_PATH = os.path.join(os.path.dirname(__file__), "openings.epd")


@dataclass
class SPRTStats:
    wins: int
    losses: int
    draws: int
    elo: float
    elo_error: float
    llr: float
    lower_bound: float
    upper_bound: float
    verdict: str

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws


# Convert expected score in (0, 1) to Elo difference.
def elo_from_score(score: float) -> float:
    clamped = max(1e-4, min(1.0 - 1e-4, score))
    return -400.0 * math.log10(1.0 / clamped - 1.0)


# Convert Elo difference to expected game score in (0, 1).
def score_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


# Compute Elo difference, 95% error bar, Log-Likelihood Ratio, and SPRT verdict.
def compute_sprt_stats(
    wins: int,
    losses: int,
    draws: int,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SPRTStats:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    total = wins + losses + draws

    if total == 0:
        return SPRTStats(
            wins=0,
            losses=0,
            draws=0,
            elo=0.0,
            elo_error=0.0,
            llr=0.0,
            lower_bound=lower,
            upper_bound=upper,
            verdict="INCONCLUSIVE",
        )

    score = (wins + 0.5 * draws) / total
    elo = elo_from_score(score)

    var_num = (
        wins * (1.0 - score) ** 2
        + draws * (0.5 - score) ** 2
        + losses * (0.0 - score) ** 2
    )
    sample_var = var_num / total
    if sample_var < 1e-6:
        sample_var = 0.25 / max(1, total)

    if total > 1:
        std_err = math.sqrt(var_num / (total * (total - 1)))
        s_clamped = max(0.01, min(0.99, score))
        deriv = 400.0 / (math.log(10.0) * s_clamped * (1.0 - s_clamped))
        elo_error = 1.96 * deriv * std_err
    else:
        elo_error = 0.0

    mu0 = score_from_elo(elo0)
    mu1 = score_from_elo(elo1)
    llr = total * ((mu1 - mu0) / (2.0 * sample_var)) * (2.0 * score - (mu0 + mu1))

    if llr >= upper:
        verdict = "H1_ACCEPTED"
    elif llr <= lower:
        verdict = "H0_ACCEPTED"
    else:
        verdict = "INCONCLUSIVE"

    return SPRTStats(
        wins=wins,
        losses=losses,
        draws=draws,
        elo=elo,
        elo_error=elo_error,
        llr=llr,
        lower_bound=lower,
        upper_bound=upper,
        verdict=verdict,
    )


# Load non-empty FEN lines from an EPD/FEN opening suite file.
def load_openings(path: str = DEFAULT_OPENINGS_PATH) -> List[str]:
    if not os.path.exists(path):
        return [STARTING_FEN]
    openings: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                openings.append(cleaned)
    return openings if openings else [STARTING_FEN]


# Build a cutechess-cli command line for SPRT testing between two UCI engines.
def build_cutechess_command(
    new_cmd: str,
    base_cmd: str,
    tc: str = "10+0.1",
    games: int = 200,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    openings_path: str = DEFAULT_OPENINGS_PATH,
    concurrency: int = 1,
    cutechess_bin: str = "cutechess-cli",
) -> List[str]:
    return [
        cutechess_bin,
        "-engine", "name=New", f"cmd={new_cmd}", "proto=uci",
        "-engine", "name=Base", f"cmd={base_cmd}", "proto=uci",
        "-each", f"tc={tc}",
        "-rounds", str(max(1, games // 2)),
        "-games", "2",
        "-repeat",
        "-openings", f"file={openings_path}", "format=epd", "order=random",
        "-sprt", f"elo0={elo0}", f"elo1={elo1}", f"alpha={alpha}", f"beta={beta}",
        "-concurrency", str(max(1, concurrency)),
    ]


# Parse W/L/D counts from cutechess-cli stdout and return computed SPRTStats.
def parse_cutechess_output(
    output: str,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SPRTStats:
    matches = re.findall(r"Score of New vs Base:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", output)
    if not matches:
        return compute_sprt_stats(0, 0, 0, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
    w_str, l_str, d_str = matches[-1]
    return compute_sprt_stats(
        int(w_str), int(l_str), int(d_str),
        elo0=elo0, elo1=elo1, alpha=alpha, beta=beta,
    )


# Play a single game from start_fen between two Searchers with independent limits.
def play_single_game(
    fen: str,
    white_searcher: Searcher,
    black_searcher: Searcher,
    white_limits: SearchLimits,
    black_limits: SearchLimits,
    max_plies: int = 200,
) -> Optional[Color]:
    board = Board(fen)
    white_searcher.clear()
    black_searcher.clear()

    for _ in range(max_plies):
        outcome = board.outcome()
        if outcome is not None:
            return outcome.winner

        if board.turn == Color.WHITE:
            info = white_searcher.search(board, limits=white_limits)
        else:
            info = black_searcher.search(board, limits=black_limits)

        if info.best_move == Move.NULL or info.best_move not in board.legal_moves:
            return board.turn.flip()
        board.push(info.best_move)

    return None


# Run an internal paired-opening SPRT match when cutechess-cli is not used.
def run_internal_sprt(
    games: int = 10,
    new_depth: int = 3,
    base_depth: int = 3,
    movetime_ms: int = 0,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    openings_path: str = DEFAULT_OPENINGS_PATH,
    max_plies: int = 200,
) -> SPRTStats:
    openings = load_openings(openings_path)
    new_searcher = Searcher(hash_size_mb=16)
    base_searcher = Searcher(hash_size_mb=16)

    new_limits = SearchLimits(depth=new_depth, movetime_ms=movetime_ms)
    base_limits = SearchLimits(depth=base_depth, movetime_ms=movetime_ms)

    wins = 0
    losses = 0
    draws = 0

    for g in range(games):
        fen = openings[(g // 2) % len(openings)]
        new_is_white = (g % 2 == 0)

        if new_is_white:
            winner = play_single_game(fen, new_searcher, base_searcher, new_limits, base_limits, max_plies=max_plies)
            if winner == Color.WHITE:
                wins += 1
            elif winner == Color.BLACK:
                losses += 1
            else:
                draws += 1
        else:
            winner = play_single_game(fen, base_searcher, new_searcher, base_limits, new_limits, max_plies=max_plies)
            if winner == Color.BLACK:
                wins += 1
            elif winner == Color.WHITE:
                losses += 1
            else:
                draws += 1

        stats = compute_sprt_stats(wins, losses, draws, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
        if stats.verdict != "INCONCLUSIVE":
            return stats

    return compute_sprt_stats(wins, losses, draws, elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)


def format_sprt_report(stats: SPRTStats) -> str:
    return (
        f"Games: {stats.total_games} (+{stats.wins} -{stats.losses} ={stats.draws})\n"
        f"Elo difference: {stats.elo:+.1f} +/- {stats.elo_error:.1f}\n"
        f"LLR: {stats.llr:.2f} [{stats.lower_bound:.2f}, {stats.upper_bound:.2f}]\n"
        f"Verdict: {stats.verdict}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run SPRT Elo testing for CastledEngine.")
    parser.add_argument("--new-cmd", default=f"{sys.executable} -m castled", help="Command for new engine")
    parser.add_argument("--base-cmd", default=f"{sys.executable} -m castled", help="Command for base engine")
    parser.add_argument("--tc", default="10+0.1", help="Time control for cutechess-cli")
    parser.add_argument("--games", type=int, default=10, help="Total games to play")
    parser.add_argument("--new-depth", type=int, default=3, help="Depth for new engine (internal runner)")
    parser.add_argument("--base-depth", type=int, default=3, help="Depth for base engine (internal runner)")
    parser.add_argument("--movetime-ms", type=int, default=0, help="Movetime in ms (internal runner)")
    parser.add_argument("--elo0", type=float, default=0.0, help="SPRT H0 Elo bound")
    parser.add_argument("--elo1", type=float, default=10.0, help="SPRT H1 Elo bound")
    parser.add_argument("--max-plies", type=int, default=200, help="Max plies per game (internal runner)")
    parser.add_argument("--openings", default=DEFAULT_OPENINGS_PATH, help="Path to EPD opening book")
    parser.add_argument("--use-cutechess", action="store_true", help="Use external cutechess-cli if installed")
    args = parser.parse_args()

    if args.use_cutechess and shutil.which("cutechess-cli"):
        cmd = build_cutechess_command(
            new_cmd=args.new_cmd,
            base_cmd=args.base_cmd,
            tc=args.tc,
            games=args.games,
            elo0=args.elo0,
            elo1=args.elo1,
            openings_path=args.openings,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(proc.stdout)
        stats = parse_cutechess_output(proc.stdout, elo0=args.elo0, elo1=args.elo1)
    else:
        stats = run_internal_sprt(
            games=args.games,
            new_depth=args.new_depth,
            base_depth=args.base_depth,
            movetime_ms=args.movetime_ms,
            elo0=args.elo0,
            elo1=args.elo1,
            openings_path=args.openings,
            max_plies=args.max_plies,
        )

    print(format_sprt_report(stats))


if __name__ == "__main__":
    main()
