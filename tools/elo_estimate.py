"""Estimate CastledEngine's Elo by playing it against Stockfish at fixed, calibrated strength levels.

Stockfish's `UCI_LimitStrength` / `UCI_Elo` options make it play at a chosen rating (roughly
CCRL-calibrated, 1320..3190). Playing a gauntlet against several levels and fitting one rating to all
results gives an offline strength estimate. It is a rough guide: ratings depend on time control and
pool, and are not directly comparable with Lichess ratings.

Example:
    sudo apt install stockfish
    python tools/elo_estimate.py --levels 1500 1800 2100 --games 20 --movetime-ms 100
"""

import argparse
import datetime
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from castled.core.board import Board  # noqa: E402
from castled.core.pgn import PGNGame  # noqa: E402
from castled.core.types import Color  # noqa: E402
from castled.search.search import Searcher  # noqa: E402
from tools.sprt import load_openings  # noqa: E402


class UciEngine:
    """Minimal synchronous UCI client for an external engine process."""

    def __init__(self, cmd: Sequence[str]):
        self.proc = subprocess.Popen(
            list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self.name = "UCI engine"
        self.send("uci")
        for line in self._read_until("uciok"):
            if line.startswith("id name "):
                self.name = line[len("id name "):].strip()

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix: str) -> List[str]:
        lines: List[str] = []
        while True:
            line = self._wait_for("")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def _wait_for(self, prefix: str) -> str:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("external engine exited unexpectedly")
            if line.startswith(prefix):
                return line.strip()

    def set_elo(self, elo: int) -> None:
        self.send("setoption name UCI_LimitStrength value true")
        self.send(f"setoption name UCI_Elo value {elo}")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self._wait_for("readyok")

    def best_move(self, fen: str, moves: List[str], movetime_ms: int) -> str:
        pos = f"position fen {fen}" + (" moves " + " ".join(moves) if moves else "")
        self.send(pos)
        self.send(f"go movetime {movetime_ms}")
        return self._wait_for("bestmove").split()[1]

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


@dataclass
class LevelResult:
    elo: int
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0


# Expected score of a player rated `r` against an opponent rated `opp`.
def expected_score(r: float, opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opp - r) / 400.0))


# Maximum-likelihood rating fitted to all level results, with an approximate 95% interval.
def fit_rating(results: Sequence[LevelResult]) -> Tuple[float, float]:
    def log_lik(r: float) -> float:
        total = 0.0
        for res in results:
            e = min(max(expected_score(r, res.elo), 1e-9), 1 - 1e-9)
            pts = res.wins + 0.5 * res.draws
            total += pts * math.log(e) + (res.games - pts) * math.log(1 - e)
        return total

    best = max(range(0, 4001), key=log_lik)
    # Fisher information of the logistic model gives the standard error.
    k = math.log(10) / 400.0
    info = sum(res.games * k * k * expected_score(best, res.elo) * (1 - expected_score(best, res.elo)) for res in results)
    se = 1.0 / math.sqrt(info) if info > 0 else float("inf")
    return float(best), 1.96 * se


# Play one game; returns (score from CastledEngine's view: 1.0 / 0.5 / 0.0, final board, termination).
def play_game(
    fen: str, castled_white: bool, searcher: Searcher, opponent: UciEngine, movetime_ms: int, max_plies: int
) -> Tuple[float, Board, str]:
    board = Board(fen)
    moves: List[str] = []
    searcher.clear()
    opponent.new_game()
    for _ in range(max_plies):
        if board.is_game_over():
            break
        castled_to_move = (board.turn == Color.WHITE) == castled_white
        if castled_to_move:
            uci = searcher.search(board, movetime_ms=movetime_ms).best_move.to_uci()
        else:
            uci = opponent.best_move(fen, moves, movetime_ms)
        board.push(board.parse_uci(uci))
        moves.append(uci)

    outcome = board.outcome()
    if outcome is None:
        return 0.5, board, f"adjudicated draw after {max_plies} plies"
    if outcome.winner is None:
        return 0.5, board, outcome.termination
    castled_color = Color.WHITE if castled_white else Color.BLACK
    return (1.0 if outcome.winner == castled_color else 0.0), board, outcome.termination


# Format one finished game as PGN, including the opening FEN so it can be replayed.
def game_to_pgn(
    board: Board, start_fen: str, score: float, castled_white: bool, opponent_name: str, opponent_elo: int,
    movetime_ms: int, round_no: str, termination: str,
) -> str:
    result = {1.0: "1-0", 0.0: "0-1"}.get(score, "1/2-1/2") if castled_white else {1.0: "0-1", 0.0: "1-0"}.get(score, "1/2-1/2")
    castled_name = "CastledEngine"
    sf_name = f"{opponent_name} (UCI_Elo {opponent_elo})"
    headers = {
        "Event": "CastledEngine Elo gauntlet",
        "Site": "local",
        "Date": datetime.date.today().strftime("%Y.%m.%d"),
        "Round": round_no,
        "White": castled_name if castled_white else sf_name,
        "Black": sf_name if castled_white else castled_name,
        "Result": result,
        "WhiteElo" if not castled_white else "BlackElo": str(opponent_elo),
        "TimeControl": f"movetime {movetime_ms}ms",
        "Termination": termination,
        "SetUp": "1",
        "FEN": start_fen,
    }
    return PGNGame(headers=headers, moves=list(board._move_stack), result=result, initial_fen=start_fen).to_pgn()


def run_gauntlet(
    opponent_cmd: Sequence[str],
    levels: Sequence[int],
    games_per_level: int,
    movetime_ms: int,
    openings: Sequence[str],
    use_nnue: bool = False,
    max_plies: int = 300,
    verbose: bool = True,
    pgn_path: Optional[str] = None,
) -> List[LevelResult]:
    searcher = Searcher(hash_size_mb=64, use_nnue=use_nnue)
    opponent = UciEngine(opponent_cmd)
    results: List[LevelResult] = []
    pgn_file = open(pgn_path, "w", encoding="utf-8") if pgn_path else None
    try:
        for elo in levels:
            opponent.set_elo(elo)
            res = LevelResult(elo=elo)
            for g in range(games_per_level):
                fen = openings[(g // 2) % len(openings)]
                castled_white = g % 2 == 0
                score, board, termination = play_game(fen, castled_white, searcher, opponent, movetime_ms, max_plies)
                if pgn_file is not None:
                    pgn_file.write(game_to_pgn(
                        board, fen, score, castled_white, opponent.name, elo, movetime_ms, f"{elo}.{g + 1}", termination
                    ) + "\n\n")
                    pgn_file.flush()
                if score == 1.0:
                    res.wins += 1
                elif score == 0.5:
                    res.draws += 1
                else:
                    res.losses += 1
                if verbose:
                    print(f"  vs {elo}: game {g + 1}/{games_per_level}  +{res.wins} ={res.draws} -{res.losses}", flush=True)
            results.append(res)
    finally:
        opponent.close()
        if pgn_file is not None:
            pgn_file.close()
    return results, opponent.name


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def write_report(args, sf_name: str, table: List[str], results: Sequence[LevelResult], rating: float, margin: float) -> None:
    games = sum(r.games for r in results)
    cmd = "python tools/elo_estimate.py " + " ".join(
        [f"--levels {' '.join(str(r.elo) for r in results)}", f"--games {args.games}", f"--movetime-ms {args.movetime_ms}"]
        + (["--nnue"] if args.nnue else []) + ([f"--pgn {args.pgn}"] if args.pgn else [])
        + [f"--report {args.report}"]
    )
    lines = [
        "# CastledEngine Elo gauntlet vs Stockfish",
        "",
        f"**Estimated Elo: {rating:.0f} ± {margin:.0f}** (95% interval, maximum-likelihood fit over {games} games), "
        "on Stockfish's `UCI_Elo` scale.",
        "",
        "## Setup",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- CastledEngine commit: `{_git_commit()}`, evaluation: {'NNUE' if args.nnue else 'classical'}",
        f"- Opponent: {sf_name} with `UCI_LimitStrength=true` and `UCI_Elo` set per level",
        f"- Time: {args.movetime_ms} ms per move for both engines, single thread",
        f"- Openings: `{args.openings}`, each played with both colors; games over 300 plies adjudicated as draws",
        f"- Hardware: {_cpu_name()}, {platform.system()}",
        "",
        "## Results",
        "",
        *table,
        "",
        "## Reproduce",
        "",
        "```bash",
        cmd,
        "```",
        "",
        "## Caveats",
        "",
        "- Stockfish calibrates `UCI_Elo` at longer time controls than used here, so the scale is approximate.",
        "- This is not a FIDE, CCRL or Lichess rating, and is not directly comparable with them.",
    ]
    if args.pgn:
        lines.insert(lines.index("## Reproduce"), f"All games: [`{os.path.basename(args.pgn)}`]({os.path.basename(args.pgn)})\n")
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate CastledEngine Elo against strength-limited Stockfish.")
    parser.add_argument("--stockfish", default="stockfish", help="Path to the Stockfish binary.")
    parser.add_argument("--levels", type=int, nargs="+", default=[1500, 1800, 2100], help="Stockfish UCI_Elo levels.")
    parser.add_argument("--games", type=int, default=20, help="Games per level (colors alternate).")
    parser.add_argument("--movetime-ms", type=int, default=100, help="Thinking time per move for both engines.")
    parser.add_argument("--openings", default=os.path.join("tools", "openings.epd"))
    parser.add_argument("--nnue", action="store_true", help="Use CastledEngine's NNUE instead of the classical eval.")
    parser.add_argument("--pgn", default=None, help="Save every game to this PGN file.")
    parser.add_argument("--report", default=None, help="Write a Markdown report (setup + results) to this file.")
    args = parser.parse_args(argv)

    sf = shutil.which(args.stockfish) or (args.stockfish if os.path.isfile(args.stockfish) else None)
    if sf is None:
        print("Stockfish not found. Install it (e.g. `sudo apt install stockfish`, `brew install stockfish`) "
              "or pass --stockfish /path/to/stockfish.", file=sys.stderr)
        return 1

    levels = [min(max(lv, 1320), 3190) for lv in args.levels]  # Stockfish's supported UCI_Elo range
    print(f"CastledEngine ({'NNUE' if args.nnue else 'classical'}) vs Stockfish levels {levels}, "
          f"{args.games} games each, {args.movetime_ms} ms/move")
    results, sf_name = run_gauntlet(
        [sf], levels, args.games, args.movetime_ms, load_openings(args.openings), args.nnue, pgn_path=args.pgn
    )

    table = ["| Stockfish UCI_Elo | Won | Drawn | Lost | CastledEngine score |", "|---|---|---|---|---|"]
    for r in results:
        table.append(f"| {r.elo} | {r.wins} | {r.draws} | {r.losses} | {r.score:.0%} |")
    rating, margin = fit_rating(results)
    print("\n" + "\n".join(table))
    print(f"\nEstimated CastledEngine Elo: {rating:.0f} ± {margin:.0f} (95%)")
    if args.report:
        write_report(args, sf_name, table, results, rating, margin)
        print(f"Report written to {args.report}")
    if all(r.score >= 0.9 for r in results):
        print("It beat every level; add higher --levels for a tighter estimate.")
    elif all(r.score <= 0.1 for r in results):
        print("It lost to every level; add lower --levels for a tighter estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
