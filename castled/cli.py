"""Command-line interface for CastledEngine (uci, perft, bench)."""

import sys
import time

from .core.board import Board
from .core.types import STARTING_FEN
from .protocol.uci import run_uci
from .search.search import Searcher

BENCH_FENS = (
    STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
)


# Run perft to the given depth and print node count and speed.
def run_perft(depth: int, fen: str = STARTING_FEN):
    board = Board(fen)
    backend = "C" if board._use_c else "Python"
    start = time.perf_counter()
    nodes = board.perft(depth)
    elapsed = max(time.perf_counter() - start, 1e-6)
    nps = int(nodes / elapsed)
    print(f"Perft({depth}) [{backend}]: {nodes} nodes in {elapsed:.3f}s ({nps:,} nps)")


# Run a fixed-depth search benchmark across standard test positions.
def run_bench(depth: int = 6):
    searcher = Searcher(hash_size_mb=32)
    total_nodes = 0
    start = time.perf_counter()

    for idx, fen in enumerate(BENCH_FENS, start=1):
        board = Board(fen)
        searcher.clear()
        info = searcher.search(board, depth=depth)
        total_nodes += info.nodes
        print(
            f"Position {idx}/{len(BENCH_FENS)}: depth {info.depth} "
            f"move {info.best_move.to_uci()} score {info.score} nodes {info.nodes}"
        )

    elapsed = max(time.perf_counter() - start, 1e-6)
    nps = int(total_nodes / elapsed)
    print(f"\n=========================== ")
    print(f"Total time (ms) : {int(elapsed * 1000)}")
    print(f"Nodes searched  : {total_nodes}")
    print(f"Nodes/second    : {nps}")


# Launch the interactive Pygame GUI with optional mode, difficulty, and PGN file.
def run_play(argv: list[str]) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="castled play", description="Launch the CastledEngine Pygame GUI.")
    parser.add_argument("--mode", choices=["white", "black", "pvp", "aivai"], default=None)
    parser.add_argument("--difficulty", choices=["Easy", "Medium", "Hard", "Master"], default="Medium")
    parser.add_argument("--pgn", default=None, help="Optional PGN file to load or save.")
    parsed = parser.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from gui.main import run_gui

    run_gui(mode=parsed.mode, difficulty=parsed.difficulty, pgn_path=parsed.pgn)


def main():
    args = sys.argv[1:]
    if not args or args[0] == "uci":
        run_uci()
        return

    cmd = args[0]
    if cmd == "perft":
        depth = int(args[1]) if len(args) >= 2 else 5
        fen = " ".join(args[2:]) if len(args) >= 3 else STARTING_FEN
        run_perft(depth, fen)
    elif cmd == "bench":
        depth = int(args[1]) if len(args) >= 2 else 6
        run_bench(depth)
    elif cmd == "play":
        run_play(args[1:])
    else:
        print(f"Unknown command: {cmd}. Usage: castled [uci | play | perft <depth> [fen] | bench [depth]]")
        sys.exit(1)


if __name__ == "__main__":
    main()

