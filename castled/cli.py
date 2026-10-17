"""Command-line interface for CastledEngine."""

import sys
import time
from . import __version__
from .core.board import Board, HAS_CFFI
from .core.types import STARTING_FEN


def run_perft(depth: int, fen: str = STARTING_FEN):
    """Run perft to the given depth and print node count and speed."""
    board = Board(fen)
    backend = "C" if board._use_c else "Python"
    start = time.perf_counter()
    nodes = board.perft(depth)
    elapsed = max(time.perf_counter() - start, 1e-6)
    nps = int(nodes / elapsed)
    print(f"Perft({depth}) [{backend}]: {nodes} nodes in {elapsed:.3f}s ({nps:,} nps)")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "uci":
        print(f"CastledEngine v{__version__} (C backend: {'enabled' if HAS_CFFI else 'disabled'})")
        print("UCI mode will be added in Phase 7.")
        return

    cmd = args[0]
    if cmd == "perft":
        depth = int(args[1]) if len(args) >= 2 else 5
        fen = " ".join(args[2:]) if len(args) >= 3 else STARTING_FEN
        run_perft(depth, fen)
    elif cmd == "bench":
        run_perft(5, STARTING_FEN)
    else:
        print(f"Unknown command: {cmd}. Usage: castled [uci | perft <depth> [fen] | bench]")
        sys.exit(1)


if __name__ == "__main__":
    main()

