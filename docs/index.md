# CastledEngine (`castled`) Quick Start

**CastledEngine** (`castled`) is a C99 + Python (CFFI) bitboard chess engine library with incremental `int16` NNUE evaluation, Principal Variation Search (PVS), Polyglot opening books, a full UCI protocol server, and an interactive Pygame GUI with Game Review.

## Installation

```bash
pip install -e ".[dev,gui]"
```

## 30-Second Python Example

```python
from castled import Board, Engine

board = Board()

with Engine(hash_mb=64, nnue=True, use_book=True) as engine:
    # Get the best move in 0.5 seconds
    move = engine.best_move(board, time=0.5)
    print("Best move:", move.to_uci())
    board.push(move)

    # MultiPV analysis (top 3 candidate lines)
    analysis = engine.analyse(board, depth=6, multipv=3)
    for line in analysis:
        pv_str = " ".join(m.to_uci() for m in line.pv)
        print(f"#{line.multipv} score={line.score} cp | pv={pv_str}")
```

## CLI Commands

```bash
# Launch the 1200x768 Pygame GUI with live eval bar and Game Review
castled play

# Start the UCI protocol loop (for CuteChess, Arena, En Croissant, or Lichess BOT)
castled uci

# Run Perft move generation verification
castled perft --depth 5

# Run fixed-depth search benchmark across standard tactical positions
castled bench --depth 7
```

