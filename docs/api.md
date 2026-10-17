# Python API Reference

## `castled.Engine`

High-level engine interface supporting context management, single-move search, MultiPV analysis, and streaming iterative deepening.

```python
from castled import Board, Engine

with Engine(hash_mb=64, threads=1, nnue=True, use_book=True) as engine:
    board = Board()
    move = engine.best_move(board, time=1.0)
```

### Constructor Parameters
- `hash_mb: int = 64` - Transposition table size in megabytes (`1..1024`).
- `threads: int = 1` - Number of search threads (`1..64`).
- `nnue: bool = True` - Enable incremental NNUE evaluation (`True`) or tapered classical evaluation (`False`).
- `eval_file: str = ""` - Optional path to a custom `.nnue` weight file.
- `book_path: Optional[str] = None` - Optional path to a Polyglot `.bin` opening book.
- `use_book: bool = False` - Probe the opening book before running tree search.

### Methods
- `best_move(board, time=None, depth=None, nodes=None, use_book=None) -> Move`
- `analyse(board, depth=None, time=None, nodes=None, multipv=1, on_info=None) -> AnalysisResult`
- `stream_analyse(board, depth=None, time=None, nodes=None) -> Iterator[SearchInfo]`
- `clear() -> None`
- `close() -> None`

---

## `castled.Board`

64-bit bitboard representation with incremental Zobrist hashing, SAN/LAN/FEN/PGN serialization, and legal move generation.

```python
from castled import Board

board = Board()
move = board.parse_san("Nf3")
board.push(move)
print(board.fen)
board.pop()
```

---

## `castled.PolyglotBook`

Reader, writer, and prober for standard 16-byte big-endian `.bin` Polyglot opening books.

```python
from castled import Board, PolyglotBook, DEFAULT_BOOK_PATH

book = PolyglotBook(DEFAULT_BOOK_PATH)
move = book.probe(Board(), weighted=True)
```

