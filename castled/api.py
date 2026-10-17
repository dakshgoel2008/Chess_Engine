"""High-level Python API for CastledEngine (Engine, AnalysisResult, MultiPV, streaming, context manager)."""

from dataclasses import dataclass, field
import time as _time
from typing import Callable, Iterator, List, Optional

from .book.polyglot import DEFAULT_BOOK_PATH, PolyglotBook
from .core.board import Board
from .core.move import Move
from .search.search import MAX_PLY, SearchInfo, Searcher


@dataclass
class AnalysisResult:
    """Structured analysis result supporting both single-line attributes and MultiPV indexing."""

    lines: List[SearchInfo] = field(default_factory=list)

    @property
    def best_move(self) -> Move:
        return self.lines[0].best_move if self.lines else Move.NULL

    @property
    def score(self) -> int:
        return self.lines[0].score if self.lines else 0

    @property
    def depth(self) -> int:
        return self.lines[0].depth if self.lines else 0

    @property
    def nodes(self) -> int:
        return self.lines[0].nodes if self.lines else 0

    @property
    def nps(self) -> int:
        return self.lines[0].nps if self.lines else 0

    @property
    def time_ms(self) -> int:
        return self.lines[0].time_ms if self.lines else 0

    @property
    def pv(self) -> List[Move]:
        return self.lines[0].pv if self.lines else []

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> SearchInfo:
        return self.lines[idx]

    def __iter__(self) -> Iterator[SearchInfo]:
        return iter(self.lines)


class Engine:
    """High-level chess engine interface with NNUE, Polyglot book, MultiPV, and context manager support."""

    def __init__(
        self,
        hash_mb: int = 64,
        threads: int = 1,
        nnue: bool = True,
        eval_file: str = "",
        book_path: Optional[str] = None,
        use_book: bool = False,
        *,
        _use_c: Optional[bool] = None,
    ):
        self.hash_mb = max(1, min(1024, int(hash_mb)))
        self.threads = max(1, min(64, int(threads)))
        self.nnue = bool(nnue)
        self.searcher = Searcher(
            hash_size_mb=self.hash_mb,
            use_nnue=self.nnue,
            eval_file=eval_file,
            _use_c=_use_c,
        )
        self.use_book = bool(use_book or book_path is not None)
        self.book: Optional[PolyglotBook] = None
        if book_path is not None:
            self.book = PolyglotBook(book_path)
        elif self.use_book:
            try:
                self.book = PolyglotBook(DEFAULT_BOOK_PATH)
            except ValueError:
                self.book = None
        self._closed = False

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # Clear transposition table and stop any active search.
    def clear(self) -> None:
        self.searcher.clear()

    # Stop any ongoing search and release engine state.
    def close(self) -> None:
        if not self._closed:
            self.searcher.stop()
            self.searcher.clear()
            self._closed = True

    # Return the best Move for `board` given `time` (seconds), `depth`, or `nodes`.
    def best_move(
        self,
        board: Board,
        time: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        use_book: Optional[bool] = None,
    ) -> Move:
        should_probe_book = use_book if use_book is not None else self.use_book
        if should_probe_book and self.book is not None:
            book_move = self.book.probe(board)
            if book_move is not None:
                return book_move

        movetime_ms = max(1, int(round(time * 1000.0))) if (time is not None and time > 0) else None
        if movetime_ms is None and depth is None and nodes is None:
            depth = 6

        info = self.searcher.search(
            board,
            depth=depth,
            movetime_ms=movetime_ms,
            nodes=nodes,
            use_nnue=self.nnue,
        )
        return info.best_move

    # Analyse `board` up to `depth`, `time` (seconds), or `nodes` and return `AnalysisResult` with `multipv` lines.
    def analyse(
        self,
        board: Board,
        depth: Optional[int] = None,
        time: Optional[float] = None,
        nodes: Optional[int] = None,
        multipv: int = 1,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> AnalysisResult:
        movetime_ms = max(1, int(round(time * 1000.0))) if (time is not None and time > 0) else None
        if movetime_ms is None and depth is None and nodes is None:
            depth = 6

        lines = self.searcher.search_multipv(
            board,
            multipv=max(1, int(multipv)),
            depth=depth,
            movetime_ms=movetime_ms,
            nodes=nodes,
            use_nnue=self.nnue,
            on_info=on_info,
        )
        return AnalysisResult(lines=lines)

    # Stream `SearchInfo` updates as an iterator across iterative deepening depths.
    def stream_analyse(
        self,
        board: Board,
        depth: Optional[int] = None,
        time: Optional[float] = None,
        nodes: Optional[int] = None,
    ) -> Iterator[SearchInfo]:
        collected: List[SearchInfo] = []
        movetime_ms = max(1, int(round(time * 1000.0))) if (time is not None and time > 0) else None
        target_depth = depth if depth is not None else (MAX_PLY - 1 if movetime_ms or nodes else 6)

        start_ms = _time.monotonic() * 1000.0
        for d in range(1, max(1, min(MAX_PLY - 1, target_depth)) + 1):
            remaining_ms: Optional[int] = None
            if movetime_ms is not None:
                elapsed = int(_time.monotonic() * 1000.0 - start_ms)
                if elapsed >= movetime_ms and d > 1:
                    break
                remaining_ms = max(1, movetime_ms - elapsed)

            info = self.searcher.search(
                board,
                depth=d,
                movetime_ms=remaining_ms,
                nodes=nodes,
                use_nnue=self.nnue,
            )
            collected.append(info)
            yield info

            if nodes is not None and info.nodes >= nodes:
                break
            if movetime_ms is not None and (int(_time.monotonic() * 1000.0 - start_ms) >= movetime_ms):
                break

