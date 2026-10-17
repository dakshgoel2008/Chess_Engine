__version__ = "0.1.0"

from .api import AnalysisResult, Engine
from .book import DEFAULT_BOOK_PATH, BookEntry, PolyglotBook, build_default_book
from .core.types import Color, PieceType, Piece, Square, MoveFlag, CastlingRights
from .core.move import Move
from .core.board import Board, Outcome
from .core.pgn import PGNGame, parse_pgn, dump_pgn
from .eval.classical import evaluate
from .eval.nnue_ref import evaluate_nnue
from .search import (
    Searcher,
    SearchInfo,
    SearchLimits,
    TimeAllocation,
    TranspositionTable,
    TTBound,
    allocate_time,
    search,
    see,
)
from .protocol import EngineOptions, UCIEngine, run_uci
