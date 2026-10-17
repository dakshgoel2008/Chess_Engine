"""Board coordinate mapping, evaluation bar math, PGN helpers, and Game Review move classification."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from castled.core.board import Board
from castled.core.pgn import PGNGame, parse_pgn
from castled.core.types import Color, PieceType, STARTING_FEN
from castled.search.search import Searcher
from castled.search.transposition import SCORE_MATE, SCORE_MATE_THRESHOLD


_PIECE_IMAGE_SUFFIX: Dict[PieceType, str] = {
    PieceType.PAWN: "p",
    PieceType.KNIGHT: "N",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.QUEEN: "Q",
    PieceType.KING: "K",
}


@dataclass
class MoveReviewItem:
    """Stores engine evaluation before/after a move and its review classification."""

    ply: int
    san: str
    uci: str
    mover_color: int
    eval_before_white_cp: int
    eval_after_white_cp: int
    cp_loss: int
    classification: str
    best_move_uci: str


def piece_to_image_key(color: int, piece_type: int) -> str:
    """Map a Castled (Color, PieceType) pair to its image asset key (e.g., 'wp', 'bN')."""
    prefix = "w" if int(color) == int(Color.WHITE) else "b"
    suffix = _PIECE_IMAGE_SUFFIX[PieceType(piece_type)]
    return f"{prefix}{suffix}"


def grid_coords_to_square(row: int, col: int, flipped: bool = False) -> int:
    """Convert 0..7 screen (row, col) coordinates into a 0..63 LERF square index."""
    if flipped:
        rank = row
        file_idx = 7 - col
    else:
        rank = 7 - row
        file_idx = col
    return (rank << 3) | file_idx


def square_to_grid_coords(square: int, flipped: bool = False) -> Tuple[int, int]:
    """Convert a 0..63 LERF square index into 0..7 screen (row, col) coordinates."""
    rank = square >> 3
    file_idx = square & 7
    if flipped:
        return rank, 7 - file_idx
    return 7 - rank, file_idx


def board_to_grid(board: Board) -> List[List[str]]:
    """Convert a Castled Board into an 8x8 grid of image keys ('wp', 'bK', '.') with row 0 = rank 8."""
    grid = [["." for _ in range(8)] for _ in range(8)]
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            row = 7 - (sq >> 3)
            col = sq & 7
            grid[row][col] = piece_to_image_key(piece.color, piece.piece_type)
    return grid


def to_white_perspective_cp(score_side_to_move: int, side_to_move: int) -> int:
    """Convert a side-to-move centipawn score into White's perspective."""
    return int(score_side_to_move) if int(side_to_move) == int(Color.WHITE) else -int(score_side_to_move)


def eval_to_bar_fraction(white_score_cp: int) -> float:
    """Map White-perspective centipawns to a [0.0, 1.0] vertical bar fill ratio using a logistic curve."""
    if white_score_cp >= SCORE_MATE_THRESHOLD:
        return 1.0
    if white_score_cp <= -SCORE_MATE_THRESHOLD:
        return 0.0
    clamped = max(-2000, min(2000, int(white_score_cp)))
    return 1.0 / (1.0 + math.exp(-clamped / 280.0))


def format_eval_label(white_score_cp: int) -> str:
    """Format a White-perspective centipawn or mate score into a compact UI label."""
    if abs(white_score_cp) >= SCORE_MATE_THRESHOLD:
        mate_plies = max(1, SCORE_MATE - abs(white_score_cp))
        mate_moves = (mate_plies + 1) // 2
        sign = "+" if white_score_cp > 0 else "-"
        return f"{sign}M{mate_moves}"
    pawns = white_score_cp / 100.0
    return f"{pawns:+.2f}"


def classify_move_cpl(cp_loss: int, is_best_move: bool = False) -> str:
    """Classify a move into Best, Good, Inaccuracy, Mistake, or Blunder based on centipawn loss."""
    if is_best_move or cp_loss <= 10:
        return "Best"
    if cp_loss <= 35:
        return "Good"
    if cp_loss <= 90:
        return "Inaccuracy"
    if cp_loss <= 200:
        return "Mistake"
    return "Blunder"


def _clamp_eval_for_cpl(white_cp: int) -> int:
    """Cap mate and extreme scores to +/-1000 cp so centipawn loss remains well-scaled."""
    return max(-1000, min(1000, int(white_cp)))


def review_game(
    moves_uci: List[str],
    start_fen: str = STARTING_FEN,
    depth: int = 3,
    searcher: Optional[Searcher] = None,
) -> List[MoveReviewItem]:
    """Run engine analysis across each move of a game and classify move quality."""
    eng = searcher if searcher is not None else Searcher(hash_size_mb=16)
    board = Board(start_fen)
    items: List[MoveReviewItem] = []

    res_initial = eng.search(board, depth=max(1, depth))
    eval_before_white = to_white_perspective_cp(res_initial.score, board.turn)
    best_before_uci = res_initial.best_move.to_uci() if int(res_initial.best_move) != 0 else ""

    for idx, uci in enumerate(moves_uci):
        mover = int(board.turn)
        move = board.parse_uci(uci)
        san_str = board.san(move)
        board.push(move)

        if board.is_checkmate():
            eval_after_white = SCORE_MATE if mover == int(Color.WHITE) else -SCORE_MATE
            next_best_uci = ""
        elif board.is_game_over():
            eval_after_white = 0
            next_best_uci = ""
        else:
            res_after = eng.search(board, depth=max(1, depth))
            eval_after_white = to_white_perspective_cp(res_after.score, board.turn)
            next_best_uci = res_after.best_move.to_uci() if int(res_after.best_move) != 0 else ""

        before_clamped = _clamp_eval_for_cpl(eval_before_white)
        after_clamped = _clamp_eval_for_cpl(eval_after_white)
        if mover == int(Color.WHITE):
            cp_loss = max(0, before_clamped - after_clamped)
        else:
            cp_loss = max(0, after_clamped - before_clamped)

        is_best = (uci == best_before_uci)
        classification = classify_move_cpl(cp_loss, is_best_move=is_best)

        items.append(
            MoveReviewItem(
                ply=idx + 1,
                san=san_str,
                uci=uci,
                mover_color=mover,
                eval_before_white_cp=eval_before_white,
                eval_after_white_cp=eval_after_white,
                cp_loss=cp_loss,
                classification=classification,
                best_move_uci=best_before_uci,
            )
        )
        eval_before_white = eval_after_white
        best_before_uci = next_best_uci

    return items


def summarize_review(items: List[MoveReviewItem]) -> Dict[int, Dict[str, int]]:
    """Aggregate counts of Best, Good, Inaccuracy, Mistake, and Blunder moves per color."""
    categories = ("Best", "Good", "Inaccuracy", "Mistake", "Blunder")
    summary: Dict[int, Dict[str, int]] = {
        int(Color.WHITE): {cat: 0 for cat in categories},
        int(Color.BLACK): {cat: 0 for cat in categories},
    }
    for item in items:
        if item.classification in summary[item.mover_color]:
            summary[item.mover_color][item.classification] += 1
    return summary


def export_game_pgn(
    moves_uci: List[str],
    start_fen: str = STARTING_FEN,
    white_name: str = "Player",
    black_name: str = "CastledEngine",
) -> str:
    """Serialize a list of UCI moves from start_fen into a complete PGN string."""
    board = Board(start_fen)
    moves = []
    for uci in moves_uci:
        m = board.parse_uci(uci)
        moves.append(m)
        board.push(m)
    outcome = board.outcome()
    result = outcome.result if outcome else "*"
    headers = {
        "Event": "CastledEngine GUI Match",
        "White": white_name,
        "Black": black_name,
        "Result": result,
    }
    if start_fen != STARTING_FEN:
        headers["FEN"] = start_fen
    game = PGNGame(headers=headers, moves=moves, result=result, initial_fen=start_fen)
    return game.to_pgn()


def import_game_pgn(pgn_text: str) -> Tuple[str, List[str]]:
    """Parse a PGN string and return (initial_fen, list_of_uci_moves)."""
    game = parse_pgn(pgn_text)
    return game.initial_fen, [m.to_uci() for m in game.moves]

