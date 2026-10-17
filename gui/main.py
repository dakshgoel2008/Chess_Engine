"""Main game controller and Pygame loop driving CastledEngine."""

import asyncio
import concurrent.futures
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pygame as p
except ImportError:  # pragma: no cover
    p = None  # type: ignore[assignment]

from castled.core.board import Board
from castled.core.move import Move
from castled.core.types import Color, PieceType, STARTING_FEN
from castled.eval.classical import evaluate
from castled.eval.nnue_ref import evaluate_nnue
from castled.search.search import SearchInfo, Searcher
from gui.config import (
    BOARD_SIZE,
    DIFFICULTY_LEVELS,
    DIMENSION,
    EVAL_BAR_WIDTH,
    HEIGHT,
    MAX_FPS,
    SQUARE_SIZE,
    WIDTH,
)
from gui.review import (
    MoveReviewItem,
    export_game_pgn,
    grid_coords_to_square,
    import_game_pgn,
    review_game,
    to_white_perspective_cp,
)
from gui.ui import (
    _require_pygame,
    animate_move,
    draw_board,
    draw_eval_bar,
    draw_game_over_banner,
    draw_labels,
    draw_menu,
    draw_pieces,
    draw_promotion_dialog,
    draw_sidebar,
    highlight_squares,
    load_images,
)


MODE_LABELS: Dict[str, str] = {
    "white": "White vs AI",
    "black": "Black vs AI",
    "pvp": "Player vs Player",
    "aivai": "AI vs AI",
}


class GameSession:
    """Manages board state, move history, engine search parameters, PGN I/O, and Game Review."""

    def __init__(
        self,
        mode: str = "white",
        difficulty: str = "Medium",
        start_fen: str = STARTING_FEN,
        hash_mb: int = 32,
        use_nnue: bool = True,
    ) -> None:
        self.use_nnue = use_nnue
        self.searcher = Searcher(hash_size_mb=hash_mb, use_nnue=use_nnue)
        self.mode = mode if mode in MODE_LABELS else "white"
        self.difficulty = difficulty if difficulty in DIFFICULTY_LEVELS else "Medium"
        self.flipped = (self.mode == "black")
        self.reset(start_fen=start_fen)

    def _static_eval(self) -> int:
        return evaluate_nnue(self.board) if self.use_nnue else evaluate(self.board)

    def toggle_nnue(self) -> bool:
        """Toggle between NNUE evaluation and Classical evaluation."""
        self.use_nnue = not self.use_nnue
        self.searcher.use_nnue = self.use_nnue
        self.searcher.clear()
        self.white_score_cp = to_white_perspective_cp(self._static_eval(), self.board.turn)
        label = "NNUE" if self.use_nnue else "Classical"
        self.status_text = f"Switched evaluator to {label}"
        return self.use_nnue

    def reset(self, start_fen: str = STARTING_FEN) -> None:
        """Reset the game to the starting or specified FEN position."""
        self.start_fen = start_fen
        self.board = Board(start_fen)
        self.searcher.clear()
        self.uci_moves: List[str] = []
        self.san_moves: List[str] = []
        self.move_objects: List[Move] = []
        self.selected_sq: Optional[int] = None
        self.pending_promotions: List[Move] = []
        self.review_items: List[MoveReviewItem] = []

        static_cp = self._static_eval()
        self.white_score_cp = to_white_perspective_cp(static_cp, self.board.turn)
        self.search_depth = 0
        self.search_nodes = 0
        self.search_nps = 0
        self.pv_text = "-"
        self.status_text = "White to move" if self.board.turn == Color.WHITE else "Black to move"

    def set_mode_and_difficulty(self, mode: str, difficulty: str) -> None:
        """Update the active game mode and AI difficulty preset."""
        if mode in MODE_LABELS:
            self.mode = mode
            self.flipped = (mode == "black")
        if difficulty in DIFFICULTY_LEVELS:
            self.difficulty = difficulty

    def is_human_turn(self) -> bool:
        """Return True if the side to move is controlled by a human player."""
        if self.board.is_game_over():
            return False
        if self.mode == "pvp":
            return True
        if self.mode == "aivai":
            return False
        if self.mode == "white":
            return int(self.board.turn) == int(Color.WHITE)
        return int(self.board.turn) == int(Color.BLACK)

    def legal_moves_from(self, square: int) -> List[Move]:
        """Return all legal moves originating from the given 0..63 square."""
        return [m for m in self.board.legal_moves if m.from_square == square]

    def select_or_move(self, square: int) -> Optional[Move]:
        """Handle a board click on `square`; return the chosen Move if a move was completed."""
        if not self.is_human_turn():
            return None

        if self.selected_sq is not None:
            candidates = [
                m for m in self.board.legal_moves
                if m.from_square == self.selected_sq and m.to_square == square
            ]
            if len(candidates) == 1:
                self.selected_sq = None
                self.pending_promotions = []
                return candidates[0]
            if len(candidates) > 1:
                self.pending_promotions = candidates
                self.status_text = "Select promotion piece"
                return None

        piece = self.board.piece_at(square)
        if piece is not None and int(piece.color) == int(self.board.turn):
            self.selected_sq = square
            self.pending_promotions = []
        else:
            self.selected_sq = None
            self.pending_promotions = []
        return None

    def resolve_promotion(self, piece_type: PieceType) -> Optional[Move]:
        """Select the promotion move matching `piece_type` from `pending_promotions`."""
        promo_map = {
            PieceType.QUEEN: "q",
            PieceType.ROOK: "r",
            PieceType.BISHOP: "b",
            PieceType.KNIGHT: "n",
        }
        suffix = promo_map.get(piece_type, "q")
        chosen = None
        for m in self.pending_promotions:
            if m.to_uci().endswith(suffix):
                chosen = m
                break
        self.pending_promotions = []
        self.selected_sq = None
        return chosen

    def apply_move(self, move: Move, search_result: Optional[SearchInfo] = None) -> None:
        """Push a move onto the board and update move history and telemetry."""
        mover_turn = self.board.turn
        san_str = self.board.san(move)
        uci_str = move.to_uci()
        self.board.push(move)

        self.uci_moves.append(uci_str)
        self.san_moves.append(san_str)
        self.move_objects.append(move)
        self.selected_sq = None
        self.pending_promotions = []
        self.review_items = []

        if search_result is not None:
            self.white_score_cp = to_white_perspective_cp(search_result.score, mover_turn)
            self.search_depth = search_result.depth
            self.search_nodes = search_result.nodes
            self.search_nps = search_result.nps
            self.pv_text = " ".join(m.to_uci() for m in search_result.pv[:5]) or uci_str
        else:
            static_cp = self._static_eval()
            self.white_score_cp = to_white_perspective_cp(static_cp, self.board.turn)

        if self.board.is_checkmate():
            winner = "White" if mover_turn == Color.WHITE else "Black"
            self.status_text = f"Checkmate - {winner} wins!"
        elif self.board.is_game_over():
            outcome = self.board.outcome()
            reason = outcome.termination.replace("_", " ") if outcome else "draw"
            self.status_text = f"Game drawn ({reason})"
        else:
            side = "White" if self.board.turn == Color.WHITE else "Black"
            chk = " (Check!)" if self.board.is_check() else ""
            self.status_text = f"{side} to move{chk}"

    def undo(self) -> bool:
        """Undo the last move (or last pair of moves when playing against AI)."""
        if not self.move_objects:
            return False

        steps = 1
        if self.mode in ("white", "black") and self.is_human_turn() and len(self.move_objects) >= 2:
            steps = 2

        for _ in range(min(steps, len(self.move_objects))):
            self.board.pop()
            self.uci_moves.pop()
            self.san_moves.pop()
            self.move_objects.pop()

        self.selected_sq = None
        self.pending_promotions = []
        self.review_items = []
        static_cp = self._static_eval()
        self.white_score_cp = to_white_perspective_cp(static_cp, self.board.turn)
        side = "White" if self.board.turn == Color.WHITE else "Black"
        self.status_text = f"Undid move - {side} to move"
        return True

    def compute_ai_move(self) -> Tuple[Optional[Move], SearchInfo]:
        """Run the CastledEngine search on a board copy according to the current difficulty."""
        preset = DIFFICULTY_LEVELS.get(self.difficulty, DIFFICULTY_LEVELS["Medium"])
        max_depth = int(preset["depth"])
        movetime_ms = int(preset["movetime_ms"])
        board_copy = self.board.copy()
        res = self.searcher.search(
            board_copy,
            depth=max_depth,
            movetime_ms=movetime_ms,
            use_nnue=self.use_nnue,
        )
        best = res.best_move if int(res.best_move) != 0 else None
        return best, res

    def run_review(self, depth: int = 2) -> List[MoveReviewItem]:
        """Analyze all played moves and populate `self.review_items`."""
        if not self.uci_moves:
            self.status_text = "Play some moves before running Game Review"
            return []
        self.review_items = review_game(
            self.uci_moves,
            start_fen=self.start_fen,
            depth=depth,
            searcher=self.searcher,
        )
        self.status_text = f"Game Review complete ({len(self.review_items)} plies analyzed)"
        return self.review_items

    def save_pgn_file(self, path: str = "castled_game.pgn") -> str:
        """Save the current game in standard PGN format to `path`."""
        white_name = "Player" if self.mode in ("white", "pvp") else "CastledEngine"
        black_name = "Player" if self.mode in ("black", "pvp") else "CastledEngine"
        pgn_str = export_game_pgn(
            self.uci_moves,
            start_fen=self.start_fen,
            white_name=white_name,
            black_name=black_name,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(pgn_str + "\n")
        self.status_text = f"Saved PGN to {os.path.basename(path)}"
        return pgn_str

    def load_pgn_file(self, path: str = "castled_game.pgn") -> bool:
        """Load a PGN file from `path` into the active session."""
        if not os.path.exists(path):
            self.status_text = f"No file found at {os.path.basename(path)}"
            return False
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        initial_fen, uci_list = import_game_pgn(text)
        self.reset(start_fen=initial_fen)
        for uci in uci_list:
            m = self.board.parse_uci(uci)
            self.apply_move(m)
        self.status_text = f"Loaded {len(uci_list)} plies from {os.path.basename(path)}"
        return True

    def sidebar_state(self) -> Dict[str, Any]:
        """Build the dictionary consumed by `draw_sidebar`."""
        return {
            "mode_label": MODE_LABELS.get(self.mode, "Player vs AI"),
            "difficulty": self.difficulty,
            "use_nnue": self.use_nnue,
            "status_text": self.status_text,
            "search_depth": self.search_depth,
            "search_nodes": self.search_nodes,
            "search_nps": self.search_nps,
            "white_score_cp": self.white_score_cp,
            "pv_text": self.pv_text,
            "san_moves": self.san_moves,
            "review_items": self.review_items,
        }


async def main(
    initial_mode: Optional[str] = None,
    difficulty: str = "Medium",
    pgn_path: Optional[str] = None,
) -> None:
    """Run the asynchronous Pygame event loop for CastledEngine."""
    _require_pygame()
    p.init()
    p.display.set_caption("CastledEngine - Chess GUI")
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    images = load_images()
    label_font = p.font.SysFont("Arial", 14, bold=True)

    session = GameSession(mode=initial_mode or "white", difficulty=difficulty)
    if pgn_path:
        session.load_pgn_file(pgn_path)

    in_menu = (initial_mode is None and pgn_path is None)
    running = True
    is_wasm = (sys.platform == "emscripten")

    executor = None if is_wasm else concurrent.futures.ThreadPoolExecutor(max_workers=1)
    ai_future: Optional[concurrent.futures.Future] = None
    review_future: Optional[concurrent.futures.Future] = None
    sidebar_buttons: Dict[str, Any] = {}
    promo_hitboxes: List[Tuple[PieceType, Any]] = []

    try:
        while running:
            if in_menu:
                mode_rects, diff_rects = draw_menu(screen, selected_difficulty=session.difficulty)
                p.display.flip()
                for e in p.event.get():
                    if e.type == p.QUIT:
                        running = False
                    elif e.type == p.MOUSEBUTTONDOWN and e.button == 1:
                        loc = p.mouse.get_pos()
                        for diff_key, rect in diff_rects:
                            if rect.collidepoint(loc):
                                session.difficulty = diff_key
                        for mode_key, rect in mode_rects:
                            if rect.collidepoint(loc):
                                session.set_mode_and_difficulty(mode_key, session.difficulty)
                                session.reset()
                                in_menu = False
                clock.tick(MAX_FPS)
                await asyncio.sleep(0)
                continue

            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
                elif e.type == p.MOUSEBUTTONDOWN and e.button == 1:
                    loc = p.mouse.get_pos()

                    # Check promotion dialog first if open
                    if session.pending_promotions and promo_hitboxes:
                        for pt, rect in promo_hitboxes:
                            if rect.collidepoint(loc):
                                chosen = session.resolve_promotion(pt)
                                if chosen is not None:
                                    animate_move(
                                        screen,
                                        session.board,
                                        images,
                                        chosen,
                                        clock,
                                        flipped=session.flipped,
                                        white_score_cp=session.white_score_cp,
                                        sidebar_state=session.sidebar_state(),
                                    )
                                    session.apply_move(chosen)
                                break
                        continue

                    # Check sidebar buttons
                    clicked_button = None
                    for btn_key, rect in sidebar_buttons.items():
                        if rect.collidepoint(loc):
                            clicked_button = btn_key
                            break

                    if clicked_button == "new":
                        if ai_future is None or ai_future.done():
                            in_menu = True
                        continue
                    elif clicked_button == "undo":
                        if ai_future is None or ai_future.done():
                            session.undo()
                        continue
                    elif clicked_button == "flip":
                        session.flipped = not session.flipped
                        continue
                    elif clicked_button == "review":
                        if (ai_future is None or ai_future.done()) and (review_future is None or review_future.done()):
                            session.status_text = "Analyzing game moves..."
                            if executor is not None:
                                review_future = executor.submit(session.run_review, 2)
                            else:
                                session.run_review(2)
                        continue
                    elif clicked_button == "save_pgn":
                        session.save_pgn_file(pgn_path or "castled_game.pgn")
                        continue
                    elif clicked_button == "load_pgn":
                        if ai_future is None or ai_future.done():
                            session.load_pgn_file(pgn_path or "castled_game.pgn")
                        continue

                    # Board square click
                    board_x = loc[0] - EVAL_BAR_WIDTH
                    board_y = loc[1]
                    if 0 <= board_x < BOARD_SIZE and 0 <= board_y < HEIGHT:
                        col = board_x // SQUARE_SIZE
                        row = board_y // SQUARE_SIZE
                        if 0 <= row < DIMENSION and 0 <= col < DIMENSION:
                            sq = grid_coords_to_square(row, col, flipped=session.flipped)
                            chosen = session.select_or_move(sq)
                            if chosen is not None:
                                animate_move(
                                    screen,
                                    session.board,
                                    images,
                                    chosen,
                                    clock,
                                    flipped=session.flipped,
                                    white_score_cp=session.white_score_cp,
                                    sidebar_state=session.sidebar_state(),
                                )
                                session.apply_move(chosen)

                elif e.type == p.KEYDOWN:
                    if e.key == p.K_z and (ai_future is None or ai_future.done()):
                        session.undo()
                    elif e.key in (p.K_n, p.K_r) and (ai_future is None or ai_future.done()):
                        in_menu = True
                    elif e.key == p.K_f:
                        session.flipped = not session.flipped
                    elif e.key == p.K_e and (ai_future is None or ai_future.done()):
                        session.toggle_nnue()
                    elif e.key == p.K_g and (ai_future is None or ai_future.done()):
                        if review_future is None or review_future.done():
                            session.status_text = "Analyzing game moves..."
                            if executor is not None:
                                review_future = executor.submit(session.run_review, 2)
                            else:
                                session.run_review(2)
                    elif e.key == p.K_s:
                        session.save_pgn_file(pgn_path or "castled_game.pgn")
                    elif e.key == p.K_l and (ai_future is None or ai_future.done()):
                        session.load_pgn_file(pgn_path or "castled_game.pgn")

            # Poll background Game Review task
            if review_future is not None and review_future.done():
                review_future = None

            # Non-blocking AI move execution (or cooperative single-frame execution on WASM)
            if not session.board.is_game_over() and not session.is_human_turn():
                if executor is None:
                    session.status_text = f"CastledEngine ({session.difficulty}) thinking..."
                    p.display.flip()
                    await asyncio.sleep(0)
                    best_move, search_res = session.compute_ai_move()
                    if best_move is not None:
                        session.apply_move(best_move, search_result=search_res)
                elif ai_future is None:
                    session.status_text = f"CastledEngine ({session.difficulty}) thinking..."
                    p.display.set_caption("CastledEngine - AI is thinking...")
                    ai_future = executor.submit(session.compute_ai_move)
                elif ai_future.done():
                    best_move, search_res = ai_future.result()
                    ai_future = None
                    p.display.set_caption("CastledEngine - Chess GUI")
                    if best_move is not None:
                        animate_move(
                            screen,
                            session.board,
                            images,
                            best_move,
                            clock,
                            flipped=session.flipped,
                            white_score_cp=session.white_score_cp,
                            sidebar_state=session.sidebar_state(),
                        )
                        session.apply_move(best_move, search_result=search_res)

            # Render frame
            draw_eval_bar(screen, session.white_score_cp, flipped=session.flipped)
            draw_board(screen)
            draw_labels(screen, label_font, flipped=session.flipped)
            legal_moves = session.board.legal_moves
            last_move = session.move_objects[-1] if session.move_objects else None
            highlight_squares(
                screen,
                session.board,
                session.selected_sq,
                legal_moves,
                last_move=last_move,
                flipped=session.flipped,
            )
            draw_pieces(screen, session.board, images, flipped=session.flipped)

            if session.pending_promotions:
                promo_hitboxes = draw_promotion_dialog(screen, int(session.board.turn), images)
            else:
                promo_hitboxes = []

            if session.board.is_game_over():
                draw_game_over_banner(screen, session.board)

            sidebar_buttons = draw_sidebar(screen, session.sidebar_state())

            clock.tick(MAX_FPS)
            await asyncio.sleep(0)
            p.display.flip()
    finally:
        executor.shutdown(wait=False)
        p.quit()


def run_gui(
    mode: Optional[str] = None,
    difficulty: str = "Medium",
    pgn_path: Optional[str] = None,
) -> int:
    """Launch the Pygame GUI synchronously."""
    asyncio.run(main(initial_mode=mode, difficulty=difficulty, pgn_path=pgn_path))
    return 0


if __name__ == "__main__":
    run_gui()

