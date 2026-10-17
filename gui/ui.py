"""Pygame rendering routines for the board, 3D piece animations, eval bar, sidebar HUD, and game review graph."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import pygame as p
except ImportError:  # pragma: no cover
    p = None  # type: ignore[assignment]

from castled.core.board import Board
from castled.core.move import Move
from castled.core.types import Color, PieceType
from gui.config import (
    ANIMATION_DURATION,
    BOARD_SIZE,
    CLASSIFICATION_COLORS,
    DIFFICULTY_LEVELS,
    DIFFICULTY_ORDER,
    DIMENSION,
    EVAL_BAR_WIDTH,
    HEIGHT,
    SIDEBAR_WIDTH,
    SQUARE_SIZE,
    WIDTH,
)
from gui.review import (
    MoveReviewItem,
    eval_to_bar_fraction,
    format_eval_label,
    piece_to_image_key,
    square_to_grid_coords,
    summarize_review,
)


def _require_pygame() -> None:
    """Raise a clear error if pygame is not installed in the active environment."""
    if p is None:
        raise RuntimeError(
            "Pygame is required for the graphical UI. Install it with: pip install pygame-ce (or pip install pygame)"
        )


def load_images(square_size: int = SQUARE_SIZE) -> Dict[str, Any]:
    """Load and scale all 12 piece PNG sprites from gui/images (with fallback to Chess/images)."""
    _require_pygame()
    pieces = ["wB", "wK", "wN", "wp", "wQ", "wR", "bB", "bK", "bN", "bp", "bQ", "bR"]
    images: Dict[str, Any] = {}
    gui_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(gui_dir)

    for piece in pieces:
        candidates = [
            os.path.join(gui_dir, "images", f"{piece}.png"),
            os.path.join(repo_root, "Chess", "images", f"{piece}.png"),
        ]
        loaded = None
        for path in candidates:
            if os.path.exists(path):
                loaded = p.transform.smoothscale(p.image.load(path), (square_size, square_size))
                break
        if loaded is None:
            surf = p.Surface((square_size, square_size), p.SRCALPHA)
            fill = (240, 240, 240) if piece.startswith("w") else (50, 50, 50)
            p.draw.circle(surf, fill, (square_size // 2, square_size // 2), square_size // 3)
            loaded = surf
        images[piece] = loaded
    return images


def ease_in_out(t: float) -> float:
    """Compute smooth cubic Hermite interpolation for piece movement animations."""
    return t * t * (3.0 - 2.0 * t)


def draw_board(screen: Any, x_offset: int = EVAL_BAR_WIDTH) -> None:
    """Render the 8x8 chessboard with beveled square edges for visual depth."""
    light_color = p.Color(240, 217, 181)
    dark_color = p.Color(181, 136, 99)
    colors = [light_color, dark_color]

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            base_color = colors[(r + c) & 1]
            rect = p.Rect(x_offset + c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            p.draw.rect(screen, base_color, rect)

            highlight = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
            p.draw.polygon(
                highlight,
                (255, 255, 255, 55),
                [
                    (0, 0),
                    (SQUARE_SIZE, 0),
                    (SQUARE_SIZE - 5, 5),
                    (5, 5),
                    (5, SQUARE_SIZE - 5),
                    (0, SQUARE_SIZE),
                ],
            )
            p.draw.polygon(
                highlight,
                (0, 0, 0, 35),
                [
                    (SQUARE_SIZE, SQUARE_SIZE),
                    (0, SQUARE_SIZE),
                    (5, SQUARE_SIZE - 5),
                    (SQUARE_SIZE - 5, SQUARE_SIZE - 5),
                    (SQUARE_SIZE - 5, 5),
                    (SQUARE_SIZE, 0),
                ],
            )
            screen.blit(highlight, rect.topleft)


def draw_labels(
    screen: Any,
    font: Any,
    flipped: bool = False,
    x_offset: int = EVAL_BAR_WIDTH,
) -> None:
    """Draw algebraic file (a-h) and rank (1-8) labels along the board edges."""
    for i in range(DIMENSION):
        file_char = chr(ord("h") - i) if flipped else chr(ord("a") + i)
        rank_str = str(i + 1) if flipped else str(DIMENSION - i)

        file_label = font.render(file_char, True, p.Color(40, 30, 20))
        screen.blit(
            file_label,
            (
                x_offset + i * SQUARE_SIZE + SQUARE_SIZE - file_label.get_width() - 6,
                HEIGHT - file_label.get_height() - 4,
            ),
        )
        rank_label = font.render(rank_str, True, p.Color(40, 30, 20))
        screen.blit(rank_label, (x_offset + 6, i * SQUARE_SIZE + 4))


def draw_drop_shadow(screen: Any, x: float, y: float) -> None:
    """Render an elliptical drop shadow beneath a piece sprite."""
    shadow = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
    shadow_width = int(SQUARE_SIZE * 0.7)
    shadow_height = int(SQUARE_SIZE * 0.3)
    shadow_x = (SQUARE_SIZE - shadow_width) // 2
    shadow_y = SQUARE_SIZE - shadow_height - 6
    p.draw.ellipse(
        shadow,
        (0, 0, 0, 80),
        p.Rect(shadow_x, shadow_y, shadow_width, shadow_height),
    )
    screen.blit(shadow, (x, y))


def draw_pieces(
    screen: Any,
    board: Board,
    images: Dict[str, Any],
    flipped: bool = False,
    skip_square: Optional[int] = None,
    x_offset: int = EVAL_BAR_WIDTH,
) -> None:
    """Draw all pieces on the board except an optional square currently being animated."""
    for sq in range(64):
        if skip_square is not None and sq == skip_square:
            continue
        piece = board.piece_at(sq)
        if piece is not None:
            key = piece_to_image_key(piece.color, piece.piece_type)
            r, c = square_to_grid_coords(sq, flipped=flipped)
            x = x_offset + c * SQUARE_SIZE
            y = r * SQUARE_SIZE
            draw_drop_shadow(screen, x, y)
            screen.blit(images[key], (x, y))


def highlight_squares(
    screen: Any,
    board: Board,
    selected_sq: Optional[int],
    legal_moves: List[Move],
    last_move: Optional[Move] = None,
    flipped: bool = False,
    x_offset: int = EVAL_BAR_WIDTH,
) -> None:
    """Highlight the last move, selected piece, legal target squares, and check status."""
    if last_move is not None:
        for sq in (last_move.from_square, last_move.to_square):
            r, c = square_to_grid_coords(sq, flipped=flipped)
            s = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
            s.fill((245, 235, 110, 95))
            screen.blit(s, (x_offset + c * SQUARE_SIZE, r * SQUARE_SIZE))

    if selected_sq is not None:
        r, c = square_to_grid_coords(selected_sq, flipped=flipped)
        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
        p.draw.circle(
            s,
            (100, 200, 255, 130),
            (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
            SQUARE_SIZE // 2 - 6,
        )
        screen.blit(s, (x_offset + c * SQUARE_SIZE, r * SQUARE_SIZE))

        seen_targets = set()
        for move in legal_moves:
            if move.from_square == selected_sq and move.to_square not in seen_targets:
                seen_targets.add(move.to_square)
                er, ec = square_to_grid_coords(move.to_square, flipped=flipped)
                move_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
                if move.is_capture():
                    p.draw.circle(
                        move_surf,
                        (255, 95, 95, 210),
                        (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                        SQUARE_SIZE // 2 - 6,
                        5,
                    )
                else:
                    p.draw.circle(
                        move_surf,
                        (140, 245, 140, 185),
                        (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                        SQUARE_SIZE // 6,
                    )
                screen.blit(move_surf, (x_offset + ec * SQUARE_SIZE, er * SQUARE_SIZE))

    if board.is_check():
        king_sq = None
        for sq in range(64):
            pc = board.piece_at(sq)
            if pc is not None and pc.color == board.turn and pc.piece_type == PieceType.KING:
                king_sq = sq
                break
        if king_sq is not None:
            kr, kc = square_to_grid_coords(king_sq, flipped=flipped)
            s = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
            for i in range(3):
                p.draw.circle(
                    s,
                    (255, 0, 0, 90 - i * 22),
                    (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                    SQUARE_SIZE // 2 - i * 3,
                )
            screen.blit(s, (x_offset + kc * SQUARE_SIZE, kr * SQUARE_SIZE))


def draw_game_over_banner(screen: Any, board: Board, x_offset: int = EVAL_BAR_WIDTH) -> None:
    """Draw a semi-transparent banner over the board when the game has ended."""
    outcome = board.outcome()
    if outcome is None:
        return

    overlay = p.Surface((BOARD_SIZE, HEIGHT), p.SRCALPHA)
    overlay.fill((20, 30, 25, 170))
    screen.blit(overlay, (x_offset, 0))

    font_main = p.font.SysFont("Helvetica", 40, bold=True)
    font_sub = p.font.SysFont("Helvetica", 22)

    if outcome.winner == Color.WHITE:
        headline = "Checkmate - White Wins!"
    elif outcome.winner == Color.BLACK:
        headline = "Checkmate - Black Wins!"
    else:
        reason = outcome.termination.replace("_", " ").title()
        headline = f"Draw ({reason})"

    text_surf = font_main.render(headline, True, p.Color(245, 245, 245))
    sub_surf = font_sub.render("Press N for New Game, Z to Undo, or G for Game Review", True, p.Color(200, 220, 255))

    cx = x_offset + BOARD_SIZE // 2
    cy = HEIGHT // 2
    screen.blit(text_surf, (cx - text_surf.get_width() // 2, cy - 34))
    screen.blit(sub_surf, (cx - sub_surf.get_width() // 2, cy + 20))


def animate_move(
    screen: Any,
    board: Board,
    images: Dict[str, Any],
    move: Move,
    clock: Any,
    flipped: bool = False,
    white_score_cp: int = 0,
    sidebar_state: Optional[Dict[str, Any]] = None,
    x_offset: int = EVAL_BAR_WIDTH,
) -> None:
    """Animate a piece moving from its origin square to its destination with a 3D lift arc."""
    piece_info = board.piece_at(move.from_square)
    if piece_info is None:
        return

    piece_key = piece_to_image_key(piece_info.color, piece_info.piece_type)
    sr, sc = square_to_grid_coords(move.from_square, flipped=flipped)
    er, ec = square_to_grid_coords(move.to_square, flipped=flipped)

    start_x = x_offset + sc * SQUARE_SIZE
    start_y = sr * SQUARE_SIZE
    end_x = x_offset + ec * SQUARE_SIZE
    end_y = er * SQUARE_SIZE

    start_time = p.time.get_ticks()
    label_font = p.font.SysFont("Arial", 16, bold=True)

    while True:
        elapsed = p.time.get_ticks() - start_time
        if elapsed >= ANIMATION_DURATION:
            break

        progress = ease_in_out(elapsed / ANIMATION_DURATION)
        current_x = start_x + (end_x - start_x) * progress
        lift_height = math.sin(progress * math.pi) * (SQUARE_SIZE * 0.65)
        current_y = start_y + (end_y - start_y) * progress - lift_height

        draw_eval_bar(screen, white_score_cp, flipped=flipped)
        draw_board(screen, x_offset=x_offset)
        draw_labels(screen, label_font, flipped=flipped, x_offset=x_offset)
        draw_pieces(screen, board, images, flipped=flipped, skip_square=move.from_square, x_offset=x_offset)

        shadow_base_y = start_y + (end_y - start_y) * progress
        shadow_scale = 1.0 - (math.sin(progress * math.pi) * 0.4)
        shadow_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
        s_width = int(SQUARE_SIZE * 0.7 * shadow_scale)
        s_height = int(SQUARE_SIZE * 0.3 * shadow_scale)
        s_x = (SQUARE_SIZE - s_width) // 2
        s_y = SQUARE_SIZE - s_height - 6
        p.draw.ellipse(
            shadow_surf,
            (0, 0, 0, int(80 * shadow_scale)),
            p.Rect(s_x, s_y, s_width, s_height),
        )
        screen.blit(shadow_surf, (current_x, shadow_base_y))
        screen.blit(images[piece_key], (current_x, current_y))

        if sidebar_state is not None:
            draw_sidebar(screen, sidebar_state)

        p.display.flip()
        clock.tick(60)


def draw_eval_bar(screen: Any, white_score_cp: int, flipped: bool = False) -> None:
    """Render the vertical evaluation bar showing White vs Black advantage."""
    bar_rect = p.Rect(0, 0, EVAL_BAR_WIDTH, HEIGHT)
    p.draw.rect(screen, (40, 42, 48), bar_rect)

    white_frac = eval_to_bar_fraction(white_score_cp)
    white_height = int(round(HEIGHT * white_frac))

    if flipped:
        white_rect = p.Rect(0, 0, EVAL_BAR_WIDTH, white_height)
        p.draw.rect(screen, (238, 238, 232), white_rect)
    else:
        black_height = HEIGHT - white_height
        white_rect = p.Rect(0, black_height, EVAL_BAR_WIDTH, white_height)
        p.draw.rect(screen, (238, 238, 232), white_rect)

    mid_y = HEIGHT // 2
    p.draw.line(screen, (120, 130, 145), (0, mid_y), (EVAL_BAR_WIDTH, mid_y), 2)
    p.draw.line(screen, (25, 25, 30), (EVAL_BAR_WIDTH - 1, 0), (EVAL_BAR_WIDTH - 1, HEIGHT), 1)

    label_str = format_eval_label(white_score_cp)
    font = p.font.SysFont("Arial", 13, bold=True)
    if white_score_cp >= 0:
        txt_color = (30, 30, 35)
        txt_y = HEIGHT - 22 if not flipped else 6
    else:
        txt_color = (235, 235, 240)
        txt_y = 6 if not flipped else HEIGHT - 22
    surf = font.render(label_str, True, txt_color)
    screen.blit(surf, (max(1, (EVAL_BAR_WIDTH - surf.get_width()) // 2), txt_y))


def _draw_eval_graph(
    screen: Any,
    rect: Any,
    review_items: List[MoveReviewItem],
) -> None:
    """Plot the per-ply White centipawn evaluation curve inside the sidebar."""
    p.draw.rect(screen, (24, 28, 36), rect, border_radius=8)
    p.draw.rect(screen, (65, 75, 95), rect, 1, border_radius=8)

    mid_y = rect.y + rect.height // 2
    p.draw.line(screen, (80, 90, 110), (rect.x + 6, mid_y), (rect.right - 6, mid_y), 1)

    if not review_items:
        font = p.font.SysFont("Arial", 15)
        msg = font.render("Click 'Review (G)' to analyze game", True, (150, 160, 180))
        screen.blit(msg, (rect.centerx - msg.get_width() // 2, rect.centery - msg.get_height() // 2))
        return

    points: List[Tuple[int, int]] = []
    values = [review_items[0].eval_before_white_cp] + [it.eval_after_white_cp for it in review_items]
    n = len(values)
    inner_w = max(1, rect.width - 16)
    inner_h = max(1, rect.height - 16)

    for idx, cp in enumerate(values):
        frac = eval_to_bar_fraction(cp)
        px = rect.x + 8 + int((idx / max(1, n - 1)) * inner_w)
        py = rect.bottom - 8 - int(frac * inner_h)
        points.append((px, py))

    if len(points) >= 2:
        p.draw.lines(screen, (110, 195, 255), False, points, 2)

    for idx, item in enumerate(review_items):
        px, py = points[idx + 1]
        color = CLASSIFICATION_COLORS.get(item.classification, (180, 180, 180))
        p.draw.circle(screen, color, (px, py), 4)


def draw_sidebar(screen: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """Render the right-side HUD containing engine telemetry, move list, review graph, and buttons."""
    x0 = EVAL_BAR_WIDTH + BOARD_SIZE
    panel_rect = p.Rect(x0, 0, SIDEBAR_WIDTH, HEIGHT)
    p.draw.rect(screen, (30, 35, 45), panel_rect)
    p.draw.line(screen, (60, 70, 90), (x0, 0), (x0, HEIGHT), 2)

    font_title = p.font.SysFont("Helvetica", 22, bold=True)
    font_small = p.font.SysFont("Arial", 15)
    font_mono = p.font.SysFont("Courier New", 15, bold=True)
    font_btn = p.font.SysFont("Helvetica", 15, bold=True)

    # Header: Mode & Difficulty
    mode_label = state.get("mode_label", "Player vs AI")
    diff_name = state.get("difficulty", "Medium")
    header_surf = font_title.render(f"{mode_label} [{diff_name}]", True, (235, 240, 250))
    screen.blit(header_surf, (x0 + 16, 14))

    status_text = state.get("status_text", "Ready")
    status_surf = font_small.render(status_text, True, (150, 200, 255))
    screen.blit(status_surf, (x0 + 16, 44))

    # Engine search telemetry box
    info_rect = p.Rect(x0 + 14, 72, SIDEBAR_WIDTH - 28, 86)
    p.draw.rect(screen, (22, 26, 34), info_rect, border_radius=8)
    p.draw.rect(screen, (55, 65, 85), info_rect, 1, border_radius=8)

    depth = state.get("search_depth", 0)
    nodes = state.get("search_nodes", 0)
    nps = state.get("search_nps", 0)
    white_cp = state.get("white_score_cp", 0)
    pv_str = state.get("pv_text", "-")

    line1 = f"Eval: {format_eval_label(white_cp)}  Depth: {depth}  NPS: {nps // 1000}k"
    line2 = f"Nodes: {nodes:,}"
    line3 = f"Best: {pv_str[:34]}"

    screen.blit(font_mono.render(line1, True, (220, 230, 245)), (x0 + 22, 80))
    screen.blit(font_small.render(line2, True, (165, 175, 195)), (x0 + 22, 105))
    screen.blit(font_small.render(line3, True, (140, 220, 170)), (x0 + 22, 130))

    # Move History List
    moves_rect = p.Rect(x0 + 14, 170, SIDEBAR_WIDTH - 28, 290)
    p.draw.rect(screen, (22, 26, 34), moves_rect, border_radius=8)
    p.draw.rect(screen, (55, 65, 85), moves_rect, 1, border_radius=8)

    san_moves: List[str] = state.get("san_moves", [])
    review_items: List[MoveReviewItem] = state.get("review_items", [])
    review_map = {item.ply: item for item in review_items}

    total_rows = (len(san_moves) + 1) // 2
    max_visible_rows = 11
    start_row = max(0, total_rows - max_visible_rows)

    for row_idx in range(start_row, total_rows):
        y_pos = moves_rect.y + 8 + (row_idx - start_row) * 24
        w_ply = row_idx * 2 + 1
        b_ply = row_idx * 2 + 2
        w_san = san_moves[w_ply - 1] if w_ply - 1 < len(san_moves) else ""
        b_san = san_moves[b_ply - 1] if b_ply - 1 < len(san_moves) else ""

        num_surf = font_mono.render(f"{row_idx + 1:>2}.", True, (130, 140, 160))
        screen.blit(num_surf, (moves_rect.x + 10, y_pos))

        w_color = (235, 235, 240)
        if w_ply in review_map:
            cls_name = review_map[w_ply].classification
            w_color = CLASSIFICATION_COLORS.get(cls_name, w_color)
            w_san = f"{w_san} [{cls_name[:4]}]"
        screen.blit(font_mono.render(w_san[:15], True, w_color), (moves_rect.x + 50, y_pos))

        if b_san:
            b_color = (210, 215, 225)
            if b_ply in review_map:
                cls_name = review_map[b_ply].classification
                b_color = CLASSIFICATION_COLORS.get(cls_name, b_color)
                b_san = f"{b_san} [{cls_name[:4]}]"
            screen.blit(font_mono.render(b_san[:15], True, b_color), (moves_rect.x + 205, y_pos))

    # Game Review Evaluation Graph & Summary
    graph_rect = p.Rect(x0 + 14, 472, SIDEBAR_WIDTH - 28, 140)
    _draw_eval_graph(screen, graph_rect, review_items)

    if review_items:
        summary = summarize_review(review_items)
        w_sum = summary[int(Color.WHITE)]
        b_sum = summary[int(Color.BLACK)]
        sum_text = (
            f"W B/G/I/M/Bl: {w_sum['Best']}/{w_sum['Good']}/{w_sum['Inaccuracy']}/{w_sum['Mistake']}/{w_sum['Blunder']} | "
            f"B: {b_sum['Best']}/{b_sum['Good']}/{b_sum['Inaccuracy']}/{b_sum['Mistake']}/{b_sum['Blunder']}"
        )
        screen.blit(font_small.render(sum_text, True, (200, 210, 230)), (x0 + 16, 620))
    else:
        hint = "Keys: Z=Undo  F=Flip  G=Review  S=Save  L=Load"
        screen.blit(font_small.render(hint, True, (150, 165, 185)), (x0 + 16, 620))

    # Action buttons grid (2 rows x 3 columns)
    buttons: Dict[str, Any] = {}
    btn_specs = [
        ("new", "New (N)"),
        ("undo", "Undo (Z)"),
        ("flip", "Flip (F)"),
        ("review", "Review (G)"),
        ("save_pgn", "Save PGN"),
        ("load_pgn", "Load PGN"),
    ]
    btn_w = (SIDEBAR_WIDTH - 38) // 3
    btn_h = 46
    mouse_pos = p.mouse.get_pos()

    for idx, (key, label) in enumerate(btn_specs):
        r = idx // 3
        c = idx % 3
        bx = x0 + 14 + c * (btn_w + 5)
        by = 650 + r * (btn_h + 10)
        rect = p.Rect(bx, by, btn_w, btn_h)
        hovered = rect.collidepoint(mouse_pos)
        bg = (70, 110, 190) if hovered else (45, 60, 90)
        p.draw.rect(screen, bg, rect, border_radius=8)
        p.draw.rect(screen, (140, 165, 210), rect, 1, border_radius=8)
        lbl_surf = font_btn.render(label, True, (245, 245, 250))
        screen.blit(
            lbl_surf,
            (rect.centerx - lbl_surf.get_width() // 2, rect.centery - lbl_surf.get_height() // 2),
        )
        buttons[key] = rect

    return buttons


def draw_promotion_dialog(
    screen: Any,
    color: int,
    images: Dict[str, Any],
    x_offset: int = EVAL_BAR_WIDTH,
) -> List[Tuple[PieceType, Any]]:
    """Render a centered modal allowing the player to choose Queen, Rook, Bishop, or Knight on promotion."""
    overlay = p.Surface((BOARD_SIZE, HEIGHT), p.SRCALPHA)
    overlay.fill((15, 20, 30, 175))
    screen.blit(overlay, (x_offset, 0))

    choices = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
    box_w = 4 * (SQUARE_SIZE + 16) + 16
    box_h = SQUARE_SIZE + 66
    bx = x_offset + (BOARD_SIZE - box_w) // 2
    by = (HEIGHT - box_h) // 2

    panel = p.Rect(bx, by, box_w, box_h)
    p.draw.rect(screen, (35, 42, 58), panel, border_radius=12)
    p.draw.rect(screen, (180, 205, 255), panel, 2, border_radius=12)

    font = p.font.SysFont("Helvetica", 20, bold=True)
    title = font.render("Choose Promotion Piece", True, (240, 245, 255))
    screen.blit(title, (panel.centerx - title.get_width() // 2, by + 12))

    hitboxes: List[Tuple[PieceType, Any]] = []
    mouse_pos = p.mouse.get_pos()

    for idx, pt in enumerate(choices):
        rx = bx + 16 + idx * (SQUARE_SIZE + 16)
        ry = by + 48
        rect = p.Rect(rx, ry, SQUARE_SIZE, SQUARE_SIZE)
        hovered = rect.collidepoint(mouse_pos)
        p.draw.rect(screen, (80, 120, 200) if hovered else (55, 65, 85), rect, border_radius=10)
        p.draw.rect(screen, (200, 220, 255), rect, 1, border_radius=10)
        key = piece_to_image_key(color, pt)
        screen.blit(images[key], (rx, ry))
        hitboxes.append((pt, rect))

    return hitboxes


def draw_menu(
    screen: Any,
    selected_difficulty: str = "Medium",
) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    """Render the startup menu for selecting game mode and engine difficulty level."""
    for y in range(HEIGHT):
        color = (max(18, 36 - y // 28), max(26, 46 - y // 28), max(44, 76 - y // 20))
        p.draw.line(screen, color, (0, y), (WIDTH, y))

    font_title = p.font.SysFont("Helvetica", 56, bold=True)
    font_sub = p.font.SysFont("Helvetica", 22)
    font_options = p.font.SysFont("Helvetica", 26, bold=True)
    font_diff = p.font.SysFont("Helvetica", 20, bold=True)

    title_text = font_title.render("CASTLED ENGINE", True, p.Color("white"))
    title_shadow = font_title.render("CASTLED ENGINE", True, p.Color("black"))
    screen.blit(title_shadow, (WIDTH // 2 - title_text.get_width() // 2 + 4, 64))
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 60))

    sub_info = DIFFICULTY_LEVELS.get(selected_difficulty, DIFFICULTY_LEVELS["Medium"])
    sub_surf = font_sub.render(f"Active AI Difficulty: {sub_info['label']}", True, (185, 215, 255))
    screen.blit(sub_surf, (WIDTH // 2 - sub_surf.get_width() // 2, 136))

    # Difficulty selector pills
    diff_rects: List[Tuple[str, Any]] = []
    pill_w = 150
    pill_h = 48
    total_pills_w = len(DIFFICULTY_ORDER) * pill_w + (len(DIFFICULTY_ORDER) - 1) * 16
    start_x = WIDTH // 2 - total_pills_w // 2
    mouse_pos = p.mouse.get_pos()

    for idx, level in enumerate(DIFFICULTY_ORDER):
        rect = p.Rect(start_x + idx * (pill_w + 16), 184, pill_w, pill_h)
        is_sel = (level == selected_difficulty)
        hovered = rect.collidepoint(mouse_pos)
        bg = (80, 155, 110) if is_sel else ((65, 95, 160) if hovered else (38, 52, 82))
        p.draw.rect(screen, bg, rect, border_radius=10)
        p.draw.rect(screen, (220, 240, 255) if is_sel else (140, 165, 210), rect, 2, border_radius=10)
        txt = font_diff.render(level, True, p.Color("white"))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
        diff_rects.append((level, rect))

    mode_options: List[Tuple[str, str, Any]] = [
        ("white", "1. Play as White vs AI", p.Rect(WIDTH // 2 - 230, 270, 460, 68)),
        ("black", "2. Play as Black vs AI", p.Rect(WIDTH // 2 - 230, 358, 460, 68)),
        ("pvp", "3. Player vs Player", p.Rect(WIDTH // 2 - 230, 446, 460, 68)),
        ("aivai", "4. Watch AI vs AI", p.Rect(WIDTH // 2 - 230, 534, 460, 68)),
    ]

    mode_rects: List[Tuple[str, Any]] = []
    for mode_key, text, rect in mode_options:
        hovered = rect.collidepoint(mouse_pos)
        bg_color = (60, 105, 205) if hovered else (40, 60, 100)
        p.draw.rect(screen, bg_color, rect, border_radius=12)
        p.draw.rect(screen, (200, 220, 255), rect, 2, border_radius=12)
        text_surf = font_options.render(text, True, p.Color("white"))
        screen.blit(
            text_surf,
            (
                rect.centerx - text_surf.get_width() // 2,
                rect.centery - text_surf.get_height() // 2,
            ),
        )
        mode_rects.append((mode_key, rect))

    return mode_rects, diff_rects
