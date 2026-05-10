import math
import os

import pygame as p
from config import ANIMATION_DURATION, DIMENSION, HEIGHT, SQUARE_SIZE, WIDTH


def load_images() -> dict[str, p.Surface]:
    pieces = ["wB", "wK", "wN", "wp", "wQ", "wR", "bB", "bK", "bN", "bp", "bQ", "bR"]
    images = {}
    for piece in pieces:
        path = os.path.join("Chess", "images", f"{piece}.png")
        images[piece] = p.transform.scale(
            p.image.load(path), (SQUARE_SIZE, SQUARE_SIZE)
        )
    return images


def draw_game_state(
    screen,
    board,
    images,
    selected_sq,
    valid_moves,
    gs,
    animating_move=None,
    animation_progress=0,
) -> None:
    draw_board(screen)
    draw_labels(screen, p.font.SysFont("Arial", 18))
    highlight_squares(screen, selected_sq, valid_moves, gs)

    # Draw pieces with animation
    if animating_move:
        draw_pieces_with_animation(
            screen, board, images, animating_move, animation_progress
        )
    else:
        draw_pieces(screen, board, images)


def animate_move(screen, board, images, move, gs, clock):
    start_time = p.time.get_ticks()

    start_x = move.startCol * SQUARE_SIZE
    start_y = move.startRow * SQUARE_SIZE
    end_x = move.endCol * SQUARE_SIZE
    end_y = move.endRow * SQUARE_SIZE

    # Store the piece being moved
    piece = move.pieceMoved

    # Temporarily remove piece from board for animation
    temp_board = board.copy()
    temp_board[move.startRow][move.startCol] = "."

    while True:
        current_time = p.time.get_ticks()
        elapsed = current_time - start_time

        if elapsed >= ANIMATION_DURATION:
            break

        # Calculate animation progress (0 to 1)
        progress = elapsed / ANIMATION_DURATION
        # Use easing function for smoother animation
        progress = ease_in_out(progress)

        # Calculate current position
        current_x = start_x + (end_x - start_x) * progress
        # 3D Lift animation arc
        lift_height = math.sin(progress * math.pi) * (SQUARE_SIZE * 0.8)
        current_y = start_y + (end_y - start_y) * progress - lift_height

        # Draw everything
        draw_board(screen)
        draw_labels(screen, p.font.SysFont("Arial", 18))

        # Draw all pieces except the moving one
        draw_pieces(screen, temp_board, images)

        # Draw dynamic shadow for animating piece
        shadow_base_y = start_y + (end_y - start_y) * progress
        shadow_scale = 1.0 - (math.sin(progress * math.pi) * 0.4)

        shadow_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
        s_width = int(SQUARE_SIZE * 0.7 * shadow_scale)
        s_height = int(SQUARE_SIZE * 0.3 * shadow_scale)
        s_x = (SQUARE_SIZE - s_width) // 2
        s_y = SQUARE_SIZE - s_height - 5
        p.draw.ellipse(
            shadow_surf,
            (0, 0, 0, int(80 * shadow_scale)),
            p.Rect(s_x, s_y, s_width, s_height),
        )
        screen.blit(shadow_surf, (current_x, shadow_base_y))

        # Draw the moving piece at current position
        screen.blit(images[piece], (current_x, current_y))

        # Handle special moves during animation
        if move.isCastleMove:
            animate_castling_rook(screen, images, move, progress)

        p.display.flip()
        clock.tick(60)  # 60 FPS for smooth animation


def animate_castling_rook(screen, images, move, progress):
    if move.endCol - move.startCol == 2:  # King side castle
        rook_start_col = 7
        rook_end_col = 5
    else:  # Queen side castle
        rook_start_col = 0
        rook_end_col = 3

    rook_piece = "wR" if move.pieceMoved == "wK" else "bR"

    # Calculate rook positions
    rook_start_x = rook_start_col * SQUARE_SIZE
    rook_end_x = rook_end_col * SQUARE_SIZE
    rook_y = move.startRow * SQUARE_SIZE

    # Current rook position
    current_rook_x = rook_start_x + (rook_end_x - rook_start_x) * progress

    screen.blit(images[rook_piece], (current_rook_x, rook_y))


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


def draw_board(screen) -> None:
    light_color = p.Color(240, 217, 181)
    dark_color = p.Color(181, 136, 99)
    colors = [light_color, dark_color]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            base_color = colors[(r + c) & 1]
            rect = p.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            p.draw.rect(screen, base_color, rect)

            # Bevel effect for depth
            highlight = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
            # Top and left bright edges
            p.draw.polygon(
                highlight,
                (255, 255, 255, 60),
                [
                    (0, 0),
                    (SQUARE_SIZE, 0),
                    (SQUARE_SIZE - 4, 4),
                    (4, 4),
                    (4, SQUARE_SIZE - 4),
                    (0, SQUARE_SIZE),
                ],
            )
            # Bottom and right dark edges
            p.draw.polygon(
                highlight,
                (0, 0, 0, 40),
                [
                    (SQUARE_SIZE, SQUARE_SIZE),
                    (0, SQUARE_SIZE),
                    (4, SQUARE_SIZE - 4),
                    (SQUARE_SIZE - 4, SQUARE_SIZE - 4),
                    (SQUARE_SIZE - 4, 4),
                    (SQUARE_SIZE, 0),
                ],
            )
            screen.blit(highlight, rect.topleft)


def draw_drop_shadow(screen, x, y):
    shadow = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
    shadow_width = int(SQUARE_SIZE * 0.7)
    shadow_height = int(SQUARE_SIZE * 0.3)
    shadow_x = (SQUARE_SIZE - shadow_width) // 2
    shadow_y = SQUARE_SIZE - shadow_height - 5
    p.draw.ellipse(
        shadow, (0, 0, 0, 80), p.Rect(shadow_x, shadow_y, shadow_width, shadow_height)
    )
    screen.blit(shadow, (x, y))


def draw_pieces(screen, board, images) -> None:
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != ".":
                x = c * SQUARE_SIZE
                y = r * SQUARE_SIZE
                draw_drop_shadow(screen, x, y)
                screen.blit(images[piece], (x, y))


def draw_pieces_with_animation(screen, board, images, animating_move, progress):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "." and not (
                r == animating_move.startRow and c == animating_move.startCol
            ):
                x = c * SQUARE_SIZE
                y = r * SQUARE_SIZE
                draw_drop_shadow(screen, x, y)
                screen.blit(images[piece], (x, y))

    # Draw the animating piece
    start_x = animating_move.startCol * SQUARE_SIZE
    start_y = animating_move.startRow * SQUARE_SIZE
    end_x = animating_move.endCol * SQUARE_SIZE
    end_y = animating_move.endRow * SQUARE_SIZE

    current_x = start_x + (end_x - start_x) * progress
    lift_height = math.sin(progress * math.pi) * (SQUARE_SIZE * 0.8)
    current_y = start_y + (end_y - start_y) * progress - lift_height

    shadow_base_y = start_y + (end_y - start_y) * progress
    shadow_scale = 1.0 - (math.sin(progress * math.pi) * 0.4)

    shadow_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
    s_width = int(SQUARE_SIZE * 0.7 * shadow_scale)
    s_height = int(SQUARE_SIZE * 0.3 * shadow_scale)
    s_x = (SQUARE_SIZE - s_width) // 2
    s_y = SQUARE_SIZE - s_height - 5
    p.draw.ellipse(
        shadow_surf,
        (0, 0, 0, int(80 * shadow_scale)),
        p.Rect(s_x, s_y, s_width, s_height),
    )
    screen.blit(shadow_surf, (current_x, shadow_base_y))

    screen.blit(images[animating_move.pieceMoved], (current_x, current_y))


def highlight_squares(screen, selected_sq, valid_moves, gs) -> None:
    if selected_sq:
        r, c = selected_sq
        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
        # Highlight selected piece with a subtle glow
        p.draw.circle(
            s,
            (100, 200, 255, 120),
            (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
            SQUARE_SIZE // 2 - 4,
        )
        screen.blit(s, (c * SQUARE_SIZE, r * SQUARE_SIZE))

        # Valid moves indicators (dots or open circles)
        for move in valid_moves:
            if move.startRow == r and move.startCol == c:
                er = move.endRow
                ec = move.endCol
                move_surf = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
                if gs.board[er][ec] != ".":
                    p.draw.circle(
                        move_surf,
                        (255, 100, 100, 200),
                        (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                        SQUARE_SIZE // 2 - 5,
                        4,
                    )
                else:
                    p.draw.circle(
                        move_surf,
                        (150, 255, 150, 180),
                        (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                        SQUARE_SIZE // 6,
                    )
                screen.blit(move_surf, (ec * SQUARE_SIZE, er * SQUARE_SIZE))

    if gs.inCheck():
        king_row, king_col = gs.whiteKingLoc if gs.white_to_move else gs.blackKingLoc
        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE), p.SRCALPHA)
        # Intense red glow for check
        for i in range(3):
            p.draw.circle(
                s,
                (255, 0, 0, 80 - i * 20),
                (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                SQUARE_SIZE // 2 - i * 2,
            )
        screen.blit(s, (king_col * SQUARE_SIZE, king_row * SQUARE_SIZE))

    if gs.checkMate() == "checkmate":
        s = p.Surface((WIDTH, HEIGHT))
        s.set_alpha(100)
        s.fill(p.Color("green"))
        screen.blit(s, (0, 0))
        text = p.font.SysFont("Times New Roman", 40).render(
            "Checkmate! " + ("White" if not gs.white_to_move else "Black") + " wins",
            True,
            p.Color("black"),
        )
        screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )

    if gs.checkMate() == "stalemate":
        s = p.Surface((WIDTH, HEIGHT))
        s.set_alpha(100)
        s.fill(p.Color("green"))
        screen.blit(s, (0, 0))
        text = p.font.SysFont("Times New Roman", 50).render(
            "Stalemate!", True, p.Color("black")
        )
        screen.blit(
            text,
            (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2),
        )


def draw_labels(screen, font) -> None:
    for i in range(DIMENSION):
        label = font.render(chr(ord("a") + i), True, p.Color("black"))
        screen.blit(
            label,
            (
                i * SQUARE_SIZE + SQUARE_SIZE // 2 - label.get_width() // 2,
                HEIGHT - label.get_height(),
            ),
        )
        # Ranks (1-8)
        label = font.render(str(DIMENSION - i), True, p.Color("black"))
        screen.blit(
            label, (0, i * SQUARE_SIZE + SQUARE_SIZE // 2 - label.get_height() // 2)
        )


def draw_menu(screen) -> None:
    for y in range(HEIGHT):
        color = (max(20, 40 - y // 20), max(30, 50 - y // 20), max(50, 80 - y // 15))
        p.draw.line(screen, color, (0, y), (WIDTH, y))

    font_title = p.font.SysFont("Helvetica", 46, bold=True)
    font_options = p.font.SysFont("Helvetica", 24, bold=True)

    title_text = font_title.render("CHESS", True, p.Color("white"))
    # Add title drop shadow
    title_shadow = font_title.render("CHESS", True, p.Color("black"))
    screen.blit(
        title_shadow, (WIDTH // 2 - title_text.get_width() // 2 + 3, HEIGHT // 4 + 3)
    )
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

    options = [
        ("1. Player vs Player", p.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 40, 300, 50)),
        ("2. Player vs AI", p.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 30, 300, 50)),
        ("3. AI vs AI", p.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 50)),
    ]

    for text, rect in options:
        # Check mouse hover
        mouse_pos = p.mouse.get_pos()
        hovered = rect.collidepoint(mouse_pos)

        bg_color = (60, 100, 200) if hovered else (40, 60, 100)
        p.draw.rect(screen, bg_color, rect, border_radius=12)
        p.draw.rect(screen, (200, 220, 255), rect, 2, border_radius=12)  # border

        text_surf = font_options.render(text, True, p.Color("white"))
        screen.blit(
            text_surf,
            (
                rect.x + rect.width // 2 - text_surf.get_width() // 2,
                rect.y + rect.height // 2 - text_surf.get_height() // 2,
            ),
        )

    return options


def animate_capture(screen, images, captured_piece, position, clock):
    start_time = p.time.get_ticks()
    fade_duration = 200

    while True:
        current_time = p.time.get_ticks()
        elapsed = current_time - start_time

        if elapsed >= fade_duration:
            break

        progress = elapsed / fade_duration
        alpha = int(255 * (1 - progress))

        # Create fading piece surface
        piece_surface = images[captured_piece].copy()
        piece_surface.set_alpha(alpha)

        screen.blit(piece_surface, position)
        p.display.flip()
        clock.tick(60)


def animate_check_warning(screen, king_pos, clock):
    start_time = p.time.get_ticks()
    pulse_duration = 500

    while True:
        current_time = p.time.get_ticks()
        elapsed = current_time - start_time

        if elapsed >= pulse_duration:
            break

        progress = elapsed / pulse_duration
        # Create pulsing effect
        alpha = int(100 + 100 * math.sin(progress * math.pi * 4))

        s = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
        s.set_alpha(alpha)
        s.fill(p.Color("red"))
        screen.blit(s, (king_pos[1] * SQUARE_SIZE, king_pos[0] * SQUARE_SIZE))

        p.display.flip()
        clock.tick(60)
