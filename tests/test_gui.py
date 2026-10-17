"""Tests for the CastledEngine GUI controller, coordinate mapping, eval bar, PGN I/O, and Game Review."""

from pathlib import Path

from castled.core.board import Board
from castled.core.types import Color, PieceType
from castled.search.transposition import SCORE_MATE
from gui.main import GameSession
from gui.review import (
    board_to_grid,
    classify_move_cpl,
    eval_to_bar_fraction,
    export_game_pgn,
    format_eval_label,
    grid_coords_to_square,
    import_game_pgn,
    review_game,
    square_to_grid_coords,
    summarize_review,
)


def test_coordinate_and_grid_mapping():
    for flipped in (False, True):
        for sq in range(64):
            r, c = square_to_grid_coords(sq, flipped=flipped)
            assert 0 <= r < 8 and 0 <= c < 8
            assert grid_coords_to_square(r, c, flipped=flipped) == sq

    board = Board()
    grid = board_to_grid(board)
    assert grid[0][0] == "bR"
    assert grid[0][4] == "bK"
    assert grid[1][3] == "bp"
    assert grid[6][4] == "wp"
    assert grid[7][4] == "wK"
    assert grid[4][4] == "."


def test_eval_bar_math_and_formatting():
    assert abs(eval_to_bar_fraction(0) - 0.5) < 1e-6
    assert eval_to_bar_fraction(300) > 0.70
    assert eval_to_bar_fraction(-300) < 0.30
    assert eval_to_bar_fraction(SCORE_MATE - 3) == 1.0
    assert eval_to_bar_fraction(-(SCORE_MATE - 2)) == 0.0

    assert format_eval_label(0) == "+0.00"
    assert format_eval_label(150) == "+1.50"
    assert format_eval_label(-75) == "-0.75"
    assert format_eval_label(SCORE_MATE - 3) == "+M2"
    assert format_eval_label(-(SCORE_MATE - 2)) == "-M1"


def test_move_classification_and_game_review():
    assert classify_move_cpl(0, is_best_move=True) == "Best"
    assert classify_move_cpl(8, is_best_move=False) == "Best"
    assert classify_move_cpl(25, is_best_move=False) == "Good"
    assert classify_move_cpl(60, is_best_move=False) == "Inaccuracy"
    assert classify_move_cpl(140, is_best_move=False) == "Mistake"
    assert classify_move_cpl(350, is_best_move=False) == "Blunder"

    # Fool's Mate (4 plies)
    fools_mate = ["f2f3", "e7e5", "g2g4", "d8h4"]
    items = review_game(fools_mate, depth=2)
    assert len(items) == 4
    assert items[2].mover_color == int(Color.WHITE)
    assert items[2].classification in ("Mistake", "Blunder")
    assert items[3].san == "Qh4#"

    summary = summarize_review(items)
    assert sum(summary[int(Color.WHITE)].values()) == 2
    assert sum(summary[int(Color.BLACK)].values()) == 2


def test_pgn_export_and_import_roundtrip():
    uci_moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    pgn_str = export_game_pgn(uci_moves, white_name="Human", black_name="CastledEngine")
    assert '[White "Human"]' in pgn_str
    assert "1. e4 e5 2. Nf3 Nc6" in pgn_str

    fen, loaded_moves = import_game_pgn(pgn_str)
    assert loaded_moves == uci_moves
    assert fen == Board().fen


def test_game_session_clicks_promotion_ai_undo_and_pgn(tmp_path: Path):
    session = GameSession(mode="pvp", difficulty="Easy", hash_mb=8)
    assert session.is_human_turn()

    # Click e2 (sq 12) then e4 (sq 28)
    assert session.select_or_move(12) is None
    assert session.selected_sq == 12
    move = session.select_or_move(28)
    assert move is not None and move.to_uci() == "e2e4"
    session.apply_move(move)
    assert session.san_moves == ["e4"]

    # AI move computation
    best_move, res = session.compute_ai_move()
    assert best_move is not None
    session.apply_move(best_move, search_result=res)
    assert len(session.uci_moves) == 2

    # Save and reload PGN
    pgn_file = tmp_path / "test_game.pgn"
    session.save_pgn_file(str(pgn_file))
    assert pgn_file.exists()

    session2 = GameSession(mode="white", difficulty="Easy", hash_mb=8)
    assert session2.load_pgn_file(str(pgn_file))
    assert len(session2.uci_moves) == 2

    # Undo in 'white' mode on human's turn undoes both AI + player move
    assert session2.undo()
    assert len(session2.uci_moves) == 0

    # Promotion dialog flow
    promo_fen = "8/P7/8/8/8/8/7k/4K3 w - - 0 1"
    session.reset(start_fen=promo_fen)
    a7 = (6 << 3) | 0
    a8 = (7 << 3) | 0
    assert session.select_or_move(a7) is None
    assert session.select_or_move(a8) is None
    assert len(session.pending_promotions) == 4
    chosen = session.resolve_promotion(PieceType.KNIGHT)
    assert chosen is not None and chosen.to_uci() == "a7a8n"


def test_headless_pygame_rendering_and_cli_play_help(monkeypatch):
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-m", "castled.cli", "play", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Launch the CastledEngine Pygame GUI" in res.stdout

    try:
        import pygame as p
    except ImportError:
        return

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from gui.config import HEIGHT, WIDTH
    from gui.ui import (
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

    p.init()
    try:
        screen = p.display.set_mode((WIDTH, HEIGHT))
        images = load_images()
        font = p.font.SysFont("Arial", 14, bold=True)
        session = GameSession(mode="white", difficulty="Easy", hash_mb=8)

        mode_rects, diff_rects = draw_menu(screen, selected_difficulty="Easy")
        assert len(mode_rects) == 4 and len(diff_rects) == 4

        draw_eval_bar(screen, session.white_score_cp, flipped=False)
        draw_board(screen)
        draw_labels(screen, font, flipped=False)
        highlight_squares(screen, session.board, 12, session.board.legal_moves, flipped=False)
        draw_pieces(screen, session.board, images, flipped=False)
        btns = draw_sidebar(screen, session.sidebar_state())
        assert "review" in btns and "undo" in btns

        hitboxes = draw_promotion_dialog(screen, 0, images)
        assert len(hitboxes) == 4

        session.apply_move(session.board.parse_uci("e2e4"))
        session.run_review(depth=1)
        draw_sidebar(screen, session.sidebar_state())
        draw_game_over_banner(screen, session.board)
    finally:
        p.quit()


def test_nnue_toggle_and_web_demo_assets():
    session = GameSession(mode="white", difficulty="Easy", hash_mb=8, use_nnue=True)
    assert session.use_nnue is True
    assert session.sidebar_state()["use_nnue"] is True

    assert session.toggle_nnue() is False
    assert session.use_nnue is False
    assert session.sidebar_state()["use_nnue"] is False

    assert session.toggle_nnue() is True
    assert session.use_nnue is True

    root = Path(__file__).resolve().parent.parent
    web_html = root / "web" / "index.html"
    main_py = root / "main.py"
    assert web_html.exists()
    assert "CastledEngine" in web_html.read_text(encoding="utf-8")
    assert main_py.exists()


