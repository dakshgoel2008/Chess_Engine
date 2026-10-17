"""Configuration constants and difficulty presets for the CastledEngine Pygame GUI."""

DIMENSION = 8
BOARD_SIZE = 768
SQUARE_SIZE = BOARD_SIZE // DIMENSION
EVAL_BAR_WIDTH = 44
SIDEBAR_WIDTH = 388

WIDTH = EVAL_BAR_WIDTH + BOARD_SIZE + SIDEBAR_WIDTH
HEIGHT = BOARD_SIZE
MAX_FPS = 60
ANIMATION_DURATION = 220

# Difficulty presets mapping level names to search depth and time limit in milliseconds.
DIFFICULTY_LEVELS: dict[str, dict[str, int | str]] = {
    "Easy": {"depth": 2, "movetime_ms": 150, "label": "Easy (Depth 2)"},
    "Medium": {"depth": 4, "movetime_ms": 400, "label": "Medium (Depth 4)"},
    "Hard": {"depth": 6, "movetime_ms": 1000, "label": "Hard (Depth 6)"},
    "Master": {"depth": 8, "movetime_ms": 2000, "label": "Master (Depth 8)"},
}

DIFFICULTY_ORDER = ["Easy", "Medium", "Hard", "Master"]

# Color definitions for move review classifications.
CLASSIFICATION_COLORS: dict[str, tuple[int, int, int]] = {
    "Best": (92, 184, 92),
    "Good": (130, 190, 110),
    "Inaccuracy": (240, 195, 65),
    "Mistake": (235, 135, 55),
    "Blunder": (220, 65, 65),
}
