"""Core constants, piece/square enums, and coordinate helpers."""

from enum import IntEnum


class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    def flip(self) -> 'Color':
        return Color(1 - self.value)


class PieceType(IntEnum):
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5


# Piece values 0-5 are White, 6-11 are Black, 12 is empty.
class Piece(IntEnum):
    WHITE_PAWN = 0
    WHITE_KNIGHT = 1
    WHITE_BISHOP = 2
    WHITE_ROOK = 3
    WHITE_QUEEN = 4
    WHITE_KING = 5
    BLACK_PAWN = 6
    BLACK_KNIGHT = 7
    BLACK_BISHOP = 8
    BLACK_ROOK = 9
    BLACK_QUEEN = 10
    BLACK_KING = 11
    NONE = 12

    @property
    def color(self) -> Color:
        return Color(self.value // 6) if self.value != 12 else None

    @property
    def piece_type(self) -> PieceType:
        return PieceType(self.value % 6) if self.value != 12 else None

    @property
    def symbol(self) -> str:
        if self.value == 12:
            return '.'
        return PIECE_SYMBOLS[self.value]


# Little-Endian Rank-File mapping: A1 = 0, H1 = 7, A8 = 56, H8 = 63.
class Square(IntEnum):
    A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
    A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
    A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
    A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
    A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
    A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
    A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
    A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)
    NONE = 64


def square_name(sq: int) -> str:
    """Convert square index (0-63) to algebraic name like 'e4'."""
    if sq == 64 or sq < 0 or sq > 63:
        return "-"
    file = sq % 8
    rank = sq // 8
    return f"{chr(ord('a') + file)}{rank + 1}"


def parse_square(name: str) -> int:
    """Convert algebraic name like 'e4' to square index (0-63)."""
    if name == "-" or len(name) != 2:
        return Square.NONE
    file = ord(name[0]) - ord('a')
    rank = int(name[1]) - 1
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return rank * 8 + file
    return Square.NONE


def file_of(sq: int) -> int:
    return sq % 8


def rank_of(sq: int) -> int:
    return sq // 8


def mirror(sq: int) -> int:
    """Flip square vertically (rank 1 <-> rank 8)."""
    return sq ^ 56


# 4-bit move flags: bit 2 = capture, bit 3 = promotion.
class MoveFlag(IntEnum):
    QUIET = 0
    DOUBLE_PAWN = 1
    KING_CASTLE = 2
    QUEEN_CASTLE = 3
    CAPTURE = 4
    EP_CAPTURE = 5
    PROMO_N = 8
    PROMO_B = 9
    PROMO_R = 10
    PROMO_Q = 11
    CAPTURE_PROMO_N = 12
    CAPTURE_PROMO_B = 13
    CAPTURE_PROMO_R = 14
    CAPTURE_PROMO_Q = 15


class CastlingRights:
    WHITE_KINGSIDE = 1
    WHITE_QUEENSIDE = 2
    BLACK_KINGSIDE = 4
    BLACK_QUEENSIDE = 8


PIECE_SYMBOLS = 'PNBRQKpnbrqk.'
PIECE_NAMES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

