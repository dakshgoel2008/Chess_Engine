"""
Move representation using a 16-bit packed integer.

Layout: [flags:4][to_sq:6][from_sq:6]
  - bits 0-5:   source square (0-63)
  - bits 6-11:  target square (0-63)
  - bits 12-15: move flags (quiet, capture, promotion, etc.)
"""

from .types import PieceType, MoveFlag, square_name, parse_square


class Move:
    """A single chess move packed into a 16-bit int."""
    __slots__ = ('_data',)

    def __init__(self, data: int):
        self._data = data

    @classmethod
    def make(cls, from_sq: int, to_sq: int, flags: int = 0) -> 'Move':
        return cls((from_sq & 0x3F) | ((to_sq & 0x3F) << 6) | ((flags & 0xF) << 12))

    # -- properties --

    @property
    def from_square(self) -> int:
        return self._data & 0x3F

    @property
    def to_square(self) -> int:
        return (self._data >> 6) & 0x3F

    @property
    def flags(self) -> int:
        return (self._data >> 12) & 0xF

    # -- flag checks --

    def is_capture(self) -> bool:
        # bit 2 of flags is set for all captures
        return (self.flags & 4) != 0

    def is_promotion(self) -> bool:
        # bit 3 of flags is set for all promotions
        return (self.flags & 8) != 0

    def is_castling(self) -> bool:
        return self.flags in (MoveFlag.KING_CASTLE, MoveFlag.QUEEN_CASTLE)

    def is_en_passant(self) -> bool:
        return self.flags == MoveFlag.EP_CAPTURE

    def promotion_piece(self):
        """Returns the PieceType we're promoting to, or None."""
        if not self.is_promotion():
            return None
        # bottom 2 bits of flags encode the promotion piece:
        # 0 -> KNIGHT(1), 1 -> BISHOP(2), 2 -> ROOK(3), 3 -> QUEEN(4)
        return PieceType((self.flags & 3) + 1)

    # -- UCI string conversion --

    def to_uci(self) -> str:
        """Convert to UCI notation like 'e2e4' or 'e7e8q'."""
        if self._data == 0:
            return "0000"
        s = square_name(self.from_square) + square_name(self.to_square)
        if self.is_promotion():
            promo_chars = {PieceType.KNIGHT: 'n', PieceType.BISHOP: 'b',
                           PieceType.ROOK: 'r', PieceType.QUEEN: 'q'}
            s += promo_chars.get(self.promotion_piece(), 'q')
        return s

    @classmethod
    def from_uci(cls, uci_str: str) -> 'Move':
        """Parse a UCI string like 'e2e4' or 'e7e8q'."""
        if uci_str == "0000":
            return cls(0)

        from_sq = parse_square(uci_str[0:2])
        to_sq = parse_square(uci_str[2:4])
        flags = MoveFlag.QUIET

        if len(uci_str) == 5:
            promo_map = {'n': MoveFlag.PROMO_N, 'b': MoveFlag.PROMO_B,
                         'r': MoveFlag.PROMO_R, 'q': MoveFlag.PROMO_Q}
            flags = promo_map.get(uci_str[4].lower(), MoveFlag.PROMO_Q)

        return cls.make(from_sq, to_sq, flags)

    # -- standard python stuff --

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self._data == other._data

    def __hash__(self):
        return hash(self._data)

    def __int__(self) -> int:
        return self._data

    def __repr__(self):
        return f"Move.from_uci('{self.to_uci()}')"

    def __str__(self):
        return self.to_uci()


# null move constant
Move.NULL = Move(0)
