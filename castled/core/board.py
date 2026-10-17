"""Board representation, move generation, game-over rules, and SAN notation."""

from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

from .types import (
    Color, PieceType, Piece, Square, MoveFlag,
    CastlingRights, STARTING_FEN, square_name, parse_square, file_of, rank_of,
)
from .move import Move
from .bitboard import (
    lsb, popcount, iter_bits,
    FILE_A, FILE_H, NOT_FILE_A, NOT_FILE_H, RANK_3, RANK_6,
    KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS,
    rook_attacks, bishop_attacks, queen_attacks,
)
from .zobrist import PIECE_KEYS, CASTLING_KEYS, EP_KEYS, SIDE_KEY, compute_hash

try:
    from ._engine import lib as _lib, ffi as _ffi
    _lib.init_engine()
    HAS_CFFI = True
except ImportError:
    _lib = None
    _ffi = None
    HAS_CFFI = False


# Precomputed castling rights mask for each square (clears rights when king/rook moves or is captured).
CASTLING_UPDATE = [15] * 64
CASTLING_UPDATE[Square.A1] = ~CastlingRights.WHITE_QUEENSIDE & 15
CASTLING_UPDATE[Square.E1] = ~(CastlingRights.WHITE_KINGSIDE | CastlingRights.WHITE_QUEENSIDE) & 15
CASTLING_UPDATE[Square.H1] = ~CastlingRights.WHITE_KINGSIDE & 15
CASTLING_UPDATE[Square.A8] = ~CastlingRights.BLACK_QUEENSIDE & 15
CASTLING_UPDATE[Square.E8] = ~(CastlingRights.BLACK_KINGSIDE | CastlingRights.BLACK_QUEENSIDE) & 15
CASTLING_UPDATE[Square.H8] = ~CastlingRights.BLACK_KINGSIDE & 15

_CHAR_TO_PIECE = {
    'P': Piece.WHITE_PAWN, 'N': Piece.WHITE_KNIGHT, 'B': Piece.WHITE_BISHOP,
    'R': Piece.WHITE_ROOK, 'Q': Piece.WHITE_QUEEN, 'K': Piece.WHITE_KING,
    'p': Piece.BLACK_PAWN, 'n': Piece.BLACK_KNIGHT, 'b': Piece.BLACK_BISHOP,
    'r': Piece.BLACK_ROOK, 'q': Piece.BLACK_QUEEN, 'k': Piece.BLACK_KING,
}

_SAN_PIECE_CHARS = {
    PieceType.KNIGHT: 'N', PieceType.BISHOP: 'B',
    PieceType.ROOK: 'R', PieceType.QUEEN: 'Q', PieceType.KING: 'K',
}


@dataclass(frozen=True)
class Outcome:
    """Result of a finished game."""
    termination: str
    winner: Optional[Color]

    @property
    def result(self) -> str:
        if self.winner == Color.WHITE:
            return "1-0"
        if self.winner == Color.BLACK:
            return "0-1"
        return "1/2-1/2"


def _should_use_c() -> bool:
    return HAS_CFFI and os.environ.get("CASTLED_PURE_PYTHON", "0") != "1"


class Board:
    """Chess board supporting both the fast C backend and a pure-Python fallback."""

    __slots__ = (
        '_c_board', '_use_c', '_move_stack',
        'pieces', 'colors', 'mailbox', 'turn', 'castling_rights',
        'ep_square', 'halfmove_clock', 'fullmove_number',
        '_hash', '_history', '_hash_history',
    )

    def __init__(self, fen: str = STARTING_FEN, *, _use_c: Optional[bool] = None):
        self._c_board = None
        self._use_c = _should_use_c() if _use_c is None else (_use_c and HAS_CFFI)
        self._move_stack: List[Move] = []

        self.pieces = [0] * 12
        self.colors = [0, 0]
        self.mailbox = [Piece.NONE] * 64
        self.turn = Color.WHITE
        self.castling_rights = 0
        self.ep_square = Square.NONE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._hash = 0
        self._history: List[Tuple[Move, int, int, int, int, int]] = []
        self._hash_history: List[int] = []

        if self._use_c:
            self._c_board = _lib.board_new()
            if fen and fen != STARTING_FEN:
                _lib.board_from_fen(self._c_board, fen.encode('utf-8'))
            self._sync_scalars_from_c()
        elif fen:
            self._set_fen(fen)

    def _sync_scalars_from_c(self):
        """Sync scalar attributes from the C board into Python fields."""
        self.turn = Color(_lib.board_turn(self._c_board))
        self.castling_rights = _lib.board_castling(self._c_board)
        self.ep_square = _lib.board_ep_square(self._c_board)
        self.halfmove_clock = _lib.board_halfmove_clock(self._c_board)
        self.fullmove_number = _lib.board_fullmove_number(self._c_board)
        self._hash = int(_lib.board_hash(self._c_board))

    def _set_fen(self, fen: str):
        """Parse a FEN string into bitboards, mailbox, and Zobrist hash."""
        self.pieces = [0] * 12
        self.colors = [0, 0]
        self.mailbox = [Piece.NONE] * 64

        parts = fen.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN string: {fen}")

        for r, row in enumerate(parts[0].split('/')):
            rank = 7 - r
            f = 0
            for char in row:
                if char.isdigit():
                    f += int(char)
                else:
                    sq = rank * 8 + f
                    p = _CHAR_TO_PIECE.get(char)
                    if p is None:
                        raise ValueError(f"Invalid piece character '{char}' in FEN: {fen}")
                    self.pieces[p.value] |= (1 << sq)
                    self.colors[p.color.value] |= (1 << sq)
                    self.mailbox[sq] = p
                    f += 1

        self.turn = Color.WHITE if parts[1] == 'w' else Color.BLACK

        cr = 0
        if 'K' in parts[2]:
            cr |= CastlingRights.WHITE_KINGSIDE
        if 'Q' in parts[2]:
            cr |= CastlingRights.WHITE_QUEENSIDE
        if 'k' in parts[2]:
            cr |= CastlingRights.BLACK_KINGSIDE
        if 'q' in parts[2]:
            cr |= CastlingRights.BLACK_QUEENSIDE
        self.castling_rights = cr

        self.ep_square = parse_square(parts[3])
        self.halfmove_clock = int(parts[4]) if len(parts) >= 5 else 0
        self.fullmove_number = int(parts[5]) if len(parts) >= 6 else 1
        self._hash = compute_hash(self)
        self._history.clear()
        self._hash_history.clear()
        self._move_stack.clear()

    @classmethod
    def from_fen(cls, fen: str) -> 'Board':
        return cls(fen)

    def to_fen(self) -> str:
        return self.fen

    def __del__(self):
        if self._c_board is not None and _lib is not None:
            _lib.board_free(self._c_board)
            self._c_board = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._c_board is not None and _lib is not None:
            _lib.board_free(self._c_board)
            self._c_board = None

    # --- Piece and square queries ---

    def piece_at(self, sq: int) -> Optional[Piece]:
        """Return the Piece on sq (0-63), or None if empty."""
        if sq < 0 or sq >= 64:
            return None
        if self._use_c:
            pc = _lib.board_piece_at(self._c_board, sq)
            return None if pc == 12 else Piece(pc)
        p = self.mailbox[sq]
        return None if p == Piece.NONE else Piece(p)

    def piece_type_at(self, sq: int) -> Optional[PieceType]:
        """Return the PieceType on sq (0-63), or None if empty."""
        p = self.piece_at(sq)
        return None if p is None else p.piece_type

    def color_at(self, sq: int) -> Optional[Color]:
        """Return the Color of the piece on sq (0-63), or None if empty."""
        p = self.piece_at(sq)
        return None if p is None else p.color

    @property
    def occupied(self) -> int:
        if self._use_c:
            occ = 0
            for sq in range(64):
                if _lib.board_piece_at(self._c_board, sq) != 12:
                    occ |= (1 << sq)
            return occ
        return self.colors[0] | self.colors[1]

    @property
    def empty(self) -> int:
        return ~self.occupied & 0xFFFFFFFFFFFFFFFF

    def _put_piece(self, pc: int, sq: int):
        bb = 1 << sq
        self.pieces[pc] |= bb
        self.colors[pc // 6] |= bb
        self.mailbox[sq] = Piece(pc)
        self._hash ^= PIECE_KEYS[pc][sq]

    def _remove_piece(self, sq: int) -> int:
        pc = int(self.mailbox[sq])
        if pc == 12:
            return 12
        bb = 1 << sq
        self.pieces[pc] &= ~bb
        self.colors[pc // 6] &= ~bb
        self.mailbox[sq] = Piece.NONE
        self._hash ^= PIECE_KEYS[pc][sq]
        return pc

    # --- Make and unmake moves ---

    def push(self, move: Move):
        """Execute a move and update state and Zobrist hash incrementally."""
        self._move_stack.append(move)
        self._hash_history.append(self.zobrist_hash)

        if self._use_c:
            _lib.board_make_move(self._c_board, move._data)
            self._sync_scalars_from_c()
            return

        from_sq = move.from_square
        to_sq = move.to_square
        flags = move.flags
        us = self.turn.value

        captured = int(self.mailbox[to_sq])
        self._history.append((
            move, self.castling_rights, self.ep_square,
            self.halfmove_clock, self._hash, captured,
        ))

        pc = int(self.mailbox[from_sq])
        if pc == 12:
            return

        if self.ep_square != Square.NONE:
            self._hash ^= EP_KEYS[self.ep_square % 8]
        self.ep_square = Square.NONE

        self._hash ^= CASTLING_KEYS[self.castling_rights]
        self.castling_rights &= CASTLING_UPDATE[from_sq] & CASTLING_UPDATE[to_sq]
        self._hash ^= CASTLING_KEYS[self.castling_rights]

        if captured != 12 or (pc % 6) == PieceType.PAWN:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self._remove_piece(from_sq)
        if captured != 12:
            self._remove_piece(to_sq)

        if flags in (MoveFlag.QUIET, MoveFlag.CAPTURE, MoveFlag.DOUBLE_PAWN):
            self._put_piece(pc, to_sq)
            if flags == MoveFlag.DOUBLE_PAWN:
                self.ep_square = to_sq - 8 if us == 0 else to_sq + 8
                self._hash ^= EP_KEYS[self.ep_square % 8]
        elif flags == MoveFlag.EP_CAPTURE:
            self._put_piece(pc, to_sq)
            cap_sq = to_sq - 8 if us == 0 else to_sq + 8
            ep_cap = self._remove_piece(cap_sq)
            self._history[-1] = (
                move, self._history[-1][1], self._history[-1][2],
                self._history[-1][3], self._history[-1][4], ep_cap,
            )
        elif flags == MoveFlag.KING_CASTLE:
            self._put_piece(pc, to_sq)
            r_from, r_to = (Square.H1, Square.F1) if us == 0 else (Square.H8, Square.F8)
            self._remove_piece(r_from)
            self._put_piece(us * 6 + PieceType.ROOK, r_to)
        elif flags == MoveFlag.QUEEN_CASTLE:
            self._put_piece(pc, to_sq)
            r_from, r_to = (Square.A1, Square.D1) if us == 0 else (Square.A8, Square.D8)
            self._remove_piece(r_from)
            self._put_piece(us * 6 + PieceType.ROOK, r_to)
        else:
            promo_type = (flags & 3) + 1
            self._put_piece(us * 6 + promo_type, to_sq)

        if self.turn == Color.BLACK:
            self.fullmove_number += 1
        self.turn = self.turn.flip()
        self._hash ^= SIDE_KEY

    def pop(self) -> Move:
        """Undo the last move and return it."""
        if not self._move_stack:
            return Move.NULL

        move = self._move_stack.pop()
        self._hash_history.pop()

        if self._use_c:
            _lib.board_unmake_move(self._c_board)
            self._sync_scalars_from_c()
            return move

        move, castling, ep, half, prev_hash, captured = self._history.pop()
        us = self.turn.flip().value
        if us == Color.BLACK:
            self.fullmove_number -= 1
        self.turn = Color(us)

        from_sq = move.from_square
        to_sq = move.to_square
        flags = move.flags

        moved_pc = int(self.mailbox[to_sq])
        self._remove_piece(to_sq)
        if flags >= MoveFlag.PROMO_N:
            self._put_piece(us * 6 + PieceType.PAWN, from_sq)
        else:
            self._put_piece(moved_pc, from_sq)

        if flags == MoveFlag.CAPTURE or flags >= MoveFlag.CAPTURE_PROMO_N:
            self._put_piece(captured, to_sq)
        elif flags == MoveFlag.EP_CAPTURE:
            cap_sq = to_sq - 8 if us == 0 else to_sq + 8
            self._put_piece(captured, cap_sq)
        elif flags == MoveFlag.KING_CASTLE:
            r_from, r_to = (Square.H1, Square.F1) if us == 0 else (Square.H8, Square.F8)
            self._remove_piece(r_to)
            self._put_piece(us * 6 + PieceType.ROOK, r_from)
        elif flags == MoveFlag.QUEEN_CASTLE:
            r_from, r_to = (Square.A1, Square.D1) if us == 0 else (Square.A8, Square.D8)
            self._remove_piece(r_to)
            self._put_piece(us * 6 + PieceType.ROOK, r_from)

        self.castling_rights = castling
        self.ep_square = ep
        self.halfmove_clock = half
        self._hash = prev_hash
        return move

    # --- Attack and check detection ---

    def attackers_to(self, sq: int, color: Optional[Color] = None) -> int:
        """Return a bitboard of pieces attacking sq (filtered by color if given)."""
        if self._use_c:
            occ = self.occupied
            att = int(_lib.board_attackers_to(self._c_board, sq, occ))
            if color is None:
                return att
            mask = 0
            for s in iter_bits(att):
                if self.color_at(s) == color:
                    mask |= (1 << s)
            return mask

        occ = self.colors[0] | self.colors[1]
        att = 0
        att |= PAWN_ATTACKS[Color.BLACK][sq] & self.pieces[Piece.WHITE_PAWN]
        att |= PAWN_ATTACKS[Color.WHITE][sq] & self.pieces[Piece.BLACK_PAWN]
        att |= KNIGHT_ATTACKS[sq] & (self.pieces[Piece.WHITE_KNIGHT] | self.pieces[Piece.BLACK_KNIGHT])
        att |= KING_ATTACKS[sq] & (self.pieces[Piece.WHITE_KING] | self.pieces[Piece.BLACK_KING])
        bishops_queens = (
            self.pieces[Piece.WHITE_BISHOP] | self.pieces[Piece.BLACK_BISHOP]
            | self.pieces[Piece.WHITE_QUEEN] | self.pieces[Piece.BLACK_QUEEN]
        )
        rooks_queens = (
            self.pieces[Piece.WHITE_ROOK] | self.pieces[Piece.BLACK_ROOK]
            | self.pieces[Piece.WHITE_QUEEN] | self.pieces[Piece.BLACK_QUEEN]
        )
        att |= bishop_attacks(sq, occ) & bishops_queens
        att |= rook_attacks(sq, occ) & rooks_queens

        if color is not None:
            att &= self.colors[color.value]
        return att

    def is_attacked_by(self, color: Color, sq: int) -> bool:
        """Return True if any piece of color attacks sq."""
        if self._use_c:
            return bool(_lib.board_is_attacked_by(self._c_board, sq, color.value))
        return bool(self.attackers_to(sq, color))

    def _attacks_to(self, sq: int, color: Color) -> int:
        return self.attackers_to(sq, color)

    def checkers(self) -> int:
        """Return bitboard of opponent pieces currently giving check."""
        c = self.turn.value
        king_pc = Piece.WHITE_KING if c == 0 else Piece.BLACK_KING
        if self._use_c:
            king_sq = -1
            for sq in range(64):
                if _lib.board_piece_at(self._c_board, sq) == king_pc:
                    king_sq = sq
                    break
            if king_sq == -1:
                return 0
            return self.attackers_to(king_sq, self.turn.flip())

        kings = self.pieces[king_pc]
        if not kings:
            return 0
        return self.attackers_to(lsb(kings), self.turn.flip())

    def is_check(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_check(self._c_board))
        c = self.turn.value
        kings = self.pieces[c * 6 + 5]
        if not kings:
            return False
        return self.is_attacked_by(self.turn.flip(), lsb(kings))

    # --- Move generation ---

    @property
    def pseudo_legal_moves(self) -> List[Move]:
        if self._use_c:
            buf = _ffi.new("uint16_t[256]")
            count = _lib.board_generate_pseudo_legal_moves(self._c_board, buf)
            return [Move(buf[i]) for i in range(count)]

        moves: List[Move] = []
        c = self.turn.value
        occ = self.colors[0] | self.colors[1]
        us = self.colors[c]
        them = self.colors[1 - c]

        # Pawn pushes, captures, promotions, and en passant
        pawns = self.pieces[c * 6 + 0]
        if c == 0:
            up1 = (pawns << 8) & ~occ
            up2 = ((up1 & RANK_3) << 8) & ~occ
            cap_r = (pawns << 9) & NOT_FILE_A & them
            cap_l = (pawns << 7) & NOT_FILE_H & them

            for t in iter_bits(up1):
                f = t - 8
                if t >= 56:
                    moves.extend([
                        Move.make(f, t, MoveFlag.PROMO_Q),
                        Move.make(f, t, MoveFlag.PROMO_R),
                        Move.make(f, t, MoveFlag.PROMO_B),
                        Move.make(f, t, MoveFlag.PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.QUIET))

            for t in iter_bits(up2):
                moves.append(Move.make(t - 16, t, MoveFlag.DOUBLE_PAWN))

            for t in iter_bits(cap_r):
                f = t - 9
                if t >= 56:
                    moves.extend([
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_R),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_B),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))

            for t in iter_bits(cap_l):
                f = t - 7
                if t >= 56:
                    moves.extend([
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_R),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_B),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))

            if self.ep_square != 64:
                ep_mask = 1 << self.ep_square
                if (pawns << 9) & NOT_FILE_A & ep_mask:
                    moves.append(Move.make(self.ep_square - 9, self.ep_square, MoveFlag.EP_CAPTURE))
                if (pawns << 7) & NOT_FILE_H & ep_mask:
                    moves.append(Move.make(self.ep_square - 7, self.ep_square, MoveFlag.EP_CAPTURE))
        else:
            up1 = (pawns >> 8) & ~occ
            up2 = ((up1 & RANK_6) >> 8) & ~occ
            cap_r = (pawns >> 7) & NOT_FILE_A & them
            cap_l = (pawns >> 9) & NOT_FILE_H & them

            for t in iter_bits(up1):
                f = t + 8
                if t <= 7:
                    moves.extend([
                        Move.make(f, t, MoveFlag.PROMO_Q),
                        Move.make(f, t, MoveFlag.PROMO_R),
                        Move.make(f, t, MoveFlag.PROMO_B),
                        Move.make(f, t, MoveFlag.PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.QUIET))

            for t in iter_bits(up2):
                moves.append(Move.make(t + 16, t, MoveFlag.DOUBLE_PAWN))

            for t in iter_bits(cap_r):
                f = t + 7
                if t <= 7:
                    moves.extend([
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_R),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_B),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))

            for t in iter_bits(cap_l):
                f = t + 9
                if t <= 7:
                    moves.extend([
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_R),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_B),
                        Move.make(f, t, MoveFlag.CAPTURE_PROMO_N),
                    ])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))

            if self.ep_square != 64:
                ep_mask = 1 << self.ep_square
                if (pawns >> 7) & NOT_FILE_A & ep_mask:
                    moves.append(Move.make(self.ep_square + 7, self.ep_square, MoveFlag.EP_CAPTURE))
                if (pawns >> 9) & NOT_FILE_H & ep_mask:
                    moves.append(Move.make(self.ep_square + 9, self.ep_square, MoveFlag.EP_CAPTURE))

        # Knights, bishops, rooks, queens
        for f in iter_bits(self.pieces[c * 6 + 1]):
            for t in iter_bits(KNIGHT_ATTACKS[f] & ~us):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        for f in iter_bits(self.pieces[c * 6 + 2]):
            for t in iter_bits(bishop_attacks(f, occ) & ~us):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        for f in iter_bits(self.pieces[c * 6 + 3]):
            for t in iter_bits(rook_attacks(f, occ) & ~us):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        for f in iter_bits(self.pieces[c * 6 + 4]):
            for t in iter_bits(queen_attacks(f, occ) & ~us):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        # King normal moves and castling
        kings = self.pieces[c * 6 + 5]
        if kings:
            f = lsb(kings)
            for t in iter_bits(KING_ATTACKS[f] & ~us):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

            if c == 0:
                if (self.castling_rights & CastlingRights.WHITE_KINGSIDE) and not (occ & 0x60):
                    if not (self.is_attacked_by(Color.BLACK, 4) or self.is_attacked_by(Color.BLACK, 5) or self.is_attacked_by(Color.BLACK, 6)):
                        moves.append(Move.make(4, 6, MoveFlag.KING_CASTLE))
                if (self.castling_rights & CastlingRights.WHITE_QUEENSIDE) and not (occ & 0x0E):
                    if not (self.is_attacked_by(Color.BLACK, 4) or self.is_attacked_by(Color.BLACK, 3) or self.is_attacked_by(Color.BLACK, 2)):
                        moves.append(Move.make(4, 2, MoveFlag.QUEEN_CASTLE))
            else:
                if (self.castling_rights & CastlingRights.BLACK_KINGSIDE) and not (occ & 0x6000000000000000):
                    if not (self.is_attacked_by(Color.WHITE, 60) or self.is_attacked_by(Color.WHITE, 61) or self.is_attacked_by(Color.WHITE, 62)):
                        moves.append(Move.make(60, 62, MoveFlag.KING_CASTLE))
                if (self.castling_rights & CastlingRights.BLACK_QUEENSIDE) and not (occ & 0x0E00000000000000):
                    if not (self.is_attacked_by(Color.WHITE, 60) or self.is_attacked_by(Color.WHITE, 59) or self.is_attacked_by(Color.WHITE, 58)):
                        moves.append(Move.make(60, 58, MoveFlag.QUEEN_CASTLE))

        return moves

    @property
    def legal_moves(self) -> List[Move]:
        if self._use_c:
            buf = _ffi.new("uint16_t[256]")
            count = _lib.board_generate_legal_moves(self._c_board, buf)
            return [Move(buf[i]) for i in range(count)]

        moves: List[Move] = []
        us = self.turn
        them = us.flip()
        king_pc = Piece.WHITE_KING if us == Color.WHITE else Piece.BLACK_KING

        for move in self.pseudo_legal_moves:
            self.push(move)
            kings = self.pieces[king_pc]
            if kings and not self.is_attacked_by(them, lsb(kings)):
                moves.append(move)
            self.pop()
        return moves

    def generate_captures(self) -> List[Move]:
        """Return all legal capture and promotion moves."""
        if self._use_c:
            buf = _ffi.new("uint16_t[256]")
            count = _lib.board_generate_captures(self._c_board, buf)
            return [Move(buf[i]) for i in range(count)]

        moves: List[Move] = []
        us = self.turn
        them = us.flip()
        king_pc = Piece.WHITE_KING if us == Color.WHITE else Piece.BLACK_KING

        for move in self.pseudo_legal_moves:
            if not (move.is_capture() or move.is_promotion()):
                continue
            self.push(move)
            kings = self.pieces[king_pc]
            if kings and not self.is_attacked_by(them, lsb(kings)):
                moves.append(move)
            self.pop()
        return moves

    # --- Terminal state and draw rules ---

    def is_checkmate(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_checkmate(self._c_board))
        return self.is_check() and len(self.legal_moves) == 0

    def is_stalemate(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_stalemate(self._c_board))
        return not self.is_check() and len(self.legal_moves) == 0

    def is_fifty_moves(self) -> bool:
        return self.halfmove_clock >= 100

    def is_repetition(self, count: int = 3) -> bool:
        """Return True if the current position has appeared at least `count` times."""
        current = self.zobrist_hash
        prior_matches = sum(1 for h in self._hash_history if h == current)
        return (prior_matches + 1) >= count

    def is_insufficient_material(self) -> bool:
        """Return True when neither side has enough material to force checkmate."""
        if self._use_c:
            return bool(_lib.board_is_insufficient_material(self._c_board))

        if (
            self.pieces[Piece.WHITE_PAWN] or self.pieces[Piece.BLACK_PAWN]
            or self.pieces[Piece.WHITE_ROOK] or self.pieces[Piece.BLACK_ROOK]
            or self.pieces[Piece.WHITE_QUEEN] or self.pieces[Piece.BLACK_QUEEN]
        ):
            return False

        wn = popcount(self.pieces[Piece.WHITE_KNIGHT])
        bn = popcount(self.pieces[Piece.BLACK_KNIGHT])
        wb = popcount(self.pieces[Piece.WHITE_BISHOP])
        bb = popcount(self.pieces[Piece.BLACK_BISHOP])
        total_minors = wn + bn + wb + bb

        if total_minors <= 1:
            return True

        if total_minors == 2 and wn == 0 and bn == 0 and wb == 1 and bb == 1:
            w_sq = lsb(self.pieces[Piece.WHITE_BISHOP])
            b_sq = lsb(self.pieces[Piece.BLACK_BISHOP])
            return ((rank_of(w_sq) + file_of(w_sq)) & 1) == ((rank_of(b_sq) + file_of(b_sq)) & 1)

        return False

    def is_draw(self) -> bool:
        return self.is_fifty_moves() or self.is_repetition(3) or self.is_insufficient_material()

    def is_game_over(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_game_over(self._c_board))
        return len(self.legal_moves) == 0 or self.is_draw()

    def outcome(self) -> Optional[Outcome]:
        """Return an Outcome if the game has ended, or None if still in progress."""
        legal = self.legal_moves
        if not legal:
            if self.is_check():
                return Outcome("checkmate", self.turn.flip())
            return Outcome("stalemate", None)
        if self.is_fifty_moves():
            return Outcome("fifty_moves", None)
        if self.is_repetition(3):
            return Outcome("repetition", None)
        if self.is_insufficient_material():
            return Outcome("insufficient_material", None)
        return None

    # --- Serialization and hashing ---

    @property
    def fen(self) -> str:
        if self._use_c:
            buf = _ffi.new("char[128]")
            _lib.board_to_fen(self._c_board, buf, 128)
            return _ffi.string(buf).decode("utf-8")

        rows = []
        for r in range(7, -1, -1):
            row = ""
            empty = 0
            for f in range(8):
                p = self.piece_at(r * 8 + f)
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p.symbol
            if empty:
                row += str(empty)
            rows.append(row)

        turn_str = "w" if self.turn == Color.WHITE else "b"

        c = ""
        if self.castling_rights & CastlingRights.WHITE_KINGSIDE:
            c += "K"
        if self.castling_rights & CastlingRights.WHITE_QUEENSIDE:
            c += "Q"
        if self.castling_rights & CastlingRights.BLACK_KINGSIDE:
            c += "k"
        if self.castling_rights & CastlingRights.BLACK_QUEENSIDE:
            c += "q"
        castling_str = c if c else "-"

        ep_str = square_name(self.ep_square) if self.ep_square != 64 else "-"
        return f"{'/'.join(rows)} {turn_str} {castling_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"

    @property
    def zobrist_hash(self) -> int:
        if self._use_c:
            return int(_lib.board_hash(self._c_board))
        return self._hash

    def perft(self, depth: int) -> int:
        if self._use_c:
            return int(_lib.board_perft(self._c_board, depth))
        if depth == 0:
            return 1
        if depth == 1:
            return len(self.legal_moves)
        nodes = 0
        for m in self.legal_moves:
            self.push(m)
            nodes += self.perft(depth - 1)
            self.pop()
        return nodes

    def copy(self) -> 'Board':
        if self._use_c:
            b = Board.__new__(Board)
            b._use_c = True
            b._c_board = _lib.board_copy(self._c_board)
            b._move_stack = list(self._move_stack)
            b._hash_history = list(self._hash_history)
            b.pieces = [0] * 12
            b.colors = [0, 0]
            b.mailbox = [Piece.NONE] * 64
            b._history = []
            b._sync_scalars_from_c()
            return b

        b = Board(self.fen, _use_c=False)
        b._history = list(self._history)
        b._hash_history = list(self._hash_history)
        b._move_stack = list(self._move_stack)
        return b

    # --- UCI and SAN move notation ---

    def parse_uci(self, uci_str: str) -> Move:
        """Parse a UCI move string (e.g. 'e2e4') and return the matching legal Move."""
        m = Move.from_uci(uci_str.strip())
        for legal in self.legal_moves:
            if legal.from_square == m.from_square and legal.to_square == m.to_square:
                if m.is_promotion():
                    if m.promotion_piece() != legal.promotion_piece():
                        continue
                elif legal.is_promotion():
                    continue
                return legal
        raise ValueError(f"Illegal or invalid UCI move: {uci_str}")

    def _san_without_suffix(self, move: Move, legal: List[Move]) -> str:
        if move.flags == MoveFlag.KING_CASTLE:
            return "O-O"
        if move.flags == MoveFlag.QUEEN_CASTLE:
            return "O-O-O"

        from_sq = move.from_square
        to_sq = move.to_square
        pt = self.piece_type_at(from_sq)
        if pt is None:
            raise ValueError(f"No piece on starting square for move {move}")

        is_cap = move.is_capture()
        dest = square_name(to_sq)

        if pt == PieceType.PAWN:
            prefix = f"{chr(ord('a') + file_of(from_sq))}x" if is_cap else ""
            promo = ""
            if move.is_promotion():
                promo = f"={_SAN_PIECE_CHARS[move.promotion_piece()]}"
            return f"{prefix}{dest}{promo}"

        piece_letter = _SAN_PIECE_CHARS[pt]
        ambiguous = [
            m for m in legal
            if m != move and m.to_square == to_sq and self.piece_type_at(m.from_square) == pt
        ]

        disambig = ""
        if ambiguous:
            same_file = any(file_of(m.from_square) == file_of(from_sq) for m in ambiguous)
            same_rank = any(rank_of(m.from_square) == rank_of(from_sq) for m in ambiguous)
            if not same_file:
                disambig = chr(ord('a') + file_of(from_sq))
            elif not same_rank:
                disambig = str(rank_of(from_sq) + 1)
            else:
                disambig = square_name(from_sq)

        cap_str = "x" if is_cap else ""
        return f"{piece_letter}{disambig}{cap_str}{dest}"

    def san(self, move: Move) -> str:
        """Convert a legal Move into Standard Algebraic Notation (with +/# suffix)."""
        legal = self.legal_moves
        if move not in legal:
            raise ValueError(f"Illegal move {move} in position {self.fen}")

        base = self._san_without_suffix(move, legal)
        self.push(move)
        suffix = ""
        if self.is_check():
            suffix = "#" if len(self.legal_moves) == 0 else "+"
        self.pop()
        return base + suffix

    def parse_san(self, san_str: str) -> Move:
        """Parse a SAN string (or UCI string) and return the matching legal Move."""
        cleaned = san_str.strip()
        while cleaned and cleaned[-1] in "+#?!":
            cleaned = cleaned[:-1]
        cleaned = cleaned.replace("0-0-0", "O-O-O").replace("0-0", "O-O")

        legal = self.legal_moves
        for move in legal:
            base = self._san_without_suffix(move, legal)
            if base == cleaned or base.replace("x", "") == cleaned:
                return move

        try:
            return self.parse_uci(cleaned)
        except ValueError:
            pass

        raise ValueError(f"Illegal or unrecognized SAN move: {san_str}")

    def __str__(self):
        return self.fen

    def __repr__(self):
        return f"Board('{self.fen}')"


