import copy
import os
from typing import List, Optional

from .types import (
    Color, PieceType, Piece, Square, MoveFlag,
    CastlingRights, STARTING_FEN, square_name, parse_square,
)
from .move import Move
from .bitboard import (
    lsb, iter_bits, KNIGHT_ATTACKS, KING_ATTACKS,
    rook_attacks, bishop_attacks, queen_attacks,
)
from .zobrist import compute_hash

try:
    from ._engine import lib as _lib, ffi as _ffi
    _lib.init_engine()
    HAS_CFFI = True
except ImportError:
    _lib = None
    _ffi = None
    HAS_CFFI = False


def _should_use_c() -> bool:
    return HAS_CFFI and os.environ.get("CASTLED_PURE_PYTHON", "0") != "1"


class Board:
    """Chess board supporting both the fast C backend and a pure-Python fallback."""

    __slots__ = (
        '_c_board', '_use_c', '_move_stack',
        'pieces', 'colors', 'turn', 'castling_rights',
        'ep_square', 'halfmove_clock', 'fullmove_number', '_history',
    )

    def __init__(self, fen: str = STARTING_FEN, *, _use_c: Optional[bool] = None):
        self._c_board = None
        self._use_c = _should_use_c() if _use_c is None else (_use_c and HAS_CFFI)
        self._move_stack: List[Move] = []

        self.pieces = [0] * 12
        self.colors = [0, 0]
        self.turn = Color.WHITE
        self.castling_rights = 0
        self.ep_square = Square.NONE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._history = []

        if self._use_c:
            self._c_board = _lib.board_new()
            if fen and fen != STARTING_FEN:
                _lib.board_from_fen(self._c_board, fen.encode('utf-8'))
            self._sync_scalars_from_c()
        elif fen:
            self._set_fen(fen)

    def _sync_scalars_from_c(self):
        """Sync lightweight scalar attributes from the C board."""
        self.turn = Color(_lib.board_turn(self._c_board))
        self.castling_rights = _lib.board_castling(self._c_board)
        self.ep_square = _lib.board_ep_square(self._c_board)
        self.halfmove_clock = _lib.board_halfmove_clock(self._c_board)
        self.fullmove_number = _lib.board_fullmove_number(self._c_board)

    def _set_fen(self, fen: str):
        """Parse a FEN string into the pure-Python bitboard state."""
        self.pieces = [0] * 12
        self.colors = [0, 0]

        parts = fen.split()
        if len(parts) < 4:
            return

        char_to_piece = {
            'P': Piece.WHITE_PAWN, 'N': Piece.WHITE_KNIGHT, 'B': Piece.WHITE_BISHOP,
            'R': Piece.WHITE_ROOK, 'Q': Piece.WHITE_QUEEN, 'K': Piece.WHITE_KING,
            'p': Piece.BLACK_PAWN, 'n': Piece.BLACK_KNIGHT, 'b': Piece.BLACK_BISHOP,
            'r': Piece.BLACK_ROOK, 'q': Piece.BLACK_QUEEN, 'k': Piece.BLACK_KING,
        }

        for r, row in enumerate(parts[0].split('/')):
            rank = 7 - r
            f = 0
            for char in row:
                if char.isdigit():
                    f += int(char)
                else:
                    sq = rank * 8 + f
                    p = char_to_piece.get(char)
                    if p is not None:
                        self.pieces[p.value] |= (1 << sq)
                        self.colors[p.color.value] |= (1 << sq)
                    f += 1

        self.turn = Color.WHITE if parts[1] == 'w' else Color.BLACK

        self.castling_rights = 0
        if 'K' in parts[2]:
            self.castling_rights |= CastlingRights.WHITE_KINGSIDE
        if 'Q' in parts[2]:
            self.castling_rights |= CastlingRights.WHITE_QUEENSIDE
        if 'k' in parts[2]:
            self.castling_rights |= CastlingRights.BLACK_KINGSIDE
        if 'q' in parts[2]:
            self.castling_rights |= CastlingRights.BLACK_QUEENSIDE

        self.ep_square = parse_square(parts[3])
        self.halfmove_clock = int(parts[4]) if len(parts) >= 5 else 0
        self.fullmove_number = int(parts[5]) if len(parts) >= 6 else 1

    @classmethod
    def from_fen(cls, fen: str) -> 'Board':
        return cls(fen)

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

    def piece_at(self, sq: int) -> Optional[Piece]:
        if sq < 0 or sq >= 64:
            return None
        if self._use_c:
            pc = _lib.board_piece_at(self._c_board, sq)
            return None if pc == 12 else Piece(pc)
        for p in range(12):
            if self.pieces[p] & (1 << sq):
                return Piece(p)
        return None

    def _remove_piece(self, sq: int):
        for p in range(12):
            if self.pieces[p] & (1 << sq):
                self.pieces[p] &= ~(1 << sq)
                self.colors[p // 6] &= ~(1 << sq)
                return p
        return None

    def _add_piece(self, p: int, sq: int):
        self.pieces[p] |= (1 << sq)
        self.colors[p // 6] |= (1 << sq)

    @property
    def occupied(self) -> int:
        if self._use_c:
            occ = 0
            for sq in range(64):
                if _lib.board_piece_at(self._c_board, sq) != 12:
                    occ |= (1 << sq)
            return occ
        return self.colors[0] | self.colors[1]

    def push(self, move: Move):
        """Apply a move to the board and record it on the history stack."""
        self._move_stack.append(move)

        if self._use_c:
            _lib.board_make_move(self._c_board, move._data)
            self._sync_scalars_from_c()
            return

        self._history.append((
            list(self.pieces), list(self.colors), self.turn,
            self.castling_rights, self.ep_square,
            self.halfmove_clock, self.fullmove_number,
        ))

        f = move.from_square
        t = move.to_square
        flags = move.flags

        p = self._remove_piece(f)
        captured = self._remove_piece(t)
        if p is None:
            return

        c = p // 6
        self.ep_square = 64
        self.halfmove_clock = 0 if (captured is not None or (p % 6) == 0) else self.halfmove_clock + 1

        new_p = p
        if flags in (MoveFlag.PROMO_Q, MoveFlag.CAPTURE_PROMO_Q):
            new_p = c * 6 + 4
        elif flags in (MoveFlag.PROMO_R, MoveFlag.CAPTURE_PROMO_R):
            new_p = c * 6 + 3
        elif flags in (MoveFlag.PROMO_B, MoveFlag.CAPTURE_PROMO_B):
            new_p = c * 6 + 2
        elif flags in (MoveFlag.PROMO_N, MoveFlag.CAPTURE_PROMO_N):
            new_p = c * 6 + 1

        self._add_piece(new_p, t)

        if flags == MoveFlag.DOUBLE_PAWN:
            self.ep_square = f + 8 if c == 0 else f - 8
        elif flags == MoveFlag.EP_CAPTURE:
            cap_sq = t - 8 if c == 0 else t + 8
            self._remove_piece(cap_sq)
        elif flags == MoveFlag.KING_CASTLE:
            r_f, r_t = (7, 5) if c == 0 else (63, 61)
            self._remove_piece(r_f)
            self._add_piece(c * 6 + 3, r_t)
        elif flags == MoveFlag.QUEEN_CASTLE:
            r_f, r_t = (0, 3) if c == 0 else (56, 59)
            self._remove_piece(r_f)
            self._add_piece(c * 6 + 3, r_t)

        cr = self.castling_rights
        if f == 4 or t == 4:
            cr &= ~(CastlingRights.WHITE_KINGSIDE | CastlingRights.WHITE_QUEENSIDE)
        if f == 60 or t == 60:
            cr &= ~(CastlingRights.BLACK_KINGSIDE | CastlingRights.BLACK_QUEENSIDE)
        if f == 0 or t == 0:
            cr &= ~CastlingRights.WHITE_QUEENSIDE
        if f == 7 or t == 7:
            cr &= ~CastlingRights.WHITE_KINGSIDE
        if f == 56 or t == 56:
            cr &= ~CastlingRights.BLACK_QUEENSIDE
        if f == 63 or t == 63:
            cr &= ~CastlingRights.BLACK_KINGSIDE
        self.castling_rights = cr

        if self.turn == Color.BLACK:
            self.fullmove_number += 1
        self.turn = self.turn.flip()

    def pop(self) -> Move:
        """Undo the last move and return it."""
        if not self._move_stack:
            return Move.NULL

        move = self._move_stack.pop()

        if self._use_c:
            _lib.board_unmake_move(self._c_board)
            self._sync_scalars_from_c()
            return move

        pieces, colors, turn, castling, ep, half, full = self._history.pop()
        self.pieces = pieces
        self.colors = colors
        self.turn = turn
        self.castling_rights = castling
        self.ep_square = ep
        self.halfmove_clock = half
        self.fullmove_number = full
        return move

    def _attacks_to(self, sq: int, color: Color) -> int:
        if self._use_c:
            return int(_lib.board_is_attacked_by(self._c_board, sq, color.value))

        c = color.value
        occ = self.colors[0] | self.colors[1]

        pawns = self.pieces[c * 6 + 0]
        p_attacks = 0
        if c == 0:
            if (1 << sq) & ((pawns << 9) & 0xfefefefefefefefe):
                p_attacks = 1
            if (1 << sq) & ((pawns << 7) & 0x7f7f7f7f7f7f7f7f):
                p_attacks = 1
        else:
            if (1 << sq) & ((pawns >> 7) & 0xfefefefefefefefe):
                p_attacks = 1
            if (1 << sq) & ((pawns >> 9) & 0x7f7f7f7f7f7f7f7f):
                p_attacks = 1

        knights = self.pieces[c * 6 + 1]
        k_attacks = KNIGHT_ATTACKS[sq] & knights

        kings = self.pieces[c * 6 + 5]
        kg_attacks = KING_ATTACKS[sq] & kings

        bishops_queens = self.pieces[c * 6 + 2] | self.pieces[c * 6 + 4]
        b_attacks = bishop_attacks(sq, occ) & bishops_queens

        rooks_queens = self.pieces[c * 6 + 3] | self.pieces[c * 6 + 4]
        r_attacks = rook_attacks(sq, occ) & rooks_queens

        return p_attacks | k_attacks | kg_attacks | b_attacks | r_attacks

    def is_check(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_check(self._c_board))
        c = self.turn.value
        kings = self.pieces[c * 6 + 5]
        if not kings:
            return False
        return bool(self._attacks_to(lsb(kings), self.turn.flip()))

    @property
    def pseudo_legal_moves(self) -> List[Move]:
        if self._use_c:
            buf = _ffi.new("uint16_t[256]")
            count = _lib.board_generate_pseudo_legal_moves(self._c_board, buf)
            return [Move(buf[i]) for i in range(count)]

        moves = []
        c = self.turn.value
        occ = self.colors[0] | self.colors[1]
        us = self.colors[c]
        them = self.colors[1 - c]

        pawns = self.pieces[c * 6 + 0]
        if c == 0:
            p_up1 = (pawns << 8) & ~occ
            p_up2 = ((p_up1 & 0x0000000000FF0000) << 8) & ~occ
            p_cap_r = (pawns << 9) & 0xfefefefefefefefe & them
            p_cap_l = (pawns << 7) & 0x7f7f7f7f7f7f7f7f & them

            for t in iter_bits(p_up1):
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

            for t in iter_bits(p_up2):
                moves.append(Move.make(t - 16, t, MoveFlag.DOUBLE_PAWN))

            for t in iter_bits(p_cap_r):
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

            for t in iter_bits(p_cap_l):
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
                ep_cap_r = (pawns << 9) & 0xfefefefefefefefe & (1 << self.ep_square)
                ep_cap_l = (pawns << 7) & 0x7f7f7f7f7f7f7f7f & (1 << self.ep_square)
                if ep_cap_r:
                    moves.append(Move.make(self.ep_square - 9, self.ep_square, MoveFlag.EP_CAPTURE))
                if ep_cap_l:
                    moves.append(Move.make(self.ep_square - 7, self.ep_square, MoveFlag.EP_CAPTURE))
        else:
            p_up1 = (pawns >> 8) & ~occ
            p_up2 = ((p_up1 & 0x0000FF0000000000) >> 8) & ~occ
            p_cap_r = (pawns >> 7) & 0xfefefefefefefefe & them
            p_cap_l = (pawns >> 9) & 0x7f7f7f7f7f7f7f7f & them

            for t in iter_bits(p_up1):
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

            for t in iter_bits(p_up2):
                moves.append(Move.make(t + 16, t, MoveFlag.DOUBLE_PAWN))

            for t in iter_bits(p_cap_r):
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

            for t in iter_bits(p_cap_l):
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
                ep_cap_r = (pawns >> 7) & 0xfefefefefefefefe & (1 << self.ep_square)
                ep_cap_l = (pawns >> 9) & 0x7f7f7f7f7f7f7f7f & (1 << self.ep_square)
                if ep_cap_r:
                    moves.append(Move.make(self.ep_square + 7, self.ep_square, MoveFlag.EP_CAPTURE))
                if ep_cap_l:
                    moves.append(Move.make(self.ep_square + 9, self.ep_square, MoveFlag.EP_CAPTURE))

        knights = self.pieces[c * 6 + 1]
        for f in iter_bits(knights):
            atts = KNIGHT_ATTACKS[f] & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        bishops = self.pieces[c * 6 + 2]
        for f in iter_bits(bishops):
            atts = bishop_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        rooks = self.pieces[c * 6 + 3]
        for f in iter_bits(rooks):
            atts = rook_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        queens = self.pieces[c * 6 + 4]
        for f in iter_bits(queens):
            atts = queen_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

        kings = self.pieces[c * 6 + 5]
        if kings:
            f = lsb(kings)
            atts = KING_ATTACKS[f] & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1 << t) & them else MoveFlag.QUIET))

            if c == 0:
                if self.castling_rights & CastlingRights.WHITE_KINGSIDE:
                    if not (occ & 0x0000000000000060):
                        if not self._attacks_to(4, Color.BLACK) and not self._attacks_to(5, Color.BLACK) and not self._attacks_to(6, Color.BLACK):
                            moves.append(Move.make(4, 6, MoveFlag.KING_CASTLE))
                if self.castling_rights & CastlingRights.WHITE_QUEENSIDE:
                    if not (occ & 0x000000000000000E):
                        if not self._attacks_to(4, Color.BLACK) and not self._attacks_to(3, Color.BLACK) and not self._attacks_to(2, Color.BLACK):
                            moves.append(Move.make(4, 2, MoveFlag.QUEEN_CASTLE))
            else:
                if self.castling_rights & CastlingRights.BLACK_KINGSIDE:
                    if not (occ & 0x6000000000000000):
                        if not self._attacks_to(60, Color.WHITE) and not self._attacks_to(61, Color.WHITE) and not self._attacks_to(62, Color.WHITE):
                            moves.append(Move.make(60, 62, MoveFlag.KING_CASTLE))
                if self.castling_rights & CastlingRights.BLACK_QUEENSIDE:
                    if not (occ & 0x0E00000000000000):
                        if not self._attacks_to(60, Color.WHITE) and not self._attacks_to(59, Color.WHITE) and not self._attacks_to(58, Color.WHITE):
                            moves.append(Move.make(60, 58, MoveFlag.QUEEN_CASTLE))

        return moves

    @property
    def legal_moves(self) -> List[Move]:
        if self._use_c:
            buf = _ffi.new("uint16_t[256]")
            count = _lib.board_generate_legal_moves(self._c_board, buf)
            return [Move(buf[i]) for i in range(count)]

        moves = []
        for move in self.pseudo_legal_moves:
            self.push(move)
            self.turn = self.turn.flip()
            if not self.is_check():
                moves.append(move)
            self.turn = self.turn.flip()
            self.pop()
        return moves

    def is_checkmate(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_checkmate(self._c_board))
        return self.is_check() and len(self.legal_moves) == 0

    def is_stalemate(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_stalemate(self._c_board))
        return not self.is_check() and len(self.legal_moves) == 0

    def is_game_over(self) -> bool:
        if self._use_c:
            return bool(_lib.board_is_game_over(self._c_board))
        return len(self.legal_moves) == 0 or self.halfmove_clock >= 100

    @property
    def fen(self) -> str:
        if self._use_c:
            buf = _ffi.new("char[128]")
            _lib.board_to_fen(self._c_board, buf, 128)
            return _ffi.string(buf).decode("utf-8")

        s = ""
        for r in range(7, -1, -1):
            emp = 0
            for f in range(8):
                sq = r * 8 + f
                p = self.piece_at(sq)
                if p is None:
                    emp += 1
                else:
                    if emp:
                        s += str(emp)
                        emp = 0
                    s += p.symbol
            if emp:
                s += str(emp)
            if r > 0:
                s += "/"

        s += " " + ("w" if self.turn == Color.WHITE else "b") + " "

        c = ""
        if self.castling_rights & CastlingRights.WHITE_KINGSIDE:
            c += "K"
        if self.castling_rights & CastlingRights.WHITE_QUEENSIDE:
            c += "Q"
        if self.castling_rights & CastlingRights.BLACK_KINGSIDE:
            c += "k"
        if self.castling_rights & CastlingRights.BLACK_QUEENSIDE:
            c += "q"
        s += c if c else "-"

        s += " " + (square_name(self.ep_square) if self.ep_square != 64 else "-")
        s += f" {self.halfmove_clock} {self.fullmove_number}"
        return s

    @property
    def zobrist_hash(self) -> int:
        if self._use_c:
            return int(_lib.board_hash(self._c_board))
        return compute_hash(self)

    def perft(self, depth: int) -> int:
        if self._use_c:
            return int(_lib.board_perft(self._c_board, depth))
        if depth == 0:
            return 1
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
            b.pieces = [0] * 12
            b.colors = [0, 0]
            b._history = []
            b._sync_scalars_from_c()
            return b

        b = Board(self.fen, _use_c=False)
        b._history = copy.deepcopy(self._history)
        b._move_stack = list(self._move_stack)
        return b

    def parse_uci(self, uci_str: str) -> Move:
        m = Move.from_uci(uci_str)
        for legal in self.legal_moves:
            if legal.from_square == m.from_square and legal.to_square == m.to_square:
                if m.is_promotion() and m.promotion_piece() != legal.promotion_piece():
                    continue
                return legal
        raise ValueError(f"Invalid UCI move: {uci_str}")

    def san(self, move: Move) -> str:
        return move.to_uci()

    def parse_san(self, san_str: str) -> Move:
        return self.parse_uci(san_str)

    def __str__(self):
        return self.fen

    def __repr__(self):
        return f"Board('{self.fen}')"

