from typing import List, Optional
import copy
from .types import Color, PieceType, Piece, Square, MoveFlag, CastlingRights, STARTING_FEN, square_name, parse_square
from .move import Move
from .bitboard import lsb, pop_lsb, popcount, iter_bits, KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, rook_attacks, bishop_attacks, queen_attacks
from .zobrist import compute_hash

try:
    from ._engine import lib as _lib, ffi as _ffi
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

class Board:
    __slots__ = ('_c_board', 'pieces', 'colors', 'turn', 'castling_rights', 'ep_square', 'halfmove_clock', 'fullmove_number', '_history')
    
    def __init__(self, fen: str = STARTING_FEN):
        self._c_board = None
        
        self.pieces = [0] * 12
        self.colors = [0, 0]
        self.turn = Color.WHITE
        self.castling_rights = 0
        self.ep_square = Square.NONE
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        self._history = []
        
        if HAS_CFFI:
            self._c_board = _lib.board_new()
            if fen != STARTING_FEN:
                _lib.board_free(self._c_board)
                self._c_board = _lib.board_from_fen(fen.encode('utf-8'))
        
        if fen:
            self._set_fen(fen)

    def _set_fen(self, fen: str):
        self.pieces = [0] * 12
        self.colors = [0, 0]
        
        parts = fen.split()
        if len(parts) < 4: return
            
        rows = parts[0].split('/')
        for r, row in enumerate(rows):
            rank = 7 - r
            f = 0
            for char in row:
                if char.isdigit():
                    f += int(char)
                else:
                    sq = rank * 8 + f
                    p = None
                    if char == 'P': p = Piece.WHITE_PAWN
                    elif char == 'N': p = Piece.WHITE_KNIGHT
                    elif char == 'B': p = Piece.WHITE_BISHOP
                    elif char == 'R': p = Piece.WHITE_ROOK
                    elif char == 'Q': p = Piece.WHITE_QUEEN
                    elif char == 'K': p = Piece.WHITE_KING
                    elif char == 'p': p = Piece.BLACK_PAWN
                    elif char == 'n': p = Piece.BLACK_KNIGHT
                    elif char == 'b': p = Piece.BLACK_BISHOP
                    elif char == 'r': p = Piece.BLACK_ROOK
                    elif char == 'q': p = Piece.BLACK_QUEEN
                    elif char == 'k': p = Piece.BLACK_KING
                    
                    if p is not None:
                        self.pieces[p.value] |= (1 << sq)
                        self.colors[p.color.value] |= (1 << sq)
                    f += 1
                    
        self.turn = Color.WHITE if parts[1] == 'w' else Color.BLACK
        
        self.castling_rights = 0
        if 'K' in parts[2]: self.castling_rights |= CastlingRights.WHITE_KINGSIDE
        if 'Q' in parts[2]: self.castling_rights |= CastlingRights.WHITE_QUEENSIDE
        if 'k' in parts[2]: self.castling_rights |= CastlingRights.BLACK_KINGSIDE
        if 'q' in parts[2]: self.castling_rights |= CastlingRights.BLACK_QUEENSIDE
        
        self.ep_square = parse_square(parts[3])
        
        if len(parts) >= 5: self.halfmove_clock = int(parts[4])
        if len(parts) >= 6: self.fullmove_number = int(parts[5])

    @classmethod
    def from_fen(cls, fen: str) -> 'Board':
        return cls(fen)
        
    def __del__(self):
        if HAS_CFFI and self._c_board:
            _lib.board_free(self._c_board)
            self._c_board = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if HAS_CFFI and self._c_board:
            _lib.board_free(self._c_board)
            self._c_board = None

    def piece_at(self, sq: int) -> Optional[Piece]:
        if sq == 64: return None
        for p in range(12):
            if (self.pieces[p] & (1 << sq)):
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
        return self.colors[0] | self.colors[1]
        
    def push(self, move: Move):
        self._history.append((
            list(self.pieces), list(self.colors), self.turn,
            self.castling_rights, self.ep_square,
            self.halfmove_clock, self.fullmove_number
        ))
        
        f = move.from_square
        t = move.to_square
        flags = move.flags
        
        p = self._remove_piece(f)
        captured = self._remove_piece(t)
        
        if p is None: # Should not happen in legal move
            return
            
        c = p // 6
        
        self.ep_square = 64
        self.halfmove_clock += 1
        
        if captured is not None or (p % 6) == 0:
            self.halfmove_clock = 0
            
        new_p = p
        if flags == MoveFlag.PROMO_Q or flags == MoveFlag.CAPTURE_PROMO_Q: new_p = c * 6 + 4
        elif flags == MoveFlag.PROMO_R or flags == MoveFlag.CAPTURE_PROMO_R: new_p = c * 6 + 3
        elif flags == MoveFlag.PROMO_B or flags == MoveFlag.CAPTURE_PROMO_B: new_p = c * 6 + 2
        elif flags == MoveFlag.PROMO_N or flags == MoveFlag.CAPTURE_PROMO_N: new_p = c * 6 + 1
        
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
            
        # Update castling rights
        cr = self.castling_rights
        if f == 4 or t == 4: cr &= ~(CastlingRights.WHITE_KINGSIDE | CastlingRights.WHITE_QUEENSIDE)
        if f == 60 or t == 60: cr &= ~(CastlingRights.BLACK_KINGSIDE | CastlingRights.BLACK_QUEENSIDE)
        if f == 0 or t == 0: cr &= ~CastlingRights.WHITE_QUEENSIDE
        if f == 7 or t == 7: cr &= ~CastlingRights.WHITE_KINGSIDE
        if f == 56 or t == 56: cr &= ~CastlingRights.BLACK_QUEENSIDE
        if f == 63 or t == 63: cr &= ~CastlingRights.BLACK_KINGSIDE
        self.castling_rights = cr
        
        if self.turn == Color.BLACK:
            self.fullmove_number += 1
            
        self.turn = self.turn.flip()
        
        if HAS_CFFI and self._c_board:
            _lib.board_make_move(self._c_board, move._data)

    def pop(self) -> Move:
        if not self._history:
            return Move.NULL
            
        pieces, colors, turn, castling, ep, half, full = self._history.pop()
        self.pieces = pieces
        self.colors = colors
        self.turn = turn
        self.castling_rights = castling
        self.ep_square = ep
        self.halfmove_clock = half
        self.fullmove_number = full
        
        if HAS_CFFI and self._c_board:
            _lib.board_unmake_move(self._c_board)
            
        return Move.NULL # Pop doesn't return the move in this simple interface

    def _attacks_to(self, sq: int, color: Color) -> int:
        c = color.value
        occ = self.occupied
        
        pawns = self.pieces[c * 6 + 0]
        p_attacks = 0
        if c == 0: # White pawns attack up
            if (1 << sq) & ((pawns << 9) & 0xfefefefefefefefe): p_attacks = 1
            if (1 << sq) & ((pawns << 7) & 0x7f7f7f7f7f7f7f7f): p_attacks = 1
        else:
            if (1 << sq) & ((pawns >> 7) & 0xfefefefefefefefe): p_attacks = 1
            if (1 << sq) & ((pawns >> 9) & 0x7f7f7f7f7f7f7f7f): p_attacks = 1
            
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
        c = self.turn.value
        kings = self.pieces[c * 6 + 5]
        if not kings: return False
        ksq = lsb(kings)
        return bool(self._attacks_to(ksq, self.turn.flip()))

    @property
    def pseudo_legal_moves(self) -> List[Move]:
        moves = []
        c = self.turn.value
        occ = self.occupied
        us = self.colors[c]
        them = self.colors[1 - c]
        
        # Pawns
        pawns = self.pieces[c * 6 + 0]
        if c == 0:
            p_up1 = (pawns << 8) & ~occ
            p_up2 = ((p_up1 & 0x0000000000FF0000) << 8) & ~occ
            p_cap_r = (pawns << 9) & 0xfefefefefefefefe & them
            p_cap_l = (pawns << 7) & 0x7f7f7f7f7f7f7f7f & them
            
            for t in iter_bits(p_up1):
                f = t - 8
                if t >= 56:
                    moves.extend([Move.make(f, t, MoveFlag.PROMO_Q), Move.make(f, t, MoveFlag.PROMO_R), Move.make(f, t, MoveFlag.PROMO_B), Move.make(f, t, MoveFlag.PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.QUIET))
                    
            for t in iter_bits(p_up2):
                moves.append(Move.make(t - 16, t, MoveFlag.DOUBLE_PAWN))
                
            for t in iter_bits(p_cap_r):
                f = t - 9
                if t >= 56:
                    moves.extend([Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q), Move.make(f, t, MoveFlag.CAPTURE_PROMO_R), Move.make(f, t, MoveFlag.CAPTURE_PROMO_B), Move.make(f, t, MoveFlag.CAPTURE_PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))
                    
            for t in iter_bits(p_cap_l):
                f = t - 7
                if t >= 56:
                    moves.extend([Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q), Move.make(f, t, MoveFlag.CAPTURE_PROMO_R), Move.make(f, t, MoveFlag.CAPTURE_PROMO_B), Move.make(f, t, MoveFlag.CAPTURE_PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))
                    
            if self.ep_square != 64:
                ep_cap_r = (pawns << 9) & 0xfefefefefefefefe & (1 << self.ep_square)
                ep_cap_l = (pawns << 7) & 0x7f7f7f7f7f7f7f7f & (1 << self.ep_square)
                if ep_cap_r: moves.append(Move.make(self.ep_square - 9, self.ep_square, MoveFlag.EP_CAPTURE))
                if ep_cap_l: moves.append(Move.make(self.ep_square - 7, self.ep_square, MoveFlag.EP_CAPTURE))
        else:
            p_up1 = (pawns >> 8) & ~occ
            p_up2 = ((p_up1 & 0x0000FF0000000000) >> 8) & ~occ
            p_cap_r = (pawns >> 7) & 0xfefefefefefefefe & them
            p_cap_l = (pawns >> 9) & 0x7f7f7f7f7f7f7f7f & them
            
            for t in iter_bits(p_up1):
                f = t + 8
                if t <= 7:
                    moves.extend([Move.make(f, t, MoveFlag.PROMO_Q), Move.make(f, t, MoveFlag.PROMO_R), Move.make(f, t, MoveFlag.PROMO_B), Move.make(f, t, MoveFlag.PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.QUIET))
                    
            for t in iter_bits(p_up2):
                moves.append(Move.make(t + 16, t, MoveFlag.DOUBLE_PAWN))
                
            for t in iter_bits(p_cap_r):
                f = t + 7
                if t <= 7:
                    moves.extend([Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q), Move.make(f, t, MoveFlag.CAPTURE_PROMO_R), Move.make(f, t, MoveFlag.CAPTURE_PROMO_B), Move.make(f, t, MoveFlag.CAPTURE_PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))
                    
            for t in iter_bits(p_cap_l):
                f = t + 9
                if t <= 7:
                    moves.extend([Move.make(f, t, MoveFlag.CAPTURE_PROMO_Q), Move.make(f, t, MoveFlag.CAPTURE_PROMO_R), Move.make(f, t, MoveFlag.CAPTURE_PROMO_B), Move.make(f, t, MoveFlag.CAPTURE_PROMO_N)])
                else:
                    moves.append(Move.make(f, t, MoveFlag.CAPTURE))
                    
            if self.ep_square != 64:
                ep_cap_r = (pawns >> 7) & 0xfefefefefefefefe & (1 << self.ep_square)
                ep_cap_l = (pawns >> 9) & 0x7f7f7f7f7f7f7f7f & (1 << self.ep_square)
                if ep_cap_r: moves.append(Move.make(self.ep_square + 7, self.ep_square, MoveFlag.EP_CAPTURE))
                if ep_cap_l: moves.append(Move.make(self.ep_square + 9, self.ep_square, MoveFlag.EP_CAPTURE))
        
        # Knights
        knights = self.pieces[c * 6 + 1]
        for f in iter_bits(knights):
            atts = KNIGHT_ATTACKS[f] & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1<<t)&them else MoveFlag.QUIET))
                
        # Bishops
        bishops = self.pieces[c * 6 + 2]
        for f in iter_bits(bishops):
            atts = bishop_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1<<t)&them else MoveFlag.QUIET))
                
        # Rooks
        rooks = self.pieces[c * 6 + 3]
        for f in iter_bits(rooks):
            atts = rook_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1<<t)&them else MoveFlag.QUIET))
                
        # Queens
        queens = self.pieces[c * 6 + 4]
        for f in iter_bits(queens):
            atts = queen_attacks(f, occ) & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1<<t)&them else MoveFlag.QUIET))
                
        # Kings
        kings = self.pieces[c * 6 + 5]
        if kings:
            f = lsb(kings)
            atts = KING_ATTACKS[f] & ~us
            for t in iter_bits(atts):
                moves.append(Move.make(f, t, MoveFlag.CAPTURE if (1<<t)&them else MoveFlag.QUIET))
                
            # Castling
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
        return self.is_check() and len(self.legal_moves) == 0

    def is_stalemate(self) -> bool:
        return not self.is_check() and len(self.legal_moves) == 0
        
    def is_game_over(self) -> bool:
        return len(self.legal_moves) == 0 or self.halfmove_clock >= 100

    @property
    def fen(self) -> str:
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
            if emp: s += str(emp)
            if r > 0: s += "/"
            
        s += " " + ("w" if self.turn == Color.WHITE else "b") + " "
        
        c = ""
        if self.castling_rights & CastlingRights.WHITE_KINGSIDE: c += "K"
        if self.castling_rights & CastlingRights.WHITE_QUEENSIDE: c += "Q"
        if self.castling_rights & CastlingRights.BLACK_KINGSIDE: c += "k"
        if self.castling_rights & CastlingRights.BLACK_QUEENSIDE: c += "q"
        s += c if c else "-"
        
        s += " " + (square_name(self.ep_square) if self.ep_square != 64 else "-")
        s += f" {self.halfmove_clock} {self.fullmove_number}"
        return s
        
    @property
    def zobrist_hash(self) -> int:
        return compute_hash(self)

    def perft(self, depth: int) -> int:
        if depth == 0:
            return 1
        nodes = 0
        for m in self.legal_moves:
            self.push(m)
            nodes += self.perft(depth - 1)
            self.pop()
        return nodes

    def copy(self) -> 'Board':
        b = Board(self.fen)
        b._history = copy.deepcopy(self._history)
        return b

    def parse_uci(self, uci_str: str) -> Move:
        m = Move.from_uci(uci_str)
        for legal in self.legal_moves:
            if legal.from_square == m.from_square and legal.to_square == m.to_square:
                if m.is_promotion():
                    if m.promotion_piece() != legal.promotion_piece():
                        continue
                return legal
        raise ValueError(f"Invalid UCI move: {uci_str}")

    def san(self, move: Move) -> str:
        # Simplistic SAN
        return move.to_uci()

    def parse_san(self, san_str: str) -> Move:
        # Simplistic
        return self.parse_uci(san_str)

    def __str__(self):
        return self.fen

    def __repr__(self):
        return f"Board('{self.fen}')"
