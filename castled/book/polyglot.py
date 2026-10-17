"""Polyglot (.bin) opening book reader, writer, and move prober."""

import bisect
from dataclasses import dataclass
import os
import random
import struct
from typing import List, Optional, Sequence, Tuple

from ..core.board import Board
from ..core.move import Move
from ..core.types import CastlingRights, Color, MoveFlag, Piece, STARTING_FEN

ENTRY_STRUCT = struct.Struct(">QHHI")
DEFAULT_BOOK_PATH = os.path.join(os.path.dirname(__file__), "default_book.bin")


# Generate the 781 deterministic 64-bit Polyglot Zobrist keys via SplitMix64.
def _init_polyglot_keys() -> Tuple[int, ...]:
    keys: List[int] = []
    state = 0x436173746C656450
    for _ in range(781):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
        keys.append(z)
    return tuple(keys)


POLYGLOT_KEYS = _init_polyglot_keys()

# Map internal Piece (0..5 White P..K, 6..11 Black p..k) to Polyglot piece index (bp=0, wp=1, bn=2, wn=3, ...).
_PIECE_TO_POLYGLOT = {
    int(Piece.BLACK_PAWN): 0,
    int(Piece.WHITE_PAWN): 1,
    int(Piece.BLACK_KNIGHT): 2,
    int(Piece.WHITE_KNIGHT): 3,
    int(Piece.BLACK_BISHOP): 4,
    int(Piece.WHITE_BISHOP): 5,
    int(Piece.BLACK_ROOK): 6,
    int(Piece.WHITE_ROOK): 7,
    int(Piece.BLACK_QUEEN): 8,
    int(Piece.WHITE_QUEEN): 9,
    int(Piece.BLACK_KING): 10,
    int(Piece.WHITE_KING): 11,
}


@dataclass(frozen=True)
class BookEntry:
    # Single 16-byte Polyglot book entry.
    key: int
    raw_move: int
    weight: int
    learn: int = 0
    move: Move = Move.NULL


# Compute the 64-bit Polyglot Zobrist hash for `board`.
def polyglot_key(board: Board) -> int:
    h = 0
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is not None and int(pc) in _PIECE_TO_POLYGLOT:
            poly_pc = _PIECE_TO_POLYGLOT[int(pc)]
            h ^= POLYGLOT_KEYS[64 * poly_pc + sq]

    cr = board.castling_rights
    if cr & int(CastlingRights.WHITE_KINGSIDE):
        h ^= POLYGLOT_KEYS[768]
    if cr & int(CastlingRights.WHITE_QUEENSIDE):
        h ^= POLYGLOT_KEYS[769]
    if cr & int(CastlingRights.BLACK_KINGSIDE):
        h ^= POLYGLOT_KEYS[770]
    if cr & int(CastlingRights.BLACK_QUEENSIDE):
        h ^= POLYGLOT_KEYS[771]

    ep = board.ep_square
    if ep is not None and 0 <= int(ep) < 64:
        ep_file = int(ep) & 7
        if board.turn == Color.WHITE:
            pawn_rank = 4
            our_pawn = int(Piece.WHITE_PAWN)
        else:
            pawn_rank = 3
            our_pawn = int(Piece.BLACK_PAWN)
        has_ep_pawn = False
        if ep_file > 0 and board.piece_at(pawn_rank * 8 + (ep_file - 1)) == our_pawn:
            has_ep_pawn = True
        if ep_file < 7 and board.piece_at(pawn_rank * 8 + (ep_file + 1)) == our_pawn:
            has_ep_pawn = True
        if has_ep_pawn:
            h ^= POLYGLOT_KEYS[772 + ep_file]

    if board.turn == Color.WHITE:
        h ^= POLYGLOT_KEYS[780]

    return h & 0xFFFFFFFFFFFFFFFF


# Encode a `castled.Move` into the 16-bit Polyglot move integer.
def encode_polyglot_move(move: Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    flags = move.flags

    # Polyglot encodes castling as king-captures-own-rook (e1h1, e1a1, e8h8, e8a8).
    if flags == MoveFlag.KING_CASTLE:
        to_sq = 7 if from_sq == 4 else 63
    elif flags == MoveFlag.QUEEN_CASTLE:
        to_sq = 0 if from_sq == 4 else 56

    promo = 0
    if flags in (MoveFlag.PROMO_N, MoveFlag.CAPTURE_PROMO_N):
        promo = 1
    elif flags in (MoveFlag.PROMO_B, MoveFlag.CAPTURE_PROMO_B):
        promo = 2
    elif flags in (MoveFlag.PROMO_R, MoveFlag.CAPTURE_PROMO_R):
        promo = 3
    elif flags in (MoveFlag.PROMO_Q, MoveFlag.CAPTURE_PROMO_Q):
        promo = 4

    to_file = to_sq & 7
    to_rank = to_sq >> 3
    from_file = from_sq & 7
    from_rank = from_sq >> 3
    return to_file | (to_rank << 3) | (from_file << 6) | (from_rank << 9) | (promo << 12)


# Decode a 16-bit Polyglot move integer into a legal `castled.Move` on `board` (or `Move.NULL`).
def decode_polyglot_move(board: Board, raw_move: int) -> Move:
    to_file = raw_move & 7
    to_rank = (raw_move >> 3) & 7
    from_file = (raw_move >> 6) & 7
    from_rank = (raw_move >> 9) & 7
    promo = (raw_move >> 12) & 7

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file

    # Translate Polyglot king-captures-rook castling coordinates to destination square.
    king_pc = int(Piece.WHITE_KING) if board.turn == Color.WHITE else int(Piece.BLACK_KING)
    if board.piece_at(from_sq) == king_pc:
        if from_sq == 4 and to_sq == 7:
            to_sq = 6
        elif from_sq == 4 and to_sq == 0:
            to_sq = 2
        elif from_sq == 60 and to_sq == 63:
            to_sq = 62
        elif from_sq == 60 and to_sq == 56:
            to_sq = 58

    promo_chars = {0: "", 1: "n", 2: "b", 3: "r", 4: "q"}
    promo_suffix = promo_chars.get(promo, "")

    for legal in board.legal_moves:
        if legal.from_square == from_sq and legal.to_square == to_sq:
            uci = legal.to_uci()
            if promo_suffix:
                if uci.endswith(promo_suffix):
                    return legal
            elif len(uci) == 4:
                return legal
    return Move.NULL


class PolyglotBook:
    """Sorted Polyglot `.bin` opening book with O(log N) binary-search lookup."""

    def __init__(self, path: Optional[str] = None):
        self._entries: List[BookEntry] = []
        self._keys: List[int] = []
        if path:
            self.load(path)

    def __len__(self) -> int:
        return len(self._entries)

    # Load a binary `.bin` Polyglot book from disk.
    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise ValueError(f"Opening book file not found: {path}")
        raw_bytes = open(path, "rb").read()
        if len(raw_bytes) % ENTRY_STRUCT.size != 0:
            raise ValueError("Invalid Polyglot book file size (must be a multiple of 16 bytes)")

        entries: List[BookEntry] = []
        for offset in range(0, len(raw_bytes), ENTRY_STRUCT.size):
            key, raw_move, weight, learn = ENTRY_STRUCT.unpack_from(raw_bytes, offset)
            entries.append(BookEntry(key=key, raw_move=raw_move, weight=weight, learn=learn))

        entries.sort(key=lambda e: (e.key, -e.weight))
        self._entries = entries
        self._keys = [e.key for e in entries]

    # Save all book entries in standard sorted 16-byte big-endian Polyglot `.bin` format.
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._entries.sort(key=lambda e: (e.key, -e.weight))
        self._keys = [e.key for e in self._entries]
        with open(path, "wb") as fp:
            for e in self._entries:
                fp.write(ENTRY_STRUCT.pack(e.key, e.raw_move, e.weight, e.learn))

    # Add or update an opening move for `board` with `weight`.
    def add_entry(self, board: Board, move: Move, weight: int = 100, learn: int = 0) -> None:
        key = polyglot_key(board)
        raw_move = encode_polyglot_move(move)
        weight_clamped = max(1, min(65535, int(weight)))
        entry = BookEntry(key=key, raw_move=raw_move, weight=weight_clamped, learn=int(learn), move=move)
        self._entries.append(entry)
        self._entries.sort(key=lambda e: (e.key, -e.weight))
        self._keys = [e.key for e in self._entries]

    # Return all matching legal `BookEntry` objects for `board`, ordered by weight descending.
    def get_entries(self, board: Board) -> List[BookEntry]:
        if not self._entries:
            return []
        matches: List[BookEntry] = []
        for candidate_key in (polyglot_key(board), board.zobrist_hash):
            lo = bisect.bisect_left(self._keys, candidate_key)
            hi = bisect.bisect_right(self._keys, candidate_key)
            for idx in range(lo, hi):
                e = self._entries[idx]
                move = decode_polyglot_move(board, e.raw_move)
                if move != Move.NULL and e.weight > 0:
                    matches.append(
                        BookEntry(
                            key=e.key,
                            raw_move=e.raw_move,
                            weight=e.weight,
                            learn=e.learn,
                            move=move,
                        )
                    )
            if matches:
                break
        matches.sort(key=lambda e: e.weight, reverse=True)
        return matches

    # Probe the opening book for `board`; selects weighted-randomly or highest-weight when `weighted=False`.
    def probe(
        self,
        board: Board,
        weighted: bool = True,
        rng: Optional[random.Random] = None,
    ) -> Optional[Move]:
        entries = self.get_entries(board)
        if not entries:
            return None
        if not weighted or len(entries) == 1:
            return entries[0].move
        total_weight = sum(e.weight for e in entries)
        if total_weight <= 0:
            return entries[0].move
        gen = rng if rng is not None else random
        pick = gen.uniform(0.0, float(total_weight))
        acc = 0.0
        for e in entries:
            acc += float(e.weight)
            if pick <= acc:
                return e.move
        return entries[0].move


_DEFAULT_OPENING_LINES: Sequence[Tuple[Sequence[str], int]] = (
    # Ruy Lopez
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"), 120),
    # Italian Game (Giuoco Piano)
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3"), 100),
    # Sicilian Najdorf / Open
    (("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"), 120),
    # French Defense
    (("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"), 80),
    # Caro-Kann Defense
    (("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"), 85),
    # Queen's Gambit Declined
    (("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"), 115),
    # Slav Defense
    (("d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"), 90),
    # King's Indian Defense
    (("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8"), 100),
    # English Opening
    (("c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"), 75),
)


# Build a default Polyglot opening book from standard grandmaster opening lines.
def build_default_book(output_path: Optional[str] = None) -> PolyglotBook:
    book = PolyglotBook()
    seen = set()
    for moves_uci, base_weight in _DEFAULT_OPENING_LINES:
        board = Board(STARTING_FEN)
        for ply, uci_str in enumerate(moves_uci):
            move = board.parse_uci(uci_str)
            key = polyglot_key(board)
            raw_move = encode_polyglot_move(move)
            if (key, raw_move) not in seen:
                seen.add((key, raw_move))
                book.add_entry(board, move, weight=max(10, base_weight - ply * 5))
            board.push(move)
    if output_path:
        book.save(output_path)
    return book
