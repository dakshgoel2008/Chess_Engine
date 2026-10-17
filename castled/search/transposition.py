"""Transposition table with bound types and ply-relative mate score adjustment."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

from ..core.board import HAS_CFFI, _lib, _ffi, _should_use_c
from ..core.move import Move

SCORE_INF = 32000
SCORE_MATE = 30000
SCORE_MATE_THRESHOLD = 29000


class TTBound(IntEnum):
    NONE = 0
    EXACT = 1
    LOWER = 2
    UPPER = 3
    LOWER_BOUND = 2
    UPPER_BOUND = 3


@dataclass
class TTEntry:
    key: int
    best_move: Move
    score: int
    depth: int
    bound: TTBound


def _score_to_tt(score: int, ply: int) -> int:
    if score > SCORE_MATE_THRESHOLD:
        score += ply
    elif score < -SCORE_MATE_THRESHOLD:
        score -= ply
    return max(-SCORE_INF, min(SCORE_INF, score))


def _score_from_tt(score: int, ply: int) -> int:
    if score > SCORE_MATE_THRESHOLD:
        return score - ply
    if score < -SCORE_MATE_THRESHOLD:
        return score + ply
    return score


class TranspositionTable:
    """Fixed-size Zobrist hash table supporting C and pure-Python backends."""

    def __init__(self, size_mb: int = 16, *, _use_c: Optional[bool] = None):
        self._use_c = _should_use_c() if _use_c is None else (_use_c and HAS_CFFI)
        self._table: Dict[int, TTEntry] = {}
        self._mask = 0
        self.size_mb = max(1, min(1024, size_mb))
        self.resize(self.size_mb)

    # Resize the table to fit within size_mb using a power-of-two slot count.
    def resize(self, size_mb: int):
        self.size_mb = max(1, min(1024, size_mb))
        max_entries = (self.size_mb * 1024 * 1024) // 16
        count = 1024
        while (count << 1) <= max_entries:
            count <<= 1
        self._mask = count - 1
        if self._use_c:
            _lib.tt_init(self.size_mb)
        else:
            self._table.clear()

    def clear(self):
        if self._use_c:
            _lib.tt_clear()
        else:
            self._table.clear()

    # Store a position evaluation and best move, adjusting mate scores by ply.
    def store(
        self,
        key: int,
        depth: int,
        score: int,
        bound: int,
        best_move: Move = Move.NULL,
        ply: int = 0,
    ):
        if self._use_c:
            _lib.tt_store(key, depth, score, int(bound), int(best_move), ply)
            return

        idx = key & self._mask
        existing = self._table.get(idx)
        move_to_save = best_move
        if existing is not None and existing.key == key and best_move == Move.NULL:
            move_to_save = existing.best_move

        clamped_depth = max(0, min(127, depth))
        self._table[idx] = TTEntry(
            key=key,
            best_move=Move(int(move_to_save)),
            score=_score_to_tt(score, ply),
            depth=clamped_depth,
            bound=TTBound(int(bound)),
        )

    # Probe the table for an alpha-beta cutoff and retrieve the stored best move.
    def probe(
        self, key: int, depth: int, alpha: int, beta: int, ply: int = 0
    ) -> Tuple[bool, Move, int]:
        if self._use_c:
            best_move_ptr = _ffi.new("uint16_t*")
            score_ptr = _ffi.new("int*")
            hit = bool(_lib.tt_probe(key, depth, alpha, beta, ply, best_move_ptr, score_ptr))
            return hit, Move(best_move_ptr[0]), int(score_ptr[0])

        entry = self._table.get(key & self._mask)
        if entry is None or entry.bound == TTBound.NONE or entry.key != key:
            return False, Move.NULL, 0

        s = _score_from_tt(entry.score, ply)
        if entry.depth >= depth:
            if entry.bound == TTBound.EXACT:
                return True, entry.best_move, s
            if entry.bound == TTBound.LOWER and s >= beta:
                return True, entry.best_move, s
            if entry.bound == TTBound.UPPER and s <= alpha:
                return True, entry.best_move, s

        return False, entry.best_move, 0

    # Return the stored entry for key (with mate score adjusted to ply), or None.
    def probe_entry(self, key: int, ply: int = 0) -> Optional[TTEntry]:
        if self._use_c:
            bm_ptr = _ffi.new("uint16_t*")
            sc_ptr = _ffi.new("int*")
            dp_ptr = _ffi.new("int*")
            bd_ptr = _ffi.new("int*")
            if not _lib.tt_probe_entry(key, ply, bm_ptr, sc_ptr, dp_ptr, bd_ptr):
                return None
            return TTEntry(
                key=key,
                best_move=Move(bm_ptr[0]),
                score=int(sc_ptr[0]),
                depth=int(dp_ptr[0]),
                bound=TTBound(int(bd_ptr[0])),
            )

        entry = self._table.get(key & self._mask)
        if entry is None or entry.bound == TTBound.NONE or entry.key != key:
            return None
        return TTEntry(
            key=entry.key,
            best_move=entry.best_move,
            score=_score_from_tt(entry.score, ply),
            depth=entry.depth,
            bound=entry.bound,
        )

