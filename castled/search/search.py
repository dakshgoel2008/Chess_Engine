"""Iterative deepening negamax search with alpha-beta pruning and quiescence."""

from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional

from ..core.board import Board, HAS_CFFI, _lib, _ffi
from ..core.move import Move
from ..core.types import MoveFlag, Piece, PieceType
from ..eval.classical import evaluate_python
from .transposition import (
    SCORE_INF,
    SCORE_MATE,
    SCORE_MATE_THRESHOLD,
    TTBound,
    TranspositionTable,
)

MAX_PLY = 64
_MVV_LVA_VALS = (100, 300, 320, 500, 900, 10000, 0)
_CENTER_WEIGHT = (0, 1, 2, 3, 3, 2, 1, 0)


@dataclass
class SearchLimits:
    depth: int = 64
    nodes: int = 0
    movetime_ms: int = 0


@dataclass
class SearchInfo:
    depth: int
    score: int
    nodes: int
    nps: int
    time_ms: int
    pv: List[Move] = field(default_factory=list)
    best_move: Move = Move.NULL


# Score a move for ordering: TT move first, then promotions, MVV-LVA captures, and center moves.
def _score_move_py(board: Board, move: Move, tt_move: Move) -> int:
    if move == tt_move and tt_move != Move.NULL:
        return 1000000

    flags = move.flags
    from_sq = move.from_square
    to_sq = move.to_square
    attacker_pc = board.mailbox[from_sq]
    attacker = (attacker_pc.value % 6) if attacker_pc != Piece.NONE else 0
    score = 0

    if flags == MoveFlag.CAPTURE or (MoveFlag.CAPTURE_PROMO_N <= flags <= MoveFlag.CAPTURE_PROMO_Q):
        victim_pc = board.mailbox[to_sq]
        victim = (victim_pc.value % 6) if victim_pc != Piece.NONE else 0
        score = 100000 + 10 * _MVV_LVA_VALS[victim] - _MVV_LVA_VALS[attacker]
    elif flags == MoveFlag.EP_CAPTURE:
        score = 100000 + 10 * _MVV_LVA_VALS[PieceType.PAWN] - _MVV_LVA_VALS[PieceType.PAWN]

    if flags >= MoveFlag.PROMO_N:
        promo_type = (int(flags) & 3) + 1
        score += 50000 + _MVV_LVA_VALS[promo_type]

    if score == 0:
        to_c = _CENTER_WEIGHT[to_sq & 7] + _CENTER_WEIGHT[to_sq >> 3]
        from_c = _CENTER_WEIGHT[from_sq & 7] + _CENTER_WEIGHT[from_sq >> 3]
        score = (to_c - from_c) * 10 + attacker

    return score


def _pick_next_move(moves: List[Move], scores: List[int], idx: int):
    best_idx = idx
    best_score = scores[idx]
    for i in range(idx + 1, len(moves)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    if best_idx != idx:
        moves[idx], moves[best_idx] = moves[best_idx], moves[idx]
        scores[idx], scores[best_idx] = scores[best_idx], scores[idx]


# Check if current board hash has already occurred since the last pawn move or capture.
def _is_search_repetition_py(board: Board) -> bool:
    hist = board._hash_history
    start = max(0, len(hist) - board.halfmove_clock)
    target = board._hash
    for i in range(len(hist) - 1, start - 1, -1):
        if hist[i] == target:
            return True
    return False


class Searcher:
    """Alpha-beta search engine with iterative deepening and transposition table."""

    def __init__(self, hash_size_mb: int = 16, *, _use_c: Optional[bool] = None):
        self.tt = TranspositionTable(size_mb=hash_size_mb, _use_c=_use_c)
        self._use_c = self.tt._use_c
        self._stop_flag = _ffi.new("int*", 0) if HAS_CFFI else None
        self._py_stop = False

    def stop(self):
        self._py_stop = True
        if self._stop_flag is not None:
            self._stop_flag[0] = 1

    def clear(self):
        self.tt.clear()

    # Search for the best move on board under the given depth, time, or node limits.
    def search(
        self,
        board: Board,
        depth: Optional[int] = None,
        movetime_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        limits: Optional[SearchLimits] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> SearchInfo:
        if limits is not None:
            max_depth = limits.depth
            max_nodes = limits.nodes
            max_time = limits.movetime_ms
        else:
            max_depth = depth if depth is not None else 64
            max_nodes = nodes if nodes is not None else 0
            max_time = movetime_ms if movetime_ms is not None else 0

        if max_depth <= 0 or max_depth >= MAX_PLY:
            max_depth = MAX_PLY - 1

        self._py_stop = False
        if self._stop_flag is not None:
            self._stop_flag[0] = 0

        if self._use_c and board._use_c:
            return self._search_c(board, max_depth, max_nodes, max_time, on_info)
        return self._search_python(board, max_depth, max_nodes, max_time, on_info)

    def _search_c(
        self,
        board: Board,
        max_depth: int,
        max_nodes: int,
        movetime_ms: int,
        on_info: Optional[Callable[[SearchInfo], None]],
    ) -> SearchInfo:
        c_limits = _ffi.new("SearchLimits*")
        c_limits.max_depth = max_depth
        c_limits.max_nodes = max(0, max_nodes)
        c_limits.movetime_ms = max(0, movetime_ms)
        c_limits.stop_flag = self._stop_flag

        c_result = _ffi.new("SearchResult*")

        if on_info is not None:
            @_ffi.callback("void(int, int, uint64_t, int, const uint16_t*, int)")
            def _cb(d, score, nodes, time_ms, pv_ptr, pv_len):
                pv = [Move(pv_ptr[i]) for i in range(pv_len)]
                nps = int(nodes * 1000 // max(1, time_ms))
                info = SearchInfo(
                    depth=int(d),
                    score=int(score),
                    nodes=int(nodes),
                    nps=nps,
                    time_ms=int(time_ms),
                    pv=pv,
                    best_move=pv[0] if pv else Move.NULL,
                )
                on_info(info)

            cb_arg = _cb
        else:
            cb_arg = _ffi.NULL

        _lib.search_best_move(board._c_board, c_limits, cb_arg, c_result)

        pv = [Move(c_result.pv[i]) for i in range(c_result.pv_length)]
        best_move = Move(c_result.best_move)
        if not pv and best_move != Move.NULL:
            pv = [best_move]
        time_ms = int(c_result.time_ms)
        nodes = int(c_result.nodes)
        nps = int(nodes * 1000 // max(1, time_ms))

        return SearchInfo(
            depth=int(c_result.depth),
            score=int(c_result.score),
            nodes=nodes,
            nps=nps,
            time_ms=time_ms,
            pv=pv,
            best_move=best_move,
        )

    # Pure-Python reference search matching the C implementation.
    def _search_python(
        self,
        board: Board,
        max_depth: int,
        max_nodes: int,
        movetime_ms: int,
        on_info: Optional[Callable[[SearchInfo], None]],
    ) -> SearchInfo:
        root_moves = board.legal_moves
        if not root_moves:
            score = -SCORE_MATE if board.is_check() else 0
            return SearchInfo(depth=0, score=score, nodes=0, nps=0, time_ms=0, pv=[], best_move=Move.NULL)

        start_ms = time.monotonic() * 1000.0
        nodes_count = 0
        current_depth = 1
        aborted = False
        pv_table = [[Move.NULL] * MAX_PLY for _ in range(MAX_PLY)]
        pv_length = [0] * MAX_PLY

        def check_stop() -> bool:
            nonlocal aborted
            if aborted:
                return True
            if current_depth <= 1:
                return False
            if max_nodes > 0 and nodes_count >= max_nodes:
                aborted = True
                return True
            if (nodes_count & 2047) == 0:
                if self._py_stop:
                    aborted = True
                    return True
                if movetime_ms > 0 and (time.monotonic() * 1000.0 - start_ms) >= movetime_ms:
                    aborted = True
                    return True
            return False

        def quiescence(alpha: int, beta: int, ply: int) -> int:
            nonlocal nodes_count
            if check_stop():
                return 0
            nodes_count += 1

            if ply >= MAX_PLY:
                return evaluate_python(board)

            if board.is_check():
                if not board.legal_moves:
                    return -SCORE_MATE + ply

            stand_pat = evaluate_python(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            captures = board.generate_captures()
            scores = [_score_move_py(board, m, Move.NULL) for m in captures]

            for i in range(len(captures)):
                _pick_next_move(captures, scores, i)
                board.push(captures[i])
                score = -quiescence(-beta, -alpha, ply + 1)
                board.pop()

                if aborted:
                    return 0
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

            return alpha

        def negamax(depth: int, alpha: int, beta: int, ply: int) -> int:
            nonlocal nodes_count
            if ply < MAX_PLY:
                pv_length[ply] = ply

            if check_stop():
                return 0

            if ply > 0:
                if board.halfmove_clock >= 100 or _is_search_repetition_py(board) or board.is_insufficient_material():
                    return 0

            if depth <= 0 or ply >= MAX_PLY:
                return quiescence(alpha, beta, ply)

            nodes_count += 1
            orig_alpha = alpha

            hit, tt_move, tt_score = self.tt.probe(board.zobrist_hash, depth, alpha, beta, ply)
            if hit and ply > 0:
                if tt_move != Move.NULL and ply + 1 < MAX_PLY:
                    pv_table[ply][ply] = tt_move
                    pv_length[ply] = ply + 1
                return tt_score

            moves = board.legal_moves
            if not moves:
                return (-SCORE_MATE + ply) if board.is_check() else 0

            scores = [_score_move_py(board, m, tt_move) for m in moves]
            best_move = moves[0]
            best_score = -SCORE_INF

            for i in range(len(moves)):
                _pick_next_move(moves, scores, i)
                m = moves[i]

                board.push(m)
                score = -negamax(depth - 1, -beta, -alpha, ply + 1)
                board.pop()

                if aborted:
                    return 0

                if score > best_score:
                    best_score = score
                    best_move = m

                if score > alpha:
                    alpha = score
                    if ply < MAX_PLY:
                        pv_table[ply][ply] = m
                        child_len = pv_length[ply + 1] if (ply + 1 < MAX_PLY) else (ply + 1)
                        for nxt in range(ply + 1, min(child_len, MAX_PLY)):
                            pv_table[ply][nxt] = pv_table[ply + 1][nxt]
                        pv_length[ply] = child_len

                if alpha >= beta:
                    break

            bound = TTBound.EXACT
            if best_score <= orig_alpha:
                bound = TTBound.UPPER
            elif best_score >= beta:
                bound = TTBound.LOWER
            self.tt.store(board.zobrist_hash, depth, best_score, bound, best_move, ply)

            return best_score

        best_info = SearchInfo(
            depth=1,
            score=0,
            nodes=0,
            nps=0,
            time_ms=0,
            pv=[root_moves[0]],
            best_move=root_moves[0],
        )

        for d in range(1, max_depth + 1):
            current_depth = d
            score = negamax(d, -SCORE_INF, SCORE_INF, 0)
            elapsed = max(0, int(time.monotonic() * 1000.0 - start_ms))

            if aborted and d > 1:
                best_info.nodes = nodes_count
                best_info.time_ms = elapsed
                best_info.nps = int(nodes_count * 1000 // max(1, elapsed))
                break

            pv = [pv_table[0][i] for i in range(pv_length[0])]
            best_move = pv[0] if pv else root_moves[0]
            if not pv:
                pv = [best_move]

            nps = int(nodes_count * 1000 // max(1, elapsed))
            best_info = SearchInfo(
                depth=d,
                score=score,
                nodes=nodes_count,
                nps=nps,
                time_ms=elapsed,
                pv=pv,
                best_move=best_move,
            )

            if on_info is not None:
                on_info(best_info)

            if self._py_stop:
                break
            if max_nodes > 0 and nodes_count >= max_nodes:
                break
            if movetime_ms > 0 and elapsed >= movetime_ms:
                break

        return best_info


_default_searcher: Optional[Searcher] = None


# Convenience function to run a search on board using a shared Searcher instance.
def search(
    board: Board,
    depth: Optional[int] = None,
    movetime_ms: Optional[int] = None,
    nodes: Optional[int] = None,
    limits: Optional[SearchLimits] = None,
    on_info: Optional[Callable[[SearchInfo], None]] = None,
) -> SearchInfo:
    global _default_searcher
    if _default_searcher is None or _default_searcher._use_c != board._use_c:
        _default_searcher = Searcher(_use_c=board._use_c)
    return _default_searcher.search(
        board,
        depth=depth,
        movetime_ms=movetime_ms,
        nodes=nodes,
        limits=limits,
        on_info=on_info,
    )
