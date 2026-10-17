"""Iterative deepening PVS search with NMP, LMR, SEE,RFP/FP pruning, killers, history, and aspiration windows."""

from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional

from ..core.bitboard import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    lsb,
    rook_attacks,
)
from ..core.board import Board, HAS_CFFI, _ffi, _lib
from ..core.move import Move
from ..core.types import Color, MoveFlag, Piece, PieceType
from ..eval.classical import evaluate_python
from ..eval.nnue_ref import evaluate_nnue_numpy, load_nnue_file, reset_default_weights
from .time_manager import scale_soft_time
from .transposition import (
    SCORE_INF,
    SCORE_MATE,
    SCORE_MATE_THRESHOLD,
    TTBound,
    TranspositionTable,
)

MAX_PLY = 64
_MVV_LVA_VALS = (100, 300, 320, 500, 900, 10000, 0)
_SEE_VALS = (100, 320, 330, 500, 900, 20000, 0)
_CENTER_WEIGHT = (0, 1, 2, 3, 3, 2, 1, 0)


@dataclass
class SearchLimits:
    depth: int = 64
    nodes: int = 0
    movetime_ms: int = 0
    soft_movetime_ms: int = 0
    use_nnue: Optional[bool] = None


@dataclass
class SearchInfo:
    depth: int
    score: int
    nodes: int
    nps: int
    time_ms: int
    pv: List[Move] = field(default_factory=list)
    best_move: Move = Move.NULL


# Compute attackers to `sq` under custom occupancy bitboard `occ` in pure Python.
def _attackers_to_occ_py(board: Board, sq: int, occ: int) -> int:
    att = 0
    att |= PAWN_ATTACKS[Color.WHITE][sq] & board.pieces[Piece.BLACK_PAWN]
    att |= PAWN_ATTACKS[Color.BLACK][sq] & board.pieces[Piece.WHITE_PAWN]
    att |= KNIGHT_ATTACKS[sq] & (board.pieces[Piece.WHITE_KNIGHT] | board.pieces[Piece.BLACK_KNIGHT])
    att |= KING_ATTACKS[sq] & (board.pieces[Piece.WHITE_KING] | board.pieces[Piece.BLACK_KING])
    bishops_queens = (
        board.pieces[Piece.WHITE_BISHOP] | board.pieces[Piece.BLACK_BISHOP]
        | board.pieces[Piece.WHITE_QUEEN] | board.pieces[Piece.BLACK_QUEEN]
    )
    rooks_queens = (
        board.pieces[Piece.WHITE_ROOK] | board.pieces[Piece.BLACK_ROOK]
        | board.pieces[Piece.WHITE_QUEEN] | board.pieces[Piece.BLACK_QUEEN]
    )
    att |= bishop_attacks(sq, occ) & bishops_queens
    att |= rook_attacks(sq, occ) & rooks_queens
    return att


# Pure-Python Static Exchange Evaluation (SEE) matching C `board_see`.
def _see_python(board: Board, move: Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    flags = move.flags

    if flags in (MoveFlag.KING_CASTLE, MoveFlag.QUEEN_CASTLE):
        return 0

    gain = [0] * 32
    d = 0

    if flags == MoveFlag.EP_CAPTURE:
        gain[0] = _SEE_VALS[PieceType.PAWN]
    else:
        captured = board.mailbox[to_sq]
        gain[0] = _SEE_VALS[captured.value % 6] if captured != Piece.NONE else 0

    if flags >= MoveFlag.PROMO_N:
        promo_pt = (int(flags) & 3) + 1
        gain[0] += _SEE_VALS[promo_pt] - _SEE_VALS[PieceType.PAWN]
        next_victim_val = _SEE_VALS[promo_pt]
    else:
        attacker_pc = board.mailbox[from_sq]
        if attacker_pc == Piece.NONE:
            return 0
        next_victim_val = _SEE_VALS[attacker_pc.value % 6]

    occ = (board.colors[0] | board.colors[1]) ^ (1 << from_sq)
    if flags == MoveFlag.EP_CAPTURE:
        cap_sq = (to_sq - 8) if board.turn == Color.WHITE else (to_sq + 8)
        occ &= ~(1 << cap_sq)
    occ |= (1 << to_sq)

    side = int(board.turn) ^ 1

    while d < 31:
        attackers = _attackers_to_occ_py(board, to_sq, occ) & occ
        stm_attackers = attackers & board.colors[side]
        if stm_attackers == 0:
            break

        found_pt = -1
        attacker_sq = -1
        for pt in range(6):
            subset = stm_attackers & board.pieces[side * 6 + pt]
            if subset != 0:
                found_pt = pt
                attacker_sq = lsb(subset)
                break
        if found_pt < 0:
            break

        occ ^= (1 << attacker_sq)

        if found_pt == PieceType.KING:
            opp_attackers = _attackers_to_occ_py(board, to_sq, occ) & occ & board.colors[side ^ 1]
            if opp_attackers != 0:
                break

        d += 1
        gain[d] = next_victim_val - gain[d - 1]
        next_victim_val = _SEE_VALS[found_pt]

        if found_pt == PieceType.PAWN and (to_sq < 8 or to_sq >= 56):
            gain[d] += _SEE_VALS[PieceType.QUEEN] - _SEE_VALS[PieceType.PAWN]
            next_victim_val = _SEE_VALS[PieceType.QUEEN]

        side ^= 1

    while d > 0:
        if -gain[d - 1] < gain[d]:
            gain[d - 1] = -gain[d]
        d -= 1

    return gain[0]


# Compute Static Exchange Evaluation (SEE) in centipawns for `move` on `board`.
def see(board: Board, move: Move) -> int:
    if board._use_c and HAS_CFFI:
        return int(_lib.board_see(board._c_board, move._data))
    return _see_python(board, move)


# Score a move for ordering: TT move, promotions, SEE-ordered captures, killers, history, and center.
def _score_move_py(
    board: Board,
    move: Move,
    tt_move: Move,
    ply: int = 0,
    killers: Optional[List[List[Move]]] = None,
    history: Optional[List[List[List[int]]]] = None,
) -> int:
    if move == tt_move and tt_move != Move.NULL:
        return 2000000

    flags = move.flags
    from_sq = move.from_square
    to_sq = move.to_square
    attacker_pc = board.mailbox[from_sq]
    attacker = (attacker_pc.value % 6) if attacker_pc != Piece.NONE else 0

    if flags >= MoveFlag.PROMO_N:
        promo_type = (int(flags) & 3) + 1
        cap_bonus = 0
        if MoveFlag.CAPTURE_PROMO_N <= flags <= MoveFlag.CAPTURE_PROMO_Q:
            victim_pc = board.mailbox[to_sq]
            victim = (victim_pc.value % 6) if victim_pc != Piece.NONE else 0
            cap_bonus = _MVV_LVA_VALS[victim]
        return 1500000 + _MVV_LVA_VALS[promo_type] + cap_bonus

    if flags in (MoveFlag.CAPTURE, MoveFlag.EP_CAPTURE):
        if flags == MoveFlag.EP_CAPTURE:
            victim = int(PieceType.PAWN)
        else:
            victim_pc = board.mailbox[to_sq]
            victim = (victim_pc.value % 6) if victim_pc != Piece.NONE else 0
        mvv_lva = 10 * _MVV_LVA_VALS[victim] - _MVV_LVA_VALS[attacker]
        see_val = _see_python(board, move)
        if see_val >= 0:
            return 1000000 + see_val * 10 + mvv_lva
        return -100000 + see_val * 10 + mvv_lva

    if killers is not None and ply < MAX_PLY:
        if move == killers[ply][0]:
            return 900000
        if move == killers[ply][1]:
            return 800000

    hist = history[int(board.turn)][from_sq][to_sq] if history is not None else 0
    to_c = _CENTER_WEIGHT[to_sq & 7] + _CENTER_WEIGHT[to_sq >> 3]
    from_c = _CENTER_WEIGHT[from_sq & 7] + _CENTER_WEIGHT[from_sq >> 3]
    return hist + (to_c - from_c) * 10 + attacker


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
    """Alpha-beta PVS search engine with transposition table, pruning, and move ordering."""

    def __init__(
        self,
        hash_size_mb: int = 16,
        use_nnue: bool = False,
        eval_file: str = "",
        *,
        _use_c: Optional[bool] = None,
    ):
        self.tt = TranspositionTable(size_mb=hash_size_mb, _use_c=_use_c)
        self._use_c = self.tt._use_c
        self.use_nnue = use_nnue
        self.eval_file = eval_file
        if eval_file:
            self.load_nnue(eval_file)
        self._stop_flag = _ffi.new("int*", 0) if HAS_CFFI else None
        self._py_stop = False

    # Load a `.nnue` weight file into the active C and NumPy NNUE evaluator.
    def load_nnue(self, path: str) -> bool:
        if not path:
            reset_default_weights()
            self.eval_file = ""
            return True
        load_nnue_file(path, activate=True)
        self.eval_file = path
        return True

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
        soft_movetime_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        use_nnue: Optional[bool] = None,
        limits: Optional[SearchLimits] = None,
        on_info: Optional[Callable[[SearchInfo], None]] = None,
    ) -> SearchInfo:
        if limits is not None:
            max_depth = limits.depth
            max_nodes = limits.nodes
            max_time = limits.movetime_ms
            soft_time = limits.soft_movetime_ms
            active_nnue = (
                use_nnue
                if use_nnue is not None
                else (limits.use_nnue if limits.use_nnue is not None else self.use_nnue)
            )
        else:
            max_depth = depth if depth is not None else 64
            max_nodes = nodes if nodes is not None else 0
            max_time = movetime_ms if movetime_ms is not None else 0
            soft_time = soft_movetime_ms if soft_movetime_ms is not None else 0
            active_nnue = use_nnue if use_nnue is not None else self.use_nnue

        if max_depth <= 0 or max_depth >= MAX_PLY:
            max_depth = MAX_PLY - 1

        self._py_stop = False
        if self._stop_flag is not None:
            self._stop_flag[0] = 0

        if self._use_c and board._use_c:
            return self._search_c(board, max_depth, max_nodes, max_time, soft_time, active_nnue, on_info)
        return self._search_python(board, max_depth, max_nodes, max_time, soft_time, active_nnue, on_info)

    def _search_c(
        self,
        board: Board,
        max_depth: int,
        max_nodes: int,
        movetime_ms: int,
        soft_movetime_ms: int,
        use_nnue: bool,
        on_info: Optional[Callable[[SearchInfo], None]],
    ) -> SearchInfo:
        c_limits = _ffi.new("SearchLimits*")
        c_limits.max_depth = max_depth
        c_limits.max_nodes = max(0, max_nodes)
        c_limits.movetime_ms = max(0, movetime_ms)
        c_limits.soft_movetime_ms = max(0, soft_movetime_ms)
        c_limits.use_nnue = 1 if use_nnue else 0
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

    # Pure-Python reference search matching the C Search v2 implementation.
    def _search_python(
        self,
        board: Board,
        max_depth: int,
        max_nodes: int,
        movetime_ms: int,
        soft_movetime_ms: int,
        use_nnue: bool,
        on_info: Optional[Callable[[SearchInfo], None]],
    ) -> SearchInfo:
        root_moves = board.legal_moves
        if not root_moves:
            score = -SCORE_MATE if board.is_check() else 0
            return SearchInfo(depth=0, score=score, nodes=0, nps=0, time_ms=0, pv=[], best_move=Move.NULL)

        eval_fn = evaluate_nnue_numpy if use_nnue else evaluate_python

        start_ms = time.monotonic() * 1000.0
        nodes_count = 0
        current_depth = 1
        aborted = False
        pv_table = [[Move.NULL] * MAX_PLY for _ in range(MAX_PLY)]
        pv_length = [0] * MAX_PLY
        killers = [[Move.NULL, Move.NULL] for _ in range(MAX_PLY)]
        history = [[[0] * 64 for _ in range(64)] for _ in range(2)]

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
                return eval_fn(board)

            in_check = board.is_check()
            if in_check:
                if not board.legal_moves:
                    return -SCORE_MATE + ply

            stand_pat = eval_fn(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

            captures = board.generate_captures()
            scores = [_score_move_py(board, m, Move.NULL, ply, killers, history) for m in captures]

            for i in range(len(captures)):
                _pick_next_move(captures, scores, i)
                m = captures[i]

                if not in_check and m.flags < MoveFlag.PROMO_N and scores[i] < 0:
                    continue

                board.push(m)
                score = -quiescence(-beta, -alpha, ply + 1)
                board.pop()

                if aborted:
                    return 0
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

            return alpha

        def negamax(depth: int, alpha: int, beta: int, ply: int, can_null: bool) -> int:
            nonlocal nodes_count
            if ply < MAX_PLY:
                pv_length[ply] = ply

            if check_stop():
                return 0

            if ply > 0:
                if board.halfmove_clock >= 100 or _is_search_repetition_py(board) or board.is_insufficient_material():
                    return 0

            in_check = board.is_check()
            if in_check and 0 < ply < MAX_PLY - 2:
                depth += 1

            if depth <= 0 or ply >= MAX_PLY:
                return quiescence(alpha, beta, ply)

            nodes_count += 1
            is_pv = (beta - alpha) > 1
            orig_alpha = alpha

            hit, tt_move, tt_score = self.tt.probe(board.zobrist_hash, depth, alpha, beta, ply)
            if hit and not is_pv and ply > 0:
                return tt_score

            static_eval = -SCORE_INF if in_check else eval_fn(board)

            # Reverse Futility Pruning (RFP)
            if (
                not is_pv
                and not in_check
                and depth <= 5
                and -SCORE_MATE_THRESHOLD < beta < SCORE_MATE_THRESHOLD
            ):
                rfp_margin = 85 * depth
                if static_eval - rfp_margin >= beta:
                    return static_eval - rfp_margin

            # Null-Move Pruning (NMP) with non-pawn material zugzwang guard
            if (
                not is_pv
                and not in_check
                and can_null
                and depth >= 3
                and board.has_non_pawn_material(board.turn)
                and static_eval >= beta
                and beta < SCORE_MATE_THRESHOLD
            ):
                r_null = 2 + (depth // 4)
                board.push_null()
                null_score = -negamax(depth - 1 - r_null, -beta, -beta + 1, ply + 1, False)
                board.pop_null()
                if aborted:
                    return 0
                if null_score >= beta:
                    return beta if null_score >= SCORE_MATE_THRESHOLD else null_score

            moves = board.legal_moves
            if not moves:
                return (-SCORE_MATE + ply) if in_check else 0

            scores = [_score_move_py(board, m, tt_move, ply, killers, history) for m in moves]
            best_move = moves[0]
            best_score = -SCORE_INF
            moves_searched = 0

            for i in range(len(moves)):
                _pick_next_move(moves, scores, i)
                m = moves[i]
                flags = m.flags
                is_tactical = flags in (MoveFlag.CAPTURE, MoveFlag.EP_CAPTURE) or flags >= MoveFlag.PROMO_N

                # SEE pruning for heavily losing captures at shallow depth
                if (
                    not is_pv
                    and not in_check
                    and moves_searched > 0
                    and depth <= 3
                    and flags in (MoveFlag.CAPTURE, MoveFlag.EP_CAPTURE)
                    and scores[i] < 0
                ):
                    if _see_python(board, m) < -100 * depth:
                        continue

                board.push(m)
                gives_check = board.is_check()

                # Futility pruning for quiet non-checking moves at shallow depth
                if (
                    not is_pv
                    and not in_check
                    and not gives_check
                    and moves_searched > 0
                    and depth <= 3
                    and not is_tactical
                    and alpha > -SCORE_MATE_THRESHOLD
                    and beta < SCORE_MATE_THRESHOLD
                ):
                    if static_eval + 110 * depth <= alpha:
                        board.pop()
                        continue

                moves_searched += 1

                if moves_searched == 1:
                    score = -negamax(depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    r_lmr = 0
                    if depth >= 3 and moves_searched >= 4 and not in_check and not gives_check and not is_tactical:
                        r_lmr = 1 + (depth // 5) + (moves_searched // 10)
                        if is_pv:
                            r_lmr -= 1
                        if ply < MAX_PLY and (m == killers[ply][0] or m == killers[ply][1]):
                            r_lmr -= 1
                        if r_lmr < 0:
                            r_lmr = 0
                        if r_lmr > depth - 2:
                            r_lmr = depth - 2

                    score = -negamax(depth - 1 - r_lmr, -alpha - 1, -alpha, ply + 1, True)
                    if r_lmr > 0 and score > alpha and not aborted:
                        score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, True)
                    if is_pv and alpha < score < beta and not aborted:
                        score = -negamax(depth - 1, -beta, -alpha, ply + 1, True)

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
                    if not is_tactical and ply < MAX_PLY:
                        if killers[ply][0] != m:
                            killers[ply][1] = killers[ply][0]
                            killers[ply][0] = m
                        bonus = depth * depth
                        us = int(board.turn)
                        from_sq = m.from_square
                        to_sq = m.to_square
                        history[us][from_sq][to_sq] = min(300000, history[us][from_sq][to_sq] + bonus)
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

        prev_best_move = Move.NULL
        prev_score = 0
        best_move_changes = 0

        for d in range(1, max_depth + 1):
            current_depth = d
            if d <= 4:
                score = negamax(d, -SCORE_INF, SCORE_INF, 0, True)
            else:
                delta = 35
                alpha = max(-SCORE_INF, prev_score - delta)
                beta = min(SCORE_INF, prev_score + delta)
                while True:
                    score = negamax(d, alpha, beta, 0, True)
                    if aborted:
                        break
                    if score <= alpha:
                        alpha = max(-SCORE_INF, score - delta)
                        delta *= 2
                    elif score >= beta:
                        beta = min(SCORE_INF, score + delta)
                        delta *= 2
                    else:
                        break
                    if delta > 500:
                        alpha = -SCORE_INF
                        beta = SCORE_INF

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

            score_drop = 0
            if d >= 2:
                if best_move != prev_best_move:
                    best_move_changes += 1
                if score < prev_score:
                    score_drop = prev_score - score
            prev_best_move = best_move
            prev_score = score

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
            if soft_movetime_ms > 0:
                effective_soft = (
                    scale_soft_time(soft_movetime_ms, movetime_ms, best_move_changes, score_drop)
                    if d >= 4
                    else soft_movetime_ms
                )
                if elapsed >= effective_soft:
                    break

        return best_info


# Alias Engine to Searcher for convenience.
Engine = Searcher

_default_searcher: Optional[Searcher] = None


# Convenience function to run a search on board using a shared Searcher instance.
def search(
    board: Board,
    depth: Optional[int] = None,
    movetime_ms: Optional[int] = None,
    soft_movetime_ms: Optional[int] = None,
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
        soft_movetime_ms=soft_movetime_ms,
        nodes=nodes,
        limits=limits,
        on_info=on_info,
    )
