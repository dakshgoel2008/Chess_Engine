#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "bitboard.h"
#include <stddef.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static int64_t now_ms(void) {
    return (int64_t)GetTickCount64();
}
#else
#include <sys/time.h>
static int64_t now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + (int64_t)(tv.tv_usec / 1000);
}
#endif

typedef struct {
    SearchLimits limits;
    int64_t start_ms;
    uint64_t nodes;
    int current_depth;
    bool aborted;
    Move pv_table[MAX_SEARCH_PLY][MAX_SEARCH_PLY];
    int pv_length[MAX_SEARCH_PLY];
    Move killers[MAX_SEARCH_PLY][2];
    int history[2][64][64];
} SearchState;

static const int MVV_LVA_VALS[7] = { 100, 300, 320, 500, 900, 10000, 0 };
static const int SEE_VALS[7] = { 100, 320, 330, 500, 900, 20000, 0 };
static const int CENTER_WEIGHT[8] = { 0, 1, 2, 3, 3, 2, 1, 0 };

// Static Exchange Evaluation (SEE): returns material gain/loss in centipawns for a move.
int board_see(const Board* board, Move move) {
    int from_sq = MOVE_FROM(move);
    int to_sq = MOVE_TO(move);
    int flags = MOVE_FLAGS(move);

    if (flags == KING_CASTLE || flags == QUEEN_CASTLE) {
        return 0;
    }

    int gain[32];
    int d = 0;

    if (flags == EP_CAPTURE) {
        gain[0] = SEE_VALS[PAWN];
    } else {
        int captured = board->mailbox[to_sq];
        gain[0] = (captured != PIECE_NONE) ? SEE_VALS[captured % 6] : 0;
    }

    int next_victim_val;
    if (flags >= PROMO_N) {
        int promo_pt = (flags & 3) + 1;
        gain[0] += SEE_VALS[promo_pt] - SEE_VALS[PAWN];
        next_victim_val = SEE_VALS[promo_pt];
    } else {
        int attacker_pc = board->mailbox[from_sq];
        if (attacker_pc == PIECE_NONE) return 0;
        next_victim_val = SEE_VALS[attacker_pc % 6];
    }

    uint64_t occ = board->occupied[2] ^ (1ULL << from_sq);
    if (flags == EP_CAPTURE) {
        int cap_sq = (board->turn == WHITE) ? (to_sq - 8) : (to_sq + 8);
        occ &= ~(1ULL << cap_sq);
    }
    occ |= (1ULL << to_sq);

    int side = board->turn ^ 1;

    while (d < 31) {
        uint64_t attackers = board_attackers_to(board, to_sq, occ) & occ;
        uint64_t stm_attackers = attackers & board->occupied[side];
        if (stm_attackers == 0) {
            break;
        }

        int found_pt = -1;
        int attacker_sq = -1;
        for (int pt = PAWN; pt <= KING; pt++) {
            uint64_t subset = stm_attackers & board->pieces[side * 6 + pt];
            if (subset != 0) {
                found_pt = pt;
                attacker_sq = lsb(subset);
                break;
            }
        }
        if (found_pt < 0) {
            break;
        }

        occ ^= (1ULL << attacker_sq);

        if (found_pt == KING) {
            uint64_t opp_attackers = board_attackers_to(board, to_sq, occ) & occ & board->occupied[side ^ 1];
            if (opp_attackers != 0) {
                break;
            }
        }

        d++;
        gain[d] = next_victim_val - gain[d - 1];
        next_victim_val = SEE_VALS[found_pt];

        if (found_pt == PAWN && (to_sq < 8 || to_sq >= 56)) {
            gain[d] += SEE_VALS[QUEEN] - SEE_VALS[PAWN];
            next_victim_val = SEE_VALS[QUEEN];
        }

        side ^= 1;
    }

    while (d > 0) {
        if (-gain[d - 1] < gain[d]) {
            gain[d - 1] = -gain[d];
        }
        d--;
    }

    return gain[0];
}

// Return true if side `color` has at least one non-pawn, non-king piece (zugzwang guard).
static bool has_non_pawn_material(const Board* board, int color) {
    int base = color * 6;
    return (board->pieces[base + KNIGHT] |
            board->pieces[base + BISHOP] |
            board->pieces[base + ROOK] |
            board->pieces[base + QUEEN]) != 0;
}

// Check node limit on every node and time/stop limits every 2048 nodes after depth 1.
static bool check_stop(SearchState* st) {
    if (st->aborted) return true;
    if (st->current_depth <= 1) return false;

    if (st->limits.max_nodes > 0 && st->nodes >= st->limits.max_nodes) {
        st->aborted = true;
        return true;
    }

    if ((st->nodes & 2047) == 0) {
        if (st->limits.stop_flag != NULL && *(st->limits.stop_flag) != 0) {
            st->aborted = true;
            return true;
        }
        if (st->limits.movetime_ms > 0 && (now_ms() - st->start_ms) >= st->limits.movetime_ms) {
            st->aborted = true;
            return true;
        }
    }
    return false;
}

// Check if current board hash has already appeared since the last irreversible move.
static bool is_search_repetition(const Board* board) {
    int start = board->hash_history_count - board->halfmove_clock;
    if (start < 0) start = 0;
    for (int i = board->hash_history_count - 1; i >= start; i--) {
        if (board->hash_history[i] == board->hash) {
            return true;
        }
    }
    return false;
}

// Score moves for ordering: TT move, promotions, SEE-ordered captures, killers, history, and center.
static int score_move(const Board* board, Move move, Move tt_move, int ply, const SearchState* st) {
    if (move == tt_move && tt_move != 0) {
        return 2000000;
    }

    int flags = MOVE_FLAGS(move);
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int attacker = board->mailbox[from] % 6;

    if (flags >= PROMO_N) {
        int promo_type = (flags & 3) + 1;
        int cap_bonus = 0;
        if (flags >= CAPTURE_PROMO_N && flags <= CAPTURE_PROMO_Q) {
            int victim = board->mailbox[to] % 6;
            cap_bonus = MVV_LVA_VALS[victim];
        }
        return 1500000 + MVV_LVA_VALS[promo_type] + cap_bonus;
    }

    if (flags == CAPTURE || flags == EP_CAPTURE) {
        int victim = (flags == EP_CAPTURE) ? PAWN : (board->mailbox[to] % 6);
        int mvv_lva = 10 * MVV_LVA_VALS[victim] - MVV_LVA_VALS[attacker];
        int see_val = board_see(board, move);
        if (see_val >= 0) {
            return 1000000 + see_val * 10 + mvv_lva;
        }
        return -100000 + see_val * 10 + mvv_lva;
    }

    if (ply < MAX_SEARCH_PLY && st != NULL) {
        if (move == st->killers[ply][0]) return 900000;
        if (move == st->killers[ply][1]) return 800000;
    }

    int hist = (st != NULL) ? st->history[board->turn][from][to] : 0;
    int to_c = CENTER_WEIGHT[to % 8] + CENTER_WEIGHT[to / 8];
    int from_c = CENTER_WEIGHT[from % 8] + CENTER_WEIGHT[from / 8];
    return hist + (to_c - from_c) * 10 + attacker;
}

// Swap the highest-scoring move from [idx..count-1] into position idx.
static void pick_next_move(Move* moves, int* scores, int count, int idx) {
    int best_idx = idx;
    int best_score = scores[idx];
    for (int i = idx + 1; i < count; i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_idx = i;
        }
    }
    if (best_idx != idx) {
        Move tmp_m = moves[idx];
        moves[idx] = moves[best_idx];
        moves[best_idx] = tmp_m;

        int tmp_s = scores[idx];
        scores[idx] = scores[best_idx];
        scores[best_idx] = tmp_s;
    }
}

// Quiescence search over tactical moves with stand-pat and SEE pruning.
static int quiescence(Board* board, int alpha, int beta, int ply, SearchState* st) {
    if (check_stop(st)) return 0;
    st->nodes++;

    if (ply >= MAX_SEARCH_PLY) {
        return board_evaluate(board);
    }

    bool in_check = board_is_check(board);
    if (in_check) {
        Move evasions[MAX_MOVES];
        if (board_generate_legal_moves(board, evasions) == 0) {
            return -SCORE_MATE + ply;
        }
    }

    int stand_pat = board_evaluate(board);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int count = board_generate_captures(board, moves);
    for (int i = 0; i < count; i++) {
        scores[i] = score_move(board, moves[i], 0, ply, st);
    }

    for (int i = 0; i < count; i++) {
        pick_next_move(moves, scores, count, i);
        Move m = moves[i];

        // Skip captures with negative SEE in quiescence when not in check.
        if (!in_check && MOVE_FLAGS(m) < PROMO_N && scores[i] < 0) {
            continue;
        }

        board_make_move(board, m);
        int score = -quiescence(board, -beta, -alpha, ply + 1, st);
        board_unmake_move(board);

        if (st->aborted) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// Principal Variation Search (PVS) with NMP, RFP, FP, SEE pruning, LMR, killers, history, and check extensions.
static int negamax(Board* board, int depth, int alpha, int beta, int ply, SearchState* st, bool can_null) {
    if (ply < MAX_SEARCH_PLY) {
        st->pv_length[ply] = ply;
    }

    if (check_stop(st)) return 0;

    if (ply > 0) {
        if (board->halfmove_clock >= 100 || is_search_repetition(board) || board_is_insufficient_material(board)) {
            return 0;
        }
    }

    bool in_check = board_is_check(board);
    if (in_check && ply > 0 && ply < MAX_SEARCH_PLY - 2) {
        depth++;
    }

    if (depth <= 0 || ply >= MAX_SEARCH_PLY) {
        return quiescence(board, alpha, beta, ply, st);
    }

    st->nodes++;

    bool is_pv = (beta - alpha) > 1;
    int orig_alpha = alpha;
    Move tt_move = 0;
    int tt_score = 0;
    if (tt_probe(board->hash, depth, alpha, beta, ply, &tt_move, &tt_score)) {
        if (!is_pv && ply > 0) {
            return tt_score;
        }
    }

    int static_eval = in_check ? -SCORE_INF : board_evaluate(board);

    // Reverse Futility Pruning (Static Null Move Pruning)
    if (!is_pv && !in_check && depth <= 5 &&
        beta < SCORE_MATE_THRESHOLD && beta > -SCORE_MATE_THRESHOLD) {
        int rfp_margin = 85 * depth;
        if (static_eval - rfp_margin >= beta) {
            return static_eval - rfp_margin;
        }
    }

    // Null-Move Pruning with non-pawn material zugzwang guard
    if (!is_pv && !in_check && can_null && depth >= 3 &&
        has_non_pawn_material(board, board->turn) &&
        static_eval >= beta && beta < SCORE_MATE_THRESHOLD) {
        int R = 2 + depth / 4;
        board_make_null_move(board);
        int null_score = -negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1, st, false);
        board_unmake_null_move(board);
        if (st->aborted) return 0;
        if (null_score >= beta) {
            return (null_score >= SCORE_MATE_THRESHOLD) ? beta : null_score;
        }
    }

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int count = board_generate_legal_moves(board, moves);
    if (count == 0) {
        return in_check ? (-SCORE_MATE + ply) : 0;
    }

    for (int i = 0; i < count; i++) {
        scores[i] = score_move(board, moves[i], tt_move, ply, st);
    }

    Move best_move = moves[0];
    int best_score = -SCORE_INF;
    int moves_searched = 0;

    for (int i = 0; i < count; i++) {
        pick_next_move(moves, scores, count, i);
        Move m = moves[i];
        int flags = MOVE_FLAGS(m);
        bool is_tactical = (flags == CAPTURE || flags == EP_CAPTURE || flags >= PROMO_N);

        // SEE pruning for heavily losing captures at shallow depth
        if (!is_pv && !in_check && moves_searched > 0 && depth <= 3 &&
            (flags == CAPTURE || flags == EP_CAPTURE) && scores[i] < 0) {
            if (board_see(board, m) < -100 * depth) {
                continue;
            }
        }

        board_make_move(board, m);
        bool gives_check = board_is_check(board);

        // Futility pruning for quiet non-checking moves at shallow depth
        if (!is_pv && !in_check && !gives_check && moves_searched > 0 && depth <= 3 && !is_tactical &&
            alpha > -SCORE_MATE_THRESHOLD && beta < SCORE_MATE_THRESHOLD) {
            if (static_eval + 110 * depth <= alpha) {
                board_unmake_move(board);
                continue;
            }
        }

        moves_searched++;
        int score;

        if (moves_searched == 1) {
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, st, true);
        } else {
            int R = 0;
            if (depth >= 3 && moves_searched >= 4 && !in_check && !gives_check && !is_tactical) {
                R = 1 + (depth / 5) + (moves_searched / 10);
                if (is_pv) R--;
                if (ply < MAX_SEARCH_PLY && (m == st->killers[ply][0] || m == st->killers[ply][1])) R--;
                if (R < 0) R = 0;
                if (R > depth - 2) R = depth - 2;
            }

            score = -negamax(board, depth - 1 - R, -alpha - 1, -alpha, ply + 1, st, true);
            if (R > 0 && score > alpha && !st->aborted) {
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, st, true);
            }
            if (is_pv && score > alpha && score < beta && !st->aborted) {
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, st, true);
            }
        }

        board_unmake_move(board);

        if (st->aborted) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }

        if (score > alpha) {
            alpha = score;
            if (ply < MAX_SEARCH_PLY) {
                st->pv_table[ply][ply] = m;
                int child_len = (ply + 1 < MAX_SEARCH_PLY) ? st->pv_length[ply + 1] : (ply + 1);
                for (int next = ply + 1; next < child_len && next < MAX_SEARCH_PLY; next++) {
                    st->pv_table[ply][next] = st->pv_table[ply + 1][next];
                }
                st->pv_length[ply] = child_len;
            }
        }

        if (alpha >= beta) {
            if (!is_tactical && ply < MAX_SEARCH_PLY) {
                if (st->killers[ply][0] != m) {
                    st->killers[ply][1] = st->killers[ply][0];
                    st->killers[ply][0] = m;
                }
                int bonus = depth * depth;
                int us = board->turn;
                int from = MOVE_FROM(m);
                int to = MOVE_TO(m);
                st->history[us][from][to] += bonus;
                if (st->history[us][from][to] > 300000) {
                    st->history[us][from][to] = 300000;
                }
            }
            break;
        }
    }

    int bound = TT_EXACT;
    if (best_score <= orig_alpha) {
        bound = TT_UPPER;
    } else if (best_score >= beta) {
        bound = TT_LOWER;
    }
    tt_store(board->hash, depth, best_score, bound, best_move, ply);

    return best_score;
}

// Run iterative deepening with aspiration windows and dynamic soft-time extension.
void search_best_move(
    Board* board,
    const SearchLimits* limits,
    SearchInfoCallback callback,
    SearchResult* result
) {
    memset(result, 0, sizeof(SearchResult));

    Move root_moves[MAX_MOVES];
    int root_count = board_generate_legal_moves(board, root_moves);
    if (root_count == 0) {
        result->best_move = 0;
        result->score = board_is_check(board) ? -SCORE_MATE : 0;
        return;
    }

    SearchState st;
    memset(&st, 0, sizeof(SearchState));
    if (limits != NULL) {
        st.limits = *limits;
    }
    if (st.limits.max_depth <= 0 || st.limits.max_depth >= MAX_SEARCH_PLY) {
        st.limits.max_depth = MAX_SEARCH_PLY - 1;
    }

    st.start_ms = now_ms();
    result->best_move = root_moves[0];

    Move prev_best_move = 0;
    int prev_score = 0;
    int best_move_changes = 0;

    for (int d = 1; d <= st.limits.max_depth; d++) {
        st.current_depth = d;
        int score = 0;

        if (d <= 4) {
            score = negamax(board, d, -SCORE_INF, SCORE_INF, 0, &st, true);
        } else {
            int delta = 35;
            int alpha = (prev_score - delta > -SCORE_INF) ? (prev_score - delta) : -SCORE_INF;
            int beta = (prev_score + delta < SCORE_INF) ? (prev_score + delta) : SCORE_INF;

            while (true) {
                score = negamax(board, d, alpha, beta, 0, &st, true);
                if (st.aborted) break;

                if (score <= alpha) {
                    alpha = (score - delta > -SCORE_INF) ? (score - delta) : -SCORE_INF;
                    delta *= 2;
                } else if (score >= beta) {
                    beta = (score + delta < SCORE_INF) ? (score + delta) : SCORE_INF;
                    delta *= 2;
                } else {
                    break;
                }

                if (delta > 500) {
                    alpha = -SCORE_INF;
                    beta = SCORE_INF;
                }
            }
        }

        int elapsed = (int)(now_ms() - st.start_ms);
        if (elapsed < 0) elapsed = 0;

        if (st.aborted && d > 1) {
            result->nodes = st.nodes;
            result->time_ms = elapsed;
            break;
        }

        result->depth = d;
        result->score = score;
        result->nodes = st.nodes;
        result->time_ms = elapsed;

        if (st.pv_length[0] > 0) {
            result->best_move = st.pv_table[0][0];
            result->pv_length = st.pv_length[0];
            for (int i = 0; i < st.pv_length[0]; i++) {
                result->pv[i] = st.pv_table[0][i];
            }
        }

        int score_drop = 0;
        if (d >= 2) {
            if (result->best_move != prev_best_move) {
                best_move_changes++;
            }
            if (score < prev_score) {
                score_drop = prev_score - score;
            }
        }
        prev_best_move = result->best_move;
        prev_score = score;

        if (callback != NULL) {
            callback(d, result->score, result->nodes, result->time_ms, result->pv, result->pv_length);
        }

        if (st.limits.stop_flag != NULL && *(st.limits.stop_flag) != 0) {
            break;
        }
        if (st.limits.max_nodes > 0 && st.nodes >= st.limits.max_nodes) {
            break;
        }
        if (st.limits.movetime_ms > 0 && elapsed >= st.limits.movetime_ms) {
            break;
        }
        if (st.limits.soft_movetime_ms > 0) {
            int effective_soft = st.limits.soft_movetime_ms;
            if (d >= 4) {
                int capped_changes = (best_move_changes < 3) ? best_move_changes : 3;
                if (capped_changes > 0) {
                    effective_soft += (st.limits.soft_movetime_ms * 35 * capped_changes) / 100;
                }
                if (score_drop >= 30) {
                    effective_soft += (st.limits.soft_movetime_ms * 35) / 100;
                }
                if (st.limits.movetime_ms > 0 && effective_soft > st.limits.movetime_ms) {
                    effective_soft = st.limits.movetime_ms;
                }
            }
            if (elapsed >= effective_soft) {
                break;
            }
        }
    }
}
