#include "search.h"
#include "movegen.h"
#include "eval.h"
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
} SearchState;

static const int MVV_LVA_VALS[7] = { 100, 300, 320, 500, 900, 10000, 0 };
static const int CENTER_WEIGHT[8] = { 0, 1, 2, 3, 3, 2, 1, 0 };

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

// Score moves for ordering: TT move first, then promotions, MVV-LVA captures, and center moves.
static int score_move(const Board* board, Move move, Move tt_move) {
    if (move == tt_move && tt_move != 0) {
        return 1000000;
    }

    int flags = MOVE_FLAGS(move);
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int attacker = board->mailbox[from] % 6;
    int score = 0;

    if (flags == CAPTURE || (flags >= CAPTURE_PROMO_N && flags <= CAPTURE_PROMO_Q)) {
        int victim = board->mailbox[to] % 6;
        score = 100000 + 10 * MVV_LVA_VALS[victim] - MVV_LVA_VALS[attacker];
    } else if (flags == EP_CAPTURE) {
        score = 100000 + 10 * MVV_LVA_VALS[PAWN] - MVV_LVA_VALS[PAWN];
    }

    if (flags >= PROMO_N) {
        int promo_type = (flags & 3) + 1;
        score += 50000 + MVV_LVA_VALS[promo_type];
    }

    if (score == 0) {
        int to_c = CENTER_WEIGHT[to % 8] + CENTER_WEIGHT[to / 8];
        int from_c = CENTER_WEIGHT[from % 8] + CENTER_WEIGHT[from / 8];
        score = (to_c - from_c) * 10 + attacker;
    }
    return score;
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

// Quiescence search over tactical moves (captures and promotions) with stand-pat.
static int quiescence(Board* board, int alpha, int beta, int ply, SearchState* st) {
    if (check_stop(st)) return 0;
    st->nodes++;

    if (ply >= MAX_SEARCH_PLY) {
        return board_evaluate(board);
    }

    if (board_is_check(board)) {
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
        scores[i] = score_move(board, moves[i], 0);
    }

    for (int i = 0; i < count; i++) {
        pick_next_move(moves, scores, count, i);
        board_make_move(board, moves[i]);
        int score = -quiescence(board, -beta, -alpha, ply + 1, st);
        board_unmake_move(board);

        if (st->aborted) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// Negamax alpha-beta search with transposition table and triangular PV tracking.
static int negamax(Board* board, int depth, int alpha, int beta, int ply, SearchState* st) {
    if ( ply < MAX_SEARCH_PLY ) {
        st->pv_length[ply] = ply;
    }

    if (check_stop(st)) return 0;

    if (ply > 0) {
        if (board->halfmove_clock >= 100 || is_search_repetition(board) || board_is_insufficient_material(board)) {
            return 0;
        }
    }

    if (depth <= 0 || ply >= MAX_SEARCH_PLY) {
        return quiescence(board, alpha, beta, ply, st);
    }

    st->nodes++;

    int orig_alpha = alpha;
    Move tt_move = 0;
    int tt_score = 0;
    if (tt_probe(board->hash, depth, alpha, beta, ply, &tt_move, &tt_score)) {
        if (ply > 0) {
            if (tt_move != 0 && ply + 1 < MAX_SEARCH_PLY) {
                st->pv_table[ply][ply] = tt_move;
                st->pv_length[ply] = ply + 1;
            }
            return tt_score;
        }
    }

    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int count = board_generate_legal_moves(board, moves);
    if (count == 0) {
        if (board_is_check(board)) {
            return -SCORE_MATE + ply;
        }
        return 0;
    }

    for (int i = 0; i < count; i++) {
        scores[i] = score_move(board, moves[i], tt_move);
    }

    Move best_move = moves[0];
    int best_score = -SCORE_INF;

    for (int i = 0; i < count; i++) {
        pick_next_move(moves, scores, count, i);
        Move m = moves[i];

        board_make_move(board, m);
        int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, st);
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

// Run iterative deepening up to limits->max_depth and populate result.
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
    if (st.limits.max_depth <= 0) {
        st.limits.max_depth = MAX_SEARCH_PLY - 1;
    }
    if (st.limits.max_depth >= MAX_SEARCH_PLY) {
        st.limits.max_depth = MAX_SEARCH_PLY - 1;
    }

    st.start_ms = now_ms();
    result->best_move = root_moves[0];

    for (int d = 1; d <= st.limits.max_depth; d++) {
        st.current_depth = d;
        int score = negamax(board, d, -SCORE_INF, SCORE_INF, 0, &st);

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
    }
}
