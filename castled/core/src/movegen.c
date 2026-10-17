#include "movegen.h"
#include "bitboard.h"
#include "magics.h"

static inline void add_move(Move* moves, int* count, int from, int to, int flags) {
    moves[(*count)++] = MAKE_MOVE(from, to, flags);
}

int board_generate_pseudo_legal_moves(const Board* board, Move* moves) {
    int count = 0;
    int us = board->turn;
    int them = us ^ 1;
    uint64_t our_pieces = board->occupied[us];
    uint64_t their_pieces = board->occupied[them];
    uint64_t all_pieces = board->occupied[2];
    uint64_t empty = ~all_pieces;

    int offset = (us == WHITE) ? 0 : 6;
    
    // Pawns
    uint64_t pawns = board->pieces[WHITE_PAWN + offset];
    while(pawns) {
        int sq = pop_lsb(&pawns);
        int r = sq / 8;
        int push_sq = (us == WHITE) ? sq + 8 : sq - 8;
        if (empty & (1ULL << push_sq)) {
            int promo_r = (us == WHITE) ? 7 : 0;
            if (push_sq / 8 == promo_r) {
                add_move(moves, &count, sq, push_sq, PROMO_Q);
                add_move(moves, &count, sq, push_sq, PROMO_R);
                add_move(moves, &count, sq, push_sq, PROMO_B);
                add_move(moves, &count, sq, push_sq, PROMO_N);
            } else {
                add_move(moves, &count, sq, push_sq, QUIET);
                int start_r = (us == WHITE) ? 1 : 6;
                if (r == start_r) {
                    int dpush_sq = (us == WHITE) ? sq + 16 : sq - 16;
                    if (empty & (1ULL << dpush_sq)) {
                        add_move(moves, &count, sq, dpush_sq, DOUBLE_PAWN);
                    }
                }
            }
        }
        
        uint64_t att = PAWN_ATTACKS[us][sq] & their_pieces;
        while(att) {
            int cap_sq = pop_lsb(&att);
            int promo_r = (us == WHITE) ? 7 : 0;
            if (cap_sq / 8 == promo_r) {
                add_move(moves, &count, sq, cap_sq, CAPTURE_PROMO_Q);
                add_move(moves, &count, sq, cap_sq, CAPTURE_PROMO_R);
                add_move(moves, &count, sq, cap_sq, CAPTURE_PROMO_B);
                add_move(moves, &count, sq, cap_sq, CAPTURE_PROMO_N);
            } else {
                add_move(moves, &count, sq, cap_sq, CAPTURE);
            }
        }
        
        if (board->ep_square != SQ_NONE) {
            uint64_t ep_bb = 1ULL << board->ep_square;
            if (PAWN_ATTACKS[us][sq] & ep_bb) {
                add_move(moves, &count, sq, board->ep_square, EP_CAPTURE);
            }
        }
    }

    // Knights
    uint64_t knights = board->pieces[WHITE_KNIGHT + offset];
    while(knights) {
        int sq = pop_lsb(&knights);
        uint64_t att = KNIGHT_ATTACKS[sq] & ~our_pieces;
        while(att) {
            int to = pop_lsb(&att);
            add_move(moves, &count, sq, to, (their_pieces & (1ULL << to)) ? CAPTURE : QUIET);
        }
    }

    // Bishops
    uint64_t bishops = board->pieces[WHITE_BISHOP + offset];
    while(bishops) {
        int sq = pop_lsb(&bishops);
        uint64_t att = bishop_attacks(sq, all_pieces) & ~our_pieces;
        while(att) {
            int to = pop_lsb(&att);
            add_move(moves, &count, sq, to, (their_pieces & (1ULL << to)) ? CAPTURE : QUIET);
        }
    }

    // Rooks
    uint64_t rooks = board->pieces[WHITE_ROOK + offset];
    while(rooks) {
        int sq = pop_lsb(&rooks);
        uint64_t att = rook_attacks(sq, all_pieces) & ~our_pieces;
        while(att) {
            int to = pop_lsb(&att);
            add_move(moves, &count, sq, to, (their_pieces & (1ULL << to)) ? CAPTURE : QUIET);
        }
    }

    // Queens
    uint64_t queens = board->pieces[WHITE_QUEEN + offset];
    while(queens) {
        int sq = pop_lsb(&queens);
        uint64_t att = queen_attacks(sq, all_pieces) & ~our_pieces;
        while(att) {
            int to = pop_lsb(&att);
            add_move(moves, &count, sq, to, (their_pieces & (1ULL << to)) ? CAPTURE : QUIET);
        }
    }

    // Kings
    if (board->pieces[WHITE_KING + offset]) {
        int king_sq = lsb(board->pieces[WHITE_KING + offset]);
        uint64_t att = KING_ATTACKS[king_sq] & ~our_pieces;
        while(att) {
            int to = pop_lsb(&att);
            add_move(moves, &count, king_sq, to, (their_pieces & (1ULL << to)) ? CAPTURE : QUIET);
        }
    }

    // Castling
    if (us == WHITE) {
        if (board->castling & WK) {
            if ((empty & ((1ULL << F1) | (1ULL << G1))) == ((1ULL << F1) | (1ULL << G1))) {
                if (!board_is_attacked_by(board, E1, BLACK) && !board_is_attacked_by(board, F1, BLACK) && !board_is_attacked_by(board, G1, BLACK)) {
                    add_move(moves, &count, E1, G1, KING_CASTLE);
                }
            }
        }
        if (board->castling & WQ) {
            if ((empty & ((1ULL << B1) | (1ULL << C1) | (1ULL << D1))) == ((1ULL << B1) | (1ULL << C1) | (1ULL << D1))) {
                if (!board_is_attacked_by(board, E1, BLACK) && !board_is_attacked_by(board, D1, BLACK) && !board_is_attacked_by(board, C1, BLACK)) {
                    add_move(moves, &count, E1, C1, QUEEN_CASTLE);
                }
            }
        }
    } else {
        if (board->castling & BK) {
            if ((empty & ((1ULL << F8) | (1ULL << G8))) == ((1ULL << F8) | (1ULL << G8))) {
                if (!board_is_attacked_by(board, E8, WHITE) && !board_is_attacked_by(board, F8, WHITE) && !board_is_attacked_by(board, G8, WHITE)) {
                    add_move(moves, &count, E8, G8, KING_CASTLE);
                }
            }
        }
        if (board->castling & BQ) {
            if ((empty & ((1ULL << B8) | (1ULL << C8) | (1ULL << D8))) == ((1ULL << B8) | (1ULL << C8) | (1ULL << D8))) {
                if (!board_is_attacked_by(board, E8, WHITE) && !board_is_attacked_by(board, D8, WHITE) && !board_is_attacked_by(board, C8, WHITE)) {
                    add_move(moves, &count, E8, C8, QUEEN_CASTLE);
                }
            }
        }
    }

    return count;
}

int board_generate_legal_moves(Board* board, Move* moves) {
    Move pseudo[MAX_MOVES];
    int pcount = board_generate_pseudo_legal_moves(board, pseudo);
    int lcount = 0;
    int us = board->turn;
    
    for (int i = 0; i < pcount; i++) {
        board_make_move(board, pseudo[i]);
        int king_pc = (us == WHITE) ? WHITE_KING : BLACK_KING;
        if (board->pieces[king_pc]) {
            int king_sq = lsb(board->pieces[king_pc]);
            if (!board_is_attacked_by(board, king_sq, us ^ 1)) {
                moves[lcount++] = pseudo[i];
            }
        }
        board_unmake_move(board);
    }
    return lcount;
}

// Generate all legal capture and promotion moves (used by quiescence search).
int board_generate_captures(Board* board, Move* moves) {
    Move legal[MAX_MOVES];
    int lcount = board_generate_legal_moves(board, legal);
    int ccount = 0;
    for (int i = 0; i < lcount; i++) {
        int flags = MOVE_FLAGS(legal[i]);
        if (flags == CAPTURE || flags == EP_CAPTURE || flags >= PROMO_N) {
            moves[ccount++] = legal[i];
        }
    }
    return ccount;
}

bool board_is_checkmate(Board* board) {
    if (!board_is_check(board)) return false;
    Move moves[MAX_MOVES];
    return board_generate_legal_moves(board, moves) == 0;
}

bool board_is_stalemate(Board* board) {
    if (board_is_check(board)) return false;
    Move moves[MAX_MOVES];
    return board_generate_legal_moves(board, moves) == 0;
}

bool board_is_game_over(Board* board) {
    return board_is_checkmate(board) || board_is_stalemate(board) || board_is_draw(board);
}
