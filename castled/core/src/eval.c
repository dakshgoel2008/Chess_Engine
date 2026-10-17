#include "eval.h"
#include "bitboard.h"
#include "magics.h"

// Material values for middlegame and endgame (centipawns).
static const int MG_VALUE[6] = { 82, 337, 365, 477, 1025, 0 };
static const int EG_VALUE[6] = { 94, 281, 297, 512,  936, 0 };

// Phase weights per piece type (max total phase = 24).
static const int PHASE_WEIGHT[6] = { 0, 1, 1, 2, 4, 0 };

// PeSTO piece-square tables in LERF order (A1 = index 0, H8 = index 63).
static const int MG_PST[6][64] = {
    // Pawn
    {
          0,   0,   0,   0,   0,   0,   0,   0,
        -35,  -1, -20, -23, -15,  24,  38, -22,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -14,  13,   6,  21,  23,  12,  17, -23,
         -6,   7,  26,  31,  65,  56,  25, -20,
         98, 134,  61,  95,  68, 126,  34, -11,
          0,   0,   0,   0,   0,   0,   0,   0
    },
    // Knight
    {
       -105, -21, -58, -33, -17, -28, -19, -23,
        -29, -53, -12,  -3,  -1,  18, -14, -19,
        -23,  -9,  12,  10,  19,  17,  25, -16,
        -13,   4,  16,  13,  28,  19,  21,  -8,
         -9,  17,  19,  53,  37,  69,  18,  22,
        -47,  60,  37,  65,  84, 129,  73,  44,
        -73, -41,  72,  36,  23,  62,   7, -17,
       -167, -89, -34, -49,  61, -97, -15,-107
    },
    // Bishop
    {
        -33,  -3, -14, -21, -13, -12, -39, -21,
          4,  15,  16,   0,   7,  21,  33,   1,
          0,  15,  15,  15,  14,  27,  18,  10,
         -6,  13,  13,  26,  34,  12,  10,   4,
         -4,   5,  19,  50,  37,  37,   7,  -2,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -29,   4, -82, -37, -25, -42,   7,  -8
    },
    // Rook
    {
        -19, -13,   1,  17,  16,   7, -37, -26,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -24, -11,   7,  26,  24,  35,  -8, -20,
         -5,  19,  26,  36,  17,  45,  61,  16,
         27,  32,  58,  62,  80,  67,  26,  44,
         32,  42,  32,  51,  63,   9,  31,  43
    },
    // Queen
    {
         -1, -18,  -9,  10, -15, -25, -31, -50,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -28,   0,  29,  12,  59,  44,  43,  45
    },
    // King
    {
        -15,  36,  12, -54,   8, -28,  24,  14,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -14, -14, -22, -46, -44, -30, -15, -27,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -17, -20, -12, -27, -30, -25, -14, -36,
         -9,  24,   2, -16, -20,   6,  22, -22,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -65,  23,  16, -15, -56, -34,   2,  13
    }
};

static const int EG_PST[6][64] = {
    // Pawn
    {
          0,   0,   0,   0,   0,   0,   0,   0,
         13,   8,   8,  10,  13,   0,   2,  -7,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
         32,  24,  13,   5,  -2,   4,  17,  17,
         94, 100,  85,  67,  56,  53,  82,  84,
        178, 173, 158, 134, 147, 132, 165, 187,
          0,   0,   0,   0,   0,   0,   0,   0
    },
    // Knight
    {
        -29, -51, -23, -15, -22, -18, -50, -64,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -58, -38, -13, -28, -31, -27, -63, -99
    },
    // Bishop
    {
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
         -3,   9,  12,   9,  14,  10,   3,   2,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
        -14, -21, -11,  -8,  -7,  -9, -17, -24
    },
    // Rook
    {
         -9,   2,   3,  -1,  -5, -13,   4, -20,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
          4,   3,  13,   1,   2,   1,  -1,   2,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
         11,  13,  13,  11,  -3,   3,   8,   3,
         13,  10,  18,  15,  12,  12,   8,   5
    },
    // Queen
    {
        -33, -28, -22, -43,  -5, -32, -20, -41,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -18,  28,  19,  47,  31,  34,  39,  23,
          3,  22,  24,  45,  57,  40,  57,  36,
        -20,   6,   9,  49,  47,  35,  19,   9,
        -17,  20,  32,  41,  58,  25,  30,   0,
         -9,  22,  22,  27,  27,  19,  10,  20
    },
    // King
    {
        -53, -34, -21, -11, -28, -14, -24, -43,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -18,  -4,  21,  24,  27,  23,   9, -11,
         -8,  22,  24,  27,  26,  33,  26,   3,
         10,  17,  23,  15,  20,  45,  44,  13,
        -12,  17,  14,  17,  17,  38,  23,  11,
        -74, -35, -18, -18, -11,  15,   4, -17
    }
};

static const uint64_t FILE_MASKS[8] = {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H
};

static const int PASSED_PAWN_MG[8] = { 0,  5, 10, 20, 35, 60, 100, 0 };
static const int PASSED_PAWN_EG[8] = { 0, 10, 20, 40, 70, 120, 200, 0 };
static const int MOBILITY_MG[6]    = { 0,  4,  4,  2,  1, 0 };
static const int MOBILITY_EG[6]    = { 0,  4,  4,  4,  2, 0 };

// Evaluate pawn structure (doubled, isolated, and passed pawns) for one side.
static void eval_pawns(uint64_t our_pawns, uint64_t their_pawns, int color, int* mg, int* eg) {
    for (int f = 0; f < 8; f++) {
        int count = popcount(our_pawns & FILE_MASKS[f]);
        if (count > 1) {
            *mg -= (count - 1) * 12;
            *eg -= (count - 1) * 20;
        }
    }

    uint64_t bb = our_pawns;
    while (bb) {
        int sq = pop_lsb(&bb);
        int file = sq % 8;
        int rank = sq / 8;
        int rel_rank = (color == WHITE) ? rank : (7 - rank);

        uint64_t adj_files = 0;
        if (file > 0) adj_files |= FILE_MASKS[file - 1];
        if (file < 7) adj_files |= FILE_MASKS[file + 1];

        if ((our_pawns & adj_files) == 0) {
            *mg -= 15;
            *eg -= 20;
        }

        uint64_t span_files = FILE_MASKS[file] | adj_files;
        uint64_t front_mask = 0;
        if (color == WHITE) {
            for (int r = rank + 1; r < 8; r++) front_mask |= (0xFFULL << (r * 8));
        } else {
            for (int r = 0; r < rank; r++) front_mask |= (0xFFULL << (r * 8));
        }

        if ((their_pawns & span_files & front_mask) == 0) {
            *mg += PASSED_PAWN_MG[rel_rank];
            *eg += PASSED_PAWN_EG[rel_rank];
        }
    }
}

// Evaluate piece bonuses, mobility, and king pawn shield for one side.
static void eval_side(const Board* board, int color, int* mg, int* eg, int* phase) {
    int offset = (color == WHITE) ? 0 : 6;
    uint64_t us = board->occupied[color];
    uint64_t all = board->occupied[2];
    uint64_t our_pawns = board->pieces[WHITE_PAWN + offset];
    uint64_t their_pawns = board->pieces[(color == WHITE) ? BLACK_PAWN : WHITE_PAWN];

    for (int pt = 0; pt < 6; pt++) {
        uint64_t bb = board->pieces[offset + pt];
        while (bb) {
            int sq = pop_lsb(&bb);
            int pst_sq = (color == WHITE) ? sq : (sq ^ 56);

            *mg += MG_VALUE[pt] + MG_PST[pt][pst_sq];
            *eg += EG_VALUE[pt] + EG_PST[pt][pst_sq];
            *phase += PHASE_WEIGHT[pt];

            if (pt == KNIGHT) {
                int mob = popcount(KNIGHT_ATTACKS[sq] & ~us);
                *mg += (mob - 4) * MOBILITY_MG[KNIGHT];
                *eg += (mob - 4) * MOBILITY_EG[KNIGHT];
            } else if (pt == BISHOP) {
                int mob = popcount(bishop_attacks(sq, all) & ~us);
                *mg += (mob - 6) * MOBILITY_MG[BISHOP];
                *eg += (mob - 6) * MOBILITY_EG[BISHOP];
            } else if (pt == ROOK) {
                int mob = popcount(rook_attacks(sq, all) & ~us);
                *mg += (mob - 7) * MOBILITY_MG[ROOK];
                *eg += (mob - 7) * MOBILITY_EG[ROOK];

                uint64_t fmask = FILE_MASKS[sq % 8];
                if ((our_pawns & fmask) == 0) {
                    if ((their_pawns & fmask) == 0) {
                        *mg += 25;
                        *eg += 15;
                    } else {
                        *mg += 12;
                        *eg += 8;
                    }
                }
            } else if (pt == QUEEN) {
                int mob = popcount(queen_attacks(sq, all) & ~us);
                *mg += (mob - 14) * MOBILITY_MG[QUEEN];
                *eg += (mob - 14) * MOBILITY_EG[QUEEN];
            }
        }
    }

    if (popcount(board->pieces[WHITE_BISHOP + offset]) >= 2) {
        *mg += 30;
        *eg += 45;
    }

    eval_pawns(our_pawns, their_pawns, color, mg, eg);

    uint64_t king_bb = board->pieces[WHITE_KING + offset];
    if (king_bb) {
        int ksq = lsb(king_bb);
        int kfile = ksq % 8;
        int krank = ksq / 8;
        int shield_rank = (color == WHITE) ? (krank + 1) : (krank - 1);

        if (shield_rank >= 0 && shield_rank < 8) {
            for (int df = -1; df <= 1; df++) {
                int f = kfile + df;
                if (f >= 0 && f < 8) {
                    int s_sq = shield_rank * 8 + f;
                    if (our_pawns & (1ULL << s_sq)) {
                        *mg += 10;
                    }
                }
            }
        }
    }
}

int board_evaluate(const Board* board) {
    int w_mg = 0, w_eg = 0;
    int b_mg = 0, b_eg = 0;
    int phase = 0;

    eval_side(board, WHITE, &w_mg, &w_eg, &phase);
    eval_side(board, BLACK, &b_mg, &b_eg, &phase);

    if (phase > 24) phase = 24;

    int mg_diff = w_mg - b_mg;
    int eg_diff = w_eg - b_eg;
    int score = (mg_diff * phase + eg_diff * (24 - phase)) / 24;

    return (board->turn == WHITE) ? score : -score;
}
