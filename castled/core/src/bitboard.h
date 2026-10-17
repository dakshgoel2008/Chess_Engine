#ifndef BITBOARD_H
#define BITBOARD_H

#include "types.h"
#include <stdint.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#define FILE_A 0x0101010101010101ULL
#define FILE_B 0x0202020202020202ULL
#define FILE_C 0x0404040404040404ULL
#define FILE_D 0x0808080808080808ULL
#define FILE_E 0x1010101010101010ULL
#define FILE_F 0x2020202020202020ULL
#define FILE_G 0x4040404040404040ULL
#define FILE_H 0x8080808080808080ULL

#define RANK_1 0x00000000000000FFULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_3 0x0000000000FF0000ULL
#define RANK_4 0x00000000FF000000ULL
#define RANK_5 0x000000FF00000000ULL
#define RANK_6 0x0000FF0000000000ULL
#define RANK_7 0x00FF000000000000ULL
#define RANK_8 0xFF00000000000000ULL

static inline int popcount(uint64_t bb) {
#if defined(_MSC_VER)
    return (int)__popcnt64(bb);
#else
    return __builtin_popcountll(bb);
#endif
}

static inline int lsb(uint64_t bb) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return (int)idx;
#else
    return __builtin_ctzll(bb);
#endif
}

static inline int msb(uint64_t bb) {
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, bb);
    return (int)idx;
#else
    return 63 - __builtin_clzll(bb);
#endif
}

static inline int pop_lsb(uint64_t *bb) {
    int sq = lsb(*bb);
    *bb &= *bb - 1;
    return sq;
}

extern uint64_t KNIGHT_ATTACKS[64];
extern uint64_t KING_ATTACKS[64];
extern uint64_t PAWN_ATTACKS[2][64];

void init_bitboard_tables(void);
void bb_to_string(uint64_t bb);

#endif // BITBOARD_H
