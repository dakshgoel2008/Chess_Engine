#include "bitboard.h"
#include <stdio.h>

uint64_t KNIGHT_ATTACKS[64];
uint64_t KING_ATTACKS[64];
uint64_t PAWN_ATTACKS[2][64];

const int CASTLING_RIGHTS_UPDATE[64] = {
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11
};

void init_bitboard_tables(void) {
    for (int sq = 0; sq < 64; sq++) {
        uint64_t bb = 1ULL << sq;
        
        // Knight attacks
        uint64_t n = 0;
        n |= (bb & ~(FILE_A | FILE_B)) >> 10;
        n |= (bb & ~FILE_A) >> 17;
        n |= (bb & ~FILE_H) >> 15;
        n |= (bb & ~(FILE_G | FILE_H)) >> 6;
        n |= (bb & ~(FILE_A | FILE_B)) << 6;
        n |= (bb & ~FILE_A) << 15;
        n |= (bb & ~FILE_H) << 17;
        n |= (bb & ~(FILE_G | FILE_H)) << 10;
        KNIGHT_ATTACKS[sq] = n;
        
        // King attacks
        uint64_t k = 0;
        k |= (bb & ~FILE_A) >> 9;
        k |= bb >> 8;
        k |= (bb & ~FILE_H) >> 7;
        k |= (bb & ~FILE_A) >> 1;
        k |= (bb & ~FILE_H) << 1;
        k |= (bb & ~FILE_A) << 7;
        k |= bb << 8;
        k |= (bb & ~FILE_H) << 9;
        KING_ATTACKS[sq] = k;
        
        // Pawn attacks
        PAWN_ATTACKS[WHITE][sq] = ((bb & ~FILE_A) << 7) | ((bb & ~FILE_H) << 9);
        PAWN_ATTACKS[BLACK][sq] = ((bb & ~FILE_A) >> 9) | ((bb & ~FILE_H) >> 7);
    }
}

void bb_to_string(uint64_t bb) {
    for (int r = 7; r >= 0; r--) {
        for (int f = 0; f < 8; f++) {
            int sq = r * 8 + f;
            printf("%c ", (bb & (1ULL << sq)) ? '1' : '.');
        }
        printf("\n");
    }
    printf("\n");
}
