#include "zobrist.h"
#include "board.h"

uint64_t PIECE_KEYS[12][64];
uint64_t CASTLING_KEYS[16];
uint64_t EP_KEYS[8];
uint64_t SIDE_KEY;

// Xoroshiro128+ or simple PRNG for deterministic keys
static uint64_t prng_state = 1070372;

static uint64_t next_random(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 2685821657736338717ULL;
}

void init_zobrist(void) {
    for (int pc = 0; pc < 12; pc++) {
        for (int sq = 0; sq < 64; sq++) {
            PIECE_KEYS[pc][sq] = next_random();
        }
    }
    for (int i = 0; i < 16; i++) {
        CASTLING_KEYS[i] = next_random();
    }
    for (int i = 0; i < 8; i++) {
        EP_KEYS[i] = next_random();
    }
    SIDE_KEY = next_random();
}

uint64_t compute_hash(const Board* board) {
    uint64_t h = 0;
    for (int sq = 0; sq < 64; sq++) {
        int pc = board_piece_at(board, sq);
        if (pc != PIECE_NONE) {
            h ^= PIECE_KEYS[pc][sq];
        }
    }
    h ^= CASTLING_KEYS[board->castling];
    if (board->ep_square != SQ_NONE) {
        h ^= EP_KEYS[board->ep_square % 8];
    }
    if (board->turn == BLACK) {
        h ^= SIDE_KEY;
    }
    return h;
}
