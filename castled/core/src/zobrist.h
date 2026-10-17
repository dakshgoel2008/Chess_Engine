#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"

extern uint64_t PIECE_KEYS[12][64];
extern uint64_t CASTLING_KEYS[16];
extern uint64_t EP_KEYS[8];
extern uint64_t SIDE_KEY;

void init_zobrist(void);
uint64_t compute_hash(const Board* board);

#endif // ZOBRIST_H
