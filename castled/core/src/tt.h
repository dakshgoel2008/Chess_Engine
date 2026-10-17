#ifndef TT_H
#define TT_H

#include "types.h"

#define SCORE_INF 32000
#define SCORE_MATE 30000
#define SCORE_MATE_THRESHOLD 29000

enum {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3
};

typedef struct {
    uint64_t key;
    Move best_move;
    int16_t score;
    int8_t depth;
    uint8_t bound;
} TTEntry;

void tt_init(int size_mb);
void tt_free(void);
void tt_clear(void);
void tt_store(uint64_t key, int depth, int score, int bound, Move best_move, int ply);
bool tt_probe(uint64_t key, int depth, int alpha, int beta, int ply, Move* best_move, int* score);
bool tt_probe_entry(uint64_t key, int ply, Move* best_move, int* score, int* depth, int* bound);

#endif // TT_H

