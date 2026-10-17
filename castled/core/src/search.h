#ifndef SEARCH_H
#define SEARCH_H

#include "types.h"
#include "board.h"
#include "tt.h"

#define MAX_SEARCH_PLY 64

typedef struct {
    int max_depth;
    uint64_t max_nodes;
    int movetime_ms;
    int soft_movetime_ms;
    int use_nnue;
    volatile int* stop_flag;
} SearchLimits;

typedef struct {
    Move best_move;
    int score;
    int depth;
    uint64_t nodes;
    int time_ms;
    Move pv[MAX_SEARCH_PLY];
    int pv_length;
} SearchResult;

typedef void (*SearchInfoCallback)(
    int depth, int score, uint64_t nodes, int time_ms, const Move* pv, int pv_len
);

int board_see(const Board* board, Move move);

void search_best_move(
    Board* board,
    const SearchLimits* limits,
    SearchInfoCallback callback,
    SearchResult* result
);

#endif // SEARCH_H
