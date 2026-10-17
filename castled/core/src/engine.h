#ifndef ENGINE_H
#define ENGINE_H

#include "types.h"
#include "bitboard.h"
#include "magics.h"
#include "zobrist.h"
#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "nnue.h"
#include "tt.h"
#include "search.h"

void init_engine(void);
uint64_t board_perft(Board* board, int depth);
void perft_divide(Board* board, int depth);

#endif // ENGINE_H
