#ifndef EVAL_H
#define EVAL_H

#include "board.h"

// Evaluate the board from the side-to-move's perspective (in centipawns).
int board_evaluate(const Board* board);

#endif // EVAL_H

