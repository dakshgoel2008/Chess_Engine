#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "board.h"

int board_generate_pseudo_legal_moves(const Board* board, Move* moves);
int board_generate_legal_moves(Board* board, Move* moves);
int board_generate_captures(Board* board, Move* moves);

bool board_is_checkmate(Board* board);
bool board_is_stalemate(Board* board);
bool board_is_game_over(Board* board);

#endif // MOVEGEN_H
