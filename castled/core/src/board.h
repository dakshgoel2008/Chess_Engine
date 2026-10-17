#ifndef BOARD_H
#define BOARD_H

#include "types.h"

typedef struct {
    Move move;
    int castling;
    int ep_square;
    int halfmove_clock;
    uint64_t hash;
    int captured_piece;
} State;

struct Board {
    uint64_t pieces[12];
    uint64_t occupied[3]; // WHITE, BLACK, BOTH
    int mailbox[64];
    int turn;
    int castling;
    int ep_square;
    int halfmove_clock;
    int fullmove_number;
    uint64_t hash;
    
    State history[1024];
    int history_count;
    
    uint64_t hash_history[2048];
    int hash_history_count;
};

Board* board_new(void);
void board_from_fen(Board* board, const char* fen);
void board_free(Board* board);
void board_to_fen(const Board* board, char* buf, int bufsize);
Board* board_copy(const Board* board);

void board_make_move(Board* board, Move move);
void board_unmake_move(Board* board);

int board_piece_at(const Board* board, int sq);
int board_turn(const Board* board);
uint64_t board_hash(const Board* board);
bool board_is_check(const Board* board);
bool board_is_attacked_by(const Board* board, int sq, int color);
uint64_t board_attackers_to(const Board* board, int sq, uint64_t occupied);

bool board_is_repetition(const Board* board);
bool board_is_draw(const Board* board);
bool board_is_insufficient_material(const Board* board);

#endif // BOARD_H
