#include "engine.h"
#include <stdio.h>
#include <time.h>

void init_engine(void) {
    init_bitboard_tables();
    init_zobrist();
    init_magics();
}

uint64_t board_perft(Board* board, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int count = board_generate_legal_moves(board, moves);
    if (depth == 1) return count;
    
    uint64_t nodes = 0;
    for (int i = 0; i < count; i++) {
        board_make_move(board, moves[i]);
        nodes += board_perft(board, depth - 1);
        board_unmake_move(board);
    }
    return nodes;
}

void perft_divide(Board* board, int depth) {
    if (depth == 0) return;
    Move moves[MAX_MOVES];
    int count = board_generate_legal_moves(board, moves);
    
    uint64_t total_nodes = 0;
    for (int i = 0; i < count; i++) {
        board_make_move(board, moves[i]);
        uint64_t nodes = board_perft(board, depth - 1);
        board_unmake_move(board);
        
        int from = MOVE_FROM(moves[i]);
        int to = MOVE_TO(moves[i]);
        int promo = MOVE_FLAGS(moves[i]);
        
        char move_str[6];
        move_str[0] = (from % 8) + 'a';
        move_str[1] = (from / 8) + '1';
        move_str[2] = (to % 8) + 'a';
        move_str[3] = (to / 8) + '1';
        move_str[4] = '\0';
        
        if (promo == PROMO_Q || promo == CAPTURE_PROMO_Q) move_str[4] = 'q';
        else if (promo == PROMO_R || promo == CAPTURE_PROMO_R) move_str[4] = 'r';
        else if (promo == PROMO_B || promo == CAPTURE_PROMO_B) move_str[4] = 'b';
        else if (promo == PROMO_N || promo == CAPTURE_PROMO_N) move_str[4] = 'n';
        if (move_str[4] != '\0') move_str[5] = '\0';
        
        printf("%s: %llu\n", move_str, (unsigned long long)nodes);
        total_nodes += nodes;
    }
    printf("\nTotal nodes: %llu\n", (unsigned long long)total_nodes);
}
