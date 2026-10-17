/* Standalone perft verification for the C core. */

#include "engine.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    init_engine();

    Board* b = board_new();
    char fen[128];
    board_to_fen(b, fen, sizeof(fen));
    assert(strcmp(fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 0);

    assert(board_perft(b, 1) == 20ULL);
    assert(board_perft(b, 2) == 400ULL);
    assert(board_perft(b, 3) == 8902ULL);
    assert(board_perft(b, 4) == 197281ULL);
    assert(board_perft(b, 5) == 4865609ULL);

    board_from_fen(b, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    assert(board_perft(b, 1) == 48ULL);
    assert(board_perft(b, 2) == 2039ULL);
    assert(board_perft(b, 3) == 97862ULL);

    board_free(b);
    printf("All C perft assertions passed.\n");
    return 0;
}

