import os
from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
    typedef struct Board Board;

    void init_engine(void);

    Board* board_new(void);
    void board_from_fen(Board* board, const char* fen);
    void board_free(Board* board);
    void board_to_fen(const Board* board, char* buf, int bufsize);
    Board* board_copy(const Board* board);

    void board_make_move(Board* board, uint16_t move);
    void board_unmake_move(Board* board);

    int board_generate_legal_moves(Board* board, uint16_t* moves);
    int board_generate_pseudo_legal_moves(const Board* board, uint16_t* moves);
    int board_generate_captures(Board* board, uint16_t* moves);

    bool board_is_check(const Board* board);
    bool board_is_checkmate(Board* board);
    bool board_is_stalemate(Board* board);
    bool board_is_draw(const Board* board);
    bool board_is_game_over(Board* board);
    bool board_is_repetition(const Board* board);
    bool board_is_insufficient_material(const Board* board);

    uint64_t board_hash(const Board* board);
    int board_turn(const Board* board);
    int board_castling(const Board* board);
    int board_ep_square(const Board* board);
    int board_halfmove_clock(const Board* board);
    int board_fullmove_number(const Board* board);
    int board_piece_at(const Board* board, int sq);
    bool board_is_attacked_by(const Board* board, int sq, int color);
    uint64_t board_attackers_to(const Board* board, int sq, uint64_t occupied);

    uint64_t board_perft(Board* board, int depth);
    void perft_divide(Board* board, int depth);

    uint64_t rook_attacks(int sq, uint64_t occupied);
    uint64_t bishop_attacks(int sq, uint64_t occupied);
    uint64_t queen_attacks(int sq, uint64_t occupied);
""")

_SRC_DIR = os.path.join("castled", "core", "src")

ffibuilder.set_source(
    "castled.core._engine",
    '#include "engine.h"',
    sources=[
        os.path.join(_SRC_DIR, "bitboard.c"),
        os.path.join(_SRC_DIR, "magics.c"),
        os.path.join(_SRC_DIR, "zobrist.c"),
        os.path.join(_SRC_DIR, "board.c"),
        os.path.join(_SRC_DIR, "movegen.c"),
        os.path.join(_SRC_DIR, "engine.c"),
    ],
    include_dirs=[_SRC_DIR],
    extra_compile_args=["-O2", "-std=c99"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

