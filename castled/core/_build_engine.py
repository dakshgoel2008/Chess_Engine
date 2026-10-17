from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
    typedef struct Board Board;
    Board* board_new(void);
    Board* board_from_fen(const char* fen);
    void board_free(Board* board);
    void board_to_fen(const Board* board, char* buf, int bufsize);
    int board_make_move(Board* board, uint16_t move);
    void board_unmake_move(Board* board);
    int board_generate_legal_moves(const Board* board, uint16_t* moves);
    int board_generate_pseudo_legal_moves(const Board* board, uint16_t* moves);
    int board_generate_captures(const Board* board, uint16_t* moves);
    int board_is_check(const Board* board);
    int board_is_checkmate(const Board* board);
    int board_is_stalemate(const Board* board);
    int board_is_draw(const Board* board);
    uint64_t board_hash(const Board* board);
    int board_turn(const Board* board);
    int board_piece_at(const Board* board, int sq);
    uint64_t board_attacks_to(const Board* board, int sq, int color);
    uint64_t board_perft(Board* board, int depth);
    Board* board_copy(const Board* board);
    
    uint64_t rook_attacks(int sq, uint64_t occupied);
    uint64_t bishop_attacks(int sq, uint64_t occupied);
    uint64_t queen_attacks(int sq, uint64_t occupied);
    uint64_t knight_attacks(int sq);
    uint64_t king_attacks(int sq);
    uint64_t pawn_attacks(int color, int sq);
""")

ffibuilder.set_source(
    "castled.core._engine",
    """
    #include "src/engine.h"
    """,
    sources=["src/engine.c"],
    include_dirs=["src"],
    extra_compile_args=["-O2", "-std=c99"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
