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
    void board_make_null_move(Board* board);
    void board_unmake_null_move(Board* board);

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
    int board_evaluate(const Board* board);

    void nnue_init(void);
    bool nnue_load_file(const char* path);
    bool nnue_save_file(const char* path);
    bool nnue_is_loaded(void);
    void nnue_set_weights(const int16_t* w1, const int16_t* b1, const int16_t* w2, int16_t b2);
    void nnue_get_weights(int16_t* w1, int16_t* b1, int16_t* w2, int16_t* b2);
    void nnue_get_accumulator(const Board* board, int16_t* acc_white, int16_t* acc_black);
    void nnue_refresh_accumulator(Board* board);
    int nnue_evaluate(Board* board);
    int nnue_evaluate_recompute(const Board* board);

    void tt_init(int size_mb);
    void tt_free(void);
    void tt_clear(void);
    void tt_store(uint64_t key, int depth, int score, int bound, uint16_t best_move, int ply);
    bool tt_probe(uint64_t key, int depth, int alpha, int beta, int ply, uint16_t* best_move, int* score);
    bool tt_probe_entry(uint64_t key, int ply, uint16_t* best_move, int* score, int* depth, int* bound);

    typedef struct {
        int max_depth;
        uint64_t max_nodes;
        int movetime_ms;
        int soft_movetime_ms;
        int use_nnue;
        volatile int* stop_flag;
    } SearchLimits;

    typedef struct {
        uint16_t best_move;
        int score;
        int depth;
        uint64_t nodes;
        int time_ms;
        uint16_t pv[64];
        int pv_length;
    } SearchResult;

    typedef void (*SearchInfoCallback)(
        int depth, int score, uint64_t nodes, int time_ms, const uint16_t* pv, int pv_len
    );

    int board_see(const Board* board, uint16_t move);

    void search_best_move(
        Board* board,
        const SearchLimits* limits,
        SearchInfoCallback callback,
        SearchResult* result
    );

    uint64_t rook_attacks(int sq, uint64_t occupied);
    uint64_t bishop_attacks(int sq, uint64_t occupied);
    uint64_t queen_attacks(int sq, uint64_t occupied);
""")

_SRC_DIR = os.path.join("castled", "core", "src")

import platform

_COMPILE_ARGS = ["-O3", "-std=c99"]
if platform.machine().lower() in ("x86_64", "amd64"):
    _COMPILE_ARGS.append("-march=native")

ffibuilder.set_source(
    "castled.core._engine",
    '#include "engine.h"',
    sources=[
        os.path.join(_SRC_DIR, "bitboard.c"),
        os.path.join(_SRC_DIR, "magics.c"),
        os.path.join(_SRC_DIR, "zobrist.c"),
        os.path.join(_SRC_DIR, "board.c"),
        os.path.join(_SRC_DIR, "movegen.c"),
        os.path.join(_SRC_DIR, "eval.c"),
        os.path.join(_SRC_DIR, "nnue.c"),
        os.path.join(_SRC_DIR, "tt.c"),
        os.path.join(_SRC_DIR, "search.c"),
        os.path.join(_SRC_DIR, "engine.c"),
    ],
    include_dirs=[_SRC_DIR],
    extra_compile_args=_COMPILE_ARGS,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)


