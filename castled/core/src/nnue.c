#include "nnue.h"
#include <stdio.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Network parameters: 768 inputs -> 256 hidden units per perspective -> 512 concatenated -> 1 output.
#if defined(__GNUC__) || defined(__clang__)
#define NNUE_ALIGNED __attribute__((aligned(32)))
#else
#define NNUE_ALIGNED
#endif

static int16_t g_w1[NNUE_INPUT_SIZE][NNUE_HIDDEN_SIZE] NNUE_ALIGNED;
static int16_t g_b1[NNUE_HIDDEN_SIZE] NNUE_ALIGNED;
static int16_t g_w2[NNUE_HIDDEN_SIZE * 2] NNUE_ALIGNED;
static int16_t g_b2 = 0;
static bool g_loaded = false;
static uint32_t g_weights_epoch = 1;

static const int DEFAULT_PIECE_CP[6] = { 100, 320, 330, 500, 900, 0 };
static const int DEFAULT_CENTER_BONUS[8] = { -6, -2, 2, 6, 6, 2, -2, -6 };

// Map (piece, square) to White-perspective feature index in [0..767].
static inline int feature_idx_white(int pc, int sq) {
    return pc * 64 + sq;
}

// Map (piece, square) to Black-perspective feature index in [0..767] via color and rank flip.
static inline int feature_idx_black(int pc, int sq) {
    int flipped_pc = (pc < 6) ? (pc + 6) : (pc - 6);
    int flipped_sq = sq ^ 56;
    return flipped_pc * 64 + flipped_sq;
}

// Initialize deterministic default int16 weights encoding material and centralization.
void nnue_init(void) {
    memset(g_w1, 0, sizeof(g_w1));
    memset(g_b1, 0, sizeof(g_b1));
    memset(g_w2, 0, sizeof(g_w2));
    g_b2 = 0;

    for (int pc = 0; pc < 12; pc++) {
        int pt = pc % 6;
        bool is_friendly = (pc < 6);
        int base_val = DEFAULT_PIECE_CP[pt];
        for (int sq = 0; sq < 64; sq++) {
            int f = sq & 7;
            int r = sq >> 3;
            int pos_bonus = DEFAULT_CENTER_BONUS[f] + DEFAULT_CENTER_BONUS[r];
            if (pt == PAWN) {
                pos_bonus += (r - 1) * 8;
            }
            int total_cp = base_val + pos_bonus;
            int idx = pc * 64 + sq;

            // Channel 0..5 accumulate friendly piece values; channel 6..11 accumulate opponent piece values.
            int target_ch = is_friendly ? pt : (6 + pt);
            int scaled = (total_cp * 64) / 900;
            if (scaled < 0) scaled = 0;
            if (scaled > NNUE_QA) scaled = NNUE_QA;
            g_w1[idx][target_ch] = (int16_t)scaled;

            // Distribute small deterministic features across remaining hidden units.
            int aux_ch = 12 + ((idx * 7 + pt * 13) % (NNUE_HIDDEN_SIZE - 12));
            g_w1[idx][aux_ch] = (int16_t)(is_friendly ? 12 : -8);
        }
    }

    for (int i = 0; i < 12; i++) {
        g_b1[i] = 0;
    }
    for (int i = 12; i < NNUE_HIDDEN_SIZE; i++) {
        g_b1[i] = 32;
    }

    for (int pt = 0; pt < 6; pt++) {
        g_w2[pt] = 48;
        g_w2[6 + pt] = -48;
        g_w2[NNUE_HIDDEN_SIZE + pt] = -48;
        g_w2[NNUE_HIDDEN_SIZE + 6 + pt] = 48;
    }
    for (int i = 12; i < NNUE_HIDDEN_SIZE; i++) {
        g_w2[i] = 2;
        g_w2[NNUE_HIDDEN_SIZE + i] = -2;
    }

    g_loaded = true;
    g_weights_epoch++;
}

bool nnue_is_loaded(void) {
    return g_loaded;
}

// Load a binary .nnue weight file with header validation.
bool nnue_load_file(const char* path) {
    if (path == NULL || path[0] == '\0') return false;
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) return false;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, fp) != 4) {
        fclose(fp);
        return false;
    }
    if (header[0] != NNUE_MAGIC ||
        header[1] != NNUE_VERSION ||
        header[2] != NNUE_INPUT_SIZE ||
        header[3] != NNUE_HIDDEN_SIZE) {
        fclose(fp);
        return false;
    }

    size_t w1_count = (size_t)NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE;
    if (fread(g_w1, sizeof(int16_t), w1_count, fp) != w1_count ||
        fread(g_b1, sizeof(int16_t), NNUE_HIDDEN_SIZE, fp) != NNUE_HIDDEN_SIZE ||
        fread(g_w2, sizeof(int16_t), NNUE_HIDDEN_SIZE * 2, fp) != (size_t)(NNUE_HIDDEN_SIZE * 2) ||
        fread(&g_b2, sizeof(int16_t), 1, fp) != 1) {
        fclose(fp);
        nnue_init();
        return false;
    }

    fclose(fp);
    g_loaded = true;
    g_weights_epoch++;
    return true;
}

// Save current active C NNUE weights to a binary .nnue file.
bool nnue_save_file(const char* path) {
    if (path == NULL || path[0] == '\0') return false;
    FILE* fp = fopen(path, "wb");
    if (fp == NULL) return false;

    uint32_t header[4] = { NNUE_MAGIC, NNUE_VERSION, NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE };
    size_t w1_count = (size_t)NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE;
    bool ok = (fwrite(header, sizeof(uint32_t), 4, fp) == 4 &&
               fwrite(g_w1, sizeof(int16_t), w1_count, fp) == w1_count &&
               fwrite(g_b1, sizeof(int16_t), NNUE_HIDDEN_SIZE, fp) == NNUE_HIDDEN_SIZE &&
               fwrite(g_w2, sizeof(int16_t), NNUE_HIDDEN_SIZE * 2, fp) == (size_t)(NNUE_HIDDEN_SIZE * 2) &&
               fwrite(&g_b2, sizeof(int16_t), 1, fp) == 1);
    fclose(fp);
    return ok;
}

// Set NNUE weights directly from in-memory arrays and bump the weights epoch.
void nnue_set_weights(
    const int16_t* w1,
    const int16_t* b1,
    const int16_t* w2,
    int16_t b2
) {
    memcpy(g_w1, w1, sizeof(g_w1));
    memcpy(g_b1, b1, sizeof(g_b1));
    memcpy(g_w2, w2, sizeof(g_w2));
    g_b2 = b2;
    g_loaded = true;
    g_weights_epoch++;
}

// Copy active NNUE weights into caller-provided buffers.
void nnue_get_weights(
    int16_t* w1,
    int16_t* b1,
    int16_t* w2,
    int16_t* b2
) {
    if (w1 != NULL) memcpy(w1, g_w1, sizeof(g_w1));
    if (b1 != NULL) memcpy(b1, g_b1, sizeof(g_b1));
    if (w2 != NULL) memcpy(w2, g_w2, sizeof(g_w2));
    if (b2 != NULL) *b2 = g_b2;
}

// Copy the board's current White and Black accumulators into caller buffers.
void nnue_get_accumulator(const Board* board, int16_t* acc_white, int16_t* acc_black) {
    if (acc_white != NULL) memcpy(acc_white, board->nnue_acc[WHITE], sizeof(int16_t) * NNUE_HIDDEN_SIZE);
    if (acc_black != NULL) memcpy(acc_black, board->nnue_acc[BLACK], sizeof(int16_t) * NNUE_HIDDEN_SIZE);
}

// Incrementally add a piece on `sq` to both perspective accumulators.
void nnue_acc_add_piece(int16_t* acc_w, int16_t* acc_b, int pc, int sq) {
    const int16_t* row_w = g_w1[feature_idx_white(pc, sq)];
    const int16_t* row_b = g_w1[feature_idx_black(pc, sq)];

#if defined(__AVX2__)
    for (int i = 0; i < NNUE_HIDDEN_SIZE; i += 16) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(acc_w + i));
        __m256i rw = _mm256_loadu_si256((const __m256i*)(row_w + i));
        _mm256_storeu_si256((__m256i*)(acc_w + i), _mm256_add_epi16(vw, rw));

        __m256i vb = _mm256_loadu_si256((const __m256i*)(acc_b + i));
        __m256i rb = _mm256_loadu_si256((const __m256i*)(row_b + i));
        _mm256_storeu_si256((__m256i*)(acc_b + i), _mm256_add_epi16(vb, rb));
    }
#else
    for (int i = 0; i < NNUE_HIDDEN_SIZE; i++) {
        acc_w[i] = (int16_t)(acc_w[i] + row_w[i]);
        acc_b[i] = (int16_t)(acc_b[i] + row_b[i]);
    }
#endif
}

// Incrementally subtract a piece on `sq` from both perspective accumulators.
void nnue_acc_sub_piece(int16_t* acc_w, int16_t* acc_b, int pc, int sq) {
    const int16_t* row_w = g_w1[feature_idx_white(pc, sq)];
    const int16_t* row_b = g_w1[feature_idx_black(pc, sq)];

#if defined(__AVX2__)
    for (int i = 0; i < NNUE_HIDDEN_SIZE; i += 16) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(acc_w + i));
        __m256i rw = _mm256_loadu_si256((const __m256i*)(row_w + i));
        _mm256_storeu_si256((__m256i*)(acc_w + i), _mm256_sub_epi16(vw, rw));

        __m256i vb = _mm256_loadu_si256((const __m256i*)(acc_b + i));
        __m256i rb = _mm256_loadu_si256((const __m256i*)(row_b + i));
        _mm256_storeu_si256((__m256i*)(acc_b + i), _mm256_sub_epi16(vb, rb));
    }
#else
    for (int i = 0; i < NNUE_HIDDEN_SIZE; i++) {
        acc_w[i] = (int16_t)(acc_w[i] - row_w[i]);
        acc_b[i] = (int16_t)(acc_b[i] - row_b[i]);
    }
#endif
}

// Recompute both perspective accumulators on `board` from scratch.
void nnue_refresh_accumulator(Board* board) {
    if (!g_loaded) {
        nnue_init();
    }
    memcpy(board->nnue_acc[WHITE], g_b1, sizeof(g_b1));
    memcpy(board->nnue_acc[BLACK], g_b1, sizeof(g_b1));

    for (int sq = 0; sq < 64; sq++) {
        int pc = board->mailbox[sq];
        if (pc != PIECE_NONE) {
            nnue_acc_add_piece(board->nnue_acc[WHITE], board->nnue_acc[BLACK], pc, sq);
        }
    }
    board->nnue_acc_valid = true;
    board->nnue_epoch = g_weights_epoch;
}

// Run SCReLU activation and output dot product on a pair of perspective accumulators.
static int forward_from_acc(const int16_t* us_acc, const int16_t* them_acc) {
    int64_t sum = 0;

    for (int i = 0; i < NNUE_HIDDEN_SIZE; i++) {
        int32_t u = us_acc[i];
        if (u < 0) u = 0;
        else if (u > NNUE_QA) u = NNUE_QA;

        int32_t t = them_acc[i];
        if (t < 0) t = 0;
        else if (t > NNUE_QA) t = NNUE_QA;

        sum += (int64_t)(u * u) * (int64_t)g_w2[i];
        sum += (int64_t)(t * t) * (int64_t)g_w2[NNUE_HIDDEN_SIZE + i];
    }

    int64_t out = (sum / NNUE_QA) + (int64_t)g_b2;
    int64_t eval_cp = (out * NNUE_SCALE) / (NNUE_QA * NNUE_QB);
    return (int)eval_cp;
}

// Evaluate `board` using its incrementally maintained NNUE accumulator.
int nnue_evaluate(Board* board) {
    if (!board->nnue_acc_valid || board->nnue_epoch != g_weights_epoch) {
        nnue_refresh_accumulator(board);
    }
    return forward_from_acc(board->nnue_acc[board->turn], board->nnue_acc[board->turn ^ 1]);
}

// Evaluate `board` by recomputing the accumulator from scratch without mutating `board`.
int nnue_evaluate_recompute(const Board* board) {
    if (!g_loaded) {
        nnue_init();
    }
    int16_t temp_acc[2][NNUE_HIDDEN_SIZE] NNUE_ALIGNED;
    memcpy(temp_acc[WHITE], g_b1, sizeof(g_b1));
    memcpy(temp_acc[BLACK], g_b1, sizeof(g_b1));

    for (int sq = 0; sq < 64; sq++) {
        int pc = board->mailbox[sq];
        if (pc != PIECE_NONE) {
            nnue_acc_add_piece(temp_acc[WHITE], temp_acc[BLACK], pc, sq);
        }
    }
    return forward_from_acc(temp_acc[board->turn], temp_acc[board->turn ^ 1]);
}
