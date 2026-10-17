#ifndef NNUE_H
#define NNUE_H

#include "types.h"
#include "board.h"

#define NNUE_INPUT_SIZE 768
#define NNUE_HIDDEN_SIZE 256
#define NNUE_QA 255
#define NNUE_QB 64
#define NNUE_SCALE 400
#define NNUE_MAGIC 0x4E4E5343U
#define NNUE_VERSION 1U

void nnue_init(void);
bool nnue_load_file(const char* path);
bool nnue_save_file(const char* path);
bool nnue_is_loaded(void);

void nnue_set_weights(
    const int16_t* w1,
    const int16_t* b1,
    const int16_t* w2,
    int16_t b2
);
void nnue_get_weights(
    int16_t* w1,
    int16_t* b1,
    int16_t* w2,
    int16_t* b2
);
void nnue_get_accumulator(const Board* board, int16_t* acc_white, int16_t* acc_black);

void nnue_refresh_accumulator(Board* board);
void nnue_acc_add_piece(int16_t* acc_w, int16_t* acc_b, int pc, int sq);
void nnue_acc_sub_piece(int16_t* acc_w, int16_t* acc_b, int pc, int sq);

int nnue_evaluate(Board* board);
int nnue_evaluate_recompute(const Board* board);

#endif // NNUE_H
