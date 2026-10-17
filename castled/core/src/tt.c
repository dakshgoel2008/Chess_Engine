#include "tt.h"
#include <stdlib.h>
#include <string.h>

static TTEntry* tt_table = NULL;
static size_t tt_count = 0;
static size_t tt_mask = 0;

// Convert root-relative mate score to ply-independent score for storage.
static int16_t score_to_tt(int score, int ply) {
    if (score > SCORE_MATE_THRESHOLD) score += ply;
    else if (score < -SCORE_MATE_THRESHOLD) score -= ply;
    if (score > SCORE_INF) score = SCORE_INF;
    if (score < -SCORE_INF) score = -SCORE_INF;
    return (int16_t)score;
}

// Convert stored ply-independent mate score back to root-relative score.
static int score_from_tt(int score, int ply) {
    if (score > SCORE_MATE_THRESHOLD) return score - ply;
    if (score < -SCORE_MATE_THRESHOLD) return score + ply;
    return score;
}

// Allocate transposition table with power-of-two entries fitting in size_mb.
void tt_init(int size_mb) {
    if (size_mb < 1) size_mb = 1;
    if (size_mb > 1024) size_mb = 1024;

    size_t bytes = (size_t)size_mb * 1024ULL * 1024ULL;
    size_t max_entries = bytes / sizeof(TTEntry);
    size_t count = 1024;
    while ((count << 1) <= max_entries) {
        count <<= 1;
    }

    if (tt_table != NULL && tt_count == count) {
        tt_clear();
        return;
    }

    free(tt_table);
    tt_table = (TTEntry*)calloc(count, sizeof(TTEntry));
    if (tt_table != NULL) {
        tt_count = count;
        tt_mask = count - 1;
    } else {
        tt_count = 0;
        tt_mask = 0;
    }
}

void tt_free(void) {
    free(tt_table);
    tt_table = NULL;
    tt_count = 0;
    tt_mask = 0;
}

void tt_clear(void) {
    if (tt_table != NULL && tt_count > 0) {
        memset(tt_table, 0, tt_count * sizeof(TTEntry));
    }
}

// Store a search result in the transposition table.
void tt_store(uint64_t key, int depth, int score, int bound, Move best_move, int ply) {
    if (tt_table == NULL) return;

    TTEntry* entry = &tt_table[key & tt_mask];
    if (entry->key != key || best_move != 0) {
        entry->best_move = best_move;
    }
    entry->key = key;
    entry->score = score_to_tt(score, ply);
    entry->depth = (int8_t)(depth > 127 ? 127 : (depth < 0 ? 0 : depth));
    entry->bound = (uint8_t)bound;
}

// Probe the TT for a cutoff score and/or best move for ordering.
bool tt_probe(uint64_t key, int depth, int alpha, int beta, int ply, Move* best_move, int* score) {
    *best_move = 0;
    if (tt_table == NULL) return false;

    const TTEntry* entry = &tt_table[key & tt_mask];
    if (entry->bound == TT_NONE || entry->key != key) {
        return false;
    }

    *best_move = entry->best_move;
    int s = score_from_tt(entry->score, ply);

    if (entry->depth >= depth) {
        if (entry->bound == TT_EXACT) {
            *score = s;
            return true;
        }
        if (entry->bound == TT_LOWER && s >= beta) {
            *score = s;
            return true;
        }
        if (entry->bound == TT_UPPER && s <= alpha) {
            *score = s;
            return true;
        }
    }
    return false;
}

// Inspect the raw TT entry for a given key.
bool tt_probe_entry(uint64_t key, int ply, Move* best_move, int* score, int* depth, int* bound) {
    if (tt_table == NULL) return false;

    const TTEntry* entry = &tt_table[key & tt_mask];
    if (entry->bound == TT_NONE || entry->key != key) {
        return false;
    }

    if (best_move) *best_move = entry->best_move;
    if (score) *score = score_from_tt(entry->score, ply);
    if (depth) *depth = entry->depth;
    if (bound) *bound = entry->bound;
    return true;
}

