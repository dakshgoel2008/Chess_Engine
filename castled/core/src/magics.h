#ifndef MAGICS_H
#define MAGICS_H

#include <stdint.h>

extern uint64_t ROOK_MAGICS[64];
extern uint64_t BISHOP_MAGICS[64];
extern uint64_t ROOK_MASKS[64];
extern uint64_t BISHOP_MASKS[64];
extern int ROOK_SHIFTS[64];
extern int BISHOP_SHIFTS[64];

extern uint64_t* ROOK_ATTACKS[64];
extern uint64_t* BISHOP_ATTACKS[64];

void init_magics(void);
uint64_t rook_attacks(int sq, uint64_t occupied);
uint64_t bishop_attacks(int sq, uint64_t occupied);
uint64_t queen_attacks(int sq, uint64_t occupied);

#endif // MAGICS_H
