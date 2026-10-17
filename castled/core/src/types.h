#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

// Color
enum { WHITE = 0, BLACK = 1 };

// PieceType
enum { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };

// Piece encoding
enum {
    WHITE_PAWN = 0, WHITE_KNIGHT = 1, WHITE_BISHOP = 2, WHITE_ROOK = 3, WHITE_QUEEN = 4, WHITE_KING = 5,
    BLACK_PAWN = 6, BLACK_KNIGHT = 7, BLACK_BISHOP = 8, BLACK_ROOK = 9, BLACK_QUEEN = 10, BLACK_KING = 11,
    PIECE_NONE = 12
};

// Square encoding
enum {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64
};

// Move flags
enum {
    QUIET = 0, DOUBLE_PAWN = 1, KING_CASTLE = 2, QUEEN_CASTLE = 3,
    CAPTURE = 4, EP_CAPTURE = 5,
    PROMO_N = 8, PROMO_B = 9, PROMO_R = 10, PROMO_Q = 11,
    CAPTURE_PROMO_N = 12, CAPTURE_PROMO_B = 13, CAPTURE_PROMO_R = 14, CAPTURE_PROMO_Q = 15
};

// Castling rights bitmask
enum { WK = 1, WQ = 2, BK = 4, BQ = 8 };

#define MAX_MOVES 256

// Move encoding
typedef uint16_t Move;

#define MOVE_FROM(m) ((m) & 0x3F)
#define MOVE_TO(m) (((m) >> 6) & 0x3F)
#define MOVE_FLAGS(m) (((m) >> 12) & 0x0F)
#define MAKE_MOVE(from, to, flags) ((Move)((from) | ((to) << 6) | ((flags) << 12)))

// Forward declaration for opaque type
typedef struct Board Board;

// Castling rights update table
extern const int CASTLING_RIGHTS_UPDATE[64];

#endif // TYPES_H
