"""
bitboard -> 64-bit integer where each bit represents one square
Bit 0 = A1, Bit 7 = H1, Bit 56 = A8, Bit 63 = H8.
"""

BB_ALL = 0xFFFFFFFFFFFFFFFF


# --- File and rank masks ---

FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7

NOT_FILE_A = BB_ALL ^ FILE_A
NOT_FILE_H = BB_ALL ^ FILE_H

RANK_1 = 0xFF
RANK_2 = RANK_1 << 8
RANK_3 = RANK_1 << 16
RANK_4 = RANK_1 << 24
RANK_5 = RANK_1 << 32
RANK_6 = RANK_1 << 40
RANK_7 = RANK_1 << 48
RANK_8 = RANK_1 << 56


# --- Bit manipulation ---

def lsb(bb):
    """Index of the lowest set bit. Returns 0 if bb is 0"""
    if not bb:
        return 0
    return (bb & -bb).bit_length() - 1


def pop_lsb(bb):
    """Clears the lowest set bit. Returns (new_bb, index_of_cleared_bit)"""
    idx = lsb(bb)
    return bb & (bb - 1), idx


def popcount(bb):
    """Number of set bits"""
    return bin(bb).count('1')


def iter_bits(bb):
    """Yields the index of each set bit, clearing as it goes."""
    while bb:
        bb, idx = pop_lsb(bb)
        yield idx


# --- Precomputed attack tables ---
# Built once at module load time.

KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
PAWN_ATTACKS = [[0] * 64, [0] * 64]   # [WHITE][sq], [BLACK][sq]


def _init_attack_tables():
    for sq in range(64):
        bb = 1 << sq

        # -- king: one step in all 8 directions --
        k = 0
        k |= (bb << 1) & NOT_FILE_A   # right
        k |= (bb >> 1) & NOT_FILE_H   # left
        k |= bb << 8                   # up
        k |= bb >> 8                   # down
        k |= (bb << 9) & NOT_FILE_A   # up-right
        k |= (bb << 7) & NOT_FILE_H   # up-left
        k |= (bb >> 7) & NOT_FILE_A   # down-right
        k |= (bb >> 9) & NOT_FILE_H   # down-left
        KING_ATTACKS[sq] = k & BB_ALL

        # -- knight: L-shaped jumps --
        n = 0
        n |= (bb << 17) & NOT_FILE_A        # up 2, right 1
        n |= (bb << 15) & NOT_FILE_H        # up 2, left 1
        n |= (bb << 10) & ~(FILE_A | FILE_B)  # right 2, up 1
        n |= (bb << 6)  & ~(FILE_G | FILE_H)  # left 2, up 1
        n |= (bb >> 6)  & ~(FILE_A | FILE_B)  # right 2, down 1
        n |= (bb >> 10) & ~(FILE_G | FILE_H)  # left 2, down 1
        n |= (bb >> 15) & NOT_FILE_A        # down 2, right 1
        n |= (bb >> 17) & NOT_FILE_H        # down 2, left 1
        KNIGHT_ATTACKS[sq] = n & BB_ALL

        # -- pawns: diagonal captures only (not pushes) --
        # white pawns attack upward
        PAWN_ATTACKS[0][sq] = (((bb << 9) & NOT_FILE_A) | ((bb << 7) & NOT_FILE_H)) & BB_ALL

        # black pawns attack downward
        PAWN_ATTACKS[1][sq] = (((bb >> 7) & NOT_FILE_A) | ((bb >> 9) & NOT_FILE_H)) & BB_ALL


_init_attack_tables()


# --- Sliding piece attacks (ray scanning) ---
# These walk along each ray direction until hitting a blocker.
# Will be replaced by magic bitboards in the C extension for speed.

def rook_attacks(sq, occ):
    """Rook attack bitboard given square and board occupancy."""
    attacks = 0
    rank = sq >> 3
    file = sq & 7

    # up
    for r in range(rank + 1, 8):
        s = r * 8 + file
        attacks |= (1 << s)
        if occ & (1 << s):
            break

    # down
    for r in range(rank - 1, -1, -1):
        s = r * 8 + file
        attacks |= (1 << s)
        if occ & (1 << s):
            break

    # right
    for f in range(file + 1, 8):
        s = rank * 8 + f
        attacks |= (1 << s)
        if occ & (1 << s):
            break

    # left
    for f in range(file - 1, -1, -1):
        s = rank * 8 + f
        attacks |= (1 << s)
        if occ & (1 << s):
            break

    return attacks


def bishop_attacks(sq, occ):
    """Bishop attack bitboard given square and board occupancy."""
    attacks = 0
    rank = sq >> 3
    file = sq & 7

    for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            s = r * 8 + f
            attacks |= (1 << s)
            if occ & (1 << s):
                break
            r += dr
            f += df

    return attacks


def queen_attacks(sq, occ):
    """Queen = rook + bishop."""
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)
