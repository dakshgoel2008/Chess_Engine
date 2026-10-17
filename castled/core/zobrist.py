"""
Zobrist hashing for board positions.

Every position gets a (nearly) unique 64-bit hash by XOR-ing together
random keys for: each piece on each square, castling rights, en passant
file, and side to move.

The keys are generated from a fixed seed so they're deterministic
across runs (and will match the C extension).
"""

import random

# random 64-bit keys, filled in by _init() below
PIECE_KEYS = [[0] * 64 for _ in range(12)]
CASTLING_KEYS = [0] * 16
EP_KEYS = [0] * 65    # index 64 = "no en passant"
SIDE_KEY = 0


def _init():
    global SIDE_KEY

    rng = random.Random(1070372)  # fixed seed for reproducibility

    for piece in range(12):
        for sq in range(64):
            PIECE_KEYS[piece][sq] = rng.getrandbits(64)

    for i in range(16):
        CASTLING_KEYS[i] = rng.getrandbits(64)

    for sq in range(65):
        EP_KEYS[sq] = rng.getrandbits(64)

    SIDE_KEY = rng.getrandbits(64)


_init()


def compute_hash(board) -> int:
    """Build the full Zobrist hash from scratch for a given board state."""
    h = 0

    # side to move
    if board.turn == 1:
        h ^= SIDE_KEY

    # pieces on squares
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None and p.value != 12:
            h ^= PIECE_KEYS[p.value][sq]

    # castling rights
    h ^= CASTLING_KEYS[board.castling_rights]

    # en passant
    if board.ep_square != 64:
        h ^= EP_KEYS[board.ep_square]

    return h
