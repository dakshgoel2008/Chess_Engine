"""
Zobrist hashing for board positions.
Keys are generated with the same 64-bit PRNG seed as the C extension so hashes match.
"""

PIECE_KEYS = [[0] * 64 for _ in range(12)]
CASTLING_KEYS = [0] * 16
EP_KEYS = [0] * 8
SIDE_KEY = 0

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _init():
    global SIDE_KEY
    state = 1070372

    def next_random():
        nonlocal state
        state ^= (state >> 12) & _MASK64
        state ^= (state << 25) & _MASK64
        state ^= (state >> 27) & _MASK64
        return (state * 2685821657736338717) & _MASK64

    for piece in range(12):
        for sq in range(64):
            PIECE_KEYS[piece][sq] = next_random()

    for i in range(16):
        CASTLING_KEYS[i] = next_random()

    for f in range(8):
        EP_KEYS[f] = next_random()

    SIDE_KEY = next_random()


_init()


def compute_hash(board) -> int:
    """Build the full Zobrist hash from scratch for a given board state."""
    h = 0

    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None and p.value != 12:
            h ^= PIECE_KEYS[p.value][sq]

    h ^= CASTLING_KEYS[board.castling_rights]

    if board.ep_square != 64:
        h ^= EP_KEYS[board.ep_square % 8]

    if board.turn == 1:
        h ^= SIDE_KEY

    return h

