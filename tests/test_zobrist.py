"""Tests for Zobrist hashing."""

import sys
sys.path.insert(0, '.')

from castled.core.zobrist import PIECE_KEYS, CASTLING_KEYS, EP_KEYS, SIDE_KEY, compute_hash
from castled.core.board import Board


def test_keys_are_nonzero():
    # all keys should be generated (non-zero)
    assert SIDE_KEY != 0
    assert all(PIECE_KEYS[p][sq] != 0 for p in range(12) for sq in range(64))
    assert all(CASTLING_KEYS[i] != 0 for i in range(1, 16))

def test_keys_are_unique():
    # no two piece keys should collide
    all_keys = [PIECE_KEYS[p][sq] for p in range(12) for sq in range(64)]
    assert len(all_keys) == len(set(all_keys))

def test_deterministic():
    # same seed should produce same keys across calls
    b1 = Board()
    b2 = Board()
    assert b1.zobrist_hash == b2.zobrist_hash

def test_different_positions_different_hash():
    b1 = Board()
    b2 = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert b1.zobrist_hash != b2.zobrist_hash

