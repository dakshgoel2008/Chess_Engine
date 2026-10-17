"""Tests for bitboard utilities and attack tables."""

import sys
sys.path.insert(0, '.')

from castled.core.bitboard import (
    lsb, pop_lsb, popcount, iter_bits,
    KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS,
    rook_attacks, bishop_attacks, queen_attacks,
    FILE_A, FILE_H, RANK_1, RANK_8,
)


# --- bit manipulation ---

def test_lsb():
    assert lsb(1) == 0          # bit 0
    assert lsb(8) == 3          # bit 3
    assert lsb(0x100) == 8      # bit 8

def test_pop_lsb():
    bb, idx = pop_lsb(0b1010)
    assert idx == 1             # lowest bit was at index 1
    assert bb == 0b1000         # that bit is now cleared

def test_popcount():
    assert popcount(0) == 0
    assert popcount(1) == 1
    assert popcount(0xFF) == 8
    assert popcount(0xFFFFFFFFFFFFFFFF) == 64

def test_iter_bits():
    bits = list(iter_bits(0b10110))
    assert bits == [1, 2, 4]


# --- attack tables ---

def test_knight_attacks_center():
    # knight on e4 (sq 28) should attack 8 squares
    assert popcount(KNIGHT_ATTACKS[28]) == 8

def test_knight_attacks_corner():
    # knight on a1 (sq 0) should attack 2 squares (b3, c2)
    assert popcount(KNIGHT_ATTACKS[0]) == 2

def test_king_attacks_center():
    # king in the center should attack 8 squares
    assert popcount(KING_ATTACKS[28]) == 8

def test_king_attacks_corner():
    # king on a1 should attack 3 squares
    assert popcount(KING_ATTACKS[0]) == 3

def test_pawn_attacks_white():
    # white pawn on e4 (sq 28) attacks d5 and f5
    atk = PAWN_ATTACKS[0][28]
    assert atk & (1 << 35)   # d5
    assert atk & (1 << 37)   # f5
    assert popcount(atk) == 2

def test_pawn_attacks_black():
    # black pawn on e5 (sq 36) attacks d4 and f4
    atk = PAWN_ATTACKS[1][36]
    assert atk & (1 << 27)   # d4
    assert atk & (1 << 29)   # f4
    assert popcount(atk) == 2

def test_pawn_attacks_edge():
    # white pawn on a2 (sq 8) attacks only b3, not wrapping
    atk = PAWN_ATTACKS[0][8]
    assert popcount(atk) == 1
    assert atk & (1 << 17)   # b3


# --- sliding piece attacks ---

def test_rook_attacks_empty():
    # rook on a1, empty board -> attacks a2-a8 and b1-h1 = 14 squares
    atk = rook_attacks(0, 0)
    assert popcount(atk) == 14

def test_rook_attacks_blocked():
    # rook on a1, blocker on a3 (sq 16) -> a2 and a3 only going up
    occ = (1 << 16)
    atk = rook_attacks(0, occ)
    assert atk & (1 << 8)    # a2
    assert atk & (1 << 16)   # a3 (the blocker itself)
    assert not (atk & (1 << 24))  # a4 is blocked

def test_bishop_attacks_empty():
    # bishop on d4 (sq 27), empty board
    atk = bishop_attacks(27, 0)
    assert popcount(atk) == 13

def test_queen_attacks():
    # queen = rook + bishop
    sq = 27  # d4
    occ = 0
    q = queen_attacks(sq, occ)
    r = rook_attacks(sq, occ)
    b = bishop_attacks(sq, occ)
    assert q == r | b

