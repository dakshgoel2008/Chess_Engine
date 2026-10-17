"""Edge-case tests for legal move generation."""

import random
import pytest
from castled.core.board import Board, HAS_CFFI
from castled.core.move import Move
from castled.core.types import Color, MoveFlag, PieceType, Square
from castled.core.zobrist import compute_hash


def test_cannot_castle_when_in_check():
    b = Board("4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1")
    uci_moves = {m.to_uci() for m in b.legal_moves}
    assert "e1g1" not in uci_moves
    assert "e1c1" not in uci_moves


def test_cannot_castle_through_attacked_square():
    # Black rook on f8 attacks f1, blocking white kingside castling; queenside is clear.
    b = Board("5r1k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    uci_moves = {m.to_uci() for m in b.legal_moves}
    assert "e1g1" not in uci_moves
    assert "e1c1" in uci_moves


def test_queenside_castle_allowed_when_b1_attacked():
    # b1 is attacked by black rook on b8, but king only passes through d1 and c1.
    b = Board("1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    uci_moves = {m.to_uci() for m in b.legal_moves}
    assert "e1c1" in uci_moves


def test_queenside_castle_blocked_when_b1_occupied():
    b = Board("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1")
    uci_moves = {m.to_uci() for m in b.legal_moves}
    assert "e1c1" not in uci_moves
    assert "e1g1" in uci_moves


def test_castling_rights_revoked_when_rook_captured():
    # White bishop on b2 captures black rook on h8; black loses kingside castling right.
    b = Board("r3k2r/8/8/8/8/8/1B6/R3K2R w KQkq - 0 1")
    m = b.parse_uci("b2h8")
    b.push(m)
    assert "k" not in b.fen.split()[2]
    assert "q" in b.fen.split()[2]


def test_en_passant_pinned_horizontally_is_illegal():
    # White king on a5, white pawn on b5, black pawn just double-pushed c7->c5, black rook on h5.
    # Capturing bxc6 e.p. removes both pawns from rank 5 and exposes king to rook on h5.
    b = Board("8/8/8/KPp4r/8/8/8/7k w - c6 0 1")
    uci_moves = {m.to_uci() for m in b.legal_moves}
    assert "b5c6" not in uci_moves


def test_en_passant_legal_capture_and_unmake():
    b = Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    fen_before = b.fen
    hash_before = b.zobrist_hash
    ep_move = b.parse_uci("e5f6")
    assert ep_move.is_en_passant()
    b.push(ep_move)
    assert b.piece_at(Square.F5) is None
    assert b.piece_at(Square.F6) is not None
    b.pop()
    assert b.fen == fen_before
    assert b.zobrist_hash == hash_before


def test_double_check_only_king_can_move():
    # White knight on f6 and white queen on e2 both check black king on e8.
    b = Board("4k3/8/5N2/8/8/8/4Q3/4K3 b - - 0 1")
    for m in b.legal_moves:
        assert m.from_square == Square.E8


def test_pinned_piece_can_move_along_pin_ray_only():
    # White rook on e2 is pinned to white king on e1 by black rook on e8.
    b = Board("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    rook_moves = [m for m in b.legal_moves if m.from_square == Square.E2]
    assert len(rook_moves) == 6
    for m in rook_moves:
        assert m.to_square % 8 == Square.E2 % 8


def test_promotion_generates_all_four_pieces_and_captures():
    b = Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promo_Quiets = [m for m in b.legal_moves if m.from_square == Square.A7 and m.to_square == Square.A8]
    promo_caps = [m for m in b.legal_moves if m.from_square == Square.A7 and m.to_square == Square.B8]
    assert len(promo_Quiets) == 4
    assert len(promo_caps) == 4
    assert {m.promotion_piece() for m in promo_Quiets} == {
        PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT,
    }


def test_generate_captures_only_returns_captures_and_promotions():
    b = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    caps = b.generate_captures()
    expected = [m for m in b.legal_moves if m.is_capture() or m.is_promotion()]
    assert set(caps) == set(expected)


def test_incremental_zobrist_matches_full_recompute():
    rng = random.Random(42)
    b = Board(_use_c=False)
    for _ in range(60):
        legal = b.legal_moves
        if not legal:
            break
        m = rng.choice(legal)
        b.push(m)
        assert b.zobrist_hash == compute_hash(b)


@pytest.mark.skipif(not HAS_CFFI, reason="C extension not built")
def test_random_walk_c_and_python_parity():
    rng = random.Random(1337)
    for _ in range(15):
        bc = Board(_use_c=True)
        bp = Board(_use_c=False)
        for _ in range(40):
            assert bc.fen == bp.fen
            assert bc.zobrist_hash == bp.zobrist_hash
            c_moves = set(bc.legal_moves)
            p_moves = set(bp.legal_moves)
            assert c_moves == p_moves
            if not c_moves:
                break
            chosen = rng.choice(list(c_moves))
            bc.push(chosen)
            bp.push(chosen)

