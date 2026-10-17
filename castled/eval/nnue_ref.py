"""Reference NumPy NNUE implementation and C backend bridge (768 -> 256x2 -> 1 SCReLU)."""

from dataclasses import dataclass
import os
import struct
from typing import List, Optional, Tuple

import numpy as np

from ..core.board import Board, HAS_CFFI, _ffi, _lib
from ..core.types import Color

NNUE_INPUT_SIZE = 768
NNUE_HIDDEN_SIZE = 256
NNUE_QA = 255
NNUE_QB = 64
NNUE_SCALE = 400
NNUE_MAGIC = 0x4E4E5343
NNUE_VERSION = 1

DEFAULT_NNUE_PATH = os.path.join(os.path.dirname(__file__), "weights", "default.nnue")

_DEFAULT_PIECE_CP = (100, 320, 330, 500, 900, 0)
_DEFAULT_CENTER_BONUS = (-6, -2, 2, 6, 6, 2, -2, -6)


@dataclass
class NNUEWeights:
    # Quantized int16 parameters for the 768 -> 256x2 -> 1 architecture.
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: int


# Map (piece, square) to White-perspective feature index in [0..767].
def feature_idx_white(pc: int, sq: int) -> int:
    return pc * 64 + sq


# Map (piece, square) to Black-perspective feature index in [0..767] via color and rank flip.
def feature_idx_black(pc: int, sq: int) -> int:
    flipped_pc = (pc + 6) if pc < 6 else (pc - 6)
    flipped_sq = sq ^ 56
    return flipped_pc * 64 + flipped_sq


# Build the deterministic default int16 weight set matching C `nnue_init()`.
def build_default_weights() -> NNUEWeights:
    w1 = np.zeros((NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE), dtype=np.int16)
    b1 = np.zeros(NNUE_HIDDEN_SIZE, dtype=np.int16)
    w2 = np.zeros(NNUE_HIDDEN_SIZE * 2, dtype=np.int16)
    b2 = 0

    for pc in range(12):
        pt = pc % 6
        is_friendly = pc < 6
        base_val = _DEFAULT_PIECE_CP[pt]
        for sq in range(64):
            f = sq & 7
            r = sq >> 3
            pos_bonus = _DEFAULT_CENTER_BONUS[f] + _DEFAULT_CENTER_BONUS[r]
            if pt == 0:
                pos_bonus += (r - 1) * 8
            total_cp = base_val + pos_bonus
            idx = pc * 64 + sq

            target_ch = pt if is_friendly else (6 + pt)
            scaled = int((total_cp * 64) / 900)
            scaled = max(0, min(NNUE_QA, scaled))
            w1[idx, target_ch] = np.int16(scaled)

            aux_ch = 12 + ((idx * 7 + pt * 13) % (NNUE_HIDDEN_SIZE - 12))
            w1[idx, aux_ch] = np.int16(12 if is_friendly else -8)

    b1[12:] = 32
    for pt in range(6):
        w2[pt] = 48
        w2[6 + pt] = -48
        w2[NNUE_HIDDEN_SIZE + pt] = -48
        w2[NNUE_HIDDEN_SIZE + 6 + pt] = 48
    w2[12:NNUE_HIDDEN_SIZE] = 2
    w2[NNUE_HIDDEN_SIZE + 12:] = -2

    return NNUEWeights(w1=w1, b1=b1, w2=w2, b2=b2)


_ACTIVE_WEIGHTS: NNUEWeights = build_default_weights()


# Return active feature index lists (white_features, black_features) for all pieces on `board`.
def extract_features(board: Board) -> Tuple[List[int], List[int]]:
    white_feats: List[int] = []
    black_feats: List[int] = []
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is not None and pc >= 0:
            white_feats.append(feature_idx_white(pc, sq))
            black_feats.append(feature_idx_black(pc, sq))
    return white_feats, black_feats


# Compute White and Black int16 accumulators from scratch using NumPy.
def compute_accumulator_numpy(
    board: Board,
    weights: Optional[NNUEWeights] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    w = weights if weights is not None else _ACTIVE_WEIGHTS
    acc_w = w.b1.astype(np.int32).copy()
    acc_b = w.b1.astype(np.int32).copy()
    white_feats, black_feats = extract_features(board)
    if white_feats:
        acc_w += np.sum(w.w1[white_feats].astype(np.int32), axis=0)
        acc_b += np.sum(w.w1[black_feats].astype(np.int32), axis=0)
    return acc_w.astype(np.int16), acc_b.astype(np.int16)


# Run SCReLU activation and output dot product in NumPy matching C integer truncation.
def forward_from_accumulators_numpy(
    acc_white: np.ndarray,
    acc_black: np.ndarray,
    turn: int,
    weights: Optional[NNUEWeights] = None,
) -> int:
    w = weights if weights is not None else _ACTIVE_WEIGHTS
    us_acc = acc_white if turn == Color.WHITE else acc_black
    them_acc = acc_black if turn == Color.WHITE else acc_white

    u = np.clip(us_acc.astype(np.int64), 0, NNUE_QA)
    t = np.clip(them_acc.astype(np.int64), 0, NNUE_QA)

    w2_us = w.w2[:NNUE_HIDDEN_SIZE].astype(np.int64)
    w2_them = w.w2[NNUE_HIDDEN_SIZE:].astype(np.int64)

    total = int(np.sum((u * u) * w2_us) + np.sum((t * t) * w2_them))
    out = int(total / NNUE_QA) + int(w.b2)
    eval_cp = int((out * NNUE_SCALE) / (NNUE_QA * NNUE_QB))
    return eval_cp


# Evaluate `board` from scratch using the NumPy reference forward pass.
def evaluate_nnue_numpy(board: Board, weights: Optional[NNUEWeights] = None) -> int:
    acc_w, acc_b = compute_accumulator_numpy(board, weights)
    return forward_from_accumulators_numpy(acc_w, acc_b, board.turn, weights)


# Push `weights` into the active Python state and the C engine if available.
def set_active_weights(weights: NNUEWeights) -> None:
    global _ACTIVE_WEIGHTS
    w1 = np.ascontiguousarray(weights.w1, dtype=np.int16)
    b1 = np.ascontiguousarray(weights.b1, dtype=np.int16)
    w2 = np.ascontiguousarray(weights.w2, dtype=np.int16)
    b2 = int(np.int16(weights.b2))
    _ACTIVE_WEIGHTS = NNUEWeights(w1=w1.copy(), b1=b1.copy(), w2=w2.copy(), b2=b2)

    if HAS_CFFI:
        w1_ptr = _ffi.cast("const int16_t*", _ffi.from_buffer(w1))
        b1_ptr = _ffi.cast("const int16_t*", _ffi.from_buffer(b1))
        w2_ptr = _ffi.cast("const int16_t*", _ffi.from_buffer(w2))
        _lib.nnue_set_weights(w1_ptr, b1_ptr, w2_ptr, b2)


# Return a copy of the currently active NNUEWeights.
def get_active_weights() -> NNUEWeights:
    if HAS_CFFI:
        w1 = np.zeros((NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE), dtype=np.int16)
        b1 = np.zeros(NNUE_HIDDEN_SIZE, dtype=np.int16)
        w2 = np.zeros(NNUE_HIDDEN_SIZE * 2, dtype=np.int16)
        b2_buf = _ffi.new("int16_t*")
        _lib.nnue_get_weights(
            _ffi.cast("int16_t*", _ffi.from_buffer(w1)),
            _ffi.cast("int16_t*", _ffi.from_buffer(b1)),
            _ffi.cast("int16_t*", _ffi.from_buffer(w2)),
            b2_buf,
        )
        return NNUEWeights(w1=w1, b1=b1, w2=w2, b2=int(b2_buf[0]))
    return NNUEWeights(
        w1=_ACTIVE_WEIGHTS.w1.copy(),
        b1=_ACTIVE_WEIGHTS.b1.copy(),
        w2=_ACTIVE_WEIGHTS.w2.copy(),
        b2=int(_ACTIVE_WEIGHTS.b2),
    )


# Read the board's current C incremental accumulators into NumPy arrays.
def get_c_accumulator(board: Board) -> Tuple[np.ndarray, np.ndarray]:
    if not (board._use_c and HAS_CFFI):
        return compute_accumulator_numpy(board)
    acc_w = np.zeros(NNUE_HIDDEN_SIZE, dtype=np.int16)
    acc_b = np.zeros(NNUE_HIDDEN_SIZE, dtype=np.int16)
    _lib.nnue_get_accumulator(
        board._c_board,
        _ffi.cast("int16_t*", _ffi.from_buffer(acc_w)),
        _ffi.cast("int16_t*", _ffi.from_buffer(acc_b)),
    )
    return acc_w, acc_b


# Save `weights` (or active weights) to a binary `.nnue` file.
def save_nnue_file(path: str, weights: Optional[NNUEWeights] = None) -> bool:
    if not path:
        return False
    w = weights if weights is not None else get_active_weights()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = struct.pack("<4I", NNUE_MAGIC, NNUE_VERSION, NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    with open(path, "wb") as fp:
        fp.write(header)
        fp.write(np.ascontiguousarray(w.w1, dtype="<i2").tobytes())
        fp.write(np.ascontiguousarray(w.b1, dtype="<i2").tobytes())
        fp.write(np.ascontiguousarray(w.w2, dtype="<i2").tobytes())
        fp.write(struct.pack("<h", int(w.b2)))
    return True


# Load a binary `.nnue` file, validate its header, and update active C/Python weights.
def load_nnue_file(path: str, activate: bool = True) -> NNUEWeights:
    if not path or not os.path.isfile(path):
        raise ValueError(f"NNUE file not found: {path}")
    with open(path, "rb") as fp:
        raw_header = fp.read(16)
        if len(raw_header) != 16:
            raise ValueError("Invalid NNUE file: truncated header")
        magic, version, in_size, hid_size = struct.unpack("<4I", raw_header)
        if (magic, version, in_size, hid_size) != (
            NNUE_MAGIC,
            NNUE_VERSION,
            NNUE_INPUT_SIZE,
            NNUE_HIDDEN_SIZE,
        ):
            raise ValueError("Invalid NNUE file header")

        w1_bytes = NNUE_INPUT_SIZE * NNUE_HIDDEN_SIZE * 2
        b1_bytes = NNUE_HIDDEN_SIZE * 2
        w2_bytes = NNUE_HIDDEN_SIZE * 2 * 2
        b2_bytes = 2

        buf_w1 = fp.read(w1_bytes)
        buf_b1 = fp.read(b1_bytes)
        buf_w2 = fp.read(w2_bytes)
        buf_b2 = fp.read(b2_bytes)
        if (
            len(buf_w1) != w1_bytes
            or len(buf_b1) != b1_bytes
            or len(buf_w2) != w2_bytes
            or len(buf_b2) != b2_bytes
        ):
            raise ValueError("Invalid NNUE file: truncated weight payload")

    w1 = np.frombuffer(buf_w1, dtype="<i2").reshape((NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)).copy()
    b1 = np.frombuffer(buf_b1, dtype="<i2").copy()
    w2 = np.frombuffer(buf_w2, dtype="<i2").copy()
    b2 = int(struct.unpack("<h", buf_b2)[0])
    weights = NNUEWeights(w1=w1, b1=b1, w2=w2, b2=b2)

    if activate:
        if HAS_CFFI:
            ok = bool(_lib.nnue_load_file(path.encode("utf-8")))
            if not ok:
                raise ValueError(f"C backend failed to load NNUE file: {path}")
        global _ACTIVE_WEIGHTS
        _ACTIVE_WEIGHTS = weights
    return weights


# Reset NNUE weights to `default.nnue` (or heuristic fallback) in both Python and C.
def reset_default_weights() -> NNUEWeights:
    if os.path.isfile(DEFAULT_NNUE_PATH):
        return load_nnue_file(DEFAULT_NNUE_PATH, activate=True)
    w = build_default_weights()
    set_active_weights(w)
    return w


# Evaluate `board` using the C incremental NNUE backend (or NumPy fallback when pure-Python).
def evaluate_nnue(board: Board, recompute: bool = False) -> int:
    if board._use_c and HAS_CFFI:
        if recompute:
            return int(_lib.nnue_evaluate_recompute(board._c_board))
        return int(_lib.nnue_evaluate(board._c_board))
    return evaluate_nnue_numpy(board)


reset_default_weights()

