"""Self-play data generator and compact binary format for NNUE training."""

from dataclasses import dataclass
import json
import os
import random
import struct
from typing import List, Optional, Sequence

from ..core.board import Board
from ..core.types import Color, MoveFlag, STARTING_FEN
from ..eval.classical import evaluate
from ..search.search import Searcher
from ..search.transposition import SCORE_MATE_THRESHOLD

DATAGEN_MAGIC = b"CSDT"
DATAGEN_VERSION = 1
# Record layout: 32 bytes (64 nibbles for mailbox squares), 1 byte turn, 2 bytes int16 score_cp, 1 byte uint8 wdl_code (0=loss, 1=draw, 2=win)
RECORD_STRUCT = struct.Struct("<32sBhi")


@dataclass
class SelfPlaySample:
    # Single training position with side-to-move relative score (cp) and WDL outcome in {0.0, 0.5, 1.0}.
    fen: str
    score_cp: int
    wdl: float
    turn: int = int(Color.WHITE)


# Pack a Board's 64 squares into 32 bytes (two 4-bit piece codes per byte; 15 = empty).
def _pack_board_Squares(board: Board) -> bytes:
    buf = bytearray(32)
    for sq in range(0, 64, 2):
        p0 = board.piece_at(sq)
        p1 = board.piece_at(sq + 1)
        c0 = 15 if (p0 is None or p0 < 0) else int(p0)
        c1 = 15 if (p1 is None or p1 < 0) else int(p1)
        buf[sq >> 1] = (c0 & 0x0F) | ((c1 & 0x0F) << 4)
    return bytes(buf)


# Unpack 32 bytes of 4-bit piece codes and a turn flag back into a minimalist FEN string.
def _unpack_board_fen(packed: bytes, turn: int) -> str:
    piece_chars = "PNBRQKpnbrqk"
    ranks: List[str] = []
    for r in range(7, -1, -1):
        empty = 0
        row: List[str] = []
        for f in range(8):
            sq = r * 8 + f
            byte_val = packed[sq >> 1]
            code = (byte_val & 0x0F) if (sq & 1) == 0 else ((byte_val >> 4) & 0x0F)
            if code == 15 or code >= 12:
                empty += 1
            else:
                if empty > 0:
                    row.append(str(empty))
                    empty = 0
                row.append(piece_chars[code])
        if empty > 0:
            row.append(str(empty))
        ranks.append("".join(row))
    stm_char = "w" if turn == int(Color.WHITE) else "b"
    return f"{'/'.join(ranks)} {stm_char} - - 0 1"


# Save a sequence of SelfPlaySamples in compact binary format (`CSDT` header + 40-byte records).
def save_binary_samples(path: str, samples: Sequence[SelfPlaySample]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<4sII", DATAGEN_MAGIC, DATAGEN_VERSION, len(samples)))
        for s in samples:
            b = Board(s.fen)
            packed = _pack_board_Squares(b)
            clamped_score = max(-30000, min(30000, int(s.score_cp)))
            wdl_int = int(round(s.wdl * 2.0))
            wdl_int = max(0, min(2, wdl_int))
            fp.write(RECORD_STRUCT.pack(packed, int(b.turn), clamped_score, wdl_int))
    return len(samples)


# Load SelfPlaySamples from a compact binary `.bin` dataset file.
def load_binary_samples(path: str) -> List[SelfPlaySample]:
    if not os.path.isfile(path):
        raise ValueError(f"Dataset file not found: {path}")
    samples: List[SelfPlaySample] = []
    with open(path, "rb") as fp:
        header = fp.read(12)
        if len(header) != 12:
            raise ValueError("Invalid binary dataset: truncated header")
        magic, version, count = struct.unpack("<4sII", header)
        if magic != DATAGEN_MAGIC or version != DATAGEN_VERSION:
            raise ValueError("Invalid binary dataset header magic or version")
        for _ in range(count):
            raw = fp.read(RECORD_STRUCT.size)
            if len(raw) != RECORD_STRUCT.size:
                raise ValueError("Invalid binary dataset: truncated record")
            packed, turn, score_cp, wdl_int = RECORD_STRUCT.unpack(raw)
            fen = _unpack_board_fen(packed, turn)
            samples.append(
                SelfPlaySample(
                    fen=fen,
                    score_cp=int(score_cp),
                    wdl=float(wdl_int) / 2.0,
                    turn=int(turn),
                )
            )
    return samples


# Save samples as newline-delimited JSON records.
def save_jsonl_samples(path: str, samples: Sequence[SelfPlaySample]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for s in samples:
            fp.write(
                json.dumps(
                    {"fen": s.fen, "score_cp": int(s.score_cp), "wdl": float(s.wdl), "turn": int(s.turn)}
                )
                + "\n"
            )
    return len(samples)


# Load samples from a newline-delimited JSON dataset file.
def load_jsonl_samples(path: str) -> List[SelfPlaySample]:
    samples: List[SelfPlaySample] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(
                SelfPlaySample(
                    fen=obj["fen"],
                    score_cp=int(obj["score_cp"]),
                    wdl=float(obj["wdl"]),
                    turn=int(obj.get("turn", 0)),
                )
            )
    return samples


# Play one self-play game with random opening plies and return labeled training samples.
def play_selfplay_game(
    searcher: Searcher,
    rng: random.Random,
    random_plies: int = 6,
    search_depth: int = 2,
    search_nodes: int = 0,
    max_plies: int = 120,
    adjudicate_cp: int = 1200,
) -> List[SelfPlaySample]:
    board = Board(STARTING_FEN)
    searcher.clear()

    for _ in range(random_plies):
        legal = board.legal_moves
        if not legal or board.is_game_over():
            break
        board.push(rng.choice(legal))

    recorded: List[tuple[str, int, int]] = []
    white_result: Optional[float] = None

    for _ in range(max_plies):
        if board.is_game_over():
            if board.is_checkmate():
                white_result = 0.0 if board.turn == Color.WHITE else 1.0
            else:
                white_result = 0.5
            break

        limits_kwargs = {"depth": search_depth}
        if search_nodes > 0:
            limits_kwargs["nodes"] = search_nodes

        info = searcher.search(board, **limits_kwargs)
        best_move = info.best_move
        if best_move not in board.legal_moves:
            break

        in_check = board.is_check()
        is_capture = best_move.flags in (MoveFlag.CAPTURE, MoveFlag.EP_CAPTURE) or best_move.flags >= MoveFlag.PROMO_N
        if not in_check and not is_capture and abs(info.score) < SCORE_MATE_THRESHOLD:
            recorded.append((board.fen, int(info.score), int(board.turn)))

        if abs(info.score) >= adjudicate_cp:
            stm_wins = info.score > 0
            if board.turn == Color.WHITE:
                white_result = 1.0 if stm_wins else 0.0
            else:
                white_result = 0.0 if stm_wins else 1.0
            break

        board.push(best_move)

    if white_result is None:
        final_eval = evaluate(board)
        white_eval = final_eval if board.turn == Color.WHITE else -final_eval
        if white_eval > 250:
            white_result = 1.0
        elif white_eval < -250:
            white_result = 0.0
        else:
            white_result = 0.5

    samples: List[SelfPlaySample] = []
    for fen, score_cp, turn in recorded:
        stm_wdl = white_result if turn == int(Color.WHITE) else (1.0 - white_result)
        samples.append(SelfPlaySample(fen=fen, score_cp=score_cp, wdl=stm_wdl, turn=turn))
    return samples


# Generate a dataset of self-play games and optionally save it to `output_path`.
def generate_dataset(
    num_games: int = 10,
    random_plies: int = 6,
    search_depth: int = 2,
    search_nodes: int = 0,
    max_plies: int = 100,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> List[SelfPlaySample]:
    rng = random.Random(seed)
    searcher = Searcher(hash_size_mb=8, use_nnue=False)
    all_samples: List[SelfPlaySample] = []

    for _ in range(num_games):
        game_samples = play_selfplay_game(
            searcher=searcher,
            rng=rng,
            random_plies=random_plies,
            search_depth=search_depth,
            search_nodes=search_nodes,
            max_plies=max_plies,
        )
        all_samples.extend(game_samples)

    if output_path:
        if output_path.endswith(".jsonl"):
            save_jsonl_samples(output_path, all_samples)
        else:
            save_binary_samples(output_path, all_samples)

    return all_samples

