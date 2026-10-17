"""PGN (Portable Game Notation) parser and serializer."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional

from .board import Board
from .move import Move
from .types import STARTING_FEN

_HEADER_RE = re.compile(r'^\s*\[\s*([A-Za-z0-9_]+)\s+"([^"]*)"\s*\]\s*$')
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


@dataclass
class PGNGame:
    """Represents a single chess game with headers and a move list."""
    headers: Dict[str, str] = field(default_factory=dict)
    moves: List[Move] = field(default_factory=list)
    result: str = "*"
    initial_fen: str = STARTING_FEN

    def board(self) -> Board:
        """Return the final board after replaying all moves."""
        b = Board(self.initial_fen)
        for m in self.moves:
            b.push(m)
        return b

    def to_pgn(self) -> str:
        """Serialize the game into standard PGN text."""
        lines: List[str] = []
        for k, v in self.headers.items():
            lines.append(f'[{k} "{v}"]')
        if lines:
            lines.append("")

        b = Board(self.initial_fen)
        tokens: List[str] = []
        for idx, m in enumerate(self.moves):
            if b.turn == 0:
                move_num = b.fullmove_number
                tokens.append(f"{move_num}. {b.san(m)}")
            else:
                if idx == 0:
                    tokens.append(f"{b.fullmove_number}... {b.san(m)}")
                else:
                    tokens.append(b.san(m))
            b.push(m)

        res = self.headers.get("Result", self.result)
        if res:
            tokens.append(res)

        lines.append(" ".join(tokens))
        return "\n".join(lines).strip()


def _strip_comments_and_variations(text: str) -> str:
    """Remove {...} comments, ; line comments, and (...) variations from PGN movetext."""
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r";[^\n]*", " ", text)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([^()]*\)", " ", text)
    return text


def parse_pgn(pgn_text: str) -> PGNGame:
    """Parse a single PGN string into a PGNGame object."""
    headers: Dict[str, str] = {}
    movetext_lines: List[str] = []

    for line in pgn_text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
        else:
            movetext_lines.append(line)

    initial_fen = headers.get("FEN", STARTING_FEN)
    board = Board(initial_fen)
    moves: List[Move] = []
    result = headers.get("Result", "*")

    body = _strip_comments_and_variations("\n".join(movetext_lines))
    for raw in body.split():
        if raw in _RESULTS:
            result = raw
            break
        if raw.startswith("$"):
            continue
        token = re.sub(r"^\d+\.+", "", raw).strip()
        if not token or token in _RESULTS:
            if token in _RESULTS:
                result = token
                break
            continue
        move = board.parse_san(token)
        moves.append(move)
        board.push(move)

    return PGNGame(headers=headers, moves=moves, result=result, initial_fen=initial_fen)


def dump_pgn(board: Board, headers: Optional[Dict[str, str]] = None) -> str:
    """Convert a Board's move history into a PGN string."""
    hdrs = dict(headers) if headers else {}
    outcome = board.outcome()
    result = outcome.result if outcome else hdrs.get("Result", "*")
    hdrs.setdefault("Result", result)
    game = PGNGame(headers=hdrs, moves=list(board._move_stack), result=result)
    return game.to_pgn()

