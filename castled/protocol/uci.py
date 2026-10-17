"""UCI protocol state machine with background worker thread for non-blocking search."""

import sys
import threading
from typing import Callable, List, Optional, TextIO

from .. import __version__
from ..book.polyglot import PolyglotBook
from ..core.board import Board
from ..core.move import Move
from ..core.types import STARTING_FEN
from ..search.search import SearchInfo, SearchLimits, Searcher
from ..search.time_manager import allocate_time
from ..search.transposition import SCORE_MATE, SCORE_MATE_THRESHOLD
from .options import EngineOptions


# Convert internal centipawn or mate score into UCI 'cp <x>' or 'mate <n>' string.
def format_uci_score(score: int) -> str:
    if score > SCORE_MATE_THRESHOLD:
        plies = max(1, SCORE_MATE - score)
        return f"mate {(plies + 1) // 2}"
    if score < -SCORE_MATE_THRESHOLD:
        plies = max(1, SCORE_MATE + score)
        return f"mate -{(plies + 1) // 2}"
    return f"cp {score}"


class UCIEngine:
    """UCI protocol handler managing board state, options, and threaded search."""

    def __init__(
        self,
        out_stream: Optional[TextIO] = None,
        *,
        _use_c: Optional[bool] = None,
    ):
        self.options = EngineOptions()
        self.board = Board(STARTING_FEN, _use_c=_use_c)
        self.searcher = Searcher(hash_size_mb=self.options.hash_size_mb, _use_c=_use_c)
        self.book: Optional[PolyglotBook] = None
        self._out = out_stream if out_stream is not None else sys.stdout
        self._out_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._is_infinite = False

    def _send(self, line: str):
        with self._out_lock:
            self._out.write(line + "\n")
            self._out.flush()

    # Stop any active background search and wait for the worker thread to finish.
    def stop_search(self):
        if self._worker is not None and self._worker.is_alive():
            self.searcher.stop()
            self._worker.join()
        self._worker = None

    # Process a single UCI command line; returns False when 'quit' is received.
    def handle_command(self, raw_line: str) -> bool:
        line = raw_line.strip()
        if not line:
            return True

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            self._send(f"id name CastledEngine {__version__}")
            self._send("id author Daksh")
            for opt_line in self.options.format_uci_options():
                self._send(opt_line)
            self._send("uciok")
        elif cmd == "isready":
            self._send("readyok")
        elif cmd == "ucinewgame":
            self.stop_search()
            self.board = Board(STARTING_FEN, _use_c=self.board._use_c)
            self.searcher.clear()
        elif cmd == "setoption":
            self.stop_search()
            self._handle_setoption(tokens[1:])
        elif cmd == "position":
            self.stop_search()
            self._handle_position(tokens[1:])
        elif cmd == "go":
            self.stop_search()
            self._handle_go(tokens[1:])
        elif cmd == "stop":
            self.stop_search()
        elif cmd == "quit":
            if self._worker is not None and self._worker.is_alive():
                if not self._is_infinite:
                    self._worker.join(timeout=0.5)
                self.stop_search()
            return False

        return True

    def _handle_setoption(self, tokens: List[str]):
        if not tokens or tokens[0].lower() != "name":
            return

        val_idx = None
        for i, tok in enumerate(tokens):
            if tok.lower() == "value":
                val_idx = i
                break

        if val_idx is None:
            name = " ".join(tokens[1:])
            value = ""
        else:
            name = " ".join(tokens[1:val_idx])
            value = " ".join(tokens[val_idx + 1:])

        prev_hash = self.options.hash_size_mb
        prev_eval_file = self.options.eval_file
        prev_book_file = self.options.book_file
        self.options.set_option(name, value)
        if self.options.hash_size_mb != prev_hash:
            self.searcher.tt.resize(self.options.hash_size_mb)
        self.searcher.use_nnue = self.options.use_nnue
        if self.options.eval_file != prev_eval_file:
            try:
                self.searcher.load_nnue(self.options.eval_file)
            except ValueError:
                pass
        if self.options.book_file != prev_book_file:
            if not self.options.book_file:
                self.book = None
            else:
                try:
                    self.book = PolyglotBook(self.options.book_file)
                except ValueError:
                    self.book = None

    def _handle_position(self, tokens: List[str]):
        if not tokens:
            return

        moves_idx = None
        for i, tok in enumerate(tokens):
            if tok == "moves":
                moves_idx = i
                break

        pos_tokens = tokens if moves_idx is None else tokens[:moves_idx]
        move_tokens = [] if moves_idx is None else tokens[moves_idx + 1:]

        if pos_tokens and pos_tokens[0] == "startpos":
            self.board = Board(STARTING_FEN, _use_c=self.board._use_c)
        elif pos_tokens and pos_tokens[0] == "fen":
            fen_str = " ".join(pos_tokens[1:])
            if fen_str:
                self.board = Board(fen_str, _use_c=self.board._use_c)

        for m_str in move_tokens:
            move = self.board.parse_uci(m_str)
            self.board.push(move)

    def _handle_go(self, tokens: List[str]):
        depth: Optional[int] = None
        nodes: Optional[int] = None
        movetime: Optional[int] = None
        wtime: Optional[int] = None
        btime: Optional[int] = None
        winc = 0
        binc = 0
        movestogo: Optional[int] = None
        infinite = False

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "infinite":
                infinite = True
                i += 1
            elif tok in ("depth", "nodes", "movetime", "wtime", "btime", "winc", "binc", "movestogo") and i + 1 < len(tokens):
                try:
                    val = int(tokens[i + 1])
                except ValueError:
                    val = 0
                if tok == "depth":
                    depth = val
                elif tok == "nodes":
                    nodes = val
                elif tok == "movetime":
                    movetime = val
                elif tok == "wtime":
                    wtime = val
                elif tok == "btime":
                    btime = val
                elif tok == "winc":
                    winc = val
                elif tok == "binc":
                    binc = val
                elif tok == "movestogo":
                    movestogo = val
                i += 2
            else:
                i += 1

        if self.book is not None and not infinite:
            book_move = self.book.probe(self.board)
            if book_move is not None:
                self._send(f"bestmove {book_move.to_uci()}")
                return

        legal_count = len(self.board.legal_moves)
        soft_ms = 0
        hard_ms = 0

        if not infinite and (movetime is not None or wtime is not None or btime is not None):
            alloc = allocate_time(
                wtime_ms=wtime,
                btime_ms=btime,
                winc_ms=winc,
                binc_ms=binc,
                movestogo=movestogo,
                movetime_ms=movetime,
                turn=self.board.turn,
                move_overhead_ms=self.options.move_overhead_ms,
                legal_move_count=legal_count,
            )
            soft_ms = alloc.soft_ms
            hard_ms = alloc.hard_ms
            if alloc.instant and depth is None and nodes is None:
                depth = 1

        max_depth = depth if depth is not None else 63
        max_nodes = nodes if nodes is not None else 0
        limits = SearchLimits(
            depth=max_depth,
            nodes=max_nodes,
            movetime_ms=hard_ms,
            soft_movetime_ms=soft_ms,
        )
        board_copy = self.board.copy()
        self._is_infinite = infinite

        def _worker_fn():
            def _on_info(info: SearchInfo):
                pv_str = " ".join(m.to_uci() for m in info.pv)
                score_str = format_uci_score(info.score)
                self._send(
                    f"info depth {info.depth} score {score_str} "
                    f"nodes {info.nodes} nps {info.nps} time {info.time_ms} pv {pv_str}"
                )

            result = self.searcher.search(board_copy, limits=limits, on_info=_on_info)
            best_str = result.best_move.to_uci() if result.best_move != Move.NULL else "0000"
            self._send(f"bestmove {best_str}")

        self._worker = threading.Thread(target=_worker_fn, daemon=True)
        self._worker.start()

    # Main blocking loop reading UCI commands from in_stream until 'quit' or EOF.
    def run_loop(self, in_stream: Optional[TextIO] = None):
        stream = in_stream if in_stream is not None else sys.stdin
        for line in stream:
            if not self.handle_command(line):
                break
        if self._worker is not None and self._worker.is_alive():
            if not self._is_infinite:
                self._worker.join(timeout=0.5)
            self.stop_search()


def run_uci():
    engine = UCIEngine()
    engine.run_loop()

