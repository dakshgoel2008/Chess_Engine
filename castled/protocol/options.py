"""UCI engine configuration options (Hash, Threads, MoveOverhead, UseNNUE, EvalFile, BookFile)."""

from dataclasses import dataclass
from typing import List


@dataclass
class EngineOptions:
    hash_size_mb: int = 16
    threads: int = 1
    move_overhead_ms: int = 10
    use_nnue: bool = False
    eval_file: str = ""
    book_file: str = ""

    # Apply a UCI setoption name/value pair with bounds clamping.
    def set_option(self, name: str, value: str):
        key = name.strip().lower()
        val = value.strip()

        if key == "hash":
            try:
                self.hash_size_mb = max(1, min(1024, int(val)))
            except ValueError:
                pass
        elif key == "threads":
            try:
                self.threads = max(1, min(64, int(val)))
            except ValueError:
                pass
        elif key == "moveoverhead":
            try:
                self.move_overhead_ms = max(0, min(5000, int(val)))
            except ValueError:
                pass
        elif key == "usennue":
            self.use_nnue = val.lower() in ("true", "1", "yes", "on")
        elif key == "evalfile":
            self.eval_file = "" if val == "<empty>" else val
        elif key == "bookfile":
            self.book_file = "" if val == "<empty>" else val

    # Return the standard UCI option declaration lines for the uci handshake.
    def format_uci_options(self) -> List[str]:
        nnue_default = "true" if self.use_nnue else "false"
        eval_default = self.eval_file if self.eval_file else "<empty>"
        book_default = self.book_file if self.book_file else "<empty>"
        return [
            f"option name Hash type spin default {self.hash_size_mb} min 1 max 1024",
            f"option name Threads type spin default {self.threads} min 1 max 64",
            f"option name MoveOverhead type spin default {self.move_overhead_ms} min 0 max 5000",
            f"option name UseNNUE type check default {nnue_default}",
            f"option name EvalFile type string default {eval_default}",
            f"option name BookFile type string default {book_default}",
        ]

