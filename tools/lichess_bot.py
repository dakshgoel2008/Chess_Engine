"""Lichess BOT client, challenge filter, and UCI bridge for CastledEngine."""

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import urllib.error
import urllib.request

from castled.core.board import Board
from castled.core.move import Move
from castled.core.types import Color, STARTING_FEN
from castled.search.search import SearchLimits, Searcher
from castled.search.time_manager import allocate_time


@dataclass
class ChallengeConfig:
    """Challenge acceptance rules for the Lichess BOT."""

    concurrency: int = 1
    accept_bot: bool = True
    only_bot: bool = False
    min_base: int = 30
    max_base: int = 900
    min_increment: int = 0
    max_increment: int = 30
    variants: List[str] = field(default_factory=lambda: ["standard"])
    time_controls: List[str] = field(default_factory=lambda: ["bullet", "blitz", "rapid"])
    modes: List[str] = field(default_factory=lambda: ["casual", "rated"])


@dataclass
class BotConfig:
    """Top-level Lichess BOT and engine configuration."""

    token: str = ""
    url: str = "https://lichess.org/"
    hash_mb: int = 128
    threads: int = 1
    move_overhead_ms: int = 30
    use_nnue: bool = True
    book_enabled: bool = False
    book_path: str = "castled/book/opening.bin"
    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)


# Parse the simple YAML subset in `tools/lichess-bot-config.yml` without requiring PyYAML.
def load_bot_config(path: str = "tools/lichess-bot-config.yml", env_token: Optional[str] = None) -> BotConfig:
    cfg = BotConfig()
    if os.path.exists(path):
        current_list: Optional[List[str]] = None
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].rstrip()
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("- "):
                    val = stripped[2:].strip().strip('"').strip("'")
                    if current_list is not None:
                        current_list.append(val)
                    continue

                current_list = None
                if ":" not in stripped:
                    continue
                key, val_raw = stripped.split(":", 1)
                key = key.strip()
                val = val_raw.strip().strip('"').strip("'")

                if key == "token":
                    cfg.token = val
                elif key == "url":
                    cfg.url = val
                elif key == "Hash" and val.isdigit():
                    cfg.hash_mb = int(val)
                elif key == "Threads" and val.isdigit():
                    cfg.threads = int(val)
                elif key == "MoveOverhead" and val.isdigit():
                    cfg.move_overhead_ms = int(val)
                elif key == "UseNNUE":
                    cfg.use_nnue = val.lower() == "true"
                elif key == "enabled":
                    cfg.book_enabled = val.lower() == "true"
                elif key == "concurrency" and val.isdigit():
                    cfg.challenge.concurrency = int(val)
                elif key == "accept_bot":
                    cfg.challenge.accept_bot = val.lower() == "true"
                elif key == "only_bot":
                    cfg.challenge.only_bot = val.lower() == "true"
                elif key == "min_base" and val.isdigit():
                    cfg.challenge.min_base = int(val)
                elif key == "max_base" and val.isdigit():
                    cfg.challenge.max_base = int(val)
                elif key == "min_increment" and val.isdigit():
                    cfg.challenge.min_increment = int(val)
                elif key == "max_increment" and val.isdigit():
                    cfg.challenge.max_increment = int(val)
                elif key == "variants" and not val:
                    cfg.challenge.variants = []
                    current_list = cfg.challenge.variants
                elif key == "time_controls" and not val:
                    cfg.challenge.time_controls = []
                    current_list = cfg.challenge.time_controls
                elif key == "modes" and not val:
                    cfg.challenge.modes = []
                    current_list = cfg.challenge.modes

    token = env_token if env_token is not None else os.environ.get("LICHESS_TOKEN", "")
    if token:
        cfg.token = token
    elif cfg.token == "${LICHESS_TOKEN}":
        cfg.token = ""
    return cfg


# Evaluate whether an incoming Lichess challenge dictionary satisfies `ChallengeConfig`.
def should_accept_challenge(
    challenge: Dict[str, Any],
    cfg: ChallengeConfig,
    active_games: int = 0,
) -> Tuple[bool, str]:
    if active_games >= cfg.concurrency:
        return False, "tooManyGames"

    variant_key = challenge.get("variant", {}).get("key", "standard")
    if variant_key not in cfg.variants:
        return False, "variant"

    speed = challenge.get("speed", "")
    if speed not in cfg.time_controls:
        return False, "timeControl"

    tc = challenge.get("timeControl", {})
    tc_type = tc.get("type", "clock")
    if tc_type != "clock":
        return False, "timeControl"

    limit_sec = int(tc.get("limit", 0))
    inc_sec = int(tc.get("increment", 0))
    if not (cfg.min_base <= limit_sec <= cfg.max_base):
        return False, "timeControl"
    if not (cfg.min_increment <= inc_sec <= cfg.max_increment):
        return False, "timeControl"

    rated = bool(challenge.get("rated", False))
    mode_str = "rated" if rated else "casual"
    if mode_str not in cfg.modes:
        return False, "casual" if rated else "rated"

    challenger_title = challenge.get("challenger", {}).get("title")
    is_bot = (challenger_title == "BOT")
    if is_bot and not cfg.accept_bot:
        return False, "noBot"
    if not is_bot and cfg.only_bot:
        return False, "onlyBot"

    return True, "accepted"


# Reconstruct the board from a Lichess gameState payload and compute our engine's reply move.
def compute_bot_move_for_game_state(
    moves_str: str,
    our_color: Color,
    wtime_ms: int,
    btime_ms: int,
    winc_ms: int = 0,
    binc_ms: int = 0,
    initial_fen: str = STARTING_FEN,
    move_overhead_ms: int = 30,
    max_depth: int = 63,
    searcher: Optional[Searcher] = None,
) -> Optional[Move]:
    fen = STARTING_FEN if initial_fen in ("startpos", "") else initial_fen
    board = Board(fen)
    for token in moves_str.split():
        m = board.parse_uci(token)
        board.push(m)

    if board.is_game_over() or board.turn != our_color:
        return None

    legal_count = len(board.legal_moves)
    alloc = allocate_time(
        wtime_ms=wtime_ms,
        btime_ms=btime_ms,
        winc_ms=winc_ms,
        binc_ms=binc_ms,
        turn=our_color,
        move_overhead_ms=move_overhead_ms,
        legal_move_count=legal_count,
    )
    depth = 1 if alloc.instant else max_depth
    eng = searcher if searcher is not None else Searcher(hash_size_mb=32)
    limits = SearchLimits(
        depth=depth,
        movetime_ms=alloc.hard_ms,
        soft_movetime_ms=alloc.soft_ms,
    )
    info = eng.search(board, limits=limits)
    return info.best_move if info.best_move != Move.NULL else None


# Generate Markdown badges for tracking a Lichess BOT's live Blitz and Rapid ratings in README.md.
def generate_lichess_badge_markdown(username: str) -> str:
    clean = username.strip() or "CastledBot"
    profile_url = f"https://lichess.org/@/{clean}"
    badge_url = f"https://img.shields.io/badge/Lichess_BOT-%40{clean}-blue?logo=lichess"
    return f"[![Lichess BOT @{clean}]({badge_url})]({profile_url})"


class LichessBotClient:
    """Minimal HTTP/NDJSON Lichess BOT client driving CastledEngine."""

    def __init__(
        self,
        config: BotConfig,
        request_fn: Optional[Callable[[str, str, Optional[Dict[str, str]]], Any]] = None,
    ) -> None:
        self.config = config
        self.searcher = Searcher(hash_size_mb=config.hash_mb)
        self._request_fn = request_fn

    def _api_call(self, method: str, endpoint: str, data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if self._request_fn is not None:
            return self._request_fn(method, endpoint, data)

        url = self.config.url.rstrip("/") + "/" + endpoint.lstrip("/")
        headers = {"Authorization": f"Bearer {self.config.token}"}
        body = None
        if data is not None:
            body = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8").strip()
            return json.loads(raw) if raw else {}

    def handle_game_stream_lines(self, game_id: str, bot_user_id: str, lines: Iterable[str]) -> List[str]:
        """Process NDJSON lines from `/api/bot/game/stream/{game_id}` and submit moves on our turn."""
        our_color = Color.WHITE
        initial_fen = STARTING_FEN
        submitted_moves: List[str] = []
        self.searcher.clear()

        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            event = json.loads(text)
            ev_type = event.get("type")

            if ev_type == "gameFull":
                white_id = event.get("white", {}).get("id", "").lower()
                our_color = Color.WHITE if white_id == bot_user_id.lower() else Color.BLACK
                initial_fen = event.get("initialFen", "startpos")
                state = event.get("state", {})
            elif ev_type == "gameState":
                state = event
            else:
                continue

            if state.get("status", "started") != "started":
                break

            move = compute_bot_move_for_game_state(
                moves_str=state.get("moves", ""),
                our_color=our_color,
                wtime_ms=int(state.get("wtime", 60000)),
                btime_ms=int(state.get("btime", 60000)),
                winc_ms=int(state.get("winc", 0)),
                binc_ms=int(state.get("binc", 0)),
                initial_fen=initial_fen,
                move_overhead_ms=self.config.move_overhead_ms,
                searcher=self.searcher,
            )
            if move is not None:
                uci_str = move.to_uci()
                self._api_call("POST", f"/api/bot/game/{game_id}/move/{uci_str}")
                submitted_moves.append(uci_str)

        return submitted_moves

    def handle_event_stream_lines(self, lines: Iterable[str], active_games: int = 0) -> List[Tuple[str, str]]:
        """Process NDJSON lines from `/api/stream/event` and accept or decline challenges."""
        actions: List[Tuple[str, str]] = []
        current_active = active_games

        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            event = json.loads(text)
            ev_type = event.get("type")

            if ev_type == "challenge":
                ch = event.get("challenge", {})
                ch_id = ch.get("id", "")
                if not ch_id:
                    continue
                ok, reason = should_accept_challenge(ch, self.config.challenge, active_games=current_active)
                if ok:
                    self._api_call("POST", f"/api/challenge/{ch_id}/accept")
                    actions.append(("accept", ch_id))
                else:
                    self._api_call("POST", f"/api/challenge/{ch_id}/decline", {"reason": reason})
                    actions.append((f"decline:{reason}", ch_id))
            elif ev_type == "gameStart":
                game_id = event.get("game", {}).get("id", "")
                if game_id:
                    current_active += 1
                    actions.append(("gameStart", game_id))
            elif ev_type == "gameFinish":
                game_id = event.get("game", {}).get("id", "")
                current_active = max(0, current_active - 1)
                if game_id:
                    actions.append(("gameFinish", game_id))

        return actions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CastledEngine Lichess BOT runner and config validator.")
    parser.add_argument("--config", default="tools/lichess-bot-config.yml", help="Path to bot config YAML.")
    parser.add_argument("--check-config", action="store_true", help="Validate config and print summary.")
    parser.add_argument("--badge", default=None, help="Print Markdown rating badge for Lichess username.")
    args = parser.parse_args(argv)

    if args.badge:
        print(generate_lichess_badge_markdown(args.badge))
        return 0

    cfg = load_bot_config(args.config)
    if args.check_config:
        print(
            f"Config OK: Hash={cfg.hash_mb}MB Threads={cfg.threads} "
            f"Overhead={cfg.move_overhead_ms}ms TimeControls={','.join(cfg.challenge.time_controls)}"
        )
        return 0

    if not cfg.token:
        print("Error: LICHESS_TOKEN environment variable is not set.", file=sys.stderr)
        return 1

    print(f"CastledEngine Lichess BOT configured for {cfg.url} (Hash={cfg.hash_mb}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
