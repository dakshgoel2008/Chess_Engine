"""Lichess BOT client, challenge filter, and UCI bridge for CastledEngine."""

import argparse
from dataclasses import dataclass, field
import json
import os
import random
import sys
import threading
import time
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
    book_path: str = "castled/book/default_book.bin"
    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)
    matchmaking: "MatchmakingConfig" = field(default_factory=lambda: MatchmakingConfig())


@dataclass
class MatchmakingConfig:
    """Settings for automatically challenging other online BOTs while idle."""

    enabled: bool = False
    base_sec: int = 180
    increment_sec: int = 2
    rated: bool = True
    rating_difference: int = 300
    challenge_timeout_sec: int = 30
    cooldown_sec: int = 20


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
                elif key == "allow_matchmaking":
                    cfg.matchmaking.enabled = val.lower() == "true"
                elif key == "challenge_initial_time" and val.isdigit():
                    cfg.matchmaking.base_sec = int(val)
                elif key == "challenge_increment" and val.isdigit():
                    cfg.matchmaking.increment_sec = int(val)
                elif key == "challenge_mode":
                    cfg.matchmaking.rated = val.lower() == "rated"
                elif key == "opponent_rating_difference" and val.isdigit():
                    cfg.matchmaking.rating_difference = int(val)
                elif key == "challenge_timeout" and val.isdigit():
                    cfg.matchmaking.challenge_timeout_sec = int(val)
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


# Lichess speed category for a clock, using its estimated duration (base + 40 * increment seconds).
def speed_for_clock(base_sec: int, inc_sec: int) -> str:
    estimate = base_sec + 40 * inc_sec
    if estimate < 30:
        return "ultraBullet"
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"


# Pick a random online BOT whose rating in `speed` is within `max_diff` of ours.
def choose_opponent(
    bots: Iterable[Dict[str, Any]],
    our_id: str,
    our_rating: int,
    speed: str,
    max_diff: int,
    recent: Iterable[str] = (),
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    recent_set = {r.lower() for r in recent}
    candidates = []
    for bot in bots:
        bot_id = str(bot.get("id", "")).lower()
        if not bot_id or bot_id == our_id.lower() or bot_id in recent_set:
            continue
        if bot.get("disabled") or bot.get("tosViolation"):
            continue
        rating = bot.get("perfs", {}).get(speed, {}).get("rating")
        if rating is None or abs(int(rating) - our_rating) > max_diff:
            continue
        candidates.append(bot.get("username", bot_id))
    if not candidates:
        return None
    return (rng or random).choice(candidates)


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
        stream_fn: Optional[Callable[[str], Iterable[str]]] = None,
    ) -> None:
        self.config = config
        self.searcher = Searcher(hash_size_mb=config.hash_mb, use_nnue=config.use_nnue)
        self._request_fn = request_fn
        self._stream_fn = stream_fn
        # The C search uses a global transposition table, so only one search may run at a time.
        self._search_lock = threading.Lock()
        self._games_lock = threading.Lock()
        self.active_games: set = set()
        self.bot_id = ""
        self.pending_challenge: Optional[Tuple[str, float]] = None  # (challenge id, sent at)
        self._last_matchmake = 0.0
        self._recent_opponents: List[str] = []

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

    # Yield NDJSON lines from a long-lived Lichess streaming endpoint (keep-alive newlines included).
    def _stream_lines(self, endpoint: str) -> Iterable[str]:
        if self._stream_fn is not None:
            yield from self._stream_fn(endpoint)
            return
        url = self.config.url.rstrip("/") + "/" + endpoint.lstrip("/")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.config.token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                yield raw.decode("utf-8")

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

            if state.get("status", "started") not in ("created", "started"):
                break

            with self._search_lock:
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
                try:
                    self._api_call("POST", f"/api/bot/game/{game_id}/move/{uci_str}")
                except urllib.error.HTTPError as exc:
                    # Usually the game ended (resign/flag/abort) while we were thinking.
                    _log(f"[{game_id}] move {uci_str} rejected: HTTP {exc.code}")
                    continue
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

    # Play one game to completion by following its Lichess game stream.
    def play_game(self, game_id: str) -> None:
        _log(f"[{game_id}] game started")
        try:
            moves = self.handle_game_stream_lines(game_id, self.bot_id, self._stream_lines(f"/api/bot/game/stream/{game_id}"))
            _log(f"[{game_id}] stream ended after {len(moves)} engine moves")
        except Exception as exc:  # keep the bot alive if one game fails
            _log(f"[{game_id}] game error: {exc!r}")
        finally:
            with self._games_lock:
                self.active_games.discard(game_id)

    # Handle one event from `/api/stream/event` in live mode.
    def _handle_live_event(self, event: Dict[str, Any]) -> None:
        ev_type = event.get("type")
        if ev_type == "challenge":
            ch = event.get("challenge", {})
            ch_id = ch.get("id", "")
            challenger = ch.get("challenger", {}).get("id", "")
            # Lichess also echoes challenges the bot itself sent; ignore those.
            if not ch_id or challenger.lower() == self.bot_id.lower():
                return
            with self._games_lock:
                active = len(self.active_games)
            ok, reason = should_accept_challenge(ch, self.config.challenge, active_games=active)
            speed = ch.get("speed", "?")
            if ok:
                self._cancel_pending_challenge()
            try:
                if ok:
                    self._api_call("POST", f"/api/challenge/{ch_id}/accept")
                    _log(f"accepted {speed} challenge {ch_id} from {challenger}")
                else:
                    self._api_call("POST", f"/api/challenge/{ch_id}/decline", {"reason": reason})
                    _log(f"declined {speed} challenge {ch_id} from {challenger} ({reason})")
            except urllib.error.HTTPError as exc:
                _log(f"challenge {ch_id}: HTTP {exc.code}")
        elif ev_type == "gameStart":
            game_id = event.get("game", {}).get("gameId") or event.get("game", {}).get("id", "")
            if not game_id:
                return
            with self._games_lock:
                self.pending_challenge = None
                if game_id in self.active_games:
                    return
                self.active_games.add(game_id)
            threading.Thread(target=self.play_game, args=(game_id,), daemon=True).start()
        elif ev_type == "gameFinish":
            game = event.get("game", {})
            _log(f"[{game.get('gameId') or game.get('id', '?')}] game finished")
        elif ev_type in ("challengeDeclined", "challengeCanceled"):
            ch_id = event.get("challenge", {}).get("id", "")
            with self._games_lock:
                if self.pending_challenge and self.pending_challenge[0] == ch_id:
                    self.pending_challenge = None
                    reason = event.get("challenge", {}).get("declineReason", "")
                    _log(f"our challenge {ch_id} was {ev_type[9:].lower()} {reason}".rstrip())

    def _cancel_pending_challenge(self) -> None:
        with self._games_lock:
            pending = self.pending_challenge
            self.pending_challenge = None
        if pending is None:
            return
        try:
            self._api_call("POST", f"/api/challenge/{pending[0]}/cancel")
        except urllib.error.HTTPError:
            pass  # already accepted, declined, or expired

    # One matchmaking step: cancel a stale outgoing challenge, or challenge a new BOT when idle.
    def matchmake_once(self, now: Optional[float] = None) -> Optional[str]:
        mm = self.config.matchmaking
        now = time.monotonic() if now is None else now
        with self._games_lock:
            busy = bool(self.active_games)
            pending = self.pending_challenge
        if busy:
            return None
        if pending is not None:
            if now - pending[1] >= mm.challenge_timeout_sec:
                _log(f"challenge {pending[0]} not answered; cancelling")
                self._cancel_pending_challenge()
                self._last_matchmake = now
            return None
        if now - self._last_matchmake < mm.cooldown_sec:
            return None
        self._last_matchmake = now

        speed = speed_for_clock(mm.base_sec, mm.increment_sec)
        account = self._api_call("GET", "/api/account")
        our_rating = int(account.get("perfs", {}).get(speed, {}).get("rating", 1500))
        bots = []
        for raw in self._stream_lines("/api/bot/online?nb=100"):
            text = raw.strip()
            if text:
                bots.append(json.loads(text))
        opponent = choose_opponent(
            bots, self.bot_id, our_rating, speed, mm.rating_difference, recent=self._recent_opponents
        )
        if opponent is None:
            _log(f"matchmaking: no online BOT within {mm.rating_difference} of {our_rating} ({speed})")
            return None

        resp = self._api_call(
            "POST",
            f"/api/challenge/{opponent}",
            {
                "rated": "true" if mm.rated else "false",
                "clock.limit": str(mm.base_sec),
                "clock.increment": str(mm.increment_sec),
                "color": "random",
                "variant": "standard",
            },
        )
        ch_id = resp.get("id") or resp.get("challenge", {}).get("id", "")
        if ch_id:
            with self._games_lock:
                self.pending_challenge = (ch_id, now)
        self._recent_opponents = (self._recent_opponents + [opponent])[-10:]
        mode = "rated" if mm.rated else "casual"
        _log(f"matchmaking: challenged {opponent} ({mode} {mm.base_sec // 60}+{mm.increment_sec}, our {speed} {our_rating})")
        return opponent

    def _matchmaking_loop(self) -> None:
        while True:
            time.sleep(5)
            try:
                self.matchmake_once()
            except urllib.error.HTTPError as exc:
                _log(f"matchmaking HTTP {exc.code}; backing off")
                self._last_matchmake = time.monotonic() + (60 if exc.code == 429 else 0)
            except Exception as exc:
                _log(f"matchmaking error {exc!r}")

    # Connect to Lichess and serve challenges forever, reconnecting on network errors.
    def run(self) -> int:
        try:
            account = self._api_call("GET", "/api/account")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                _log("HTTP 401: LICHESS_TOKEN is invalid or expired. Create one at https://lichess.org/account/oauth/token")
            else:
                _log(f"Could not read the Lichess account: HTTP {exc.code}")
            return 1
        except urllib.error.URLError as exc:
            _log(f"Could not reach Lichess: {exc.reason}")
            return 1
        self.bot_id = account.get("id", "")
        if account.get("title") != "BOT":
            _log(
                f"Account '{account.get('username', self.bot_id)}' is not a BOT account. "
                "Upgrade it once with: python tools/lichess_bot.py --upgrade-account"
            )
            return 1
        eval_name = "NNUE" if self.config.use_nnue else "classical"
        _log(f"Logged in as BOT {account.get('username', self.bot_id)} ({eval_name} eval). Waiting for challenges...")
        if self.config.matchmaking.enabled:
            mm = self.config.matchmaking
            _log(f"matchmaking ON: challenging BOTs within {mm.rating_difference} rating at {mm.base_sec // 60}+{mm.increment_sec}")
            threading.Thread(target=self._matchmaking_loop, daemon=True).start()

        backoff = 5
        while True:
            try:
                for raw in self._stream_lines("/api/stream/event"):
                    text = raw.strip()
                    if not text:
                        continue  # keep-alive
                    backoff = 5
                    self._handle_live_event(json.loads(text))
                _log("event stream closed by server; reconnecting")
            except urllib.error.HTTPError as exc:
                wait = 60 if exc.code == 429 else backoff
                if exc.code == 401:
                    _log("HTTP 401: the token is invalid or lacks the bot:play scope.")
                    return 1
                _log(f"event stream HTTP {exc.code}; retrying in {wait}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                _log(f"event stream error {exc!r}; retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CastledEngine Lichess BOT runner and config validator.")
    parser.add_argument("--config", default="tools/lichess-bot-config.yml", help="Path to bot config YAML.")
    parser.add_argument("--check-config", action="store_true", help="Validate config and print summary.")
    parser.add_argument("--badge", default=None, help="Print Markdown rating badge for Lichess username.")
    parser.add_argument(
        "--matchmaking",
        action="store_true",
        help="Automatically challenge other online BOTs while idle (overrides allow_matchmaking).",
    )
    parser.add_argument(
        "--upgrade-account",
        action="store_true",
        help="Permanently upgrade the token's (unused) Lichess account to a BOT account.",
    )
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

    if args.matchmaking:
        cfg.matchmaking.enabled = True
    client = LichessBotClient(cfg)
    if args.upgrade_account:
        try:
            client._api_call("POST", "/api/bot/account/upgrade")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            print(f"Upgrade failed: HTTP {exc.code} {detail}", file=sys.stderr)
            return 1
        print("Account upgraded to BOT.")
        return 0

    try:
        return client.run()
    except KeyboardInterrupt:
        _log("stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
