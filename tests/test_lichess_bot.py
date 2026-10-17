"""Unit and edge-case tests for the Lichess BOT configuration, challenge filters, and stream bridge."""

import json
from typing import Any, Dict, List, Optional, Tuple

from castled.core.types import Color
from tools.lichess_bot import (
    ChallengeConfig,
    LichessBotClient,
    compute_bot_move_for_game_state,
    generate_lichess_badge_markdown,
    load_bot_config,
    main,
    should_accept_challenge,
)


def test_load_bot_config_defaults_and_yaml():
    cfg = load_bot_config("tools/lichess-bot-config.yml", env_token="lip_test123")
    assert cfg.token == "lip_test123"
    assert cfg.hash_mb == 128
    assert cfg.threads == 1
    assert cfg.move_overhead_ms == 30
    assert cfg.use_nnue is True
    assert cfg.challenge.concurrency == 1
    assert "standard" in cfg.challenge.variants
    assert set(cfg.challenge.time_controls) == {"bullet", "blitz", "rapid"}
    assert set(cfg.challenge.modes) == {"casual", "rated"}


def test_should_accept_challenge_filters():
    rules = ChallengeConfig(
        concurrency=1,
        accept_bot=True,
        only_bot=False,
        min_base=30,
        max_base=900,
        min_increment=0,
        max_increment=30,
        variants=["standard"],
        time_controls=["bullet", "blitz", "rapid"],
        modes=["casual", "rated"],
    )

    valid_blitz = {
        "id": "ch1",
        "variant": {"key": "standard"},
        "speed": "blitz",
        "timeControl": {"type": "clock", "limit": 180, "increment": 2},
        "rated": True,
        "challenger": {"id": "user1", "title": None},
    }
    ok, reason = should_accept_challenge(valid_blitz, rules, active_games=0)
    assert ok is True and reason == "accepted"

    # Reject when active_games >= concurrency
    ok, reason = should_accept_challenge(valid_blitz, rules, active_games=1)
    assert ok is False and reason == "tooManyGames"

    # Reject non-standard variant
    bad_variant = {**valid_blitz, "variant": {"key": "chess960"}}
    ok, reason = should_accept_challenge(bad_variant, rules)
    assert ok is False and reason == "variant"

    # Reject ultraBullet or correspondence
    bad_speed = {**valid_blitz, "speed": "ultraBullet"}
    ok, reason = should_accept_challenge(bad_speed, rules)
    assert ok is False and reason == "timeControl"

    # Reject base time out of range
    bad_time = {**valid_blitz, "timeControl": {"type": "clock", "limit": 15, "increment": 0}}
    ok, reason = should_accept_challenge(bad_time, rules)
    assert ok is False and reason == "timeControl"

    # Reject BOT when accept_bot is False
    no_bot_rules = ChallengeConfig(accept_bot=False)
    bot_chal = {**valid_blitz, "challenger": {"id": "otherbot", "title": "BOT"}}
    ok, reason = should_accept_challenge(bot_chal, no_bot_rules)
    assert ok is False and reason == "noBot"


def test_compute_bot_move_for_game_state():
    # White to move at startpos
    m_white = compute_bot_move_for_game_state(
        moves_str="",
        our_color=Color.WHITE,
        wtime_ms=1000,
        btime_ms=1000,
        max_depth=3,
    )
    assert m_white is not None

    # Not our turn -> returns None
    m_none = compute_bot_move_for_game_state(
        moves_str="e2e4",
        our_color=Color.WHITE,
        wtime_ms=1000,
        btime_ms=1000,
        max_depth=3,
    )
    assert m_none is None

    # Black's turn after 1. e4 -> returns legal Black move
    m_black = compute_bot_move_for_game_state(
        moves_str="e2e4",
        our_color=Color.BLACK,
        wtime_ms=1000,
        btime_ms=1000,
        max_depth=3,
    )
    assert m_black is not None

    # Already checkmated position (Fool's Mate) -> returns None
    m_over = compute_bot_move_for_game_state(
        moves_str="f2f3 e7e5 g2g4 d8h4",
        our_color=Color.WHITE,
        wtime_ms=1000,
        btime_ms=1000,
        max_depth=3,
    )
    assert m_over is None


def test_lichess_bot_client_streams_and_cli(capsys):
    calls: List[Tuple[str, str, Optional[Dict[str, str]]]] = []

    def mock_req(method: str, endpoint: str, data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        calls.append((method, endpoint, data))
        return {"ok": True}

    cfg = load_bot_config("tools/lichess-bot-config.yml", env_token="lip_mock")
    cfg.hash_mb = 8
    client = LichessBotClient(cfg, request_fn=mock_req)

    events = [
        json.dumps({
            "type": "challenge",
            "challenge": {
                "id": "good123",
                "variant": {"key": "standard"},
                "speed": "blitz",
                "timeControl": {"type": "clock", "limit": 180, "increment": 2},
                "rated": False,
                "challenger": {"id": "alice", "title": None},
            },
        }),
        json.dumps({
            "type": "challenge",
            "challenge": {
                "id": "bad456",
                "variant": {"key": "atomic"},
                "speed": "blitz",
                "timeControl": {"type": "clock", "limit": 180, "increment": 2},
                "rated": False,
                "challenger": {"id": "bob", "title": None},
            },
        }),
    ]
    actions = client.handle_event_stream_lines(events)
    assert actions == [("accept", "good123"), ("decline:variant", "bad456")]

    game_lines = [
        json.dumps({
            "type": "gameFull",
            "white": {"id": "castledbot"},
            "black": {"id": "alice"},
            "initialFen": "startpos",
            "state": {
                "type": "gameState",
                "moves": "",
                "wtime": 500,
                "btime": 500,
                "winc": 0,
                "binc": 0,
                "status": "started",
            },
        }),
        json.dumps({
            "type": "gameState",
            "moves": "e2e4 e7e5",
            "wtime": 500,
            "btime": 500,
            "winc": 0,
            "binc": 0,
            "status": "started",
        }),
    ]
    moves_played = client.handle_game_stream_lines("game999", "castledbot", game_lines)
    assert len(moves_played) == 2
    assert any(ep.startswith("/api/bot/game/game999/move/") for _, ep, _ in calls)

    assert main(["--check-config"]) == 0
    assert main(["--badge", "CastledBot"]) == 0
    out = capsys.readouterr().out
    assert "Config OK:" in out
    assert "https://lichess.org/@/CastledBot" in out

