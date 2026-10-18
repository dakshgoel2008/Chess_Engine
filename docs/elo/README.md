# CastledEngine Elo gauntlet vs Stockfish

**Estimated Elo: 2501 ± 101** (95% interval, maximum-likelihood fit over 120 games), on Stockfish's `UCI_Elo` scale.

## Setup

- Date: 2026-10-18
- CastledEngine commit: `805524d`, evaluation: classical
- Opponent: Stockfish 19 with `UCI_LimitStrength=true` and `UCI_Elo` set per level
- Time: 100 ms per move for both engines, single thread
- Openings: `tools/openings.epd`, each played with both colors; games over 300 plies adjudicated as draws
- Hardware: 12th Gen Intel(R) Core(TM) i5-1235U, Linux

## Results

| Stockfish UCI_Elo | Won | Drawn | Lost | CastledEngine score |
|---|---|---|---|---|
| 1500 | 19 | 0 | 1 | 95% |
| 1800 | 19 | 0 | 1 | 95% |
| 2100 | 17 | 0 | 3 | 85% |
| 2400 | 12 | 1 | 7 | 62% |
| 2700 | 5 | 3 | 12 | 32% |
| 3000 | 1 | 3 | 16 | 12% |

All games: [`gauntlet.pgn`](gauntlet.pgn)

## Reproduce

```bash
python tools/elo_estimate.py --levels 1500 1800 2100 2400 2700 3000 --games 20 --movetime-ms 100 --pgn docs/elo/gauntlet.pgn --report docs/elo/README.md
```

## Caveats

- Stockfish calibrates `UCI_Elo` at longer time controls than used here, so the scale is approximate.
- This is not a FIDE, CCRL or Lichess rating, and is not directly comparable with them.
