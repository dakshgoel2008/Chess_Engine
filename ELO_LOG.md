# CastledEngine Elo Progression Log

Every search and evaluation improvement is tested via Sequential Probability Ratio Testing (`tools/sprt.py`) over the 50-position opening suite (`tools/openings.epd`) with color swapping.

| Phase / Feature | Baseline | W / D / L | Elo Gain (95% CI) | SPRT Verdict | Notes |
|---|---|---|---|---|---|
| Legacy Pygame AI (`v0.0.1`) | - | - | `0.0` (Anchor) | - | Depth-3 Python array search, no qsearch |
| Phase 5: Tapered PeSTO + Mobility + Pawn/King Eval | `v0.0.1` | - | Baseline C Eval | - | Bit-exact C + Python tapered evaluation |
| Phase 6: Search v1 (ID + Alpha-Beta + QSearch + TT + MVV-LVA) | Depth-2 Search v1 | `10 / 0 / 0` | `+600.0+` | `H1_ACCEPTED` | Reaches depth 8 in 0.67s (~2.7M NPS) |
| Phase 7: UCI Protocol + Soft/Hard Time Manager | Phase 6 | - | `+0.0` (Infrastructure) | `PASS` | Zero time losses at `10s+0.1s` |

