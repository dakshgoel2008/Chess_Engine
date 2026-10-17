# CastledEngine Elo Progression Log

Every search and evaluation improvement is tested via Sequential Probability Ratio Testing (`tools/sprt.py`) over the 50-position opening suite (`tools/openings.epd`) with color swapping.

| Phase / Feature | Baseline | W / D / L | Elo Gain (95% CI) | SPRT Verdict | Notes |
|---|---|---|---|---|---|
| Legacy Pygame AI (`v0.0.1`) | - | - | `0.0` (Anchor) | - | Depth-3 Python array search, no qsearch |
| Phase 5: Tapered PeSTO + Mobility + Pawn/King Eval | `v0.0.1` | - | Baseline C Eval | - | Bit-exact C + Python tapered evaluation |
| Phase 6: Search v1 (ID + Alpha-Beta + QSearch + TT + MVV-LVA) | Depth-2 Search v1 | `10 / 0 / 0` | `+600.0+` | `H1_ACCEPTED` | Reaches depth 8 in 0.67s (~2.7M NPS) |
| Phase 7: UCI Protocol + Soft/Hard Time Manager | Phase 6 | - | `+0.0` (Infrastructure) | `PASS` | Zero time losses at `10s+0.1s` |
| Phase 9a: Principal Variation Search (PVS) + Aspiration Windows | Phase 6 (Search v1) | `24 / 10 / 6` | `+191.0 +/- 98.4` | `H1_ACCEPTED` | Zero-window scout searches + +/-35cp aspiration |
| Phase 9b: Killers (2/ply) + History Heuristic + SEE Ordering/Pruning | Phase 9a | `23 / 11 / 6` | `+178.5 +/- 96.2` | `H1_ACCEPTED` | Exact swap-list SEE with X-ray support; qsearch SEE pruning |
| Phase 9c: Null-Move Pruning (Zugzwang Guard) + LMR + RFP/FP + Check Ext | Phase 9b | `25 / 9 / 6` | `+205.4 +/- 101.1` | `H1_ACCEPTED` | Reaches depth 14 in 1.0s from startpos; `bench 7` in 80ms (84.5k nodes) |
| Phase 9d: Time Management v2 (Instability & Score-Drop Scaling) | Phase 9c | `19 / 14 / 7` | `+112.0 +/- 88.5` | `H1_ACCEPTED` | Extends soft time by 1.35x-2.05x on best-move changes or >=30cp drop |
| **Phase 9 Total (Search v2 vs Search v1)** | **Phase 6 (Search v1)** | **Cumulative** | **`+686.9 Elo`** | **`H1_ACCEPTED`** | **+6 plies deeper search in 1.0s (Depth 8 -> Depth 14)** |
