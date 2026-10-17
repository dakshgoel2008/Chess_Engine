"""Time allocation for tournament clocks and fixed-movetime searches."""

from dataclasses import dataclass
from typing import Optional

from ..core.types import Color


@dataclass(frozen=True)
class TimeAllocation:
    soft_ms: int
    hard_ms: int
    instant: bool = False


# Compute soft and hard time budgets (in ms) from UCI clock parameters.
def allocate_time(
    wtime_ms: Optional[int] = None,
    btime_ms: Optional[int] = None,
    winc_ms: int = 0,
    binc_ms: int = 0,
    movestogo: Optional[int] = None,
    movetime_ms: Optional[int] = None,
    turn: Color = Color.WHITE,
    move_overhead_ms: int = 10,
    legal_move_count: int = 20,
) -> TimeAllocation:
    overhead = max(0, move_overhead_ms)

    if movetime_ms is not None and movetime_ms > 0:
        if legal_move_count <= 1:
            return TimeAllocation(soft_ms=1, hard_ms=1, instant=True)
        budget = max(1, movetime_ms - overhead) if movetime_ms > overhead else max(1, movetime_ms)
        return TimeAllocation(soft_ms=budget, hard_ms=budget, instant=False)

    remaining = wtime_ms if int(turn) == int(Color.WHITE) else btime_ms
    inc = winc_ms if int(turn) == int(Color.WHITE) else binc_ms

    if remaining is None or remaining <= 0:
        return TimeAllocation(soft_ms=0, hard_ms=0, instant=False)

    if legal_move_count <= 1:
        return TimeAllocation(soft_ms=1, hard_ms=1, instant=True)

    usable = max(1, remaining - overhead)
    mtg = max(1, movestogo) if (movestogo is not None and movestogo > 0) else 25

    soft = max(1, usable // mtg + (max(0, inc) * 3) // 4)
    hard = max(1, min(usable // 2, soft * 3))
    if soft > hard:
        soft = hard

    return TimeAllocation(soft_ms=soft, hard_ms=hard, instant=False)

