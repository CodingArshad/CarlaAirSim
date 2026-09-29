"""What the agent actually knows, updated after every skill call - kept
deliberately minimal for Phase 5 (a plain snapshot, no history/confidence/
staleness tracking). Phase 6 is exactly the phase that formalizes belief
properly and draws the line between "ground truth" and "what the agent
believes" - building that machinery now would duplicate that phase's work
before it's been designed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.skill_result import SkillResult


@dataclass
class SearchLeg:
    point: Tuple[float, float, float]  # local frame - already offset-adjusted
    is_target: bool
    target_id: Optional[str] = None
    dwell_s: float = 0.0


@dataclass
class Belief:
    position: Tuple[float, float, float]
    elapsed_s: float
    battery_s: float
    phase: str = "pre_takeoff"
    search_queue: List[SearchLeg] = field(default_factory=list)
    target_found: Optional[str] = None
    last_skill_result: Optional[SkillResult] = None
    nav_retries: int = 0

    @property
    def battery_frac_remaining(self) -> float:
        if self.battery_s <= 0:
            return 0.0
        return max(0.0, (self.battery_s - self.elapsed_s) / self.battery_s)
