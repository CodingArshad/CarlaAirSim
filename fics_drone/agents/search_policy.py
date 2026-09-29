"""The deterministic decision policy - built and proven before any LLM policy
replaces it later. A small explicit state machine (belief.phase), not
inferred implicitly from history, so every decision is traceable to a named
state and a named event: exactly what "auditable" has to mean for a system
whose whole point is replacing an opaque one-shot plan with something you can
actually reason about after the fact.
"""

from typing import Tuple

from ..core.skill_result import SkillStatus
from .belief import Belief
from .objectives import Objective, ReplanEvent

DEFAULT_LOW_BATTERY_FRAC = 0.30
DEFAULT_CRITICAL_BATTERY_FRAC = 0.12
DEFAULT_MAX_NAV_RETRIES = 2


class SearchAgentPolicy:
    def __init__(self, low_battery_frac: float = DEFAULT_LOW_BATTERY_FRAC,
                 critical_battery_frac: float = DEFAULT_CRITICAL_BATTERY_FRAC,
                 max_nav_retries: int = DEFAULT_MAX_NAV_RETRIES):
        self.low_battery_frac = low_battery_frac
        self.critical_battery_frac = critical_battery_frac
        self.max_nav_retries = max_nav_retries

    def decide(self, belief: Belief, event: ReplanEvent) -> Tuple[Objective, str]:
        """Returns (objective, next phase) - a pure function of belief+event,
        never a bare timer tick. Caller applies the phase; this never mutates
        belief itself, so it stays trivially testable in isolation."""
        abortable = belief.phase not in ("returning", "landing", "done")
        if abortable and belief.battery_frac_remaining <= self.critical_battery_frac:
            return Objective.RETURN_HOME, "returning"

        if belief.phase == "pre_takeoff":
            return Objective.TAKE_OFF, "climbing"

        if belief.phase in ("climbing", "transit"):
            if event in (ReplanEvent.SKILL_FAILED, ReplanEvent.GUARDIAN_BLOCKED):
                # a blocked/failed climb or transit is a bigger problem than a
                # single bad search leg - head home rather than loop on it.
                return Objective.RETURN_HOME, "returning"
            next_phase = "transit" if belief.phase == "climbing" else "searching"
            objective = Objective.GO_TO_SECTOR if belief.phase == "climbing" else Objective.SEARCH_SECTOR
            return objective, next_phase

        if belief.phase == "searching":
            if event == ReplanEvent.TARGET_DETECTED:
                return Objective.REPORT, "reporting"
            if event == ReplanEvent.SKILL_FAILED:
                return self._retry_or_abort(belief, Objective.SEARCH_SECTOR, "searching")
            if event == ReplanEvent.GUARDIAN_BLOCKED:
                # the agent already dropped the offending leg without flying it -
                # not a nav_retries case, retrying the same illegal point would
                # never succeed. Just keep going on whatever's left in the queue.
                return Objective.SEARCH_SECTOR, "searching"
            if not belief.search_queue:  # sector fully swept, nothing left to find
                return Objective.RETURN_HOME, "returning"
            return Objective.SEARCH_SECTOR, "searching"

        if belief.phase == "reporting":
            return Objective.RETURN_HOME, "returning"

        if belief.phase == "returning":
            if event in (ReplanEvent.SKILL_FAILED, ReplanEvent.GUARDIAN_BLOCKED):
                return self._retry_or_abort(belief, Objective.RETURN_HOME, "returning")
            return Objective.LAND, "landing"

        if belief.phase == "landing":
            return Objective.DONE, "done"

        return Objective.DONE, "done"

    def _retry_or_abort(self, belief: Belief, retry_objective: Objective, retry_phase: str) -> Tuple[Objective, str]:
        if belief.nav_retries >= self.max_nav_retries:
            return Objective.RETURN_HOME, "returning"
        return retry_objective, retry_phase
