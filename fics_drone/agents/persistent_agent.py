"""Phase 5 exit criterion: one agent completes a search task with no
preflight plan. The lifecycle: observe -> update belief -> select objective
-> choose skill -> validate (Guardian) -> execute -> verify - repeated until
the policy returns DONE. Every objective is a response to a named event,
never a bare timer tick.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..control.navigation import DEFAULT_HEIGHT
from ..control.skills import go_to_waypoint, hold_position, land, take_off
from ..core.interfaces import VehicleAdapter
from ..core.scenario import Scenario, Sector, Target
from ..core.skill_result import SkillStatus
from ..experiments.mission_runner import DWELL_MARGIN_S, lawnmower_waypoints
from .belief import Belief, SearchLeg
from .guardian import Guardian
from .objectives import Objective, ReplanEvent
from .search_policy import SearchAgentPolicy

MAX_IDLE_ROUNDS = 3  # safety valve: DONE must be reached, this just bounds a runaway loop


def _to_local(point: Tuple[float, float, float], spawn_offset: Tuple[float, float, float]):
    return (point[0] - spawn_offset[0], point[1] - spawn_offset[1], point[2] - spawn_offset[2])


def _to_world(point: Tuple[float, float, float], spawn_offset: Tuple[float, float, float]):
    return (point[0] + spawn_offset[0], point[1] + spawn_offset[1], point[2] + spawn_offset[2])


def _build_search_queue(sector: Sector, targets: List[Target], spawn_offset, height: float) -> List[SearchLeg]:
    legs = [SearchLeg(point=_to_local(wp, spawn_offset), is_target=False)
            for wp in lawnmower_waypoints(sector, height)]
    for t in targets:
        if sector.x_min <= t.x <= sector.x_max and sector.y_min <= t.y <= sector.y_max:
            legs.append(SearchLeg(point=_to_local((t.x, t.y, height), spawn_offset),
                                   is_target=True, target_id=t.id, dwell_s=t.dwell_s))
    return legs


@dataclass
class AgentReport:
    trace: List[str]
    target_found: Optional[str]
    battery_frac_at_end: float


class PersistentAgent:
    def __init__(self, adapter: VehicleAdapter, scenario: Scenario, sector_id: str,
                 spawn_offset: Tuple[float, float, float], battery_s: float,
                 policy: SearchAgentPolicy = None, guardian: Guardian = None,
                 cruise_height: float = DEFAULT_HEIGHT):
        self.adapter = adapter
        self.scenario = scenario
        self.sector = scenario.sector(sector_id)
        self.spawn_offset = spawn_offset
        self.cruise_height = cruise_height
        self.policy = policy or SearchAgentPolicy()
        self.guardian = guardian or Guardian(scenario.no_fly_zones)
        self.belief = Belief(
            position=(0.0, 0.0, 0.0), elapsed_s=0.0, battery_s=battery_s,
            search_queue=_build_search_queue(self.sector, scenario.targets, spawn_offset, cruise_height),
            phase="pre_takeoff",
        )

    def run(self) -> AgentReport:
        start = time.monotonic()
        event = ReplanEvent.TASK_ASSIGNED
        trace = []
        idle_rounds = 0

        while True:
            objective, next_phase = self.policy.decide(self.belief, event)
            trace.append(f"{event.value}->{objective.value}")
            self.belief.phase = next_phase

            if objective == Objective.DONE:
                break

            event = self._execute(objective, trace)
            self.belief.elapsed_s = time.monotonic() - start

            idle_rounds = idle_rounds + 1 if objective == Objective.SEARCH_SECTOR and not self.belief.search_queue else 0
            if idle_rounds > MAX_IDLE_ROUNDS:
                trace.append("idle_limit->return_home")
                self.belief.phase = "returning"
                event = ReplanEvent.SKILL_SUCCEEDED

        return AgentReport(trace=trace, target_found=self.belief.target_found,
                            battery_frac_at_end=self.belief.battery_frac_remaining)

    def _execute(self, objective: Objective, trace: List[str]) -> ReplanEvent:
        if objective == Objective.TAKE_OFF:
            result = take_off(self.adapter)
            self.belief.position = result.final_position or self.belief.position
            return ReplanEvent.SKILL_SUCCEEDED if result.status == SkillStatus.SUCCESS else ReplanEvent.SKILL_FAILED

        if objective == Objective.GO_TO_SECTOR:
            local_target = (0.0, 0.0, self.cruise_height)
            return self._guarded_fly(local_target, trace)

        if objective == Objective.SEARCH_SECTOR:
            if not self.belief.search_queue:
                return ReplanEvent.SKILL_SUCCEEDED
            leg = self.belief.search_queue.pop(0)
            event = self._guarded_fly(leg.point, trace)
            if event == ReplanEvent.SKILL_SUCCEEDED and leg.is_target:
                self.belief.target_found = leg.target_id
                self._pending_dwell = leg.dwell_s
                return ReplanEvent.TARGET_DETECTED
            return event

        if objective == Objective.REPORT:
            dwell_s = getattr(self, "_pending_dwell", 0.0) + DWELL_MARGIN_S
            result = hold_position(self.adapter, dwell_s)
            self.belief.position = result.final_position or self.belief.position
            return ReplanEvent.REPORT_SENT

        if objective == Objective.RETURN_HOME:
            local_target = (0.0, 0.0, DEFAULT_HEIGHT)
            return self._guarded_fly(local_target, trace)

        if objective == Objective.LAND:
            result = land(self.adapter)
            self.belief.position = result.final_position or self.belief.position
            return ReplanEvent.SKILL_SUCCEEDED if result.status == SkillStatus.SUCCESS else ReplanEvent.SKILL_FAILED

        return ReplanEvent.SKILL_SUCCEEDED

    def _guarded_fly(self, local_target: Tuple[float, float, float], trace: List[str]) -> ReplanEvent:
        world_target = _to_world(local_target, self.spawn_offset)
        allowed, reason = self.guardian.check(world_target)
        if not allowed:
            trace.append(f"guardian_blocked({reason})")
            return ReplanEvent.GUARDIAN_BLOCKED
        result = go_to_waypoint(self.adapter, *local_target)
        self.belief.position = result.final_position or self.belief.position
        if result.status == SkillStatus.SUCCESS:
            return ReplanEvent.SKILL_SUCCEEDED
        self.belief.nav_retries += 1
        return ReplanEvent.SKILL_FAILED
