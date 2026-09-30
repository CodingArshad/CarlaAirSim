"""Phase 5+6: one agent completes a search task with no preflight plan, and
now does it honestly - discovering targets by sensing, not by being handed
the answer. The lifecycle: observe -> update belief -> select objective ->
choose skill -> validate (Guardian) -> execute -> verify - repeated until the
policy returns DONE. Every objective is a response to a named event, never a
bare timer tick.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..control.navigation import DEFAULT_HEIGHT
from ..control.skills import go_to_waypoint, hold_position, land, take_off
from ..core.interfaces import VehicleAdapter
from ..core.scenario import Scenario, Sector
from ..core.skill_result import SkillStatus
from ..experiments.mission_runner import lawnmower_waypoints
from .belief import Belief, SearchLeg
from .belief_schema import MissionBelief, SelfState, TargetSighting
from .decision_log import DecisionLogger
from .ground_truth import SensorModel
from .guardian import Guardian
from .objectives import Objective, ReplanEvent
from .search_policy import SearchAgentPolicy

MAX_IDLE_ROUNDS = 3  # safety valve: DONE must be reached, this just bounds a runaway loop
AGENT_REPORT_HOLD_S = 3.0  # how long THIS agent holds position to confirm a sighting - an agent-
# owned protocol constant, deliberately not read from the scenario's Target.dwell_s (that would be
# the same ground-truth leak Phase 6 exists to close, just moved to a different field)
SEARCH_WAYPOINT_SPACING_M = 3.0  # sub-waypoints along each sweep leg, close enough together that a
# target sitting mid-lane (not at a sector corner, where the raw lawnmower waypoints all are) still
# gets a sensor check near its closest approach - same lesson as Phase 4's coverage sampling gap


def _to_local(point: Tuple[float, float, float], spawn_offset: Tuple[float, float, float]):
    return (point[0] - spawn_offset[0], point[1] - spawn_offset[1], point[2] - spawn_offset[2])


def _to_world(point: Tuple[float, float, float], spawn_offset: Tuple[float, float, float]):
    return (point[0] + spawn_offset[0], point[1] + spawn_offset[1], point[2] + spawn_offset[2])


def _subdivide(a: Tuple[float, float, float], b: Tuple[float, float, float], spacing: float):
    dist = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5
    if dist <= spacing:
        return [b]
    steps = max(1, int(dist / spacing))
    return [tuple(a[k] + (b[k] - a[k]) * (i / steps) for k in range(3)) for i in range(1, steps + 1)]


def _build_search_queue(sector: Sector, spawn_offset, height: float) -> List[SearchLeg]:
    """Pure geometry - no target information goes into this. An agent's
    search pattern must not depend on where the things it's searching for
    actually are."""
    corners = lawnmower_waypoints(sector, height)
    points = [corners[0]] if corners else []
    for a, b in zip(corners, corners[1:]):
        points.extend(_subdivide(a, b, SEARCH_WAYPOINT_SPACING_M))
    return [SearchLeg(point=_to_local(p, spawn_offset)) for p in points]


@dataclass
class AgentReport:
    trace: List[str]
    target_found: Optional[str]
    battery_frac_at_end: float


class PersistentAgent:
    def __init__(self, adapter: VehicleAdapter, scenario: Scenario, sector_id: str,
                 spawn_offset: Tuple[float, float, float], battery_s: float,
                 policy: SearchAgentPolicy = None, guardian: Guardian = None,
                 cruise_height: float = DEFAULT_HEIGHT, logger: DecisionLogger = None,
                 drone_name: str = "drone"):
        self.adapter = adapter
        self.scenario = scenario
        self.sector = scenario.sector(sector_id)
        self.spawn_offset = spawn_offset
        self.cruise_height = cruise_height
        self.policy = policy or SearchAgentPolicy()
        self.guardian = guardian or Guardian(scenario.no_fly_zones)
        self.sensor = SensorModel(scenario)  # the ONLY thing here allowed to read scenario.targets
        self.logger = logger
        self.drone_name = drone_name
        self.belief = Belief(
            self_state=SelfState(position=(0.0, 0.0, 0.0), elapsed_s=0.0, battery_s=battery_s),
            mission=MissionBelief(sector_id=sector_id,
                                   search_queue=_build_search_queue(self.sector, spawn_offset, cruise_height)),
        )

    def run(self) -> AgentReport:
        start = time.monotonic()
        event = ReplanEvent.TASK_ASSIGNED
        trace = []
        idle_rounds = 0
        step = 0

        while True:
            objective, next_phase = self.policy.decide(self.belief, event)
            trace.append(f"{event.value}->{objective.value}")
            self.belief.phase = next_phase
            if self.logger:
                self.logger.record(step, self.drone_name, event, self.belief, objective)
            step += 1

            if objective == Objective.DONE:
                break

            event = self._execute(objective, trace)
            self.belief.elapsed_s = time.monotonic() - start

            if objective == Objective.TAKE_OFF and event == ReplanEvent.SKILL_FAILED:
                # No ground reference was ever recorded - RETURN_HOME/LAND both need
                # one, so there is nothing safe left to command. Stop here, don't
                # route through the normal failure path (which assumes airborne).
                trace.append("takeoff_failed->aborted")
                break

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
            return self._guarded_fly((0.0, 0.0, self.cruise_height), trace)

        if objective == Objective.SEARCH_SECTOR:
            if not self.belief.mission.search_queue:
                return ReplanEvent.SKILL_SUCCEEDED
            leg = self.belief.mission.search_queue.pop(0)
            event = self._guarded_fly(leg.point, trace)
            if event != ReplanEvent.SKILL_SUCCEEDED:
                return event
            return self._sense_after_arrival()

        if objective == Objective.REPORT:
            result = hold_position(self.adapter, AGENT_REPORT_HOLD_S)
            self.belief.position = result.final_position or self.belief.position
            unconfirmed = [s for s in self.belief.mission.targets_known.values() if not s.confirmed]
            if unconfirmed:
                unconfirmed[0].confirmed = True
            self.belief.communication.last_report_sent = unconfirmed[0].target_id if unconfirmed else None
            return ReplanEvent.REPORT_SENT

        if objective == Objective.RETURN_HOME:
            return self._guarded_fly((0.0, 0.0, DEFAULT_HEIGHT), trace)

        if objective == Objective.LAND:
            result = land(self.adapter)
            self.belief.position = result.final_position or self.belief.position
            return ReplanEvent.SKILL_SUCCEEDED if result.status == SkillStatus.SUCCESS else ReplanEvent.SKILL_FAILED

        return ReplanEvent.SKILL_SUCCEEDED

    def _sense_after_arrival(self) -> ReplanEvent:
        """The only place a target can enter belief: a real sensor reading at
        the drone's actual current position, after actually flying there."""
        world_pos = _to_world(self.belief.position, self.spawn_offset)
        for seen in self.sensor.perceive(world_pos):
            if seen.target_id not in self.belief.mission.targets_known:
                self.belief.mission.targets_known[seen.target_id] = TargetSighting(
                    target_id=seen.target_id, local_position=self.belief.position,
                    first_seen_t=self.belief.elapsed_s)
                return ReplanEvent.TARGET_DETECTED
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
