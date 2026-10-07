"""The structured shape of what an agent knows. Six sections, matching
FICS's own layout - self/mission/local_map/team/communication/assumptions -
deliberately not a free-form blob or an LLM conversation transcript, because
a transcript can't answer "what did this agent know at t=140?" the way a
typed snapshot can.

Only `mission.targets_known` and `local_map.visited` are populated for now -
`team`/`communication` exist so Phase 7-9 (other agents, messaging) don't
need to redesign this schema, but stay empty until there's a second agent to
have beliefs about.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_TTL_S = 60.0
CONFIDENCE_HALF_LIFE_S = 20.0


@dataclass
class SelfState:
    position: Tuple[float, float, float]
    elapsed_s: float
    battery_s: float
    phase: str = "pre_takeoff"
    nav_retries: int = 0
    listen_rounds: int = 0  # Phase 7: how many times this agent has lingered post-search for messages
    last_block_reason: Optional[str] = None  # Phase 12: why the SafetyGuardian refused this agent's
    # most recent flight command, in the guardian's own words - purely informational, shown to a model
    # policy so it can stop re-proposing the same unsafe thing. Cleared by the next approved flight.

    @property
    def battery_frac_remaining(self) -> float:
        if self.battery_s <= 0:
            return 0.0
        return max(0.0, (self.battery_s - self.elapsed_s) / self.battery_s)


@dataclass
class TargetSighting:
    """What THIS agent knows about a target - never the scenario's own Target
    instance. source distinguishes a real sensor reading from this agent's
    own flight (Phase 6) from a teammate's TARGET_FOUND broadcast (Phase 7) -
    FICS's own exit criterion requires a drone that never visited a target's
    sector to still be able to say it only knows about it second-hand."""
    target_id: str
    local_position: Tuple[float, float, float]
    first_seen_t: float
    confirmed: bool = False  # True once held long enough to count as reported
    source: str = "sensor"   # "sensor" | "message"


@dataclass
class MissionBelief:
    sector_id: str
    search_queue: List = field(default_factory=list)  # List[SearchLeg], forward ref to avoid a cycle
    targets_known: Dict[str, TargetSighting] = field(default_factory=dict)


@dataclass
class VisitedPoint:
    local_position: Tuple[float, float, float]
    t: float


@dataclass
class LocalMap:
    visited: List[VisitedPoint] = field(default_factory=list)


@dataclass
class Provenance:
    """Ages a belief about a teammate - not exercised until Phase 7+ actually
    delivers messages between agents, but the decay math is real now."""
    timestamp: float
    source: str
    base_confidence: float = 1.0

    def decayed_confidence(self, now: float) -> float:
        age = max(0.0, now - self.timestamp)
        return self.base_confidence * (0.5 ** (age / CONFIDENCE_HALF_LIFE_S))

    def is_stale(self, now: float, ttl_s: float = DEFAULT_TTL_S) -> bool:
        return (now - self.timestamp) > ttl_s


@dataclass
class TeammateRecord:
    name: str
    last_known_position: Optional[Tuple[float, float, float]]
    provenance: Provenance


@dataclass
class TeamBelief:
    teammates: Dict[str, TeammateRecord] = field(default_factory=dict)


@dataclass
class CommunicationBelief:
    last_report_sent: Optional[str] = None  # target_id last reported, Phase 7 gives this a real destination


@dataclass
class Assumptions:
    notes: List[str] = field(default_factory=lambda: ["operating with no other agents observed"])
