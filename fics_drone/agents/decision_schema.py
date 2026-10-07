"""The model's output contract. A model never writes code or coordinates - it
returns one small JSON object choosing from a menu the CODE built for this
exact moment. Validation is strict and TYPED, because the fallback chain has
to know why an output was rejected: a malformed answer is worth one
correction, a timeout is not (explaining the problem to a hung model does not
un-hang it, and a second wait costs another timeout on a drone that is
currently airborne).
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from .objectives import Objective

# Closed list on purpose: a free-form rationale invites treating the model's
# self-report as evidence about its own reasoning, which it is not - and a
# handful of codes can be counted across a thousand decisions where a
# thousand paragraphs cannot.
REASON_CODES = (
    "waiting_for_report",       # a teammate may still be about to broadcast a target
    "nothing_more_to_wait_for",
    "teammate_may_need_help",
    "battery_conserving",
    "my_work_is_done",
    "search_elsewhere",         # Phase 12: repositioning to look somewhere new
)

ALLOWED_KEYS = {"objective", "reason_code"}
WAYPOINT_KEYS = {"x", "y"}   # allowed ONLY alongside objective == go_to_waypoint, and required then


class RejectionKind(str, Enum):
    MALFORMED_JSON = "malformed_json"
    UNKNOWN_OBJECTIVE = "unknown_objective"   # not a real objective, OR real but not offered right now
    BAD_FIELDS = "bad_fields"
    TIMEOUT = "timeout"
    BACKEND_ERROR = "backend_error"


# Only these are worth a single "that was invalid, try again" correction.
CORRECTABLE = frozenset({RejectionKind.MALFORMED_JSON, RejectionKind.UNKNOWN_OBJECTIVE,
                         RejectionKind.BAD_FIELDS})


class DecisionRejected(Exception):
    def __init__(self, kind: RejectionKind, detail: str):
        super().__init__(f"{kind.value}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class Decision:
    objective: Objective
    reason_code: str
    waypoint: Optional[Tuple[float, float]] = None   # world (x, y); only for go_to_waypoint


def parse_decision(raw: str, legal: Sequence[Objective]) -> Decision:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise DecisionRejected(RejectionKind.MALFORMED_JSON, f"not valid JSON ({exc})")
    if not isinstance(data, dict):
        raise DecisionRejected(RejectionKind.MALFORMED_JSON, "top level must be a JSON object")

    # An unknown key is an error, not something to ignore: a model that invents a
    # field has misunderstood the contract, and silently dropping the extra key
    # executes a decision nobody intended.
    wants_waypoint = data.get("objective") == Objective.GO_TO_WAYPOINT.value
    allowed = ALLOWED_KEYS | (WAYPOINT_KEYS if wants_waypoint else set())
    extra = set(data) - allowed
    if extra:
        raise DecisionRejected(RejectionKind.BAD_FIELDS, f"unknown field(s): {sorted(extra)}")
    missing = allowed - set(data)
    if missing:
        raise DecisionRejected(RejectionKind.BAD_FIELDS, f"missing field(s): {sorted(missing)}")

    name, reason = data["objective"], data["reason_code"]
    if not isinstance(name, str) or not isinstance(reason, str):
        raise DecisionRejected(RejectionKind.BAD_FIELDS, "objective and reason_code must be strings")
    if reason not in REASON_CODES:
        raise DecisionRejected(RejectionKind.BAD_FIELDS, f"reason_code {reason!r} not in the allowed list")

    legal_names = [o.value for o in legal]
    if name not in legal_names:
        raise DecisionRejected(RejectionKind.UNKNOWN_OBJECTIVE,
                               f"{name!r} is not one of the offered options {legal_names}")

    waypoint = None
    if wants_waypoint:
        # Type check ONLY. A bool must never quietly become 1.0, and a string must not
        # reach arithmetic - but whether the point is sensible (finite, in range, inside
        # the geofence, outside a no-fly zone, away from teammates) is deliberately NOT
        # judged here. That is the SafetyGuardian's job, and it has to stay the single,
        # independent authority on it: a schema that pre-filtered unsafe coordinates would
        # hide from the guardian exactly the commands it exists to catch.
        for key in ("x", "y"):
            v = data[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DecisionRejected(RejectionKind.BAD_FIELDS, f"{key} must be a number")
        waypoint = (float(data["x"]), float(data["y"]))
    return Decision(objective=Objective(name), reason_code=reason, waypoint=waypoint)
