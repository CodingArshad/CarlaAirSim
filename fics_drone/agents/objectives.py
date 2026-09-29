"""What the agent can decide to do next, and what makes it decide again.
Kept separate from the loop itself (persistent_agent.py) so the vocabulary
of "things that can happen" is visible in one place."""

from enum import Enum


class Objective(str, Enum):
    TAKE_OFF = "take_off"
    GO_TO_SECTOR = "go_to_sector"
    SEARCH_SECTOR = "search_sector"
    REPORT = "report"
    RETURN_HOME = "return_home"
    LAND = "land"
    DONE = "done"


class ReplanEvent(str, Enum):
    """What the agent re-decides in response to - never a bare timer tick."""
    TASK_ASSIGNED = "task_assigned"
    SKILL_SUCCEEDED = "skill_succeeded"
    SKILL_FAILED = "skill_failed"
    TARGET_DETECTED = "target_detected"
    REPORT_SENT = "report_sent"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    GUARDIAN_BLOCKED = "guardian_blocked"
