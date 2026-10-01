"""One MissionTask per sector, and a per-agent TaskBoard - each agent keeps
its OWN local copy, updated only by messages it actually received, never by
reaching into another agent's board. That's what makes the version-conflict
bug FICS's own doc describes a real possibility here too, not something
designed away: two agents can each believe they won the same task if their
view of the bids differs.

version is a per-task Lamport clock (next = this task's own highest version
this agent has ever seen, +1), not a private per-agent counter - two
independent per-agent counters can both read "3" while meaning nothing in
common, which is exactly the bug FICS's own build hit and fixed.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from ..core.scenario import Scenario


class TaskStatus(str, Enum):
    OPEN = "open"
    CLAIMED = "claimed"
    COMPLETE = "complete"


@dataclass
class MissionTask:
    task_id: str
    sector_id: str
    status: TaskStatus = TaskStatus.OPEN
    assignee: Optional[str] = None
    winning_bid: Optional[float] = None
    version: int = 0
    lease_expires_at: Optional[float] = None  # structural only this phase - lease
    # expiry/reclaim on a gone-silent holder is real Phase 9 territory ("keep going
    # when something fails"), not this phase's exit criterion


def tasks_from_scenario(scenario: Scenario) -> Dict[str, MissionTask]:
    """One task per sector - this is shared mission structure every agent
    starts knowing (the team was briefed 'search these N sectors'), not
    secret ground truth like a target's position. Phase 6's wall is about
    targets, not about the task list itself."""
    return {f"search_{s.id}": MissionTask(task_id=f"search_{s.id}", sector_id=s.id)
            for s in scenario.sectors}


class TaskBoard:
    """One agent's own local view. apply_bid/apply_claim are the only ways
    it changes - both driven by messages, never by peeking at anyone else's
    board or the bus directly."""

    def __init__(self, scenario: Scenario):
        self.tasks: Dict[str, MissionTask] = tasks_from_scenario(scenario)
        self._bids: Dict[str, Dict[str, float]] = {tid: {} for tid in self.tasks}  # task_id -> {bidder: bid}

    def apply_bid(self, task_id: str, bidder: str, bid: float):
        self._bids.setdefault(task_id, {})[bidder] = bid

    def bids_for(self, task_id: str) -> Dict[str, float]:
        return dict(self._bids.get(task_id, {}))

    def apply_claim(self, task_id: str, claimant: str, bid: float, version: int) -> bool:
        """Lamport-clock acceptance rule: a higher version always wins; on a
        tie, the lower bid wins; on a further tie, the lower vehicle name
        wins. Returns True if this claim was accepted (the board changed)."""
        task = self.tasks[task_id]
        if task.status == TaskStatus.OPEN:
            accept = True
        else:
            # Higher version wins outright. On a tie, LOWER bid should win - but raw
            # tuple comparison on (version, bid, name) would make a HIGHER bid win a
            # tie instead, the opposite of the actual rule. Negate bid so "greater
            # key" correctly means "better claim" on every term.
            current = (task.version, -(task.winning_bid if task.winning_bid is not None else float("inf")),
                       task.assignee or "")
            incoming = (version, -bid, claimant)
            accept = incoming > current
        if accept:
            task.status = TaskStatus.CLAIMED
            task.assignee = claimant
            task.winning_bid = bid
            task.version = version
        return accept

    def next_version(self, task_id: str) -> int:
        return self.tasks[task_id].version + 1

    def greedy_global_assignment(self, own_name: str, own_bids: Dict[str, float]) -> Dict[str, str]:
        """One-to-one, not one task independently winner-picked per task -
        the naive per-task version let a single drone be the cheapest
        bidder on MULTIPLE tasks at once, winning more than one while
        another drone won none and a sector went unclaimed by anyone.
        Since every bid is broadcast to everyone, every agent already has
        the SAME full bid matrix after the window closes - so every agent
        can run this exact greedy procedure independently (cheapest
        (task, bidder) pair first, remove both, repeat) and land on the
        same one-to-one result without any central coordinator."""
        entries = []
        for task_id in self.tasks:
            bids = self.bids_for(task_id)
            bids[own_name] = own_bids[task_id]
            for bidder, bid in bids.items():
                entries.append((bid, task_id, bidder))
        entries.sort(key=lambda e: (e[0], e[1], e[2]))  # bid, then deterministic tie-break

        assigned_task_to_bidder: Dict[str, str] = {}
        taken_bidders = set()
        for bid, task_id, bidder in entries:
            if task_id in assigned_task_to_bidder or bidder in taken_bidders:
                continue
            assigned_task_to_bidder[task_id] = bidder
            taken_bidders.add(bidder)
        return assigned_task_to_bidder
