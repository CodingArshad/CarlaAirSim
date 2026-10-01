"""Central simulation infrastructure - but agents never hold a MessageBus
directly. Each agent holds an AgentLink exposing exactly two methods,
send() and receive_available(). An agent cannot enumerate peers, read
another agent's inbox, see what is in flight, or inspect what was dropped -
that visibility exists only for the test harness and the log, never for the
agents being tested.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .protocols import TTL_BY_TYPE, Message, MessageType


@dataclass
class LogEntry:
    message_id: int
    type: str
    sender: str
    recipient: str
    created_t: float
    delivered: bool
    reason: str  # "delivered" | "blackout" | "expired"


class MessageBus:
    """drop_all=True is the blackout test: every send is logged as dropped
    and no inbox ever receives anything - the strongest possible check that
    team belief updates only via delivered messages, never a back channel."""

    def __init__(self, drop_all: bool = False):
        self.drop_all = drop_all
        self._inboxes: Dict[str, List[Message]] = {}
        self._lock = threading.Lock()
        self._next_id = 0
        self.log: List[LogEntry] = []
        self.start_time = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self.start_time

    def register(self, name: str):
        with self._lock:
            self._inboxes.setdefault(name, [])

    def send(self, type_: MessageType, sender: str, payload: dict,
              recipients: Optional[List[str]] = None, confidence: float = 1.0) -> Message:
        with self._lock:
            self._next_id += 1
            msg = Message(id=self._next_id, type=type_, sender=sender, recipients=recipients,
                          created_t=self.now(), ttl_s=TTL_BY_TYPE[type_], payload=payload,
                          confidence=confidence)
            targets = recipients if recipients is not None else [n for n in self._inboxes if n != sender]
            for name in targets:
                if self.drop_all:
                    self.log.append(LogEntry(msg.id, type_.value, sender, name, msg.created_t,
                                              delivered=False, reason="blackout"))
                    continue
                self._inboxes.setdefault(name, []).append(msg)
                self.log.append(LogEntry(msg.id, type_.value, sender, name, msg.created_t,
                                          delivered=True, reason="delivered"))
            return msg

    def receive_available(self, name: str) -> List[Message]:
        """Pops every not-yet-expired message waiting for `name`. An expired
        one is logged as dropped (separately from a blackout drop, per
        FICS's own lesson: a short TTL looks identical to link loss unless
        the two are tracked apart) and never handed to the agent."""
        now = self.now()
        with self._lock:
            waiting = self._inboxes.get(name, [])
            self._inboxes[name] = []
        delivered = []
        for msg in waiting:
            if now > msg.expires_at():
                self.log.append(LogEntry(msg.id, msg.type.value, msg.sender, name, msg.created_t,
                                          delivered=False, reason="expired"))
            else:
                delivered.append(msg)
        return delivered


class AgentLink:
    """The only thing an agent ever holds - two methods, nothing else."""

    def __init__(self, bus: MessageBus, name: str):
        self._bus = bus
        self.name = name
        bus.register(name)

    def send(self, type_: MessageType, payload: dict,
              recipients: Optional[List[str]] = None, confidence: float = 1.0):
        self._bus.send(type_, self.name, payload, recipients, confidence)

    def receive_available(self) -> List[Message]:
        return self._bus.receive_available(self.name)
