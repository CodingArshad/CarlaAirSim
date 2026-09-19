"""The two contracts the rest of the codebase depends on. Orchestration code
imports these, never LlamaPlanner or AirSimVehicleAdapter directly, so a
mock or a different backend can be swapped in without touching it."""

from abc import ABC, abstractmethod
from typing import List

from .models import SkillCommand


class MissionPlanner(ABC):
    @abstractmethod
    def plan(self, instruction: str) -> List[SkillCommand]:
        """instruction -> validated commands, or raises PlanError."""


class VehicleAdapter(ABC):
    @abstractmethod
    def connect_and_takeoff(self) -> None:
        """Call exactly once per real takeoff (records ground level)."""

    @abstractmethod
    def execute(self, commands: List[SkillCommand]) -> None:
        """Run one validated plan, blocking until done. Does not take off."""

    @abstractmethod
    def land(self) -> None:
        """Land and disarm."""
