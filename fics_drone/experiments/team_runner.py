"""Runs all four agents together, each on its own thread and its own
AgentLink into one shared MessageBus - this is FICS's run_team_threaded(),
the real-time runner selected for --airsim. A fully deterministic
simulated-clock interleaver (FICS's other runner) isn't built here: this
repo's skills layer is inherently real-time (actual time.sleep polling
loops), so a sim-clock version would need a parallel non-real-time skill
execution path - a real rearchitecture, not something to invent cheaply
under this phase's scope. Threaded is the one mechanism this build has, and
it is the one FICS itself uses for live flight anyway.
"""

import threading
from typing import Dict, List, Optional

from ..agents.decision_log import DecisionLogger
from ..agents.persistent_agent import AgentReport, PersistentAgent
from ..coordination.message_bus import AgentLink, MessageBus
from ..coordination.task_allocator import TaskAllocator
from ..core.scenario import Scenario


def run_team_threaded(scenario: Scenario, adapters: Dict[str, object], bus: MessageBus = None,
                       loggers: Dict[str, DecisionLogger] = None) -> Dict[str, AgentReport]:
    """Static assignment (scenario's own spec.sector), Phase 7 behavior -
    unchanged, still used where a fixed assignment is what's wanted."""
    bus = bus or MessageBus()
    reports: Dict[str, AgentReport] = {}
    agents: Dict[str, PersistentAgent] = {}

    for spec in scenario.drones:
        adapter = adapters[spec.name]
        link = AgentLink(bus, spec.name)
        logger = loggers.get(spec.name) if loggers else None
        agents[spec.name] = PersistentAgent(adapter, scenario, spec.sector, spec.spawn_offset,
                                             spec.battery_s, logger=logger, drone_name=spec.name, link=link)

    def fly_one(name):
        reports[name] = agents[name].run()

    threads = [threading.Thread(target=fly_one, args=(name,)) for name in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return reports, agents, bus


def allocate_sectors(scenario: Scenario, links: Dict[str, AgentLink]) -> Dict[str, Optional[str]]:
    """Phase 8: every drone bids on every sector, independently, in parallel
    (real threads - the contract-net protocol only means something if agents
    are genuinely deciding at the same time, not taking turns). Returns the
    sector_id each drone ends up responsible for."""
    results: Dict[str, Optional[str]] = {}

    def run_one(spec):
        allocator = TaskAllocator(links[spec.name], scenario, spec.name)
        # Pre-flight: no real position yet, so each drone bids from its own
        # spawn point at full battery - the only honest "current state" available
        # before anyone has taken off.
        results[spec.name] = allocator.allocate(world_position=spec.spawn_offset, battery_frac=1.0)

    threads = [threading.Thread(target=run_one, args=(spec,)) for spec in scenario.drones]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def run_team_with_allocation(scenario: Scenario, adapters: Dict[str, object], bus: MessageBus = None,
                              loggers: Dict[str, DecisionLogger] = None):
    """Phase 8 exit criterion: the team divides sectors itself, no central
    assignment - then flies using whatever each drone actually won."""
    bus = bus or MessageBus()
    links = {spec.name: AgentLink(bus, spec.name) for spec in scenario.drones}

    assignment = allocate_sectors(scenario, links)

    reports: Dict[str, AgentReport] = {}
    agents: Dict[str, PersistentAgent] = {}
    for spec in scenario.drones:
        sector_id = assignment.get(spec.name)
        if sector_id is None:
            continue  # didn't win anything - nothing to fly
        logger = loggers.get(spec.name) if loggers else None
        agents[spec.name] = PersistentAgent(adapters[spec.name], scenario, sector_id, spec.spawn_offset,
                                             spec.battery_s, logger=logger, drone_name=spec.name,
                                             link=links[spec.name])

    def fly_one(name):
        reports[name] = agents[name].run()

    threads = [threading.Thread(target=fly_one, args=(name,)) for name in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return reports, agents, bus, assignment
