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
from typing import Dict, List

from ..agents.decision_log import DecisionLogger
from ..agents.persistent_agent import AgentReport, PersistentAgent
from ..coordination.message_bus import AgentLink, MessageBus
from ..core.scenario import Scenario


def run_team_threaded(scenario: Scenario, adapters: Dict[str, object], bus: MessageBus = None,
                       loggers: Dict[str, DecisionLogger] = None) -> Dict[str, AgentReport]:
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
