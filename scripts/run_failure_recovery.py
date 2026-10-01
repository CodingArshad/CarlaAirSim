"""One drone is switched off mid-mission - the team detects it and finishes
the work without it.

    python scripts/run_failure_recovery.py                    # kill Drone2 at t=15s (mock)
    python scripts/run_failure_recovery.py --kill Drone3 --at 40
    python scripts/run_failure_recovery.py --airsim
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fics_drone.core.scenario import load_scenario
from fics_drone.coordination.message_bus import MessageBus
from fics_drone.experiments.team_runner import run_team_with_faults
from fics_drone.simulator.kinematic_mock_adapter import KinematicMockVehicleAdapter

DEFAULT_SCENARIO = os.path.join(os.path.dirname(__file__), "..", "configs", "missions",
                                 "search_relay_001.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--airsim", action="store_true")
    parser.add_argument("--kill", default="Drone2")
    parser.add_argument("--at", type=float, default=15.0, help="elapsed seconds into the mission to kill --kill")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    if args.airsim:
        from fics_drone.simulator.airsim_adapter import AirSimVehicleAdapter
        adapters = {d.name: AirSimVehicleAdapter(d.name) for d in scenario.drones}
    else:
        adapters = {d.name: KinematicMockVehicleAdapter(d.name, speed_mps=15.0) for d in scenario.drones}

    bus = MessageBus()
    print(f"!! {args.kill} will be switched off at t={args.at}s (teammates not informed)")
    reports, agents, bus, assignment = run_team_with_faults(
        scenario, adapters, bus=bus, kill_name=args.kill, kill_at_s=args.at)

    print("\n--- allocation result ---")
    for name, sector_id in assignment.items():
        print(f"{name}: {sector_id}")

    print("\n--- results ---")
    for name, report in reports.items():
        killed = " (KILLED)" if name == args.kill and "killed" in (report.trace[-1] if report.trace else "") else ""
        print(f"{name}{killed}: target_found={report.target_found} battery_frac={report.battery_frac_at_end:.2f}")

    sectors_done = {agents[n].belief.mission.sector_id for n, r in reports.items()
                    if r.trace and "killed" not in r.trace[-1]}
    print(f"\nsectors completed: {len(sectors_done)}/{len(scenario.sectors)}")


if __name__ == "__main__":
    main()
