"""Interactive entry point - a persistent session (Arshad's design, 2026-09-16):
connect and take off exactly once, then accept as many instructions as given,
landing only when actually told to - directly, or because a plan's own last
step is 'land'. This is what keeps ground_z correct across multiple missions.
"""

from .core.enums import ActionType
from .planners.base_planner import PlanError
from .planners.llama_planner import LlamaPlanner
from .simulator.airsim_adapter import AirSimVehicleAdapter


def main():
    planner = LlamaPlanner()
    drone = AirSimVehicleAdapter()
    drone.connect_and_takeoff()
    print("Connected and airborne. Type an instruction, or 'stop' to land and exit.")

    while True:
        instruction = input("> ").strip()
        if not instruction:
            continue
        if instruction.lower() in ("stop", "exit", "quit"):
            drone.land()
            break

        try:
            commands = planner.plan(instruction)
        except PlanError as e:
            print(f"Couldn't make a safe plan: {e}. Try rewording.")
            continue

        drone.execute(commands)

        if any(cmd.action == ActionType.LAND.value for cmd in commands):
            print("Landed as part of that plan - session over.")
            break

    print("Done.")


if __name__ == "__main__":
    main()
