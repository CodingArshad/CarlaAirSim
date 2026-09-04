import argparse
import json
import os

import airsim
import ollama

from drone import Drone, DEFAULT_HEIGHT
from llm_planner import LLMPlanner, DEFAULT_MODEL
from swarm import Swarm

SPACING = 4.0  # metres between drones when they spawn


def update_airsim_settings(num_agents):
    """Writes settings.json listing every drone.

    AirSim only reads this file at launch, so changing the number of drones
    means restarting the simulator afterward.
    """
    settings_dir = os.path.join(os.path.expanduser("~"), "Documents", "AirSim")
    os.makedirs(settings_dir, exist_ok=True)
    settings_path = os.path.join(settings_dir, "settings.json")

    camera_settings = {
        "0": {
            "CaptureSettings": [
                {"ImageType": 0, "Width": 1280, "Height": 960}
            ],
            "X": 0.5, "Y": 0.0, "Z": 0.1,
            "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        },
        "front_center": {
            "CaptureSettings": [
                {"ImageType": 0, "Width": 1280, "Height": 960, "FOV_Degrees": 90}
            ],
            "X": 0.2, "Y": 0.0, "Z": -0.1,
            "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
    }

    vehicles = {}
    # Starts at 1, not 2: every drone including Drone1 must be listed here,
    # or this overwrites settings.json with an empty vehicle list and AirSim
    # falls back to a default-named vehicle nothing else in this code expects.
    for i in range(1, num_agents + 1):
        vehicles[f"Drone{i}"] = {
            "VehicleType": "SimpleFlight",
            "AutoCreate": True,
            "X": i * SPACING,
            "Y": 0.0,
            "Z": 0.0,
            "Cameras": camera_settings
        }

    settings_data = {
        "SeeDocsAt": "https://github.com/microsoft/AirSim/blob/main/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "Vehicles": vehicles
    }

    with open(settings_path, "w") as f:
        json.dump(settings_data, f, indent=4)

    print(f"[Config] Wrote {num_agents} drone(s) to {settings_path}")


def spawn_missing_drones(num_agents):
    """Adds any Drone1..DroneN not already in the running simulator.

    Normally unnecessary since update_airsim_settings + a restart already
    creates them, but this lets a run recover without a restart if the
    settings file didn't have the vehicles it needed.
    """
    if num_agents < 1:
        return

    client = airsim.MultirotorClient()
    client.confirmConnection()

    existing = client.listVehicles()

    for i in range(1, num_agents + 1):
        vehicle_name = f"Drone{i}"

        if vehicle_name in existing:
            continue

        print(f"[Swarm] Adding {vehicle_name}...")
        client.simAddVehicle(
            vehicle_name=vehicle_name,
            vehicle_type="SimpleFlight",
            pose=airsim.Pose(
                airsim.Vector3r(i * SPACING, 0.0, 0.0),
                airsim.Quaternionr(0.0, 0.0, 0.0, 1.0)
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Natural-language drone control.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model to use for planning (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    try:
        ollama.show(args.model)
    except Exception:
        print(f"Error: couldn't reach the model '{args.model}'.")
        print(f"Make sure Ollama is running and you've run: ollama pull {args.model}")
        return

    try:
        num_agents = int(input("How many drones do you want? "))
    except ValueError:
        print("That needs to be a whole number.")
        return

    if num_agents < 1:
        print("Need at least 1 drone.")
        return

    update_airsim_settings(num_agents)

    print("\nIf you changed the number of drones, restart Town10HD now.")
    input("Press Enter once the map has loaded and you can see the drone...")

    spawn_missing_drones(num_agents)

    planner = LLMPlanner(model=args.model, default_height=DEFAULT_HEIGHT)
    missions = {}

    for i in range(1, num_agents + 1):
        drone_name = f"Drone{i}"

        while True:
            user_prompt = input(f"What should {drone_name} do? ").strip()

            if not user_prompt:
                print("  Can't be empty.")
                continue

            print(f"[{args.model}] Planning for {drone_name}...")

            try:
                task_list = planner.plan(user_prompt)
            except Exception as e:
                print(f"  Couldn't make a plan: {e}")
                print("  Try wording it differently.")
                continue

            print(f"  {len(task_list)} step(s):")
            for step in task_list:
                print(f"    {step['action']} {step.get('params', {})}")

            missions[drone_name] = task_list
            break

    if num_agents == 1:
        drone = Drone(vehicle_name="Drone1", cruise_height=DEFAULT_HEIGHT)
        try:
            drone.connect_and_takeoff()
            drone.run_mission(missions["Drone1"])
        finally:
            drone.shutdown()
    else:
        swarm = Swarm(Drone, missions)
        swarm.fly_all()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
