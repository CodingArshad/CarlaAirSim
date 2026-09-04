import time

import airsim

DEFAULT_HEIGHT = -8.0   # cruising height; NED coordinates, so negative = up
MOVE_SPEED = 5.0        # m/s for directional moves


class Drone:
    """One simulated drone: connects, takes off, and runs a list of actions."""

    def __init__(self, vehicle_name="", cruise_height=DEFAULT_HEIGHT):
        self.vehicle_name = vehicle_name
        self.client = None
        # The height this drone holds while moving. set_height and fly_to
        # update this so later moves keep whatever height it's at.
        self.height = cruise_height
        # Ground level under this drone, recorded before takeoff so landing
        # can return to exactly this height (the drone doesn't collide with
        # the ground, so nothing else tells it when it's "arrived").
        self.ground_z = 0.0

    def connect_and_takeoff(self):
        print(f"[{self.vehicle_name}] Connecting...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self.ground_z = self.client.getMultirotorState(
            vehicle_name=self.vehicle_name
        ).kinematics_estimated.position.z_val
        print(f"[{self.vehicle_name}] Ground level is Z={self.ground_z:.2f}")

        print(f"[{self.vehicle_name}] Taking off...")
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

        self.client.moveToZAsync(
            z=self.height, velocity=3.0, vehicle_name=self.vehicle_name
        ).join()
        print(f"[{self.vehicle_name}] Ready at {self.height} m.")

    def get_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        print(f"  -> [{self.vehicle_name}] X: {position.x_val:.2f}, "
              f"Y: {position.y_val:.2f}, Z: {position.z_val:.2f}")
        return position

    # --- movement helper ---

    def _move(self, vx, vy, duration, label):
        """Fly in a direction for a set time, holding the current height."""
        print(f"  |__ [{self.vehicle_name}] {label} for {duration}s")
        self.client.moveByVelocityZAsync(
            vx=vx,
            vy=vy,
            z=self.height,
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=self.vehicle_name
        ).join()

        # Velocity commands don't brake on their own.
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    # --- actions (names match llm_planner.NEEDED_PARAMS) ---

    def fly_to(self, x, y, z):
        print(f"  |__ [{self.vehicle_name}] Flying to ({x}, {y}, {z})")
        self.client.moveToPositionAsync(
            x, y, z, 4.0, vehicle_name=self.vehicle_name
        ).join()
        self.height = z

    def fly_forward(self, duration):
        self._move(MOVE_SPEED, 0.0, duration, "Forward")

    def fly_backward(self, duration):
        self._move(-MOVE_SPEED, 0.0, duration, "Backward")

    def fly_left(self, duration):
        # y is "right" in NED, so left is negative y.
        self._move(0.0, -MOVE_SPEED, duration, "Left")

    def fly_right(self, duration):
        self._move(0.0, MOVE_SPEED, duration, "Right")

    def hover(self, duration):
        print(f"  |__ [{self.vehicle_name}] Hovering {duration}s")
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(duration)

    def set_height(self, z):
        print(f"  |__ [{self.vehicle_name}] Changing height to {z}")
        self.height = z
        self.client.moveToZAsync(
            z, 3.0, vehicle_name=self.vehicle_name
        ).join()

    def land(self):
        print(f"  |__ [{self.vehicle_name}] Landing...")

        # Stage 1: fast descent to 4m above the recorded ground level.
        self.client.moveToZAsync(
            self.ground_z - 4.0, 5.0, vehicle_name=self.vehicle_name
        ).join()

        # Settle to kill the fast descent's momentum before the slow approach.
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(0.5)

        # Stage 2: slow final descent onto the ground.
        self.client.moveToZAsync(
            self.ground_z, 1.5, vehicle_name=self.vehicle_name
        ).join()

        self.client.armDisarm(False, vehicle_name=self.vehicle_name)
        print(f"  |__ [{self.vehicle_name}] Landed.")
        self.height = 0.0

    # --- mission runner ---

    def run_mission(self, task_list):
        print(f"\n[{self.vehicle_name}] Starting mission, {len(task_list)} steps.")

        for i, step in enumerate(task_list, start=1):
            print(f"\n[{self.vehicle_name} - Step {i}/{len(task_list)}]")
            self.get_position()

            action = step["action"]
            params = step.get("params", {})

            if action == "fly_to":
                self.fly_to(params["x"], params["y"], params["z"])
            elif action == "fly_forward":
                self.fly_forward(params["duration"])
            elif action == "fly_backward":
                self.fly_backward(params["duration"])
            elif action == "fly_left":
                self.fly_left(params["duration"])
            elif action == "fly_right":
                self.fly_right(params["duration"])
            elif action == "hover":
                self.hover(params["duration"])
            elif action == "set_height":
                self.set_height(params["z"])
            elif action == "land":
                self.land()

        print(f"[{self.vehicle_name}] Mission complete.")

    def shutdown(self):
        if self.client is None:
            return

        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
            print(f"[{self.vehicle_name}] Control released.")
        except Exception as e:
            print(f"[{self.vehicle_name}] Couldn't release control: {e}")
