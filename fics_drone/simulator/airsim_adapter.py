"""The one place this codebase talks to AirSim, and the one place NED
(negative-up) gets converted to/from plain positive-up altitude.

ground_z is recorded once, at connect_and_takeoff() time, from wherever the
drone happens to be - so it must only ever be called once per real physical
takeoff. The caller (main.py) is responsible for that: connect_and_takeoff()
once per session, then execute() as many plans as needed without re-taking
off, landing only when the user actually says stop. Calling
connect_and_takeoff() again while still airborne would treat the current
altitude as ground level, and a later land() would disarm mid-air.
"""

from ..control.navigation import (
    DEFAULT_HEIGHT, LAND_FAST_ABOVE, LAND_FAST_SPEED, LAND_SETTLE_SECS,
    LAND_SLOW_SPEED, MOVE_SPEED,
)
from ..core.enums import ActionType
from ..core.interfaces import VehicleAdapter

# world-frame (x, y) unit direction per strafe action - heading is held fixed,
# the drone strafes rather than turning to face its travel direction.
_DIRECTIONS = {
    ActionType.FLY_FORWARD.value: (1.0, 0.0),
    ActionType.FLY_BACKWARD.value: (-1.0, 0.0),
    ActionType.FLY_LEFT.value: (0.0, -1.0),
    ActionType.FLY_RIGHT.value: (0.0, 1.0),
}


class AirSimVehicleAdapter(VehicleAdapter):
    def __init__(self, vehicle_name="Drone1"):
        import airsim  # lazy import, only needed when actually flying
        self._airsim = airsim
        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name)
        self.client.armDisarm(True, vehicle_name)
        self._ground_ned = None  # set on connect_and_takeoff()

    # --- the one NED boundary ---

    def _to_ned(self, height_above_ground: float) -> float:
        return self._ground_ned - height_above_ground

    def _from_ned(self, z_ned: float) -> float:
        return self._ground_ned - z_ned

    def get_height(self) -> float:
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        return self._from_ned(pos.z_val)

    def get_xy(self):
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        return pos.x_val, pos.y_val

    # --- lifecycle ---

    def connect_and_takeoff(self):
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        self._ground_ned = pos.z_val  # recorded here - see the module docstring
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.set_height(DEFAULT_HEIGHT)

    # --- non-blocking primitives, for skills.py's polling loops ---

    def get_position(self):
        x, y = self.get_xy()
        return (x, y, self.get_height())

    def get_speed(self):
        v = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.linear_velocity
        return (v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2) ** 0.5

    def start_move_to(self, x, y, z):
        self.client.moveToPositionAsync(
            x, y, self._to_ned(z), MOVE_SPEED, vehicle_name=self.vehicle_name,
        )

    def start_hover(self):
        self.client.hoverAsync(vehicle_name=self.vehicle_name)

    # --- action executors, one per ActionType ---

    def fly_to(self, x, y, z):
        self.client.moveToPositionAsync(
            x, y, self._to_ned(z), MOVE_SPEED, vehicle_name=self.vehicle_name,
        ).join()

    def set_height(self, z):
        x, y = self.get_xy()
        self.client.moveToPositionAsync(
            x, y, self._to_ned(z), MOVE_SPEED, vehicle_name=self.vehicle_name,
        ).join()

    def _strafe(self, action, distance):
        dx, dy = _DIRECTIONS[action]
        x, y = self.get_xy()
        target_z = self.client.getMultirotorState(
            self.vehicle_name).kinematics_estimated.position.z_val
        self.client.moveToPositionAsync(
            x + dx * distance, y + dy * distance, target_z, MOVE_SPEED,
            vehicle_name=self.vehicle_name,
        ).join()

    def hover(self, duration):
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        import time
        time.sleep(duration)

    def land(self):
        """Fast descent to LAND_FAST_ABOVE, settle, slow final approach, disarm.
        Most of the drop doesn't need precision; only the final few metres,
        where impact speed matters, get the slow speed."""
        x, y = self.get_xy()
        fast_target_ned = self._to_ned(LAND_FAST_ABOVE)
        self.client.moveToPositionAsync(
            x, y, fast_target_ned, LAND_FAST_SPEED, vehicle_name=self.vehicle_name,
        ).join()

        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        import time
        time.sleep(LAND_SETTLE_SECS)

        self.client.moveToPositionAsync(
            x, y, self._ground_ned, LAND_SLOW_SPEED, vehicle_name=self.vehicle_name,
        ).join()
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
        self.client.armDisarm(False, self.vehicle_name)

    # --- run a validated plan ---

    def execute(self, commands):
        """Run one validated plan. Does NOT take off first - call
        connect_and_takeoff() once per session before the first execute()."""
        for cmd in commands:
            if cmd.action == ActionType.FLY_TO.value:
                self.fly_to(cmd.params["x"], cmd.params["y"], cmd.params["z"])
            elif cmd.action == ActionType.SET_HEIGHT.value:
                self.set_height(cmd.params["z"])
            elif cmd.action in _DIRECTIONS:
                self._strafe(cmd.action, cmd.params["distance"])
            elif cmd.action == ActionType.HOVER.value:
                self.hover(cmd.params["duration"])
            elif cmd.action == ActionType.LAND.value:
                self.land()
