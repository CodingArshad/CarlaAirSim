import math
import time

import airsim

DEFAULT_HEIGHT = 8.0    # cruising altitude, positive = up
MOVE_SPEED = 5.0        # m/s for directional moves


def _to_ned(altitude):
    """Positive-up altitude -> AirSim's NED z (down-positive)."""
    return -altitude


def _from_ned(z):
    """AirSim's NED z (down-positive) -> positive-up altitude."""
    return -z

# --- object avoidance ---
# Must match the sensor name in main.py's settings.json Lidar declaration.
LIDAR_NAME = "Lidar1"
# How often the reactive loop re-checks the lidar and re-issues a move, in
# seconds. At MOVE_SPEED that's ~1.25m of blind travel per tick - tight
# enough to react well within the lidar's 20m range, without spamming AirSim
# with commands every few centimeters.
TICK_DURATION = 0.25
# Distance (m) at which something ahead counts as "in the way." Needs
# headroom over one tick's travel distance plus one more tick to actually
# start diverting, so it can't be smaller than ~2x TICK_DURATION*MOVE_SPEED.
SAFETY_DISTANCE = 6.0
# Half-angle (degrees) of the forward cone checked for obstacles - a point
# dead ahead matters, a point 80 degrees off to the side doesn't.
AVOID_CONE_DEGREES = 30.0
# How far sideways/up to sidestep on each diversion attempt.
AVOID_STEP = 3.0
# "Close enough" to a target to stop navigating toward it.
ARRIVAL_RADIUS = 0.5
# How many consecutive blocked attempts (no successful clear-path tick in
# between) to try sidestepping sideways before trying to climb over instead.
CONSECUTIVE_BEFORE_VERTICAL = 4
# Give up once blocked this many attempts in a row - not a lifetime cap, so
# a long mission that dodges many separate obstacles never trips this as
# long as it keeps making progress between them. Only a genuinely stuck
# drone (no clear tick resetting the streak) hits this.
MAX_CONSECUTIVE_DIVERSIONS = 8


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
            z=_to_ned(self.height), velocity=3.0, vehicle_name=self.vehicle_name
        ).join()
        print(f"[{self.vehicle_name}] Ready at {self.height} m.")

    def get_position(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        print(f"  -> [{self.vehicle_name}] X: {position.x_val:.2f}, "
              f"Y: {position.y_val:.2f}, Altitude: {_from_ned(position.z_val):.2f}")
        return position

    # --- object avoidance ---

    def _min_obstacle_distance(self, direction):
        """Closest lidar point within a forward cone around `direction`.

        `direction` is an (dx, dy) unit vector in the world XY plane. Yaw is
        always held at 0 (see the yaw_mode below), so lidar points - which
        come back in the sensor's local frame - already line up with world
        directions; no rotation needed to compare them.
        """
        lidar_data = self.client.getLidarData(
            lidar_name=LIDAR_NAME, vehicle_name=self.vehicle_name
        )
        points = lidar_data.point_cloud
        if len(points) < 3:
            return float("inf")

        dx, dy = direction
        cos_half_cone = math.cos(math.radians(AVOID_CONE_DEGREES / 2))
        nearest = float("inf")

        for i in range(0, len(points), 3):
            px, py, pz = points[i], points[i + 1], points[i + 2]
            distance = math.sqrt(px * px + py * py + pz * pz)
            if distance == 0:
                continue

            # Cosine of the angle between this point and our direction of
            # travel - only points inside the forward cone count.
            alignment = (px * dx + py * dy) / distance
            if alignment >= cos_half_cone and distance < nearest:
                nearest = distance

        return nearest

    def _navigate_to(self, x, y, z):
        """Fly toward (x, y, z), diverting around anything the lidar sees.

        Replaces one long blocking moveToPositionAsync/moveByVelocityZAsync
        call with a loop of short TICK_DURATION moves - .join() on a single
        long call blocks this thread for its entire duration, so a long
        blocking call can't react to anything that appears mid-flight.

        Tries sidestepping around an obstacle first; if that keeps failing
        with no clear tick in between (boxed in, not just one obstacle after
        another), it tries climbing over instead. Only gives up once blocked
        MAX_CONSECUTIVE_DIVERSIONS times in a row - a mission that dodges
        many separate obstacles never trips this, since every clear tick
        resets the streak.

        z is altitude (positive = up), converted to AirSim's NED convention
        once here - the one real boundary. Every other method in this class
        works in plain altitude terms and never touches NED directly.
        """
        target_z = _to_ned(z)
        consecutive_diversions = 0

        while True:
            position = self.get_position()
            remaining_x = x - position.x_val
            remaining_y = y - position.y_val
            remaining_z = target_z - position.z_val
            horizontal_distance = math.hypot(remaining_x, remaining_y)

            if horizontal_distance < ARRIVAL_RADIUS and abs(remaining_z) < ARRIVAL_RADIUS:
                # Velocity commands don't brake on their own - without this
                # the drone coasts on whatever velocity it had the instant
                # it crossed the arrival threshold, instead of stopping.
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
                self.height = z
                return

            if horizontal_distance > 0:
                heading = (remaining_x / horizontal_distance, remaining_y / horizontal_distance)
            else:
                heading = (1.0, 0.0)

            if self._min_obstacle_distance(heading) < SAFETY_DISTANCE:
                consecutive_diversions += 1
                if consecutive_diversions > MAX_CONSECUTIVE_DIVERSIONS:
                    raise RuntimeError(
                        f"[{self.vehicle_name}] Stuck near "
                        f"({position.x_val:.1f}, {position.y_val:.1f}, "
                        f"{position.z_val:.1f}) - still blocked after "
                        f"{MAX_CONSECUTIVE_DIVERSIONS} attempts trying to "
                        f"reach ({x}, {y}, {z})"
                    )

                if consecutive_diversions <= CONSECUTIVE_BEFORE_VERTICAL:
                    print(f"  |__ [{self.vehicle_name}] Obstacle ahead, diverting sideways")
                    # Sidestep perpendicular to the blocked heading, at the
                    # drone's actual current altitude - not self.height,
                    # which may be stale once climbing (below) is involved.
                    divert_x = position.x_val - heading[1] * AVOID_STEP
                    divert_y = position.y_val + heading[0] * AVOID_STEP
                    self.client.moveToPositionAsync(
                        divert_x, divert_y, position.z_val, MOVE_SPEED,
                        vehicle_name=self.vehicle_name
                    ).join()
                else:
                    print(f"  |__ [{self.vehicle_name}] Still blocked, climbing")
                    self.client.moveToZAsync(
                        position.z_val - AVOID_STEP, MOVE_SPEED,
                        vehicle_name=self.vehicle_name
                    ).join()
                continue

            consecutive_diversions = 0

            # Every velocity-based vertical mechanism tried this week
            # misbehaved somehow: moveByVelocityZAsync's `z` parameter
            # (unreliable both with near-zero horizontal velocity and in
            # general - dipped/wobbled enough to fly into a building), then
            # moveByVelocityAsync's `vz` (climbed straight past every real
            # target, never reversing, well past the highest landmark).
            # moveToZAsync and moveToPositionAsync - both position-target
            # commands, not velocity commands - are the two mechanisms that
            # have behaved correctly every single time they were used (the
            # sideways-diversion branch below has used moveToPositionAsync
            # since day one with no issues).
            #
            # But a step sized to fit inside one TICK_DURATION (tried a rate
            # cap of MOVE_SPEED/3.0 m/s * TICK_DURATION) falls inside
            # AirSim's own "close enough, already there" tolerance for these
            # calls - it reports instant success without moving at all, so
            # the drone gets stuck repeating the same position forever.
            # AVOID_STEP (3.0m) is the smallest distance already proven to
            # reliably move (used for diversions since day one, always with
            # a full .join()). Tried firing an AVOID_STEP-sized move without
            # waiting for it (sleep TICK_DURATION, let a fresh command
            # supersede it) to keep a fast recheck cadence - that also froze
            # in place, most likely because re-issuing every 0.25s cancels
            # each attempt before the flight controller has time to build
            # any real motion. Matching the diversion branches exactly
            # instead: full .join(), same size step. Costs a coarser
            # recheck cadence (~AVOID_STEP/MOVE_SPEED seconds instead of
            # TICK_DURATION) but it's the one pattern proven reliable every
            # time this week.
            horizontal_step = min(AVOID_STEP, horizontal_distance) if horizontal_distance > 0 else 0.0
            step_x = position.x_val + heading[0] * horizontal_step
            step_y = position.y_val + heading[1] * horizontal_step
            step_z = position.z_val + max(-AVOID_STEP, min(AVOID_STEP, remaining_z))

            self.client.moveToPositionAsync(
                step_x, step_y, step_z, MOVE_SPEED, vehicle_name=self.vehicle_name
            ).join()

    def _navigate_to_via_altitude(self, x, y, z):
        """Adjust altitude in place first, then close horizontal distance.

        Prevents ramming into the side of a tall obstacle while still
        climbing toward a target that sits on top of one - e.g. a landmark
        surveyed on a rooftop, where a straight-line blend of x/y/z reaches
        the building's footprint long before it reaches roof height. Cheap
        no-op for plain duration-based moves, which keep the same height
        throughout (the first _navigate_to call returns immediately).
        """
        position = self.get_position()
        self._navigate_to(position.x_val, position.y_val, z)
        self._navigate_to(x, y, z)

    # --- actions (names match llm_planner.NEEDED_PARAMS) ---

    def fly_to(self, x, y, z):
        print(f"  |__ [{self.vehicle_name}] Flying to ({x}, {y}, {z})")
        self._navigate_to_via_altitude(x, y, z)

    def fly_forward(self, distance):
        print(f"  |__ [{self.vehicle_name}] Forward {distance}m")
        position = self.get_position()
        self._navigate_to(position.x_val + distance, position.y_val, self.height)

    def fly_backward(self, distance):
        print(f"  |__ [{self.vehicle_name}] Backward {distance}m")
        position = self.get_position()
        self._navigate_to(position.x_val - distance, position.y_val, self.height)

    def fly_left(self, distance):
        # y is "right" in NED, so left is negative y.
        print(f"  |__ [{self.vehicle_name}] Left {distance}m")
        position = self.get_position()
        self._navigate_to(position.x_val, position.y_val - distance, self.height)

    def fly_right(self, distance):
        print(f"  |__ [{self.vehicle_name}] Right {distance}m")
        position = self.get_position()
        self._navigate_to(position.x_val, position.y_val + distance, self.height)

    def hover(self, duration):
        print(f"  |__ [{self.vehicle_name}] Hovering {duration}s")
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        time.sleep(duration)

    def set_height(self, z):
        print(f"  |__ [{self.vehicle_name}] Changing height to {z}")
        self.height = z
        self.client.moveToZAsync(
            _to_ned(z), 3.0, vehicle_name=self.vehicle_name
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
                self.fly_forward(params["distance"])
            elif action == "fly_backward":
                self.fly_backward(params["distance"])
            elif action == "fly_left":
                self.fly_left(params["distance"])
            elif action == "fly_right":
                self.fly_right(params["distance"])
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
