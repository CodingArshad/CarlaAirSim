"""Flight constants, positive-up altitude throughout (flipped to AirSim's
native NED only at the one boundary inside the adapter that talks to AirSim)."""

DEFAULT_HEIGHT = 8.0   # default cruise height, metres above the recorded ground
MOVE_SPEED = 5.0       # m/s for directional moves and set_height

# Landing profile: fast descent for most of the drop (precision doesn't matter
# yet), then a slow final approach only once close to the ground (this is
# where an over-fast touchdown could damage the drone or overshoot the mesh).
LAND_FAST_ABOVE = 4.0   # descend fast until this many metres above ground_z
LAND_FAST_SPEED = 5.0
LAND_SLOW_SPEED = 1.5
LAND_SETTLE_SECS = 0.5  # brief hover to kill momentum before the slow approach
