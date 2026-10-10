"""GMB look-ahead: reject a leg that a dynamic zone will cover by the time the aircraft arrives.
Mock-only. Off by default (lookahead_s = 0), so earlier results are unchanged."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fics_drone.agents.belief import Belief
from fics_drone.agents.belief_schema import MissionBelief, SelfState
from fics_drone.agents.safety_guardian import Command, GuardianOutcome, SafetyGuardian, SafetyLimits
from fics_drone.core.scenario import NoFlyZone


def _belief(position=(0.0, 0.0, 8.0), t=0.0):
    return Belief(self_state=SelfState(position=position, elapsed_s=t, battery_s=1000.0),
                  mission=MissionBelief(sector_id="A", search_queue=[]))


def _guardian(lookahead, *zones):
    return SafetyGuardian(limits=SafetyLimits(restricted_zones=list(zones), geofence_half_extent_m=300.0,
                                              lookahead_s=lookahead))


def _fly(x, y, speed=5.0):
    return Command(kind="fly", target=(x, y, 8.0), speed_mps=speed, timeout_s=60.0)


# starts at x 40-60, drifts toward -x at 2 m/s from t=0: covers x=30 around t=15
def _drifting():
    return NoFlyZone("D", 40.0, 60.0, -10.0, 10.0, vx=-2.0)


class TestLookahead(unittest.TestCase):
    def test_off_by_default_approves_a_leg_the_zone_will_reach(self):
        g = _guardian(0.0, _drifting())
        self.assertEqual(g.evaluate(_fly(30, 0), _belief()).outcome, GuardianOutcome.APPROVE)

    def test_on_rejects_a_destination_the_zone_will_cover_on_arrival(self):
        g = _guardian(60.0, _drifting())   # 30 m at 5 m/s = 6 s; zone edge at 40-12 = 28 -> covers x=30
        ev = g.evaluate(_fly(30, 0), _belief())
        self.assertEqual(ev.outcome, GuardianOutcome.REJECT_AND_REPLAN)
        self.assertIn("predicted_zone_conflict", ev.failed_checks)

    def test_on_rejects_a_path_that_crosses_the_zone_mid_leg(self):
        g = _guardian(60.0, NoFlyZone("M", 20.0, 30.0, -10.0, 10.0, vx=0.0, active_from_s=1.0))
        ev = g.evaluate(_fly(80, 0, speed=5.0), _belief())   # destination clear, path crosses x 20-30
        self.assertIn("predicted_zone_conflict", ev.failed_checks)

    def test_on_approves_a_clear_leg(self):
        g = _guardian(60.0, _drifting())
        self.assertEqual(g.evaluate(_fly(-30, 0), _belief()).outcome, GuardianOutcome.APPROVE)

    def test_horizon_limits_how_far_ahead_it_looks(self):
        g = _guardian(2.0, NoFlyZone("M", 20.0, 30.0, -10.0, 10.0, active_from_s=1.0))
        self.assertEqual(g.evaluate(_fly(80, 0), _belief()).outcome, GuardianOutcome.APPROVE)  # 2 s = 10 m

    def test_uses_world_position_when_given(self):
        g = _guardian(60.0, NoFlyZone("M", 20.0, 30.0, -10.0, 10.0, active_from_s=1.0))
        ev = g.evaluate(_fly(80, 0), _belief(), position_world=(100.0, 100.0, 8.0))
        self.assertEqual(ev.outcome, GuardianOutcome.APPROVE)   # leg starts far from the zone

    def test_static_zones_are_left_to_the_existing_check(self):
        g = _guardian(60.0, NoFlyZone("S", 20.0, 30.0, -10.0, 10.0))
        ev = g.evaluate(_fly(80, 0), _belief())
        self.assertNotIn("predicted_zone_conflict", ev.failed_checks)

    def test_bad_speed_or_nan_does_not_crash(self):
        g = _guardian(60.0, _drifting())
        for speed in (0.0, -5.0, float("nan")):
            g.evaluate(_fly(30, 0, speed), _belief())
        g.evaluate(_fly(float("nan"), 0), _belief())

    def test_return_home_is_never_blocked_by_the_lookahead(self):
        g = _guardian(60.0, _drifting())
        home = Command(kind="fly", target=(30.0, 0.0, 8.0), speed_mps=5.0, timeout_s=60.0, purpose="home")
        self.assertEqual(g.evaluate(home, _belief()).outcome, GuardianOutcome.APPROVE)

    def test_preview_matches_evaluate(self):
        g = _guardian(60.0, _drifting())
        self.assertEqual([c.name for c in g.preview(_fly(30, 0), _belief())], ["predicted_zone_conflict"])


if __name__ == "__main__":
    unittest.main()
