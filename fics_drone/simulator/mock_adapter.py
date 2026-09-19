"""Simulator-free VehicleAdapter: records what it was asked to do, flies nothing."""

import time

from ..core.interfaces import VehicleAdapter


class MockVehicleAdapter(VehicleAdapter):
    def __init__(self, vehicle_name="MockDrone", step_delay=0.0):
        self.vehicle_name = vehicle_name
        self.step_delay = step_delay  # lets tests prove drones overlap in time
        self.log = []                 # ("takeoff"|"execute"|"land", detail)
        self.start_time = None
        self.end_time = None

    def connect_and_takeoff(self):
        self.log.append(("takeoff", None))

    def execute(self, commands):
        self.start_time = time.monotonic()
        self.log.append(("execute", [c.action for c in commands]))
        time.sleep(self.step_delay)
        self.end_time = time.monotonic()

    def land(self):
        self.log.append(("land", None))
