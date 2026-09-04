import threading
import time

# drone_class is passed into the constructor rather than imported at the top
# of this file. main.py imports swarm.py, so if swarm.py also imported from
# main.py (or from drone.py in a way that looped back), that's a circular
# import - Python would try to load the two files into each other before
# either one finished loading. Taking the class as a parameter avoids that
# entirely: this file never needs to import a drone class at all.


class Swarm:
    """Flies several drones at the same time, one thread each."""

    def __init__(self, drone_class, missions_by_drone):
        self.drone_class = drone_class
        self.missions_by_drone = missions_by_drone
        self.threads = []
        self.errors = {}

    def _fly_one(self, drone_name, task_list):
        drone = self.drone_class(vehicle_name=drone_name)

        try:
            drone.connect_and_takeoff()
            drone.run_mission(task_list)
        except Exception as e:
            # Exceptions raised inside a thread don't propagate to the main
            # thread on their own - without this they'd fail silently.
            print(f"\n[{drone_name}] Error: {e}")
            self.errors[drone_name] = str(e)
        finally:
            drone.shutdown()

    def fly_all(self):
        print(f"\n[Swarm] Starting {len(self.missions_by_drone)} drones...")

        for drone_name, task_list in self.missions_by_drone.items():
            t = threading.Thread(
                target=self._fly_one,
                args=(drone_name, task_list),
                name=f"Thread-{drone_name}"
            )
            self.threads.append(t)
            t.start()

            # Connecting to AirSim for every drone at the exact same instant
            # makes some connections fail - a short stagger avoids it.
            time.sleep(0.25)

        for t in self.threads:
            t.join()

        finished = len(self.missions_by_drone) - len(self.errors)
        print(f"\n[Swarm] Done. {finished} finished, {len(self.errors)} failed.")

        for drone_name, error in self.errors.items():
            print(f"  {drone_name}: {error}")
