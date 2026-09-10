# CarlaAirSim

Natural-language control of a simulated drone in the CarlaAir simulator. Type a plain-English
instruction, a local LLM turns it into a short list of flight actions, the actions get validated,
then the drone actually flies the plan.

Built as a from-scratch replication of `niranjanpillai2009-altr/AirSimRepo` for the FICS Lab
agentic-drone project (Dr. Akbas), understanding-first rather than copied: same core pipeline and
mechanics, own file structure, own naming, own system prompt wording, plus a few deliberate
fixes/improvements over the original (see below) and three follow-on research directions built
on top (object avoidance, landmark navigation, positive-up coordinates - see below).

## How it works

The LLM is restricted to 9 fixed actions (`fly_to`, `fly_to_landmark`, `fly_forward`,
`fly_backward`, `fly_left`, `fly_right`, `hover`, `set_height`, `land`), each with fixed
parameters, enforced with a JSON schema. Keeping the vocabulary small and structured means the
output is predictable and can be checked before the drone leaves the ground.

Beyond the base replication, three follow-on directions are built in:

- **Reactive object avoidance** - a lidar sensor and a reactive navigation loop mean the drone
  senses and diverts around obstacles mid-flight instead of blindly executing a pre-planned path.
  See "Object avoidance" below.
- **Landmark-based navigation** - `fly_to_landmark` lets an instruction reference a named place
  ("fly to the beach") instead of raw coordinates. See "Landmark navigation" below.
- **Positive-up altitude** - every number outside `drone.py`'s one internal boundary (the
  landmark table, the system prompt, mission logs) uses plain positive-up altitude, not AirSim's
  native NED convention. See "Coordinates" below.

## Files

| File | What it does |
|---|---|
| `main.py` | Entry point - CLI, drone-count/mission prompts, sim setup, dispatches to single-drone or swarm flight |
| `drone.py` | `Drone` class - connect/takeoff, the flight actions, the reactive object-avoidance loop, staged landing |
| `llm_planner.py` | `LLMPlanner` class - talks to Ollama, enforces the JSON schema, resolves landmarks, validates the plan before anything flies |
| `swarm.py` | `Swarm` class - runs multiple drones concurrently, one thread each |

## Running it

Requires a running CarlaAir simulator (see `niranjanpillai2009-altr/AirSimRepo`'s `SETUP_GUIDE.md`
for the full environment setup - simulator download, conda env, `Drone1` rename) and
[Ollama](https://ollama.com) with a local model pulled (default `llama3.1:8b`):

```
python main.py
python main.py --model mistral-nemo   # or any other Ollama model
```

## Object avoidance

`drone.py` mounts a lidar sensor (declared in the generated `settings.json`, full horizontal
sweep, narrow vertical band) and every movement action funnels through one reactive loop,
`_navigate_to()`, instead of a single long blocking flight command. Each short tick re-checks the
lidar for anything within a forward cone before committing to the next step, so the drone reacts
to obstacles mid-flight rather than only between pre-planned steps.

When something's in the way: sidestep perpendicular to the blocked heading first; if that keeps
failing with no clear tick in between (boxed in, not just one obstacle after another), climb over
instead. Only gives up once genuinely stuck (blocked several times in a row with zero progress) -
a mission that successfully dodges many separate obstacles across a long flight never trips this,
since every clear tick resets the streak.

All position/altitude movement is done with position-target AirSim calls (`moveToPositionAsync`,
`moveToZAsync`), not velocity commands - every velocity-based approach tried during development
(`moveByVelocityZAsync`'s altitude-hold, `moveByVelocityAsync`'s `vz`) misbehaved in some way
(drifted the wrong direction, wobbled enough to fly into a building, or climbed straight past
every real target without ever reversing). Position-target commands have behaved reliably every
time they were used instead.

## Landmark navigation

`fly_to_landmark` (param: `name`) lets an instruction reference a named place instead of raw
coordinates - "fly to the beach" instead of `fly_to x=11 y=-70 z=-18`. CARLA exposes no queryable
building names or types anywhere in its API (actors are only vehicles/pedestrians/traffic signs;
static building meshes carry no metadata), so there's no way to resolve a landmark name
automatically. Instead, `llm_planner.py`'s `LANDMARKS` table is a small, hand-surveyed lookup -
each entry recorded by flying to the spot and reading `Drone.get_position()` - explicitly scoped
to landmarks actually surveyed on this map, not a general solution.

The model can only reference a name in that table (schema-enforced, same trick used to constrain
the 8 base actions), and `validate_plan()` cross-checks whatever name it picks against the words
actually in the original instruction - if asked for a landmark that isn't in the table, the model
tends to guess a real-but-wrong one rather than admit it doesn't know, and this catches that
instead of silently flying somewhere wrong.

## Coordinates

Every place `z`/altitude appears outside `drone.py` - the landmark table, the system prompt,
mission logs, `fly_to`'s own params - is plain positive-up altitude (`z = 8` is 8m up). AirSim
itself uses NED (`z` negative going up), but that convention is converted to and from exactly
once, at the top of `drone.py`'s `_navigate_to()` (plus two direct-AirSim-call spots in
`connect_and_takeoff`/`set_height` that bypass it) - nowhere else in the codebase needs to think
about the sign flip.

## Notes

- **Ports:** AirSim `41451`, CARLA `2000`.
- Ollama is run with `num_gpu: 0` (CPU inference) deliberately - running an 8B model on GPU
  alongside the simulator maxed out an 8GB card's VRAM during testing and caused severe slowdown
  from memory contention, even on hardware well above the original repo's tested 4GB card.

## Fixes / improvements over the reference repo

- **Settings.json bug fixed:** the reference's vehicle-population loop starts at `range(2, ...)`,
  so a single-drone run silently omits `Drone1` from `settings.json`, and a later sim restart
  comes back with a default `SimpleFlight`-named vehicle nothing in the code recognizes. Fixed to
  start at 1.
- **`land` position enforced, not just requested:** the reference's prompt tells the model `land`
  must be the last step (since there's no re-arm/takeoff action afterward) but never actually
  checks it. This version validates it in code and rejects a plan that puts `land` anywhere but
  last.
- **Relative height math:** a plain schema + prose rules weren't enough for the local model to
  correctly compute relative height changes ("go up 5 meters") - it defaulted to a fixed,
  ungrounded value regardless of the actual number asked, both in this build and in the reference
  (which additionally got the direction wrong on the same test). Fixed with a worked few-shot
  example showing the actual arithmetic (`default_height + meters`, in this build's positive-up
  terms), confirmed to generalize to new magnitudes/directions not in the example.
- **Model selection via CLI flag** (`--model`) instead of three near-duplicate scripts - the
  reference's `llama_airsim_agent.py` and `mistral_airsim_agent.py` differ by only 3 lines (a
  comment, the model name, one print label).
