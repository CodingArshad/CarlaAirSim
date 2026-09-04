# CarlaAirSim

Natural-language control of a simulated drone in the CarlaAir simulator. Type a plain-English
instruction, a local LLM turns it into a short list of flight actions, the actions get validated,
then the drone actually flies the plan.

Built as a from-scratch replication of `niranjanpillai2009-altr/AirSimRepo` for the FICS Lab
agentic-drone project (Dr. Akbas), understanding-first rather than copied: same 8-action pipeline
and mechanics, own file structure, own naming, own system prompt wording, plus a few deliberate
fixes/improvements over the original (see below).

## How it works

The LLM is restricted to 8 fixed actions (`fly_to`, `fly_forward`, `fly_backward`, `fly_left`,
`fly_right`, `hover`, `set_height`, `land`), each with fixed parameters, enforced with a JSON
schema. Keeping the vocabulary small and structured means the output is predictable and can be
checked before the drone leaves the ground.

## Files

| File | What it does |
|---|---|
| `main.py` | Entry point - CLI, drone-count/mission prompts, sim setup, dispatches to single-drone or swarm flight |
| `drone.py` | `Drone` class - connect/takeoff, the 8 flight actions, staged landing |
| `llm_planner.py` | `LLMPlanner` class - talks to Ollama, enforces the JSON schema, validates the plan before anything flies |
| `swarm.py` | `Swarm` class - runs multiple drones concurrently, one thread each |

## Running it

Requires a running CarlaAir simulator (see `niranjanpillai2009-altr/AirSimRepo`'s `SETUP_GUIDE.md`
for the full environment setup - simulator download, conda env, `Drone1` rename) and
[Ollama](https://ollama.com) with a local model pulled (default `llama3.1:8b`):

```
python main.py
python main.py --model mistral-nemo   # or any other Ollama model
```

## Notes

- **Coordinates are NED:** `z` is negative going up (`z = -8` is 8m above the reference height).
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
  example showing the actual arithmetic (`default_height - meters`), confirmed to generalize to
  new magnitudes/directions not in the example.
- **Model selection via CLI flag** (`--model`) instead of three near-duplicate scripts - the
  reference's `llama_airsim_agent.py` and `mistral_airsim_agent.py` differ by only 3 lines (a
  comment, the model name, one print label).
