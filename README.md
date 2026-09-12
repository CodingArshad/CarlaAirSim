# FICSAgenticDroneSim (individual build)

**Repurposed 2026-09-12.** This repo previously held a from-scratch replication of
`niranjanpillai2009-altr/AirSimRepo` (Niranjan's solo baseline) for the FICS Lab agentic-drone
project, Dr. Akbas. That work is done, was reviewed, and stays fully available in this repo's git
history - nothing was deleted, just moved on from.

The team's actual project structure, clarified by Akbas: each member individually builds their own
implementation of Phases 1-10 of the team's shared architecture
(`AkbasLab/FICSAgenticDroneSim`), open-endedly - the team repo (and everyone's own prior work,
including the replication above) is reference material, not a spec to copy. Best ideas across
everyone's independent builds get combined at the weekly team meeting and on Discord. This repo is
now that individual build, starting from Phase 1.

## Status

Phase 1 (natural-language flight planning: instruction -> validated action plan -> executed
flight, plus the ground-truth landing profile) is designed but not yet coded. Design notes live
outside this repo, in the personal planning doc that tracks this work.

## Where the old replication went

Still here - `git log` before this commit has the full `drone.py` / `llm_planner.py` / `swarm.py` /
`main.py` implementation, object-avoidance, landmark navigation, and NED-simplification work.
Nothing here is lost, just not what this repo is building toward anymore.
