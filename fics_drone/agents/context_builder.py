"""The model's prompt, built from belief alone - never the raw flight log, never
ground truth, never another agent's state. Every section is bounded: an agent
whose prompt grows with mission length has a decision quality that varies with
how long it has been flying, which would confound any comparison this project
exists to make. Turn 200 should be the same kind of decision as turn 3.
"""

from typing import Optional, Sequence, Tuple

from .belief import Belief
from .decision_schema import (COMMUNICATION_STATUS, CURRENT_RISK, MAX_OUTGOING_MESSAGES, MISSION_PROGRESS,
                              REASON_CODES)
from .objectives import Objective, ReplanEvent

MAX_TEAMMATES_SHOWN = 8

OPTION_MEANING = {
    Objective.LISTEN: "stay here a few more seconds in case a teammate is about to report a target",
    Objective.CHECK_FOR_ORPHANS: "check whether a teammate's sector has been abandoned, and take it over if so",
    Objective.RETURN_HOME: "stop and fly home to land",
    Objective.GO_TO_WAYPOINT: 'fly to world position (x, y) in meters at your current altitude '
                              '(add "x" and "y" number fields), e.g. to look somewhere new',
}


def build_prompt(belief: Belief, event: ReplanEvent, legal: Sequence[Objective],
                 listen_cap: int, correction: Optional[str] = None,
                 sectors: Sequence = (), spawn_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 checked_points: Sequence[str] = ()) -> str:
    own = [s for s in belief.mission.targets_known.values() if s.source == "sensor"]
    second_hand = [s for s in belief.mission.targets_known.values() if s.source != "sensor"]

    teammates = list(belief.team.teammates.values())[:MAX_TEAMMATES_SHOWN]
    team_lines = [f"  {t.name}: last heard {max(0.0, belief.elapsed_s - t.provenance.timestamp):.0f}s ago"
                  + (f", last at ({t.last_known_position[0]:.0f}, {t.last_known_position[1]:.0f})"
                     if t.last_known_position else "")
                  for t in teammates] or ["  (none known)"]
    pos = belief.position
    map_lines = []
    if Objective.GO_TO_WAYPOINT in legal:
        # Mission geometry the team is already given (sector boxes) - no target locations,
        # and deliberately no no-fly zones: keeping those out is the guardian's job to enforce.
        map_lines = ["MAP (world meters)",
                     f"- you are at ({pos[0] + spawn_offset[0]:.0f}, {pos[1] + spawn_offset[1]:.0f})",
                     *[f"- sector {s.id}: x {s.x_min:.0f}..{s.x_max:.0f}, y {s.y_min:.0f}..{s.y_max:.0f}"
                       for s in sectors[:8]]]
        if checked_points:
            # Phase 12.5: computed by code (reasoning_tools.py), so the model reads verdicts
            # instead of doing distance arithmetic it is bad at.
            map_lines += ["CHECKED POINTS (already computed by code; you do not need to calculate anything)",
                          *list(checked_points)[:8]]

    lines = [
        "You are one drone in a search team. Your own sector is already finished.",
        "Choose your next step. You may ONLY pick one of the options below.",
        "",
        "SITUATION",
        f"- last_event: {event.value}",
        f"- battery_remaining: {belief.battery_frac_remaining * 100:.0f}%",
        f"- listen_rounds_used: {belief.listen_rounds} of {listen_cap}",
        f"- own_targets: {len(own)}",
        f"- second_hand_targets: {len(second_hand)}",
        *([f"- your_last_move_was_blocked_by_the_safety_layer: {belief.self_state.last_block_reason}",
           "  (the safety layer is independent and cannot be argued with - choose something different)"]
          if event == ReplanEvent.GUARDIAN_BLOCKED and belief.self_state.last_block_reason else []),
        "TEAMMATES",
        *team_lines,
        *([f"HELP REQUESTS RECEIVED (informational only; you are not obliged to act on them)",
           *[f"- {sender} asked for help ({code}) {max(0.0, belief.elapsed_s - t):.0f}s ago"
             for sender, code, t in belief.communication.help_requests[-3:]]]
          if belief.communication.help_requests else []),
        *map_lines,
        "OPTIONS",
        *[f"- {o.value}: {OPTION_MEANING[o]}" for o in legal],
        "",
        "Reply with ONLY a JSON object of exactly this shape:",
        '{"situation_assessment": {"mission_progress": "<' + '|'.join(MISSION_PROGRESS) + '>", '
        '"communication_status": "<' + '|'.join(COMMUNICATION_STATUS) + '>", '
        '"current_risk": "<' + '|'.join(CURRENT_RISK) + '>"}, '
        '"selected_tool": "<option>", "parameters": {"reason_code": "<code>"}, '
        '"outgoing_messages": [], "confidence": <number from 0 to 1>}',
        *(['If (and only if) you choose go_to_waypoint, parameters must also contain the coordinates: '
           '{"reason_code": "<code>", "x": <number>, "y": <number>}']
          if Objective.GO_TO_WAYPOINT in legal else []),
        f"reason_code must be one of: {', '.join(REASON_CODES)}",
        *(['outgoing_messages is [] unless you want to ask teammates for help; then each message is '
           '{"message_type": "help_request", "recipients": ["<teammate name from TEAMMATES>"], '
           '"payload": {"reason_code": "<code>"}} (at most ' + str(MAX_OUTGOING_MESSAGES) + ', help_request is the only type allowed)']
          if teammates else ['outgoing_messages must be [] (you have no teammates to message)']),
    ]
    if correction:
        lines += ["", f"YOUR PREVIOUS ANSWER WAS REJECTED: {correction}. Answer again, following the format exactly."]
    return "\n".join(lines)
