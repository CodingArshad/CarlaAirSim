import json

import ollama

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HEIGHT = -8.0

# Forces Ollama to return {"plan": [ ...steps... ]} instead of a single
# action object. Local models tend to answer only the first part of a
# multi-step instruction without this.
TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["fly_to", "fly_forward", "fly_backward",
                                 "fly_left", "fly_right", "hover",
                                 "set_height", "land"],
                    },
                    "params": {"type": "object"},
                },
                "required": ["action", "params"],
            },
        }
    },
    "required": ["plan"],
}

NEEDED_PARAMS = {
    "fly_to": ["x", "y", "z"],
    "fly_forward": ["duration"],
    "fly_backward": ["duration"],
    "fly_left": ["duration"],
    "fly_right": ["duration"],
    "hover": ["duration"],
    "set_height": ["z"],
    "land": [],
}


class LLMPlanner:
    """Turns a plain-English instruction into a validated list of drone actions."""

    def __init__(self, model=DEFAULT_MODEL, default_height=DEFAULT_HEIGHT):
        self.model = model
        self.default_height = default_height

    def _system_prompt(self):
        return (
            "You are a mission planner for a simulated drone.\n"
            "Turn the user's instruction into a plan: a JSON object with a "
            '"plan" array, one step per thing the user asks for.\n\n'
            "Use ONLY these actions:\n"
            "- fly_to        params: x, y, z\n"
            "- fly_forward   params: duration\n"
            "- fly_backward  params: duration\n"
            "- fly_left      params: duration\n"
            "- fly_right     params: duration\n"
            "- hover         params: duration\n"
            "- set_height    params: z\n"
            "- land          params: (empty)\n\n"
            "RULES:\n"
            "- Z is height, and a LOWER (more negative) number means HIGHER "
            f"altitude. Default cruising height is {self.default_height}.\n"
            "- 'return to home' / 'reset position' / 'come back' means "
            f"fly_to x=0.0, y=0.0, z={self.default_height}.\n"
            "- 'touch the ground' or 'land' means the land action, and it "
            "must always be the LAST step - there is no action to take off "
            "again afterward.\n"
            "- Add a step for everything the user asks for. Never skip "
            "instructions.\n"
            "- Duration is measured in seconds and is always greater than 0."
        )

    def _examples(self):
        return [
            ("Fly to 4, 8, -30, then hover for 2 seconds, come back home and then land.",
             '{"plan": [{"action": "fly_to", "params": {"x": 4.0, "y": 8.0, "z": -30.0}}, '
             '{"action": "hover", "params": {"duration": 2.0}}, '
             '{"action": "fly_to", "params": {"x": 0.0, "y": 0.0, "z": -8.0}}, '
             '{"action": "land", "params": {}}]}'),
            # Relative height math: default cruising height is -8.0, so
            # climbing 5 meters higher subtracts 5 (more negative = higher),
            # giving -13.0 - not just "-5.0" from pattern-matching the digit.
            ("Climb 5 meters higher, then hover for 2 seconds.",
             '{"plan": [{"action": "set_height", "params": {"z": -13.0}}, '
             '{"action": "hover", "params": {"duration": 2.0}}]}'),
            ("Hover in place for 9.3 seconds then land",
             '{"plan": [{"action": "hover", "params": {"duration": 9.3}}, '
             '{"action": "land", "params": {}}]}'),
        ]

    def plan(self, user_prompt):
        messages = [{"role": "system", "content": self._system_prompt()}]
        for ex_in, ex_out in self._examples():
            messages.append({"role": "user", "content": ex_in})
            messages.append({"role": "assistant", "content": ex_out})
        messages.append({"role": "user", "content": user_prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            format=TASK_SCHEMA,
            # Forces CPU inference. CarlaAir + an 8B model both wanting GPU
            # VRAM at once maxed out an 8GB card during testing (~7.8/8.1GB
            # used) and made planning extremely slow from memory pressure -
            # this trades slower inference for the sim keeping full GPU
            # headroom, matching what the reference repo does.
            options={"temperature": 0, "num_gpu": 0},
        )

        raw = response["message"]["content"]
        print(f"  [model raw output] {raw}")
        data = json.loads(raw)

        actions = extract_actions(data)
        if actions is None:
            raise ValueError(f"No list of actions in the model's reply: {raw}")

        return validate_plan(actions)


def extract_actions(data):
    """Pull the list of steps out of whatever shape the model returned."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "action" in data:
            return [data]

        for value in data.values():
            if isinstance(value, list):
                return value

        for value in data.values():
            if isinstance(value, dict):
                found = extract_actions(value)
                if found is not None:
                    return found

    return None


def validate_plan(task_list):
    """Checks the model's plan before any of it gets flown."""
    if not isinstance(task_list, list):
        raise ValueError("Expected a list of steps")

    if len(task_list) == 0:
        raise ValueError("The model returned an empty plan")

    for i, step in enumerate(task_list, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i} isn't in the right format")

        action = step.get("action")
        if action not in NEEDED_PARAMS:
            raise ValueError(f"Step {i}: don't know how to do '{action}'")

        params = step.get("params", {})

        for key in NEEDED_PARAMS[action]:
            if key not in params:
                raise ValueError(f"Step {i} ({action}): missing '{key}'")

            try:
                params[key] = float(params[key])
            except (TypeError, ValueError):
                raise ValueError(f"Step {i} ({action}): '{key}' isn't a number")

        if "duration" in params and params["duration"] <= 0:
            raise ValueError(f"Step {i} ({action}): duration must be positive")

        if action == "land" and i != len(task_list):
            raise ValueError("'land' must be the last step in the plan")

    return task_list
