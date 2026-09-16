"""Local Llama planner via Ollama - no API key needed (Arshad's 17, can't get
a Gemini key). CPU-only (num_gpu=0): running the model on GPU alongside
CarlaAir maxes shared VRAM and slows the sim - found the hard way last time."""

import json

from ..core.schema import build_plan_schema
from .base_planner import FEWSHOT, SYSTEM_PROMPT, PlanError, validate_plan

MODEL = "llama3.1:8b"


class LlamaPlanner:
    def __init__(self, model: str = MODEL):
        import ollama  # lazy import
        self._ollama = ollama
        self.model = model
        self._schema = build_plan_schema()

    def plan(self, instruction: str):
        """instruction -> list[SkillCommand], or raises PlanError."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_in, ex_out in FEWSHOT:
            messages.append({"role": "user", "content": ex_in})
            messages.append({"role": "assistant", "content": ex_out})
        messages.append({"role": "user", "content": instruction})

        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            format=self._schema,
            options={"num_gpu": 0, "temperature": 0},
        )
        raw = response["message"]["content"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PlanError(f"model returned invalid JSON: {e}")

        steps = data.get("plan")
        if not isinstance(steps, list):
            raise PlanError(f"no 'plan' array in the model's reply: {raw}")

        return validate_plan(steps)
