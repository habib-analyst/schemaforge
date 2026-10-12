"""Counterexample-guided repair: compile a ValidationError into a targeted
repair prompt, retry with a budget. Supporting feature — the fuzzer is the
headline; this is what you run AFTER the fuzzer tells you what's broken.

The repair prompt is the deliverable: it contains the schema, the broken
value, and every validation error with its exact location, constraint, and
the offending input — so a fixer (LLM or rule-based) has everything needed
in one shot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from pydantic import ValidationError


@dataclass
class RepairResult:
    success: bool
    value: Any | None
    attempts: int
    history: list[dict[str, Any]] = dc_field(default_factory=list)


def compile_repair_prompt(model_cls: type, broken_value: Any,
                          error: ValidationError) -> str:
    """Build the targeted repair prompt from a live ValidationError."""
    schema = model_cls.model_json_schema()
    err_lines = []
    for er in error.errors():
        loc = ".".join(map(str, er.get("loc", ()))) or "root"
        err_lines.append(
            f"- field `{loc}`: {er['msg']} "
            f"(rule: {er['type']}; received: {er.get('input')!r:.120})"
        )
    return (
        f"Your JSON output failed schema validation for `{model_cls.__name__}`.\n"
        f"Schema:\n{json.dumps(schema, indent=2)[:3000]}\n\n"
        f"Your broken output:\n{json.dumps(broken_value, default=str, indent=2)[:2000]}\n\n"
        f"Exact failures ({len(err_lines)}):\n" + "\n".join(err_lines) + "\n\n"
        f"Return ONLY corrected JSON satisfying the schema. Fix exactly the "
        f"fields listed above; do not change anything else."
    )


def repair_loop(model_cls: type, broken_value: Any,
                fix_callable: Callable[[str], str],
                max_retries: int = 3) -> RepairResult:
    """Try to repair a broken value. fix_callable(prompt) -> JSON string.

    Returns success with the repaired value, or failure after the budget.
    """
    history: list[dict[str, Any]] = []
    current = broken_value
    for attempt in range(1, max_retries + 1):
        try:
            obj = model_cls.model_validate(current)
            return RepairResult(True, obj.model_dump(mode="json"),
                                attempt, history)
        except ValidationError as e:
            prompt = compile_repair_prompt(model_cls, current, e)
            history.append({"attempt": attempt, "prompt": prompt,
                            "errors": e.errors()})
            try:
                fixed_raw = fix_callable(prompt)
                current = json.loads(fixed_raw)
            except Exception as ex:  # fixer itself failed
                history.append({"attempt": attempt, "fixer_error": str(ex)[:200]})
                return RepairResult(False, None, attempt, history)
    # final validation of last fix
    try:
        obj = model_cls.model_validate(current)
        return RepairResult(True, obj.model_dump(mode="json"), max_retries, history)
    except ValidationError:
        return RepairResult(False, None, max_retries, history)
