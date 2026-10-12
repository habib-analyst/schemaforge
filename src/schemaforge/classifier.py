"""Failure classifier: REJECTED (would crash) vs SILENT_MISPARSE (validates but
corrupts) vs CLEAN.

The silent-misparse class is schemaforge's core finding: schemas that LOOK
fine but silently change data semantics — lax coercions ("42" -> 42),
dropped extra fields, whitespace not stripped, 1.0 -> 1. These never raise,
so they sail through every try/except in production and corrupt downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from .mutators import Candidate


@dataclass
class Verdict:
    candidate: Candidate
    outcome: str          # "rejected" | "silent_misparse" | "clean"
    error: str | None     # ValidationError summary when rejected
    details: str          # human-readable: what changed / why rejected


def _is_iso_datetime(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def semantic_diff(original: Any, parsed: Any, path: str = "root") -> list[str]:
    """Type-strict recursive diff. Returns a list of change descriptions;
    empty means the parse preserved semantics exactly."""
    diffs: list[str] = []
    if isinstance(original, dict) and isinstance(parsed, dict):
        for k in original:
            if k not in parsed:
                diffs.append(f"{path}.{k}: field silently DROPPED")
            else:
                diffs.extend(semantic_diff(original[k], parsed[k], f"{path}.{k}"))
        for k in parsed:
            if k not in original:
                diffs.append(f"{path}.{k}: field ADDED by parsing ({parsed[k]!r:.40})")
        return diffs
    if isinstance(original, list) and isinstance(parsed, list):
        if len(original) != len(parsed):
            diffs.append(f"{path}: list length {len(original)} -> {len(parsed)}")
        for i, (a, b) in enumerate(zip(original, parsed)):
            diffs.extend(semantic_diff(a, b, f"{path}[{i}]"))
        return diffs
    # expected coercions that preserve semantics exactly
    if isinstance(original, str) and isinstance(parsed, datetime):
        if _is_iso_datetime(original) is not None:
            return diffs
    if isinstance(original, str) and isinstance(parsed, UUID):
        try:
            if str(parsed) == original.lower():
                return diffs
        except Exception:
            pass
    if isinstance(parsed, Enum) and not isinstance(original, Enum):
        return semantic_diff(original, parsed.value, path)
    if type(original) is not type(parsed):
        # bool is subclass of int: True vs 1 must still flag
        if not (isinstance(original, bool) or isinstance(parsed, bool)):
            if original == parsed and isinstance(original, (int, float)) \
                    and isinstance(parsed, (int, float)):
                diffs.append(f"{path}: numeric type {type(original).__name__} -> "
                             f"{type(parsed).__name__} ({original!r})")
            else:
                diffs.append(f"{path}: TYPE CHANGED {type(original).__name__} -> "
                             f"{type(parsed).__name__} ({original!r} -> {parsed!r:.40})")
        else:
            diffs.append(f"{path}: TYPE CHANGED {type(original).__name__} -> "
                         f"{type(parsed).__name__} ({original!r} -> {parsed!r:.40})")
        return diffs
    if original != parsed:
        diffs.append(f"{path}: VALUE CHANGED {original!r:.40} -> {parsed!r:.40}")
    return diffs


def classify(model_cls: type, candidate: Candidate) -> Verdict:
    """Validate one candidate; return its verdict."""
    try:
        obj = model_cls.model_validate(candidate.value)
    except ValidationError as e:
        errs = e.errors()
        summary = "; ".join(
            f"{'.'.join(map(str, er['loc'])) or 'root'}: {er['msg']} "
            f"(got {er.get('input')!r:.40})" for er in errs[:4]
        )
        if len(errs) > 4:
            summary += f" (+{len(errs) - 4} more)"
        return Verdict(candidate, "rejected", summary,
                        f"rejected by schema: {summary}")
    dumped = obj.model_dump(mode="python")
    diffs = semantic_diff(candidate.value, dumped)
    if diffs:
        return Verdict(candidate, "silent_misparse", None,
                        "silently misparsed: " + "; ".join(diffs[:4]))
    return Verdict(candidate, "clean", None, "validates with identical semantics")
