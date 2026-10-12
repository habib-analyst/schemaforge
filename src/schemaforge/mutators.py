"""Adversarial mutators: each turns a valid baseline instance into candidates
designed to break the schema — fuzzing AT the schema's own constraint boundaries.

Every mutator yields Candidate objects tagged with the mutation class, the
dotted target path, and a human-readable description. The mapping from
mutation class to real LLM output behaviour is documented in REPORT_LLM_MAP.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterator

from .introspect import FieldSpec, ModelSpec, iter_leaf_fields


@dataclass
class Candidate:
    value: Any            # mutated input (usually a dict)
    mutation_class: str   # e.g. "boundary", "type_confusion"
    target: str           # dotted field path, "" for whole-input mutations
    description: str


# What each mutation class corresponds to in real LLM output behaviour.
LLM_BEHAVIOUR_MAP = {
    "boundary": "LLM emits values at/over declared limits (counts, lengths)",
    "type_confusion": "LLM emits numbers as strings, bools as 'yes'/'true'",
    "missing_required": "LLM omits a field it deems obvious",
    "extra_field": "LLM hallucinates extra keys not in the schema",
    "null_field": "LLM emits null for unknown/optional-ish values",
    "whitespace": "LLM pads values with spaces/newlines",
    "unicode": "LLM emits emoji, fullwidth chars, RTL text",
    "empty_container": "LLM returns [] / {} instead of populated data",
    "enum_violation": "LLM invents a new enum value",
    "pattern_violation": "LLM emits plausible-but-malformed formatted strings",
    "union_ambiguity": "LLM emits a value matching no (or several) union arms",
    "coercion_trap": "LLM emits loosely-typed values pydantic silently coerces",
    "deep_nesting": "LLM nests structures pathologically deep",
    "huge_container": "LLM dumps enormous lists (DoS-shaped output)",
}


def _get(d: dict, path: tuple[str, ...]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


def _set(d: dict, path: tuple[str, ...], value: Any) -> dict:
    d = copy.deepcopy(d)
    cur = d
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value
    return d


def _delete(d: dict, path: tuple[str, ...]) -> dict:
    d = copy.deepcopy(d)
    cur = d
    for p in path[:-1]:
        cur = cur.get(p, {})
    cur.pop(path[-1], None)
    return d


class _Missing:
    pass


_MISSING = _Missing()


def _variants_for(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Unwrap anyOf/oneOf into concrete variants (skip bare null)."""
    if "anyOf" in schema:
        return [v for v in schema["anyOf"] if v.get("type") != "null"]
    if "oneOf" in schema:
        return [v for v in schema["oneOf"] if v.get("type") != "null"]
    return [schema]


def _eff_type(schema: dict[str, Any]) -> str | None:
    """Effective type of the first non-null variant."""
    for v in _variants_for(schema):
        t = v.get("type")
        if t:
            return t
        if "$ref" in v or "properties" in v:
            return "object"
        if "enum" in v:
            return "enum"
    return None


# ---------------------------------------------------------------- mutators

def mutate_boundary(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if cur is _MISSING:
            continue
        tgt = ".".join(f.path)
        for v in _variants_for(f.schema):
            t = v.get("type")
            if t in ("integer", "number"):
                lo, hi = v.get("minimum"), v.get("maximum")
                ex_lo, ex_hi = v.get("exclusiveMinimum"), v.get("exclusiveMaximum")
                step = 1 if t == "integer" else 1e-9
                if lo is not None:
                    yield Candidate(_set(baseline, f.path, lo - step), "boundary", tgt,
                                    f"{tgt}: below minimum ({lo - step} < {lo})")
                if ex_lo is not None:
                    yield Candidate(_set(baseline, f.path, ex_lo), "boundary", tgt,
                                    f"{tgt}: equals exclusiveMinimum ({ex_lo})")
                if hi is not None:
                    yield Candidate(_set(baseline, f.path, hi + step), "boundary", tgt,
                                    f"{tgt}: above maximum ({hi + step} > {hi})")
                if ex_hi is not None:
                    yield Candidate(_set(baseline, f.path, ex_hi), "boundary", tgt,
                                    f"{tgt}: equals exclusiveMaximum ({ex_hi})")
                if lo is None and hi is None and isinstance(cur, (int, float)):
                    # unbounded numeric: extremes LLMs love to emit
                    yield Candidate(_set(baseline, f.path, 10 ** 18), "boundary", tgt,
                                    f"{tgt}: huge value 10^18 (no maximum declared)")
                    yield Candidate(_set(baseline, f.path, -(10 ** 18)), "boundary", tgt,
                                    f"{tgt}: huge negative value (no minimum declared)")
            elif t == "string":
                minl, maxl = v.get("minLength"), v.get("maxLength")
                if minl:
                    yield Candidate(_set(baseline, f.path, "x" * (minl - 1)), "boundary", tgt,
                                    f"{tgt}: string shorter than minLength ({minl - 1} < {minl})")
                if maxl:
                    yield Candidate(_set(baseline, f.path, "x" * (maxl + 1)), "boundary", tgt,
                                    f"{tgt}: string longer than maxLength ({maxl + 1} > {maxl})")
            elif t == "array":
                min_i, max_i = v.get("minItems"), v.get("maxItems")
                if min_i and isinstance(cur, list) and len(cur) >= min_i:
                    yield Candidate(_set(baseline, f.path, cur[:min_i - 1]), "boundary", tgt,
                                    f"{tgt}: fewer items than minItems")
                if max_i and isinstance(cur, list):
                    item = cur[0] if cur else 0
                    yield Candidate(_set(baseline, f.path, [item] * (max_i + 1)), "boundary", tgt,
                                    f"{tgt}: more items than maxItems")


def mutate_type_confusion(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    swaps: dict[str, list[Any]] = {
        "integer": ["5", 5.0, True, [5], {"n": 5}],
        "number": ["3.14", True],
        "string": [5, 5.5, True, ["x"], {"s": "x"}],
        "boolean": ["true", "yes", 1, 0, "1"],
        "array": [{"a": 1}, "notalist", 5],
        "object": [[1, 2], "notadict", 5],
    }
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if cur is _MISSING:
            continue
        tgt = ".".join(f.path)
        t = _eff_type(f.schema)
        for alt in swaps.get(t, []):
            if type(alt) is type(cur):
                continue
            yield Candidate(_set(baseline, f.path, alt), "type_confusion", tgt,
                            f"{tgt}: {type(cur).__name__} replaced by {type(alt).__name__} ({alt!r:.40})")


def mutate_missing_required(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        if f.required and _get(baseline, f.path) is not _MISSING:
            tgt = ".".join(f.path)
            yield Candidate(_delete(baseline, f.path), "missing_required", tgt,
                            f"{tgt}: required field deleted")


def mutate_extra_field(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    extras = [
        ("zzz_hallucinated", "surprise"),
        ("zzz_hallucinated", {"nested": [1, 2, 3]}),
        ("zzz_hallucinated", 42),
    ]
    for key, val in extras:
        d = copy.deepcopy(baseline)
        d[key] = val
        yield Candidate(d, "extra_field", "", f"root: unexpected field {key!r} added")


def mutate_null_field(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if cur is _MISSING or cur is None:
            continue
        tgt = ".".join(f.path)
        yield Candidate(_set(baseline, f.path, None), "null_field", tgt,
                        f"{tgt}: set to null")


def mutate_whitespace(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    tricks = ["  padded  ", "", "   ", "\t\n ", "\u200bzero-width\u200b", "trailing space "]
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if not isinstance(cur, str):
            continue
        tgt = ".".join(f.path)
        for tr in tricks:
            yield Candidate(_set(baseline, f.path, tr), "whitespace", tgt,
                            f"{tgt}: whitespace trick {tr!r}")


def mutate_unicode(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    tricks = ["\U0001f600 emoji", "ＳＴＲ fullwidth", "\u202eRTL override", "naïve café ☃"]
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if not isinstance(cur, str):
            continue
        tgt = ".".join(f.path)
        for tr in tricks:
            yield Candidate(_set(baseline, f.path, tr), "unicode", tgt,
                            f"{tgt}: unicode trick {tr!r}")


def mutate_empty_container(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        tgt = ".".join(f.path)
        if isinstance(cur, list) and cur:
            yield Candidate(_set(baseline, f.path, []), "empty_container", tgt,
                            f"{tgt}: list emptied")
        if isinstance(cur, dict) and cur and _eff_type(f.schema) == "object":
            yield Candidate(_set(baseline, f.path, {}), "empty_container", tgt,
                            f"{tgt}: object emptied")


def mutate_enum_violation(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if cur is _MISSING:
            continue
        for v in _variants_for(f.schema):
            if "enum" in v:
                tgt = ".".join(f.path)
                yield Candidate(_set(baseline, f.path, "NOT_A_REAL_CHOICE"), "enum_violation", tgt,
                                f"{tgt}: value outside enum {v['enum']!r:.60}")


def mutate_pattern_violation(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if not isinstance(cur, str):
            continue
        for v in _variants_for(f.schema):
            if "pattern" in v:
                tgt = ".".join(f.path)
                yield Candidate(_set(baseline, f.path, "!!!-definitely-not-matching-!!!"),
                                "pattern_violation", tgt,
                                f"{tgt}: violates pattern {v['pattern']!r:.60}")


def mutate_union_ambiguity(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        variants = _variants_for(f.schema)
        if len(variants) < 2:
            continue
        tgt = ".".join(f.path)
        # value matching NO arm
        yield Candidate(_set(baseline, f.path, {"__no_arm_matches_this__": [None]}),
                        "union_ambiguity", tgt, f"{tgt}: matches no union arm")
        # value plausibly matching several arms (dict with merged keys)
        merged: dict[str, Any] = {}
        for v in variants:
            for k, ps in v.get("properties", {}).items():
                if k not in merged:
                    merged[k] = _sample_value(ps)
        if merged:
            yield Candidate(_set(baseline, f.path, merged), "union_ambiguity", tgt,
                            f"{tgt}: merged keys, may match several arms")


def mutate_coercion_trap(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    """Values pydantic LAX mode silently coerces — the silent-corruption class."""
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if cur is _MISSING:
            continue
        tgt = ".".join(f.path)
        t = _eff_type(f.schema)
        traps: list[Any] = []
        if t == "integer":
            traps = ["42", "  42  ", 42.0, True]
        elif t == "number":
            traps = ["3.14", True]
        elif t == "boolean":
            traps = ["true", "yes", "1", 1, 0, "false"]
        elif t == "string":
            traps = [42, 4.5, True]  # lax str() coercion
        for tr in traps:
            if type(tr) is type(cur):
                continue
            yield Candidate(_set(baseline, f.path, tr), "coercion_trap", tgt,
                            f"{tgt}: lax-coercible {tr!r} for {t}")


def mutate_deep_nesting(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    # find self-similar nested object fields (recursive models)
    for f in spec.fields:
        if _eff_type(f.schema) == "object" and f.name in str(f.schema):
            deep: Any = {"level": 0}
            cur = deep
            for i in range(1, 60):
                cur["child"] = {"level": i}
                cur = cur["child"]
            yield Candidate(_set(baseline, f.path, deep), "deep_nesting", ".".join(f.path),
                            f"{'.'.join(f.path)}: 60-deep nesting")


def mutate_huge_container(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if isinstance(cur, list):
            tgt = ".".join(f.path)
            item = cur[0] if cur else 0
            yield Candidate(_set(baseline, f.path, [item] * 20000), "huge_container", tgt,
                            f"{tgt}: 20k-element list (DoS-shaped)")


def mutate_malformed_string(baseline: dict, spec: ModelSpec) -> Iterator[Candidate]:
    """Plausible-but-malformed payloads for string fields WITHOUT patterns —
    the exact class a missing format constraint lets through."""
    payloads = ["not-an-email", "999-999-999", "<script>alert(1)</script>",
                "http://", "....", "rm -rf /", "yesterday", "N/A"]
    for f in iter_leaf_fields(spec):
        cur = _get(baseline, f.path)
        if not isinstance(cur, str):
            continue
        if any("pattern" in v for v in _variants_for(f.schema)):
            continue  # patterned fields are covered by pattern_violation
        tgt = ".".join(f.path)
        for p in payloads:
            yield Candidate(_set(baseline, f.path, p), "malformed", tgt,
                            f"{tgt}: malformed payload {p!r} (no pattern declared)")


LLM_BEHAVIOUR_MAP["malformed"] = "LLM emits plausible-but-malformed formatted strings (fake emails, bad zips)"


def _sample_value(schema: dict[str, Any]) -> Any:
    t = _eff_type(schema)
    return {"string": "x", "integer": 1, "number": 1.0, "boolean": True,
            "array": [], "object": {}}.get(t, "x")
    t = _eff_type(schema)
    return {"string": "x", "integer": 1, "number": 1.0, "boolean": True,
            "array": [], "object": {}}.get(t, "x")


MUTATORS = [
    mutate_boundary,
    mutate_type_confusion,
    mutate_missing_required,
    mutate_extra_field,
    mutate_null_field,
    mutate_whitespace,
    mutate_unicode,
    mutate_empty_container,
    mutate_enum_violation,
    mutate_pattern_violation,
    mutate_union_ambiguity,
    mutate_malformed_string,
    mutate_coercion_trap,
    mutate_deep_nesting,
    mutate_huge_container,
]
