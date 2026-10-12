"""Valid baseline generation: build a schema-conformant instance from a ModelSpec.

Used as the mutation baseline for the fuzzer, and as the negative control for
false-positive measurement (a valid instance must NEVER be flagged).
"""
from __future__ import annotations

from typing import Any

from .introspect import ModelSpec
from .mutators import _variants_for, _eff_type


def _valid_for(schema: dict[str, Any], overrides: dict[str, Any],
               path: tuple[str, ...], depth: int = 0) -> Any:
    key = ".".join(path)
    if key in overrides:
        return overrides[key]
    if depth > 6:
        return None
    variants = _variants_for(schema)
    v = variants[0] if variants else {}
    if "const" in v:
        return v["const"]
    if "enum" in v and v["enum"]:
        return v["enum"][0]
    if "default" in v:
        return v["default"]
    # formatted strings (datetime, date, uuid, decimal) before generic string
    fmt = v.get("format", "")
    if fmt == "date-time":
        return "2024-01-15T10:30:00"
    if fmt == "date":
        return "2024-01-15"
    if fmt == "uuid":
        return "12345678-1234-5678-1234-567812345678"
    if fmt == "decimal":
        from decimal import Decimal
        return Decimal("9.99")
    t = v.get("type")
    if t == "string":
        if "pattern" in v:
            # caller must override patterned fields with known-good values
            raise ValueError(f"patterned field {key!r} needs an override")
        n = v.get("minLength", 0)
        s = "valid_" + key.replace(".", "_")
        return (s + "x" * max(0, n - len(s)))[: max(n, len(s))]
    if t == "integer":
        lo = v.get("minimum", v.get("exclusiveMinimum", 0))
        base = lo if isinstance(lo, int) else 5
        if "exclusiveMinimum" in v and base == v["exclusiveMinimum"]:
            base += 1
        return base
    if t == "number":
        lo = v.get("minimum", 0)
        return float(lo) + 0.5 if isinstance(lo, (int, float)) else 1.5
    if t == "boolean":
        return True
    if t == "array":
        items = v.get("items", {})
        n = max(v.get("minItems", 1), 1)
        return [_valid_for(items, overrides, path + (str(i),), depth + 1) for i in range(min(n, 3))]
    if t == "object" or "properties" in v:
        out: dict[str, Any] = {}
        for pn, ps in v.get("properties", {}).items():
            out[pn] = _valid_for(ps, overrides, path + (pn,), depth + 1)
        return out
    if t == "null":
        return None
    return "valid"


def generate_valid(spec: ModelSpec, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one valid instance dict for the model. Raises ValueError if a
    patterned field lacks an override (caller supplies known-good values)."""
    overrides = overrides or {}
    out: dict[str, Any] = {}
    for f in spec.fields:
        out[f.name] = _valid_for(f.schema, overrides, (f.name,))
    return out


def generate_valid_random(spec: ModelSpec, overrides: dict[str, Any] | None = None,
                          seed: int = 0) -> dict[str, Any]:
    """Build a RANDOMIZED valid instance (for false-positive measurement:
    many distinct valid inputs must all classify clean)."""
    import random
    import string as _string
    rng = random.Random(seed)
    overrides = overrides or {}

    def _rand(schema: dict[str, Any], path: tuple[str, ...], depth: int = 0) -> Any:
        from .mutators import _variants_for
        key = ".".join(path)
        if key in overrides:
            return overrides[key]
        if depth > 5:
            return None
        v = rng.choice([x for x in _variants_for(schema) if x.get("type") != "null"] or [{}])
        if "const" in v:
            return v["const"]
        if "enum" in v and v["enum"]:
            return rng.choice(v["enum"])
        if "default" in v and rng.random() < 0.5:
            return v["default"]
        fmt = v.get("format", "")
        if fmt == "date-time":
            return f"2024-0{rng.randint(1,9)}-{rng.randint(10,28):02d}T10:30:00"
        if fmt == "date":
            return f"2024-0{rng.randint(1,9)}-{rng.randint(10,28):02d}"
        if fmt == "uuid":
            return "12345678-1234-5678-1234-567812345678"
        if fmt == "decimal":
            from decimal import Decimal
            return Decimal(f"{rng.randint(1,99)}.{rng.randint(10,99)}")
        t = v.get("type")
        if t == "string":
            if "pattern" in v:
                raise ValueError(f"patterned field {key!r} needs an override")
            lo, hi = v.get("minLength", 0), v.get("maxLength", 24)
            n = rng.randint(lo, max(lo, min(hi, lo + 20)))
            return "".join(rng.choice(_string.ascii_letters) for _ in range(max(n, 1)))
        if t == "integer":
            lo = v.get("minimum", v.get("exclusiveMinimum", 0))
            hi = v.get("maximum", v.get("exclusiveMaximum", 100))
            lo = lo + 1 if "exclusiveMinimum" in v and lo == v["exclusiveMinimum"] else lo
            hi = hi - 1 if "exclusiveMaximum" in v and hi == v["exclusiveMaximum"] else hi
            return rng.randint(int(lo), max(int(lo), int(hi)))
        if t == "number":
            lo = float(v.get("minimum", 0))
            hi = float(v.get("maximum", 100))
            return round(rng.uniform(lo, hi), 2)
        if t == "boolean":
            return rng.choice([True, False])
        if t == "array":
            items = v.get("items", {})
            lo, hi = v.get("minItems", 0), v.get("maxItems", 3)
            return [_rand(items, path + (str(i),), depth + 1)
                    for i in range(rng.randint(lo, max(lo, hi)))]
        if t == "object" or "properties" in v:
            return {pn: _rand(ps, path + (pn,), depth + 1)
                    for pn, ps in v.get("properties", {}).items()}
        if t == "null":
            return None
        return "valid"

    return {f.name: _rand(f.schema, (f.name,)) for f in spec.fields}
