"""Schema introspection: normalize a Pydantic model into fuzzable field specs.

Works from the model's JSON schema (uniform across nested models, unions,
constraints) with $refs resolved, so mutators can reason about every field's
type and constraints without pydantic-version-specific FieldInfo spelunking.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class FieldSpec:
    """One fuzzable field, with its resolved JSON-schema fragment."""
    name: str
    path: tuple[str, ...]          # dotted path from model root, e.g. ("address", "zip")
    required: bool
    schema: dict[str, Any]         # resolved JSON schema for this field (no $refs)
    nested: list["FieldSpec"] = dc_field(default_factory=list)  # sub-fields if object


@dataclass
class ModelSpec:
    model_cls: type
    name: str
    fields: list[FieldSpec]
    raw_schema: dict[str, Any]


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$refs so every fragment is self-contained."""
    defs = schema.get("$defs", {})

    def _res(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
                merged = {k: _res(v) for k, v in node.items() if k != "$ref"}
                base = _res(target)
                if isinstance(base, dict):
                    base.update(merged)
                    return base
                return base
            return {k: _res(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_res(v) for v in node]
        return node

    return _res(schema)


def _field_specs(props: dict[str, Any], required: set[str],
                 prefix: tuple[str, ...]) -> list[FieldSpec]:
    out: list[FieldSpec] = []
    for name, fschema in props.items():
        spec = FieldSpec(
            name=name,
            path=prefix + (name,),
            required=name in required,
            schema=fschema,
        )
        # recurse into plain object properties for targeted nested mutation
        if fschema.get("type") == "object" and isinstance(fschema.get("properties"), dict):
            spec.nested = _field_specs(
                fschema["properties"],
                set(fschema.get("required", [])),
                spec.path,
            )
        out.append(spec)
    return out


def introspect(model_cls: type) -> ModelSpec:
    """Build a ModelSpec for any pydantic BaseModel subclass."""
    raw = model_cls.model_json_schema()
    resolved = resolve_refs(raw)
    props = resolved.get("properties", {})
    required = set(resolved.get("required", []))
    return ModelSpec(
        model_cls=model_cls,
        name=model_cls.__name__,
        fields=_field_specs(props, required, ()),
        raw_schema=resolved,
    )


def iter_leaf_fields(spec: ModelSpec) -> list[FieldSpec]:
    """All fields including nested ones, depth-first."""
    out: list[FieldSpec] = []

    def _walk(fs: FieldSpec) -> None:
        out.append(fs)
        for child in fs.nested:
            _walk(child)

    for f in spec.fields:
        _walk(f)
    return out
