"""schemaforge test suite: 18 tests covering introspection, mutation,
classification, reporting, repair, and the valid generators."""
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, StrictInt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemaforge import (
    Candidate, Fuzzer, classify, compile_repair_prompt, forge_report, fuzz,
    generate_valid, generate_valid_random, introspect, repair_loop, semantic_diff,
)
from schemaforge.mutators import LLM_BEHAVIOUR_MAP, MUTATORS


class ToyModel(BaseModel):
    name: str = Field(min_length=2, max_length=10)
    age: int = Field(ge=0, le=150)
    nickname: str | None = None


class LaxModel(BaseModel):
    count: int
    label: str


class ForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int


# ------------------------------------------------------------- introspect

def test_introspect_fields():
    spec = introspect(ToyModel)
    assert spec.name == "ToyModel"
    names = {f.name for f in spec.fields}
    assert {"name", "age", "nickname"} <= names
    age = next(f for f in spec.fields if f.name == "age")
    assert age.required
    assert age.schema.get("minimum") == 0
    assert age.schema.get("maximum") == 150


def test_introspect_nested():
    class Inner(BaseModel):
        z: int

    class Outer(BaseModel):
        inner: Inner
        tag: str

    spec = introspect(Outer)
    inner = next(f for f in spec.fields if f.name == "inner")
    assert len(inner.nested) == 1 and inner.nested[0].name == "z"


# ------------------------------------------------------------- mutators

def test_mutators_all_registered():
    assert len(MUTATORS) == 15
    # every mutator's name (minus prefix, minus "_string" suffix variants)
    # must have a documented LLM-behaviour mapping
    names = {m.__name__.replace("mutate_", "").replace("_string", "") for m in MUTATORS}
    assert set(LLM_BEHAVIOUR_MAP) >= names


def test_boundary_mutations_hit_limits():
    spec = introspect(ToyModel)
    base = generate_valid(spec)
    cands = [c for m in MUTATORS for c in m(base, spec) if c.mutation_class == "boundary"]
    targets = {(c.target, c.value[c.target]) for c in cands}
    assert ("age", -1) in targets or ("age", 151) in targets
    assert any(c.target == "name" and len(c.value["name"]) == 11 for c in cands)


def test_missing_required_and_extra():
    spec = introspect(ToyModel)
    base = generate_valid(spec)
    cands = [c for m in MUTATORS for c in base and m(base, spec)]
    assert any(c.mutation_class == "missing_required" and "age" not in c.value for c in cands)
    assert any(c.mutation_class == "extra_field" and "zzz_hallucinated" in c.value for c in cands)


# ------------------------------------------------------------- classifier

def test_classify_rejected():
    c = Candidate({"name": "x", "age": -5}, "boundary", "age", "t")
    v = classify(ToyModel, c)
    assert v.outcome == "rejected"
    assert v.error is not None


def test_classify_clean():
    spec = introspect(ToyModel)
    base = generate_valid(spec)
    v = classify(ToyModel, Candidate(base, "baseline", "", "t"))
    assert v.outcome == "clean"


def test_classify_silent_misparse_lax_coercion():
    v = classify(LaxModel, Candidate({"count": "42", "label": "ok"}, "coercion_trap", "count", "t"))
    assert v.outcome == "silent_misparse"
    assert "TYPE CHANGED" in v.details


def test_extra_drop_is_misparse():
    class IgnorDefault(BaseModel):
        x: int
    v = classify(IgnorDefault, Candidate({"x": 1, "zzz": 2}, "extra_field", "", "t"))
    assert v.outcome == "silent_misparse"
    assert "DROPPED" in v.details


def test_extra_forbid_rejects():
    v = classify(ForbidModel, Candidate({"x": 1, "zzz": 2}, "extra_field", "", "t"))
    assert v.outcome == "rejected"


def test_semantic_diff_datetime_clean():
    from datetime import datetime
    assert semantic_diff("2024-01-15T10:30:00", datetime(2024, 1, 15, 10, 30, 0)) == []


def test_semantic_diff_nested_drop():
    diffs = semantic_diff({"a": {"b": 1, "c": 2}}, {"a": {"b": 1}})
    assert any("DROPPED" in d for d in diffs)


# ------------------------------------------------------------- fuzzer + report

def test_fuzz_end_to_end():
    r = fuzz(LaxModel, max_candidates=300)
    assert r.n_candidates > 20
    assert len(r.misparsed) > 0  # lax model must show silent coercions
    rep = forge_report(r)
    assert any(f.severity == "silent_corruption" for f in rep.findings)
    md = rep.markdown()
    assert "schemaforge report" in md and "Fix:" in md


def test_valid_generators():
    spec = introspect(ToyModel)
    b = generate_valid(spec)
    assert classify(ToyModel, Candidate(b, "x", "", "t")).outcome == "clean"
    for seed in range(10):
        r = generate_valid_random(spec, seed=seed)
        v = classify(ToyModel, Candidate(r, "x", "", "t"))
        assert v.outcome == "clean", f"seed {seed}: {v.details}"


# ------------------------------------------------------------- repair

def test_repair_loop_fixes_broken():
    from pydantic import ValidationError

    class M(BaseModel):
        age: int = Field(ge=0, le=150)
        name: str = Field(min_length=2)

    broken = {"age": -5}  # missing name, age out of range

    def healer(prompt: str) -> str:
        import json
        assert "greater_than_equal" in prompt or "missing" in prompt
        return json.dumps({"age": 30, "name": "ada"})

    res = repair_loop(M, broken, healer, max_retries=3)
    assert res.success and res.value == {"age": 30, "name": "ada"}


def test_repair_loop_gives_up():
    class M(BaseModel):
        age: int = Field(ge=0)

    def bad_healer(prompt: str) -> str:
        return "not json at all"

    res = repair_loop(M, {"age": -1}, bad_healer, max_retries=2)
    assert not res.success


def test_compile_repair_prompt_content():
    class M(BaseModel):
        age: int = Field(ge=0)

    try:
        M.model_validate({"age": -3})
    except Exception as e:
        prompt = compile_repair_prompt(M, {"age": -3}, e)
    assert "age" in prompt and "greater_than_equal" in prompt and "-3" in prompt
