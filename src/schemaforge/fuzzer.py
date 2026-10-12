"""Fuzzer engine: generate adversarial candidates, classify them, aggregate."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from .introspect import ModelSpec, introspect
from .mutators import MUTATORS, Candidate, LLM_BEHAVIOUR_MAP
from .classifier import Verdict, classify
from .valid import generate_valid


@dataclass
class FuzzResult:
    model_name: str
    n_candidates: int
    verdicts: list[Verdict] = dc_field(default_factory=list)

    @property
    def rejected(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.outcome == "rejected"]

    @property
    def misparsed(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.outcome == "silent_misparse"]

    @property
    def clean(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.outcome == "clean"]

    def by_class(self, mutation_class: str) -> list[Verdict]:
        return [v for v in self.verdicts if v.candidate.mutation_class == mutation_class]

    def kill_rate(self, mutation_class: str) -> float:
        """Fraction of candidates in a class that failed loudly or silently."""
        vs = self.by_class(mutation_class)
        if not vs:
            return 0.0
        return sum(1 for v in vs if v.outcome != "clean") / len(vs)


class Fuzzer:
    def __init__(self, model_cls: type,
                 overrides: dict[str, Any] | None = None,
                 mutators: list[Callable] | None = None,
                 max_candidates: int = 4000):
        self.spec: ModelSpec = introspect(model_cls)
        self.model_cls = model_cls
        self.overrides = overrides or {}
        self.mutators = mutators if mutators is not None else MUTATORS
        self.max_candidates = max_candidates

    def generate(self, baseline: dict[str, Any] | None = None) -> list[Candidate]:
        base = baseline if baseline is not None else generate_valid(self.spec, self.overrides)
        cands: list[Candidate] = []
        for mut in self.mutators:
            for c in mut(base, self.spec):
                cands.append(c)
                if len(cands) >= self.max_candidates:
                    return cands
        return cands

    def run(self, baseline: dict[str, Any] | None = None) -> FuzzResult:
        res = FuzzResult(model_name=self.spec.name, n_candidates=0)
        for c in self.generate(baseline):
            res.verdicts.append(classify(self.model_cls, c))
        res.n_candidates = len(res.verdicts)
        return res


def fuzz(model_cls: type, overrides: dict[str, Any] | None = None,
         baseline: dict[str, Any] | None = None,
         max_candidates: int = 4000) -> FuzzResult:
    """One-call fuzzing: introspect, mutate, classify."""
    return Fuzzer(model_cls, overrides=overrides,
                  max_candidates=max_candidates).run(baseline)
