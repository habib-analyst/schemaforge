"""Forged reports: turn raw verdicts into an actionable pre-deployment report.

For each mutation class the report shows what broke, maps it to the real LLM
behaviour it simulates, and emits a concrete schema fix. Fixes are ordered by
severity: silent misparses first (they corrupt data without raising), then
invalid-accepted, then correctly-rejected (informational only).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from .fuzzer import FuzzResult
from .mutators import LLM_BEHAVIOUR_MAP


@dataclass
class Finding:
    severity: str          # "silent_corruption" | "invalid_accepted" | "ok_rejected"
    mutation_class: str
    target: str
    llm_behaviour: str
    evidence: str
    fix: str


@dataclass
class ForgedReport:
    model_name: str
    n_candidates: int
    n_rejected: int
    n_misparsed: int
    n_clean: int
    findings: list[Finding] = dc_field(default_factory=list)
    class_summary: dict[str, dict[str, int]] = dc_field(default_factory=dict)

    def markdown(self) -> str:
        lines = [
            f"# schemaforge report: {self.model_name}",
            "",
            f"Candidates: {self.n_candidates} — rejected: {self.n_rejected}, "
            f"silently misparsed: {self.n_misparsed}, clean: {self.n_clean}",
            "",
            "## Findings (silent corruption first)",
            "",
        ]
        for f in self.findings:
            lines += [
                f"### [{f.severity}] {f.mutation_class} → {f.target or 'root'}",
                f"- Simulates: {f.llm_behaviour}",
                f"- Evidence: {f.evidence}",
                f"- Fix: {f.fix}",
                "",
            ]
        if not self.findings:
            lines.append("No weaknesses found. The schema survived the forge.")
        return "\n".join(lines)


_FIX_HINTS = {
    "type_confusion": "use StrictStr/StrictInt/StrictBool (or model_config strict=True) so wrong types raise instead of coercing",
    "coercion_trap": "use Strict types; lax coercion is silent data corruption",
    "extra_field": "set model_config = ConfigDict(extra='forbid') so hallucinated keys raise instead of being silently dropped",
    "boundary": "add Field(ge=..., le=...) / minLength/maxLength so out-of-range values raise",
    "null_field": "if null is never valid, keep the field non-Optional and add a validator with a clear error",
    "whitespace": "add strip_whitespace=True or a cleaning validator before validation",
    "empty_container": "add minLength/minItems so empty payloads raise",
    "enum_violation": "enum correctly rejected — no fix needed (informational)",
    "pattern_violation": "pattern correctly rejected — no fix needed (informational)",
    "missing_required": "required field correctly rejected — no fix needed (informational)",
    "union_ambiguity": "use a discriminated union (Field(discriminator=...)) so ambiguous payloads raise",
    "unicode": "decide explicitly: normalize (NFKC) on input or reject non-ASCII with a pattern",
    "deep_nesting": "add max-depth validation or restructure to avoid unbounded recursion",
    "huge_container": "add maxLength/maxItems caps to bound payload size",
}


def forge_report(result: FuzzResult) -> ForgedReport:
    rep = ForgedReport(
        model_name=result.model_name,
        n_candidates=result.n_candidates,
        n_rejected=len(result.rejected),
        n_misparsed=len(result.misparsed),
        n_clean=len(result.clean),
    )
    seen: set[tuple[str, str, str]] = set()
    for v in result.verdicts:
        mc = v.candidate.mutation_class
        key = (mc, v.candidate.target, v.outcome)
        if key in seen:
            continue
        seen.add(key)
        llm = LLM_BEHAVIOUR_MAP.get(mc, "")
        if v.outcome == "silent_misparse":
            rep.findings.append(Finding(
                severity="silent_corruption", mutation_class=mc,
                target=v.candidate.target, llm_behaviour=llm,
                evidence=v.details, fix=_FIX_HINTS.get(mc, "tighten the schema"),
            ))
        elif v.outcome == "rejected" and mc in ("boundary", "union_ambiguity",
                                                "deep_nesting", "huge_container",
                                                "null_field", "empty_container"):
            # rejected here can mean "missing constraint that should exist" only
            # when the input was actually invalid per intent; report as info.
            # (Weakness confirmation for these comes from planted is_bad checks.)
            rep.findings.append(Finding(
                severity="ok_rejected", mutation_class=mc,
                target=v.candidate.target, llm_behaviour=llm,
                evidence=v.details, fix=_FIX_HINTS.get(mc, ""),
            ))
    # per-class summary
    for v in result.verdicts:
        s = rep.class_summary.setdefault(v.candidate.mutation_class,
                                         {"rejected": 0, "silent_misparse": 0, "clean": 0})
        s[v.outcome] += 1
    # severity order
    order = {"silent_corruption": 0, "invalid_accepted": 1, "ok_rejected": 2}
    rep.findings.sort(key=lambda f: order[f.severity])
    return rep
