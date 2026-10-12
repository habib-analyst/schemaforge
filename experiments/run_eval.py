"""schemaforge evaluation: planted-weakness kill rate, false positives, repair loop.

Kill-gate: fuzzer must detect >= 80% of planted weaknesses, else exit(1)
with an honest report (the invention dies; we do not ship a weak fuzzer).
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import ValidationError

from schemaforge import Fuzzer, classify, generate_valid, generate_valid_random, introspect
from schemaforge.mutators import Candidate, _variants_for
from schemaforge.repair import repair_loop
from schemaforge.valid import _valid_for
from corpus import build_corpus, CorpusCase

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
LOG = []


def log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def dump_of(case: CorpusCase, value: dict):
    return case.model.model_validate(value).model_dump(mode="python")


def check_weakness(case: CorpusCase, w, verdicts) -> tuple[bool, str]:
    """A weakness is detected if any candidate either misparses silently in
    one of the expected classes, or validates clean while violating intent."""
    for v in verdicts:
        if v.outcome == "silent_misparse" and v.candidate.mutation_class in w.classes:
            return True, f"silent_misparse via {v.candidate.mutation_class}: {v.details[:100]}"
    for v in verdicts:
        if v.outcome == "clean":
            try:
                d = dump_of(case, v.candidate.value)
            except ValidationError:
                continue
            try:
                if w.is_bad(v.candidate.value, d):
                    return True, (f"invalid accepted via {v.candidate.mutation_class} "
                                  f"({v.candidate.target}): is_bad fired")
            except Exception:
                continue
    return False, ""


def benign_variants(case: CorpusCase, baseline: dict) -> list[dict]:
    """Valid edge values that must NEVER be flagged (false-positive probe)."""
    spec = introspect(case.model)
    variants: list[dict] = []

    def _walk(fields, prefix):
        for f in fields:
            path = prefix + (f.name,)
            for v in _variants_for(f.schema):
                t = v.get("type")
                key = ".".join(path)
                if key in case.overrides:
                    continue
                if t in ("integer", "number") and isinstance(baseline.get(f.name), (int, float)):
                    for bound, bkey in (("minimum", "ge"), ("maximum", "le"),
                                        ("exclusiveMinimum", "gt"), ("exclusiveMaximum", "lt")):
                        if bound in v:
                            bv = v[bound] + (1 if bkey in ("gt",) and t == "integer" else
                                            (1e-9 if bkey in ("gt", "lt") else 0))
                            d = copy.deepcopy(baseline)
                            d[f.name] = bv
                            variants.append(d)
                if t == "string" and isinstance(baseline.get(f.name), str):
                    if v.get("minLength"):
                        d = copy.deepcopy(baseline)
                        d[f.name] = "x" * v["minLength"]
                        variants.append(d)
                    if v.get("maxLength"):
                        d = copy.deepcopy(baseline)
                        d[f.name] = "x" * v["maxLength"]
                        variants.append(d)
            _walk(f.nested, path)

    _walk(spec.fields, ())
    return variants


# ------------------------------------------------------------------ healers

def _valid_sample_for_fragment(frag: dict, path=(("heal",))):
    try:
        return _valid_for(frag, {}, path)
    except Exception:
        return "healed"


class MockHealer:
    """Rule-based fixer that reads ONLY the compiled repair prompt.

    This measures the prompt's information content: every fix decision is
    derived from the error lines (field path + rule + received value) and the
    embedded schema. A competent LLM with the same prompt should do at least
    this well.
    """

    def __init__(self, model_cls):
        self.model_cls = model_cls
        self.schema = model_cls.model_json_schema()

    def _frag(self, dotted: str):
        node = self.schema
        for part in dotted.split("."):
            node = node.get("properties", {}).get(part, {})
        return node

    def __call__(self, prompt: str) -> str:
        m = re.search(r"Your broken output:\n(\{.*?\n\})\n\nExact failures",
                      prompt, re.S)
        if not m:
            raise ValueError("could not extract broken JSON from prompt")
        value = json.loads(m.group(1))
        for line in re.findall(r"^- field `([^`]+)`: .*?\(rule: ([^;]+);", prompt, re.M):
            loc, rule = line
            frag = self._frag(loc)
            cur = value
            parts = loc.split(".")
            for p in parts[:-1]:
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            leaf = parts[-1]
            if "missing" in rule:
                cur[leaf] = _valid_sample_for_fragment(frag)
            elif "extra_forbidden" in rule:
                cur.pop(leaf, None)
            elif rule in ("greater_than_equal", "greater_than"):
                bound = frag.get("minimum", frag.get("exclusiveMinimum", 0))
                cur[leaf] = bound + (1 if isinstance(bound, int) else 1e-9)
            elif rule in ("less_than_equal", "less_than"):
                bound = frag.get("maximum", frag.get("exclusiveMaximum", 100))
                cur[leaf] = bound - (1 if isinstance(bound, int) else 1e-9)
            elif "enum" in rule:
                enum_vals = frag.get("enum", ["healed"])
                cur[leaf] = enum_vals[0]
            else:
                # wrong type / pattern / union / datetime: replace with valid sample
                cur[leaf] = _valid_sample_for_fragment(frag)
        return json.dumps(value)


class MockNaiveHealer:
    """Baseline: schema-aware but ERROR-BLIND. Knows the schema, never sees
    which fields failed. Fills missing required fields and strips unknown
    keys, but cannot fix wrong values it cannot identify."""

    def __init__(self, model_cls):
        self.model_cls = model_cls
        self.schema = model_cls.model_json_schema()

    def __call__(self, prompt: str) -> str:
        m = re.search(r"Your broken output:\n(\{.*?\n\})\n\nExact failures",
                      prompt, re.S)
        value = json.loads(m.group(1)) if m else {}
        props = self.schema.get("properties", {})
        required = set(self.schema.get("required", []))
        # strip unknown top-level keys, fill missing required
        for k in list(value.keys()):
            if k not in props:
                del value[k]
        for k in required:
            if k not in value:
                value[k] = _valid_sample_for_fragment(props.get(k, {}))
        return json.dumps(value)


def main() -> None:
    t0 = time.time()
    corpus = build_corpus()
    total_w, detected_w = 0, 0
    per_case = {}
    fp_total, fp_flagged, intent_violations = 0, 0, 0
    repair_tests = []  # (case, broken_value)

    for case in corpus:
        fz = Fuzzer(case.model, overrides=case.overrides)
        baseline = generate_valid(fz.spec, case.overrides)
        # baseline must be clean and trigger no weakness
        bv = classify(case.model, Candidate(baseline, "baseline", "", "valid baseline"))
        assert bv.outcome == "clean", f"{case.name}: baseline not clean: {bv.details}"
        for w in case.weaknesses:
            assert not w.is_bad(baseline, dump_of(case, baseline)), \
                f"{case.name}/{w.id}: is_bad fires on valid baseline!"

        result = fz.run(baseline)
        log(f"[{case.name}] candidates={result.n_candidates} "
            f"rejected={len(result.rejected)} misparsed={len(result.misparsed)} "
            f"clean={len(result.clean)}")

        case_det = []
        for w in case.weaknesses:
            total_w += 1
            ok, how = check_weakness(case, w, result.verdicts)
            case_det.append({"id": w.id, "description": w.description,
                             "detected": ok, "how": how})
            if ok:
                detected_w += 1
                log(f"  ✓ {w.id} DETECTED: {how[:110]}")
            else:
                log(f"  ✗ {w.id} MISSED: {w.description}")
        if case.strong_control:
            bad = [v for v in result.verdicts
                   if v.outcome == "silent_misparse"
                   and not any(a in v.details for a in case.control_allow)]
            log(f"  [control] silent misparses (excl. allowlisted): {len(bad)} (must be 0)")
            assert not bad, f"{case.name}: control leaked misparses!"

        # false positives: benign valid variants must all be clean
        for bvar in benign_variants(case, baseline):
            fp_total += 1
            vv = classify(case.model, Candidate(bvar, "benign", "", "benign valid variant"))
            if vv.outcome != "clean":
                fp_flagged += 1
                log(f"  FP FLAG: {vv.details[:100]}")
        # false positives: randomized valid instances must all be clean.
        # NOTE: random schema-valid inputs may still violate a planted intent
        # (e.g. lat=95 with no bounds) — those are counted separately as
        # intent violations (they DEMONSTRATE the weakness), not as FPs.
        # A false positive is the classifier flagging an intent-valid input.
        for seed in range(25):
            try:
                rvar = generate_valid_random(fz.spec, case.overrides, seed=seed)
            except ValueError:
                break
            fp_total += 1
            vv = classify(case.model, Candidate(rvar, "benign", "", f"random valid seed={seed}"))
            if vv.outcome != "clean":
                fp_flagged += 1
                log(f"  FP FLAG (seed {seed}): {vv.details[:120]}")
                continue
            try:
                d = dump_of(case, rvar)
            except ValidationError:
                fp_flagged += 1
                log(f"  FP FLAG (seed {seed}): random valid failed validation!")
                continue
            for w in case.weaknesses:
                try:
                    if w.is_bad(rvar, d):
                        intent_violations += 1
                except Exception:
                    pass

        # collect rejected candidates for repair measurement
        for v in result.rejected[:3]:
            repair_tests.append((case, v.candidate.value))
        per_case[case.name] = {
            "candidates": result.n_candidates,
            "rejected": len(result.rejected),
            "misparsed": len(result.misparsed),
            "clean": len(result.clean),
            "weaknesses": case_det,
            "strong_control": case.strong_control,
        }

    kill_rate = detected_w / total_w if total_w else 0.0
    fp_rate = fp_flagged / fp_total if fp_total else 0.0
    log(f"\nKILL RATE: {detected_w}/{total_w} = {kill_rate:.3f} (gate: >= 0.80)")
    log(f"FALSE POSITIVE RATE (classifier): {fp_flagged}/{fp_total} = {fp_rate:.4f}")
    log(f"INTENT VIOLATIONS in random schema-valid inputs: {intent_violations}/{fp_total} "
        f"(these demonstrate the planted weaknesses, not FPs)")

    # ---- repair loop measurement
    log(f"\nREPAIR LOOP: {len(repair_tests)} broken outputs, max_retries=3")
    smart_ok, naive_ok = 0, 0
    for case, broken in repair_tests:
        r1 = repair_loop(case.model, copy.deepcopy(broken),
                         MockHealer(case.model), max_retries=3)
        r2 = repair_loop(case.model, copy.deepcopy(broken),
                         MockNaiveHealer(case.model), max_retries=3)
        smart_ok += r1.success
        naive_ok += r2.success
    n = len(repair_tests)
    log(f"  compiled-prompt repair success: {smart_ok}/{n} = {smart_ok/n:.3f}")
    log(f"  naive-prompt repair success:    {naive_ok}/{n} = {naive_ok/n:.3f}")

    summary = {
        "kill_rate": kill_rate,
        "weaknesses_detected": detected_w,
        "weaknesses_total": total_w,
        "false_positive_rate": fp_rate,
        "fp_flagged": fp_flagged,
        "fp_total": fp_total,
        "intent_violations": intent_violations,
        "repair_compiled": smart_ok / n if n else 0,
        "repair_naive": naive_ok / n if n else 0,
        "repair_n": n,
        "elapsed_s": round(time.time() - t0, 1),
        "cases": per_case,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    (RESULTS / "eval.log").write_text("\n".join(LOG))
    log(f"\nWrote results/summary.json + eval.log in {time.time()-t0:.1f}s")

    if kill_rate < 0.80:
        log("KILL-GATE FAILED: fuzzer too weak to ship. Honest negative recorded.")
        sys.exit(1)
    log("KILL-GATE PASSED.")


if __name__ == "__main__":
    main()
