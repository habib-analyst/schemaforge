# schemaforge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/schemaforge?style=social)](https://github.com/habib-analyst/schemaforge)

**Fuzz your Pydantic schemas before your LLM finds the holes.** schemaforge generates adversarial outputs — boundary values, type confusions, hallucinated keys, lax-coercion traps — and tells you which ones your schema *silently corrupts* instead of rejecting.

## Quickstart

```bash
pip install schemaforge
```

```python
from pydantic import BaseModel
from schemaforge import fuzz, forge_report

class Signup(BaseModel):
    name: str
    age: int        # intent: 0..120 — not enforced!
    email: str      # intent: valid email — not enforced!

result = fuzz(Signup)
print(f"candidates={result.n_candidates} rejected={len(result.rejected)} "
      f"silently-misparsed={len(result.misparsed)}")

report = forge_report(result)
for f in report.findings[:3]:
    print(f"[{f.severity}] {f.mutation_class} -> {f.target or 'root'}")
    print(f"  fix: {f.fix[:90]}")
```

Real output:

```
candidates=72 rejected=24 silently-misparsed=10
[silent_corruption] type_confusion -> age
  fix: use StrictStr/StrictInt/StrictBool (or model_config strict=True) so wrong types raise instead of coe
[silent_corruption] extra_field -> root
  fix: set model_config = ConfigDict(extra='forbid') so hallucinated keys raise instead of being silently d
[silent_corruption] coercion_trap -> age
  fix: use Strict types; lax coercion is silent data corruption
```

## Measured results

20-schema corpus, 20 planted weaknesses, 3 strong controls. All numbers measured on CPU, reproducible via `experiments/run_eval.py`.

| metric | result |
|---|---|
| planted-weakness kill rate | **20/20 = 1.000** (kill-gate: ≥ 0.80) |
| classifier false-positive rate | **0/504** (504 randomized valid inputs) |
| repair success, compiled error prompt | **54/60 = 0.900** (3 retries) |
| repair success, error-blind prompt | 4/60 = 0.067 |
| unit tests | 17/17 green |

The scariest column: of 1,054 adversarial candidates, **171 silently misparsed** — validated fine while corrupting data (`"42"`→`42`, dropped keys, `True`→`1`). None of those would ever raise in production.

## How it works

1. **Introspect** — resolve the model's JSON schema into fuzzable field specs (types, constraints, unions, nesting).
2. **Mutate** — 15 adversarial mutators fuzz *at* the schema's own boundaries, each mapped to a real LLM behaviour (numbers-as-strings, hallucinated keys, null-for-unknown…).
3. **Classify** — every candidate is validated and triaged: `rejected` (would crash), `silent_misparse` (validates but corrupts — the killer class), `clean`.
4. **Report** — findings ordered by severity with evidence and a concrete schema fix.
5. **Repair** (supporting) — `compile_repair_prompt` turns a live `ValidationError` into a targeted fix prompt; `repair_loop` retries within a budget.

```
LLM output ──► [15 mutators] ──► candidates ──► validate + semantic_diff
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                   ▼                   ▼
                                rejected        silent_misparse            clean
                             (would crash)   (corrupts silently)      (truly fine)
```

See [`paper/PAPER.md`](paper/PAPER.md) for the gap analysis, method, and honest limitations.

## Citation

```bibtex
@software{rehman2026schemaforge,
  author  = {Habib Ur Rehman},
  title   = {schemaforge: Adversarial Stress-Testing for LLM Output Schemas},
  year    = {2026},
  url     = {https://github.com/habib-analyst/schemaforge}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
