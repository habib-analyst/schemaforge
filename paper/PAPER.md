# schemaforge: Adversarial Stress-Testing for LLM Output Schemas, Before Deployment

**Habib Ur Rehman** · 2026-10-12

## Abstract

Everyone validates LLM structured outputs at runtime. Nobody stress-tests the
schema itself before deployment. We present **schemaforge**, a fuzzer that
takes a Pydantic model and generates adversarial candidate outputs — boundary
values, type confusions, missing fields, hallucinated keys, unicode tricks,
lax-coercion traps — then classifies each as **rejected** (would crash),
**silently misparsed** (validates but corrupts data semantics), or **clean**.
On a corpus of 20 realistic schemas with 20 planted weaknesses, schemaforge
detects **20/20 (100%)** with a **0/504 false-positive rate**, and its
counterexample-guided repair prompts achieve **90% repair success** vs 6.7%
for error-blind prompts. The headline finding: pydantic's lax mode silently
corrupts data far more often than it rejects it — and every silent corruption
is a production incident that no try/except will ever catch.

## 1. The gap

Structured-output tooling (instructor, pydantic-ai, LangChain's
`with_structured_output`, OpenAI strict mode) solves the *generation* side:
constrain the LLM, validate at runtime, retry on parse errors. All of it
assumes the schema is correct. It rarely is. A missing `Field(ge=0)`, a lax
`int` instead of `StrictInt`, the default `extra="ignore"` — each is a hole
that runtime validation cannot see, because the output *validates*. The
failure mode is not a crash; it is silent data corruption: `"42"` becomes
`42`, a hallucinated key is dropped, `True` becomes `1`.

Property-based testing (Hypothesis) generates *valid* inputs to test code.
Mutation testing mutates *code* to test suites. Nothing mutates *outputs*
against *schemas* to test the contract between an LLM and its consumer.
That is the gap schemaforge fills: **adversarial output fuzzing, pre-deployment**.

## 2. Method

**Introspection.** From `model_json_schema()` with `$ref`s resolved, we build a
uniform field spec: type, constraints (min/max, length, pattern, enum),
required-ness, nested structure, union variants.

**15 mutators**, each mapped to a real LLM output behaviour:

| mutator | simulates |
|---|---|
| boundary | values at/over declared limits |
| type_confusion | numbers as strings, `"yes"` as bool |
| missing_required | LLM omits an "obvious" field |
| extra_field | LLM hallucinates extra keys |
| null_field | LLM emits null for unknowns |
| whitespace / unicode | padding, zero-width chars, emoji |
| empty_container | `[]` instead of data |
| enum_violation / pattern_violation | invented enum values, malformed formats |
| malformed | plausible-but-wrong strings where no pattern exists |
| union_ambiguity | values matching no (or several) union arms |
| coercion_trap | values pydantic lax-mode silently coerces |
| deep_nesting / huge_container | pathological shapes |

**Classification.** Each candidate is validated. `ValidationError` →
*rejected*. Otherwise a type-strict recursive `semantic_diff` compares input
to parsed output: any type change, dropped key, or value change →
*silent_misparse*. Datetime-from-ISO, UUID-from-string, and Enum-from-value
are allowlisted as semantics-preserving parses.

**Reporting.** Findings ordered by severity (silent corruption first), each
with the simulated LLM behaviour, evidence, and a concrete schema fix
(e.g. `StrictInt`, `extra="forbid"`, discriminated unions).

**Repair (supporting).** `compile_repair_prompt` turns a live
`ValidationError` into a targeted prompt: schema + broken value + every
failure with exact location, rule, and offending input. `repair_loop` retries
within a budget.

## 3. Evaluation

Corpus: 20 realistic schemas (nested, unions, enums, datetimes, constrained
fields), 20 planted weaknesses (unbounded numerics, missing formats, lax
types, `extra="ignore"`, non-discriminated unions, bare-`str` timestamps),
plus 3 strong controls (strict-everything, discriminated union, constrained
model). Each weakness carries an `is_bad` intent predicate; detection =
silent misparse in an expected class OR clean validation that violates intent.

| metric | result |
|---|---|
| planted-weakness kill rate | **20/20 = 1.000** (gate: ≥ 0.80) |
| classifier false-positive rate | **0/504 = 0.0000** (504 randomized valid inputs) |
| strong controls: silent misparses | 0 (excl. documented datetime-from-timestamp coercion) |
| total candidates generated | 1,054 (426 rejected, 171 silently misparsed) |
| repair success, compiled prompt | **54/60 = 0.900** (3 retries, rule-based fixer reading only the prompt) |
| repair success, error-blind prompt | 4/60 = 0.067 |

The repair comparison is the honest headline for the supporting feature: the
compiled error prompt carries ~13× more repair power than "fix this JSON".

## 4. Limitations (reported, not buried)

- The fuzzer tests the schema, not the LLM: it proves a *class* of output
  would corrupt, not that your model emits it. Pair with production sampling.
- `semantic_diff` is type-strict by design; exotic-but-valid coercions
  (custom validators, `BeforeValidator` normalizers) may flag as misparses.
  Allowlist them per schema.
- Repair-loop numbers use a rule-based fixer, not a live LLM: they measure
  the prompt's *information content*, a lower bound on what a competent LLM
  achieves.
- Fuzzing is only as good as the baseline: highly conditional schemas
  (cross-field validators) need hand-written baselines via `overrides`.

## 5. Conclusion

Runtime validation tells you an output failed. schemaforge tells you, before
you ship, which outputs your schema will *silently corrupt*. The 20/20 kill
rate with zero false positives says the approach works; the repair numbers
say the error compilation is worth stealing even if you never fuzz. Test the
contract, not just the output.

---
*Code: github.com/habib-analyst/schemaforge · MIT License*
