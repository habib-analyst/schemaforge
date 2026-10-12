"""schemaforge quickstart: fuzz a schema in 30 seconds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pydantic import BaseModel

from schemaforge import fuzz, forge_report

# 1. The schema your LLM must satisfy (note: NO bounds on age, NO email format)
class Signup(BaseModel):
    name: str
    age: int            # intent: 0..120 — not enforced!
    email: str          # intent: valid email — not enforced!


# 2. Forge it: generate adversarial candidates, classify every one
result = fuzz(Signup)
print(f"candidates={result.n_candidates} rejected={len(result.rejected)} "
      f"silently-misparsed={len(result.misparsed)} clean={len(result.clean)}")

# 3. Read the findings — silent corruption first
report = forge_report(result)
for f in report.findings[:6]:
    print(f"\n[{f.severity}] {f.mutation_class} -> {f.target or 'root'}")
    print(f"  simulates : {f.llm_behaviour}")
    print(f"  evidence  : {f.evidence[:100]}")
    print(f"  fix       : {f.fix[:100]}")

# 4. Full markdown report
Path = __import__("pathlib").Path
Path("forge_report.md").write_text(report.markdown())
print("\nFull report written to forge_report.md")
