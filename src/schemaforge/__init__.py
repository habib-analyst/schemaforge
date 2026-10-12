"""schemaforge: adversarial stress-testing for Pydantic schemas, before deployment."""

from .introspect import FieldSpec, ModelSpec, introspect
from .fuzzer import Candidate, Fuzzer, fuzz
from .classifier import Verdict, classify, semantic_diff
from .report import ForgedReport, forge_report
from .repair import RepairResult, compile_repair_prompt, repair_loop
from .valid import generate_valid, generate_valid_random

__version__ = "0.1.0"
__all__ = [
    "FieldSpec", "ModelSpec", "introspect",
    "Candidate", "Fuzzer", "fuzz",
    "Verdict", "classify", "semantic_diff",
    "ForgedReport", "forge_report",
    "RepairResult", "compile_repair_prompt", "repair_loop",
    "generate_valid", "generate_valid_random",
]
