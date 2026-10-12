"""Evaluation corpus: 20 realistic schemas with PLANTED weaknesses.

Each CorpusCase carries Weakness entries: an intent the schema fails to
enforce, the mutation classes expected to catch it, and an is_bad predicate
(original_input, parsed_dump) -> True when the weakness manifested.

3 cases are STRONG controls (no planted weaknesses): the fuzzer must find
zero silent misparses and zero is_bad hits on them, except explicitly
allowlisted documented coercions (control_allow).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

IsBad = Callable[[dict, dict], bool]


@dataclass
class Weakness:
    id: str
    description: str
    classes: list[str]
    is_bad: IsBad


@dataclass
class CorpusCase:
    name: str
    model: type[BaseModel]
    overrides: dict[str, Any] = dc_field(default_factory=dict)
    weaknesses: list[Weakness] = dc_field(default_factory=list)
    strong_control: bool = False
    control_allow: list[str] = dc_field(default_factory=list)  # substrings permitted in control misparse details


def _type_changed(*fields: str) -> IsBad:
    def _check(o: dict, d: dict) -> bool:
        for f in fields:
            if f in o and f in d and type(o[f]) is not type(d[f]):
                return True
        return False
    return _check


def _is_iso(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------ cases

class UserProfile(BaseModel):
    name: str
    age: int            # intent: 0..150 — NOT enforced
    email: str          # intent: valid email — NOT enforced


class ProductReview(BaseModel):
    title: str          # lax: int 5 coerces to "5" silently
    rating: int         # intent: 1..5 — NOT enforced
    verified: bool = False


class CardPayment(BaseModel):
    method: str = "card"
    card_number: str
    amount: float


class BankPayment(BaseModel):
    method: str = "bank"
    iban: str
    amount: float


class PaymentOrder(BaseModel):
    payment: Union[CardPayment, BankPayment]  # NOT discriminated; overlapping amount
    order_id: str


class SearchQuery(BaseModel):
    q: str
    limit: int = 10     # intent: 1..100 — NOT enforced
    offset: int = 0


class EmailMessage(BaseModel):
    to: str             # intent: valid email — NOT enforced
    subject: str
    body: str = ""


class GeoPoint(BaseModel):
    lat: float          # intent: -90..90 — NOT enforced
    lon: float          # intent: -180..180 — NOT enforced


from enum import Enum


class TaskStatus(str, Enum):
    TODO = "todo"
    DOING = "doing"
    DONE = "done"


class TaskSpec(BaseModel):  # STRONG control
    model_config = ConfigDict(extra="forbid")
    title: StrictStr
    status: TaskStatus
    due: datetime
    priority: StrictInt = Field(ge=1, le=5)


class NestedConfig(BaseModel):
    name: str
    settings: dict      # extra="ignore" default: hallucinated keys silently dropped


class ApiResponse(BaseModel):
    items: list[str]    # intent: non-empty — NOT enforced
    total: int


class StrictRecord(BaseModel):  # STRONG control
    model_config = ConfigDict(extra="forbid", strict=True)
    name: StrictStr
    count: StrictInt
    active: StrictBool
    score: float = Field(ge=0.0, le=1.0)


class MoneyTransfer(BaseModel):
    amount: float       # intent: positive Decimal — NOT enforced
    currency: str
    to_account: str


class TagList(BaseModel):
    tags: list[str]     # intent: <= 50 tags — NOT enforced


class Cat(BaseModel):
    pet_type: Literal["cat"]
    meows: bool


class Dog(BaseModel):
    pet_type: Literal["dog"]
    barks: bool


class ModelChoice(BaseModel):  # STRONG control
    model_config = ConfigDict(extra="forbid")
    pet: Union[Cat, Dog] = Field(discriminator="pet_type")


class LooseRecord(BaseModel):
    name: str           # lax: 42 -> "42"
    count: int          # lax: "42" -> 42
    active: bool        # lax: "true" -> True


class AddressForm(BaseModel):
    street: str
    city: str
    zip_code: str       # intent: US zip — NOT enforced


class SurveyAnswer(BaseModel):
    question_id: str
    answer: Union[int, str]  # ambiguous: "5" vs 5 both validate, different meaning


class EventLog(BaseModel):
    timestamp: str      # intent: ISO datetime — typed as bare str!
    level: str          # intent: enum — NOT enforced
    message: str


class Preferences(BaseModel):
    newsletter: bool    # lax: "yes" -> True silently
    dark_mode: bool


class FileUpload(BaseModel):
    filename: str
    size_bytes: int     # intent: <= 10_000_000 — NOT enforced
    content_type: str


class AgentAction(BaseModel):
    tool: str           # intent: one of {search, calculator, send_email, read_file}
    args: dict


KNOWN_TOOLS = {"search", "calculator", "send_email", "read_file"}


def build_corpus() -> list[CorpusCase]:
    return [
        CorpusCase("user_profile", UserProfile,
                   overrides={"email": "user@example.com"}, weaknesses=[
            Weakness("W01", "age unbounded (intent 0..150)", ["boundary"],
                     lambda o, d: not (0 <= d["age"] <= 150)),
            Weakness("W02", "email has no format (intent: valid email)",
                     ["malformed", "whitespace"],
                     lambda o, d: "@" not in d["email"]),
        ]),
        CorpusCase("product_review", ProductReview,
                   overrides={"rating": 4}, weaknesses=[
            Weakness("W03", "rating unbounded (intent 1..5)", ["boundary"],
                     lambda o, d: not (1 <= d["rating"] <= 5)),
            Weakness("W04", "title lax str coerces int->str silently",
                     ["coercion_trap", "type_confusion"], _type_changed("title")),
        ]),
        CorpusCase("payment_order", PaymentOrder, weaknesses=[
            Weakness("W05", "non-discriminated union silently drops non-matching arm fields",
                     ["union_ambiguity"],
                     lambda o, d: "iban" in o.get("payment", {}) and "iban" not in d.get("payment", {})),
        ]),
        CorpusCase("search_query", SearchQuery,
                   overrides={"limit": 10}, weaknesses=[
            Weakness("W06", "limit unbounded (intent 1..100)", ["boundary"],
                     lambda o, d: not (1 <= d["limit"] <= 100)),
        ]),
        CorpusCase("email_message", EmailMessage,
                   overrides={"to": "user@example.com"}, weaknesses=[
            Weakness("W07", "'to' has no email format", ["malformed", "whitespace"],
                     lambda o, d: "@" not in d["to"]),
        ]),
        CorpusCase("geo_point", GeoPoint, weaknesses=[
            Weakness("W08", "lat/lon unbounded (intent: -90..90 / -180..180)",
                     ["boundary"],
                     lambda o, d: not (-90 <= d["lat"] <= 90 and -180 <= d["lon"] <= 180)),
        ]),
        CorpusCase("task_spec", TaskSpec, strong_control=True,
                   control_allow=["datetime"]),  # pydantic datetimes accept unix timestamps by design
        CorpusCase("nested_config", NestedConfig, weaknesses=[
            Weakness("W09", "extra='ignore' default silently drops hallucinated keys",
                     ["extra_field"],
                     lambda o, d: "zzz_hallucinated" in o and "zzz_hallucinated" not in d),
        ]),
        CorpusCase("api_response", ApiResponse, weaknesses=[
            Weakness("W10", "items allows empty list (intent: non-empty)",
                     ["empty_container"], lambda o, d: d["items"] == []),
        ]),
        CorpusCase("strict_record", StrictRecord, strong_control=True),
        CorpusCase("money_transfer", MoneyTransfer, weaknesses=[
            Weakness("W11", "amount allows negatives (intent: positive)",
                     ["boundary"], lambda o, d: d["amount"] < 0),
            Weakness("W12", "amount lax float coerces numeric strings",
                     ["coercion_trap"], _type_changed("amount")),
        ]),
        CorpusCase("tag_list", TagList, weaknesses=[
            Weakness("W13", "tags unbounded (intent: <= 50)", ["huge_container"],
                     lambda o, d: len(d["tags"]) > 50),
        ]),
        CorpusCase("model_choice", ModelChoice, strong_control=True),
        CorpusCase("loose_record", LooseRecord, weaknesses=[
            Weakness("W14", "all fields lax: silent coercions (42->'42', '42'->42, 'true'->True)",
                     ["coercion_trap", "type_confusion"],
                     _type_changed("name", "count", "active")),
        ]),
        CorpusCase("address_form", AddressForm,
                   overrides={"zip_code": "12345"}, weaknesses=[
            Weakness("W15", "zip_code has no format (intent: US zip)",
                     ["malformed"],
                     lambda o, d: not d["zip_code"].replace("-", "").isdigit()),
        ]),
        CorpusCase("survey_answer", SurveyAnswer, weaknesses=[
            Weakness("W16", "Union[int, str] ambiguous: '5' vs 5 validate differently",
                     ["coercion_trap", "type_confusion"], _type_changed("answer")),
        ]),
        CorpusCase("event_log", EventLog,
                   overrides={"timestamp": "2024-01-15T10:30:00"}, weaknesses=[
            Weakness("W17", "timestamp typed as bare str; accepts 'yesterday'",
                     ["malformed"], lambda o, d: not _is_iso(d["timestamp"])),
        ]),
        CorpusCase("preferences", Preferences, weaknesses=[
            Weakness("W18", "lax bools coerce 'yes'->True silently",
                     ["coercion_trap"], _type_changed("newsletter", "dark_mode")),
        ]),
        CorpusCase("file_upload", FileUpload, weaknesses=[
            Weakness("W19", "size_bytes unbounded (intent <= 10MB)", ["boundary"],
                     lambda o, d: d["size_bytes"] > 10_000_000 or d["size_bytes"] < 0),
        ]),
        CorpusCase("agent_action", AgentAction,
                   overrides={"tool": "search"}, weaknesses=[
            Weakness("W20", "tool unrestricted (intent: known tool names)",
                     ["malformed"], lambda o, d: d["tool"] not in KNOWN_TOOLS),
        ]),
    ]
