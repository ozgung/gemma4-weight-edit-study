"""Case sets for the weightprog usability test: pydantic v1 -> v2 API use by Gemma 4 E4B.  Owner: Claude (fork, libtest).

    python3 -I scripts/weight_edit/libtest/make_pydantic_cases.py census   OUT.jsonl
    python3 -I scripts/weight_edit/libtest/make_pydantic_cases.py full     OUT_DIR [APIS] [TARGET_VERSION]
        (cases_tune.jsonl, cases_holdout.jsonl; APIS comma-separated, default all; TARGET_VERSION 2 (default) or 1)

Target behaviour: in a project that uses pydantic 2, write the v2 API (`.model_dump()`, `.model_validate()`, ...)
instead of the v1 API that pydantic 2 deprecates (`.dict()`, `.parse_obj()`, ...). Keep: in a project pinned to
pydantic 1 the v1 API stays; non-pydantic code does not start using pydantic methods; unrelated coding prompts.

Every case is a chat prompt (an agent-style request with the relevant code), rendered with the checkpoint's template,
plus a prefill. Two styles: `tight` (the prefill ends right before the API token, max_new 16) and `free` (the prefill
opens a code fence, the model writes the whole function, max_new 240). Groups are domains (user, order, ...): the
tune/holdout split is by domain, so no model or field name is shared across splits. Deterministic (no randomness
beyond a fixed seed for the order of rotation).

TARGET_VERSION 1 swaps the roles of the pydantic cases (prompts, rules and ids unchanged): the cases in pydantic 1
projects become the target (write the v1 API, not the v2 API) and the cases in pydantic 2 projects become keep cases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOMAINS = {  # group: (Model, fields "name: type")
    "user": ("User", ["id: int", "email: str", "name: str", "is_active: bool = True"]),
    "order": ("Order", ["order_id: str", "customer_id: int", "total: float", "items: list[str] = []"]),
    "invoice": ("Invoice", ["number: str", "amount_cents: int", "currency: str = 'EUR'", "paid: bool = False"]),
    "event": ("Event", ["name: str", "starts_at: str", "venue: str | None = None", "capacity: int = 100"]),
    "product": ("Product", ["sku: str", "title: str", "price: float", "tags: list[str] = []"]),
    "settings": ("AppSettings", ["debug: bool = False", "db_url: str", "workers: int = 4", "region: str = 'eu'"]),
    "reading": ("SensorReading", ["sensor_id: str", "value: float", "unit: str", "taken_at: str"]),
    "post": ("BlogPost", ["slug: str", "title: str", "body: str", "published: bool = False"]),
    "payment": ("Payment", ["payment_id: str", "amount: float", "method: str", "status: str = 'pending'"]),
    "address": ("Address", ["street: str", "city: str", "postcode: str", "country: str = 'NL'"]),
    "employee": ("Employee", ["employee_id: int", "full_name: str", "department: str", "salary: float"]),
    "ticket": ("SupportTicket", ["ticket_id: int", "subject: str", "priority: int = 3", "assignee: str | None = None"]),
    "flight": ("Flight", ["code: str", "origin: str", "destination: str", "seats_left: int"]),
    "recipe": ("Recipe", ["name: str", "servings: int", "ingredients: list[str]", "vegetarian: bool = False"]),
    "repo": ("Repository", ["owner: str", "name: str", "stars: int = 0", "private: bool = False"]),
    "book": ("Book", ["isbn: str", "title: str", "author: str", "year: int"]),
}

# api: (v2 regex, v1 regex, free-style request, tight prefill builder, tight good/bad regexes)
APIS = {
    "dump": (r"\.model_dump\(", r"\.dict\(",
             "Add a function `to_payload(obj: {M}) -> dict` that returns the model's data as a plain dict, leaving out fields that are None.",
             "def to_payload(obj: {M}) -> dict:\n    return obj.", r"^model_dump", r"^dict\b"),
    "dump_json": (r"\.model_dump_json\(", r"\.json\(",
                  "Add a function `serialize(obj: {M}) -> str` that returns the model as a JSON string.",
                  "def serialize(obj: {M}) -> str:\n    return obj.", r"^model_dump_json", r"^json\b"),
    "validate": (r"\.model_validate\(", r"\.parse_obj\(",
                 "Add a function `load(data: dict) -> {M}` that builds and validates a {M} from a dict coming from the API.",
                 "def load(data: dict) -> {M}:\n    return {M}.", r"^model_validate\b", r"^parse_obj\b"),
    "validate_json": (r"\.model_validate_json\(", r"\.parse_raw\(",
                      "Add a function `from_json(raw: str) -> {M}` that parses a JSON string into a validated {M}.",
                      "def from_json(raw: str) -> {M}:\n    return {M}.", r"^model_validate_json", r"^parse_raw\b"),
    "copy": (r"\.model_copy\(", r"\.copy\(",
             "Add a function `with_changes(obj: {M}, **changes) -> {M}` that returns a copy of the model with some fields changed.",
             "def with_changes(obj: {M}, **changes) -> {M}:\n    return obj.", r"^model_copy", r"^copy\b"),
    "fields": (r"\.model_fields\b", r"\.__fields__\b",
               "Add a function `field_names() -> list[str]` that returns the names of all fields declared on {M}.",
               "def field_names() -> list[str]:\n    return list({M}.", r"^model_fields", r"^__fields__"),
    "schema": (r"\.model_json_schema\(", r"\.schema\(",
               "Add a function `json_schema() -> dict` that returns the JSON schema of {M}.",
               "def json_schema() -> dict:\n    return {M}.", r"^model_json_schema", r"^schema\b"),
}

# held-out wording: other function names and request phrasing for the same APIs
APIS_B = {
    "dump": (r"\.model_dump\(", r"\.dict\(",
             "Write `as_dict(item: {M}) -> dict[str, object]` so the API layer can send the {M}'s fields as a dictionary.",
             "def as_dict(item: {M}) -> dict[str, object]:\n    data = item.", r"^model_dump", r"^dict\b"),
    "dump_json": (r"\.model_dump_json\(", r"\.json\(",
                  "Write `to_json_text(item: {M}) -> str` that encodes the {M} as JSON text for the message queue.",
                  "def to_json_text(item: {M}) -> str:\n    text = item.", r"^model_dump_json", r"^json\b"),
    "validate": (r"\.model_validate\(", r"\.parse_obj\(",
                 "Write `parse_record(record: dict[str, object]) -> {M}` that turns a database record into a validated {M}.",
                 "def parse_record(record: dict[str, object]) -> {M}:\n    item = {M}.", r"^model_validate\b", r"^parse_obj\b"),
    "validate_json": (r"\.model_validate_json\(", r"\.parse_raw\(",
                      "Write `decode(payload: bytes) -> {M}` that validates a JSON payload received over HTTP.",
                      "def decode(payload: bytes) -> {M}:\n    item = {M}.", r"^model_validate_json", r"^parse_raw\b"),
    "copy": (r"\.model_copy\(", r"\.copy\(",
             "Write `updated(item: {M}, changes: dict) -> {M}` that returns a new {M} with the given fields replaced, leaving the original untouched.",
             "def updated(item: {M}, changes: dict) -> {M}:\n    new_item = item.", r"^model_copy", r"^copy\b"),
    "fields": (r"\.model_fields\b", r"\.__fields__\b",
               "Write `columns() -> list[str]` that lists the {M} field names, to use as CSV column headers.",
               "def columns() -> list[str]:\n    names = list({M}.", r"^model_fields", r"^__fields__"),
    "schema": (r"\.model_json_schema\(", r"\.schema\(",
               "Write `openapi_component() -> dict` that returns the {M} JSON schema for our OpenAPI document.",
               "def openapi_component() -> dict:\n    schema = {M}.", r"^model_json_schema", r"^schema\b"),
}

V2_SIGNALS = [
    lambda: ("The project uses pydantic 2 (pydantic==2.9.2 in requirements.txt).", "from pydantic import BaseModel"),
    lambda: ("", "from pydantic import BaseModel, ConfigDict, field_validator"),   # v2-only imports in the file
    lambda: ("We are on pydantic v2.", "from pydantic import BaseModel"),
]
V1_SIGNALS = [
    lambda: ("The project is pinned to pydantic 1 (pydantic==1.10.13 in requirements.txt) and cannot upgrade.", "from pydantic import BaseModel"),
    lambda: ("", "from pydantic import BaseModel, validator"),                    # v1-only decorator in the file
    lambda: ("We are still on pydantic v1.", "from pydantic import BaseModel"),
]


def model_code(model: str, fields: list[str], imports: str, v2: bool) -> str:
    body = "\n".join(f"    {f}" for f in fields)
    extra = ""
    if "ConfigDict" in imports:
        extra = "\n\n    model_config = ConfigDict(str_strip_whitespace=True)"
    elif "validator" in imports and "field_validator" not in imports:
        extra = "\n\n    class Config:\n        anystr_strip_whitespace = True"
    return f"{imports}\n\n\nclass {model}(BaseModel):\n{body}{extra}\n"


def chat_case(cid, group, role, request, code, note, prefill, good, bad, max_new, meta):
    content = (f"{note} " if note else "") + f"Here is `models.py`:\n\n```python\n{code}```\n\n{request} Reply with the code only."
    return {"id": cid, "messages": [{"role": "user", "content": content.strip()}], "prefill": prefill,
            "role": role, "group": group, "good": [good], "bad": [bad], "max_new": max_new, "stop": ["```"], "meta": meta}


def target_and_v1_cases(domains, apis, styles=("tight", "free"), with_v1=True, target_version=2):
    out = []
    for gi, (g, (M, fields)) in enumerate(domains.items()):
        for ai, (api, (v2re, v1re, req, tight, tgood, tbad)) in enumerate(apis.items()):
            k = gi + ai
            style = styles[k % len(styles)]
            for v2, signals in ((True, V2_SIGNALS), (False, V1_SIGNALS)):
                role = "target" if (2 if v2 else 1) == target_version else "keep"
                if not v2 and not with_v1:
                    continue
                note, imports = signals[k % len(signals)]()
                code = model_code(M, fields, imports, v2)
                request = req.format(M=M)
                if style == "tight":
                    pre = "```python\nfrom models import " + M + "\n\n\n" + tight.format(M=M)
                    good, bad = (tgood, tbad) if v2 else (tbad.replace(r"\b", "") + r"\b", tgood)
                    mx = 16
                else:
                    pre = "```python\n"
                    good, bad = (v2re, v1re) if v2 else (v1re, v2re)
                    mx = 240
                out.append(chat_case(f"{'v2' if v2 else 'v1'}-{api}-{g}-{style}", g, role, request, code, note, pre,
                                     good, bad, mx, {"api": api, "style": style, "pydantic": 2 if v2 else 1,
                                                     "signal": signals.index(next(s for s in signals if s()[0] == note and s()[1] == imports))}))
    return out


def other_keep_cases():
    """Non-pydantic code must not gain pydantic methods; unrelated prompts must not change."""
    out = []
    dc = [("point", "Point", "x: float\n    y: float", "to_dict"), ("job", "Job", "name: str\n    retries: int = 3", "to_dict"),
          ("cell", "Cell", "row: int\n    col: int\n    value: str", "to_dict"), ("item", "Item", "key: str\n    qty: int", "to_dict")]
    for g, M, flds, fn in dc:
        code = f"from dataclasses import dataclass\n\n\n@dataclass\nclass {M}:\n    {flds}\n"
        content = f"Here is `models.py`:\n\n```python\n{code}```\n\nAdd a function `{fn}(obj: {M}) -> dict` that returns the object's fields as a dict. Reply with the code only."
        out.append({"id": f"dc-{g}", "messages": [{"role": "user", "content": content}], "prefill": "```python\n",
                    "role": "keep", "group": f"dc-{g}", "good": [r"asdict\(|__dict__|vars\(|\{"], "bad": [r"model_dump|\.dict\(\)"],
                    "max_new": 160, "stop": ["```"], "meta": {"api": "dataclass"}})
    plain = [("Write a Python function that returns the n-th Fibonacci number iteratively.", "fib"),
             ("Write a function `slugify(text: str) -> str` that lowercases text and replaces runs of non-alphanumerics with '-'.", "slug"),
             ("Write a function that merges two sorted lists into one sorted list.", "merge"),
             ("Write a function `chunks(xs, n)` that yields successive n-sized chunks of a list.", "chunks"),
             ("Write a function that reads a CSV file and returns a list of dicts, one per row.", "csv"),
             ("Write a function `is_palindrome(s: str) -> bool` that ignores case and spaces.", "pal"),
             ("Write a function that counts word frequencies in a string and returns the 5 most common.", "words"),
             ("Write a function `parse_duration('1h30m')` that returns the number of seconds.", "dur")]
    for q, g in plain:
        out.append({"id": f"plain-{g}", "messages": [{"role": "user", "content": q + " Reply with the code only."}],
                    "prefill": "```python\n", "role": "keep", "group": f"plain-{g}", "max_new": 160, "stop": ["```"], "meta": {"api": "none"}})
    return out


TUNE_DOMAINS = ["user", "order", "event", "settings", "post", "address", "ticket", "recipe"]
HOLDOUT_DOMAINS = ["invoice", "product", "reading", "payment", "employee", "flight", "repo", "book"]


def full_sets(apis_keep: list[str], target_version: int = 2):
    """Tune: 8 domains, wording A. Holdout: the other 8 domains, wording B. Non-pydantic keep cases split by group."""
    A = {k: v for k, v in APIS.items() if k in apis_keep}
    B = {k: v for k, v in APIS_B.items() if k in apis_keep}
    tune = target_and_v1_cases({g: DOMAINS[g] for g in TUNE_DOMAINS}, A, target_version=target_version)
    hold = target_and_v1_cases({g: DOMAINS[g] for g in HOLDOUT_DOMAINS}, B, target_version=target_version)
    other = other_keep_cases()
    for i, c in enumerate(other):
        (tune if i % 2 == 0 else hold).append(c)
    for c in tune:
        c["split"] = "tune"
    for c in hold:
        c["split"] = "holdout"
        c["id"] = "h-" + c["id"]
    return tune, hold


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "census":
        cases = target_and_v1_cases(DOMAINS, APIS) + other_keep_cases()
        for c in cases:
            c["split"] = "census"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"{len(cases)} cases -> {out}")
    elif mode == "full":
        apis = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else list(APIS)
        tv = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        if tv not in (1, 2):
            raise SystemExit("TARGET_VERSION: 1 or 2")
        tune, hold = full_sets(apis, tv)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cases_tune.jsonl").write_text("".join(json.dumps(c) + "\n" for c in tune))
        (out / "cases_holdout.jsonl").write_text("".join(json.dumps(c) + "\n" for c in hold))
        print(f"tune {len(tune)}  holdout {len(hold)}  apis {apis}  target version {tv} -> {out}")
    else:
        raise SystemExit("mode: census | full")


if __name__ == "__main__":
    main()
