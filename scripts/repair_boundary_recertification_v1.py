"""Repair and archive the chapter3-boundary-sync-v1 adjudication results.

This migration is intentionally deterministic and idempotent.  It expands the
thirteen endpoint-only exploited_by evidence spans, removes the unsupported
CVE-2020-1472 -> T1202 relation, replaces generic acceptance notes with
relation-specific scope decisions, and materializes the rejection quarantine.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SOURCE_GOLD_DIR = EXP_DIR.parent / "v4.5" / "data" / "annotations" / "v3_text_grounded"
REVIEW_CSV = EXP_DIR / "results" / "boundary_sync_review_v1.csv"
QUARANTINE = (
    EXP_DIR
    / "data"
    / "annotations"
    / "quarantine"
    / "chapter3-boundary-sync-v1"
    / "rejected_items.json"
)
CONTRACT = "chapter3-boundary-sync-v1"
REVIEWER = "expert_adjudicator"
REVIEW_DATE = "2026-09-14"


# Each end marker is excluded.  The selected block therefore ends immediately
# before the next sentence/list item while retaining the governing predicate.
SPAN_REVISIONS = {
    ("aa22-074a", "R2"): (
        "Using the compromised account,",
        " The actors also modified a domain controller file",
    ),
    ("aa22-321a-hive-exchange", "R7"): (
        "Hive actors have also gained initial access to victim networks",
        "CVE-2021-34473",
    ),
    ("aa23-158a", "R11"): (
        "In May 2023, the CL0P ransomware group exploited",
        " Lemurloot was used as a method of persistence",
    ),
    ("aa23-278a", "R1"): (
        "Assessment teams have observed threat actors exploiting many CVEs",
        "CVE-2021-44228",
    ),
    ("aa23-347a-play-ransomware", "R2"): (
        "The SVR started to exploit Internet-connected JetBrains TeamCity servers",
        " The authoring agencies' observations show",
    ),
    ("aa23-352a", "R6"): (
        "The Play ransomware group gains initial access to victim networks",
        " Play ransomware actors have been observed using external-facing services",
    ),
    ("aa24-016a-androxgh0st", "R2"): (
        "In particular, threat actors deploying Androxgh0st have been observed exploiting",
        " Websites using the PHPUnit module",
    ),
    ("aa24-109a", "R4"): (
        "FBI and cybersecurity researchers",
        "CVE-2023-20269",
    ),
    ("aa24-109a", "R12"): (
        "After tunneling through a targeted router,",
        "\n7\nEnd Update",
    ),
    ("aa24-109a", "R13"): (
        "Akira threat actors have also been observed leveraging services like",
        "\nEnd Update",
    ),
    ("aa24-131a-black-basta-cve-1709", "R7"): (
        "Starting in February 2024, Black Basta affiliates began exploiting",
        " In some instances, affiliates have been observed abusing valid credentials",
    ),
    ("aa24-131a-black-basta-cve-1709", "R11"): (
        "According to cybersecurity researchers, Black Basta affiliates have also exploited",
        "[\n1\n],[\n2\n]\nLateral Movement",
    ),
    ("aa24-131a-black-basta-cve-1709", "R20"): (
        "According to cybersecurity researchers, Black Basta affiliates have also exploited",
        "[\n1\n],[\n2\n]\nLateral Movement",
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relation(document: dict, relation_id: str) -> dict | None:
    return next(
        (item for item in document["relations"] if item.get("id") == relation_id),
        None,
    )


def _select_span(text: str, relation: dict, start_anchor: str, end_before: str) -> tuple[int, int]:
    pivot = int(relation["evidence_start"])
    search_left = max(0, pivot - 3000)
    start = text.rfind(start_anchor, search_left, pivot + len(start_anchor) + 1)
    if start < 0:
        raise ValueError(f"start anchor not found near {pivot}: {start_anchor!r}")
    end = text.find(end_before, max(pivot, int(relation["evidence_end"])))
    if end < 0:
        raise ValueError(f"end marker not found after {pivot}: {end_before!r}")
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _normalized_endpoint(entity: dict) -> str:
    return str(entity.get("normalized_id") or entity.get("text") or entity.get("id"))


def _scope_kind(evidence: str) -> str:
    cves = {item.upper() for item in re.findall(r"CVE[- ]\d{4}[- ]\d+", evidence, re.I)}
    if len(cves) > 1 or "including:" in evidence.casefold():
        return "explicit governing list scope"
    return "same sentence or self-contained clause"


def _acceptance_basis(doc_id: str, relation: dict, entity_by_id: dict[str, dict]) -> str:
    head = _normalized_endpoint(entity_by_id[relation["head"]])
    tail = _normalized_endpoint(entity_by_id[relation["tail"]])
    scope = _scope_kind(relation["evidence"])
    return (
        f"{CONTRACT}; {scope} with an explicit exploitation predicate; "
        f"{doc_id}/{relation['id']} directly binds {head} to {tail} as the exploit behavior"
    )


def _source_snapshot(doc_id: str, item_id: str) -> dict:
    source = _load(SOURCE_GOLD_DIR / f"{doc_id}.json")
    if item_id.startswith("E"):
        item = next(entity for entity in source["entities"] if entity.get("id") == item_id)
        return {"item_type": "entity", "item": item}
    item = next(relation for relation in source["relations"] if relation.get("id") == item_id)
    return {"item_type": "relation", "item": item}


def main() -> None:
    with REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 118:
        raise ValueError(f"expected 118 adjudication rows, found {len(rows)}")

    # Make the semantic correction explicit in the adjudication ledger first.
    r6_row = next(
        row for row in rows
        if row["doc_id"] == "aa24-290a" and row["item_id"] == "R6"
    )
    r6_row["decision"] = "REJECT"
    r6_row["adjudication_basis"] = (
        f"rejected under {CONTRACT}: the duplicated T1202 table entry classifies "
        "adjacent Indirect Command Execution behavior and does not state that "
        "CVE-2020-1472 directly instantiates T1202; the direct exploitation mapping "
        "is retained only as T1068 in R5"
    )
    r6_row["reviewer"] = REVIEWER
    r6_row["review_date"] = REVIEW_DATE

    documents: dict[str, dict] = {}
    changed_docs: set[str] = set()
    for doc_id, relation_id in SPAN_REVISIONS:
        document = documents.setdefault(doc_id, _load(GOLD_DIR / f"{doc_id}.json"))
        relation = _relation(document, relation_id)
        if relation is None:
            raise ValueError(f"missing relation {doc_id}/{relation_id}")
        start, end = _select_span(
            document["text"], relation, *SPAN_REVISIONS[(doc_id, relation_id)]
        )
        relation["evidence_start"] = start
        relation["evidence_end"] = end
        relation["evidence"] = document["text"][start:end]
        changed_docs.add(doc_id)

    # Remove the semantically unsupported duplicate mapping.
    r6_doc = documents.setdefault("aa24-290a", _load(GOLD_DIR / "aa24-290a.json"))
    r6 = _relation(r6_doc, "R6")
    if r6 is not None:
        r6_row["evidence_preview"] = r6["evidence"]
        r6_doc["relations"] = [item for item in r6_doc["relations"] if item.get("id") != "R6"]
        changed_docs.add("aa24-290a")

    # Every accepted exploited_by relation receives a concrete, auditable basis.
    for path in sorted(GOLD_DIR.glob("*.json")):
        doc_id = path.stem
        document = documents.setdefault(doc_id, _load(path))
        entity_by_id = {entity["id"]: entity for entity in document["entities"]}
        for relation in document["relations"]:
            if relation.get("type") != "exploited_by":
                continue
            relation["adjudication_basis"] = _acceptance_basis(
                doc_id, relation, entity_by_id
            )
            changed_docs.add(doc_id)

    # Synchronize accepted ledger rows to the repaired Gold evidence and basis.
    for row in rows:
        if (
            "exploited_by_boundary_recertification_required" not in row["kind"]
            or row["decision"] != "ACCEPT"
        ):
            continue
        document = documents[row["doc_id"]]
        relation = _relation(document, row["item_id"])
        if relation is None:
            raise ValueError(f"accepted ledger relation is absent: {row['doc_id']}/{row['item_id']}")
        row["evidence_preview"] = relation["evidence"]
        row["adjudication_basis"] = relation["adjudication_basis"]
        row["reviewer"] = REVIEWER
        row["review_date"] = REVIEW_DATE

    quarantine_rel = "data/annotations/quarantine/chapter3-boundary-sync-v1/rejected_items.json"
    for row in rows:
        if row["decision"] != "REJECT":
            continue
        if row["item_id"] == "R6" and row["doc_id"] == "aa24-290a":
            continue
        if row["kind"] == "weakness_without_local_explicit_cwe":
            row["adjudication_basis"] = (
                f"rejected under {CONTRACT}: descriptive weakness phrase lacks a "
                f"locally explicit CWE in the source fact block; archived in {quarantine_rel}"
            )
        elif "explicit_post_exploitation_sequence" in row["kind"]:
            row["adjudication_basis"] = (
                f"rejected under {CONTRACT}: evidence states post-exploitation activity "
                f"rather than the CVE's direct ATT&CK exploit role; archived in {quarantine_rel}"
            )

    for doc_id in sorted(changed_docs):
        document = documents[doc_id]
        for relation in document["relations"]:
            start, end = relation["evidence_start"], relation["evidence_end"]
            if document["text"][start:end] != relation["evidence"]:
                raise ValueError(f"invalid evidence after repair: {doc_id}/{relation['id']}")
            endpoints = {entity["id"]: entity for entity in document["entities"]}
            for endpoint in (relation["head"], relation["tail"]):
                entity = endpoints[endpoint]
                if not (start <= entity["start"] < entity["end"] <= end):
                    raise ValueError(f"endpoint outside evidence: {doc_id}/{relation['id']}/{endpoint}")
        _write_json(GOLD_DIR / f"{doc_id}.json", document)

    # Materialize a recoverable, source-faithful quarantine instead of merely
    # claiming that rejected annotations were moved elsewhere.
    rejected = []
    rejected_entity_ids: dict[str, set[str]] = {}
    for row in rows:
        if row["decision"] != "REJECT":
            continue
        snapshot = _source_snapshot(row["doc_id"], row["item_id"])
        rejected.append({
            "doc_id": row["doc_id"],
            "item_id": row["item_id"],
            "kind": row["kind"],
            "decision": row["decision"],
            "adjudication_basis": row["adjudication_basis"],
            "source_snapshot": snapshot,
        })
        if snapshot["item_type"] == "entity":
            rejected_entity_ids.setdefault(row["doc_id"], set()).add(row["item_id"])

    cascaded = []
    for doc_id, entity_ids in sorted(rejected_entity_ids.items()):
        source = _load(SOURCE_GOLD_DIR / f"{doc_id}.json")
        for relation in source["relations"]:
            if relation.get("type") == "instantiates" and (
                relation.get("head") in entity_ids or relation.get("tail") in entity_ids
            ):
                cascaded.append({"doc_id": doc_id, "relation": relation})

    counts = Counter(item["kind"] for item in rejected)
    quarantine = {
        "contract_version": CONTRACT,
        "reviewer": REVIEWER,
        "review_date": REVIEW_DATE,
        "source_gold": "../v4.5/data/annotations/v3_text_grounded",
        "policy": "rejected annotations are excluded from formal Gold but preserved verbatim for audit and recovery",
        "summary": {
            "rejected_ledger_items": len(rejected),
            "cascade_removed_instantiates": len(cascaded),
            "by_kind": dict(sorted(counts.items())),
        },
        "rejected_items": rejected,
        "cascade_removed_relations": cascaded,
    }
    if len(rejected) != 62 or len(cascaded) != 47:
        raise ValueError(
            f"unexpected quarantine counts: rejected={len(rejected)}, cascaded={len(cascaded)}"
        )
    _write_json(QUARANTINE, quarantine)

    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(
        1 for row in rows
        if "exploited_by_boundary_recertification_required" in row["kind"]
        and row["decision"] == "ACCEPT"
    )
    print(json.dumps({
        "evidence_spans_revised": len(SPAN_REVISIONS),
        "accepted_exploited_by": accepted,
        "rejected_ledger_items": len(rejected),
        "cascade_removed_instantiates": len(cascaded),
        "changed_gold_documents": len(changed_docs),
        "quarantine": str(QUARANTINE.relative_to(EXP_DIR)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
