"""Apply and audit the prediction-blind Configuration MCPU v2 migration.

The migration is deliberately narrow.  It applies only the explicit span and
CPE decisions in ``config/configuration_boundary_mcpu_v2.json``, refreshes the
protocol metadata, and emits a complete 406-mention decision ledger.  It never
reads model predictions or optimization outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
POLICY_FILE = EXP_DIR / "config" / "configuration_boundary_mcpu_v2.json"
AUDIT_FILE = EXP_DIR / "results" / "configuration_boundary_mcpu_v2_audit.json"
RECEIPT_FILE = EXP_DIR / "results" / "configuration_boundary_mcpu_v2_receipt.json"
REVIEW_CSV = EXP_DIR / "results" / "configuration_boundary_mcpu_v2_review.csv"
REVIEW_MD = EXP_DIR / "results" / "configuration_boundary_mcpu_v2_review.md"

OLD_PROTOCOL = "4.5-mention-fact-dual-layer-v1"
OLD_CONTRACT = "chapter3-boundary-sync-v1"

GENERIC_SUFFIX = re.compile(
    r"(?:\b(?:software\s+)?library|\bemail\s+servers?|\bwebmail\s+clients?|"
    r"\bGateway\s+appliances?|\bSMA\s+100\s+Series\s+Appliances|"
    r"\bSD-WAN\s+WANOP\s+appliance|"
    r"\bweb\s+application\s+delivery\s+control\s*\(ADC\))$",
    re.IGNORECASE,
)
ORDINARY_RELEASE_SURFACE = re.compile(
    r"\bversions?\s+\d|\bColdFusion\s+(?:11|2016|2018|2021)\b",
    re.IGNORECASE,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpe_version(value: object) -> str | None:
    parts = str(value or "").split(":")
    if len(parts) != 13 or parts[:2] != ["cpe", "2.3"]:
        return None
    return parts[5]


def _entity_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entity["id"]): entity for entity in document.get("entities", [])}


def _preflight(
    policy: dict[str, Any], documents: dict[str, dict[str, Any]]
) -> None:
    seen_spans: set[tuple[str, str]] = set()
    for revision in policy["span_revisions"]:
        key = (revision["doc_id"], revision["entity_id"])
        if key in seen_spans:
            raise ValueError(f"duplicate span revision: {key}")
        seen_spans.add(key)
        document = documents.get(revision["doc_id"])
        if document is None:
            raise ValueError(f"span revision document missing: {key}")
        entity = _entity_index(document).get(revision["entity_id"])
        if entity is None:
            raise ValueError(f"span revision entity missing: {key}")
        old_state = (
            entity.get("text"), entity.get("start"), entity.get("end")
        )
        expected_old = (
            revision["old_text"], revision["old_start"], revision["old_end"]
        )
        expected_new = (
            revision["new_text"], revision["new_start"], revision["new_end"]
        )
        if old_state not in {expected_old, expected_new}:
            raise ValueError(
                f"span precondition failed for {key}: {old_state!r}"
            )
        text = document["text"]
        if text[revision["new_start"] : revision["new_end"]] != revision["new_text"]:
            raise ValueError(f"new span is not an exact source slice: {key}")

    seen_cpes: set[tuple[str, str]] = set()
    for revision in policy["normalization_revisions"]:
        key = (revision["doc_id"], revision["entity_id"])
        if key in seen_cpes:
            raise ValueError(f"duplicate normalization revision: {key}")
        seen_cpes.add(key)
        document = documents.get(revision["doc_id"])
        if document is None:
            raise ValueError(f"normalization revision document missing: {key}")
        entity = _entity_index(document).get(revision["entity_id"])
        if entity is None:
            raise ValueError(f"normalization revision entity missing: {key}")
        current = str(entity.get("normalized_id") or "")
        if current not in {revision["old_cpe"], revision["new_cpe"]}:
            raise ValueError(
                f"normalization precondition failed for {key}: {current!r}"
            )


def _apply(
    policy: dict[str, Any], documents: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    span_by_key = {
        (item["doc_id"], item["entity_id"]): item
        for item in policy["span_revisions"]
    }
    cpe_by_key = {
        (item["doc_id"], item["entity_id"]): item
        for item in policy["normalization_revisions"]
    }
    before_hashes = {
        doc_id: _sha256(GOLD_DIR / f"{doc_id}.json") for doc_id in documents
    }
    semantic_docs: set[str] = set()
    applied: list[dict[str, Any]] = []

    for doc_id, document in sorted(documents.items()):
        document["annotation_protocol_version"] = policy[
            "annotation_protocol_version"
        ]
        document["boundary_contract_version"] = policy[
            "boundary_contract_version"
        ]
        review = document.setdefault("gold_review", {})
        review["configuration_boundary"] = {
            "version": policy["version"],
            "prediction_outputs_used": False,
            "policy_file": "config/configuration_boundary_mcpu_v2.json",
        }

        entity_by_id = _entity_index(document)
        for entity_id, entity in entity_by_id.items():
            key = (doc_id, entity_id)
            span_revision = span_by_key.get(key)
            cpe_revision = cpe_by_key.get(key)
            if span_revision is not None:
                entity["text"] = span_revision["new_text"]
                entity["start"] = span_revision["new_start"]
                entity["end"] = span_revision["new_end"]
                semantic_docs.add(doc_id)
            if cpe_revision is not None:
                entity["normalized_id"] = cpe_revision["new_cpe"]
                semantic_docs.add(doc_id)
            if span_revision is not None or cpe_revision is not None:
                basis = []
                if span_revision is not None:
                    basis.append(span_revision["basis"])
                if cpe_revision is not None:
                    basis.append(cpe_revision["basis"])
                entity["boundary_adjudication"] = {
                    "contract_version": policy["boundary_contract_version"],
                    "decision": (
                        "REVISE_SPAN_AND_CPE"
                        if span_revision is not None and cpe_revision is not None
                        else "REVISE_SPAN"
                        if span_revision is not None
                        else "REVISE_CPE"
                    ),
                    "prediction_outputs_used": False,
                    "basis": " ".join(basis),
                }
                applied.append(
                    {
                        "doc_id": doc_id,
                        "entity_id": entity_id,
                        "decision": entity["boundary_adjudication"]["decision"],
                        "old_text": (
                            span_revision["old_text"] if span_revision else entity["text"]
                        ),
                        "new_text": entity["text"],
                        "old_cpe": (
                            cpe_revision["old_cpe"]
                            if cpe_revision
                            else entity.get("normalized_id")
                        ),
                        "new_cpe": entity.get("normalized_id"),
                        "basis": " ".join(basis),
                    }
                )

        for relation in document.get("relations", []):
            if relation.get("type") != "exploited_by":
                continue
            basis = str(relation.get("adjudication_basis") or "")
            if OLD_CONTRACT in basis:
                relation["adjudication_basis"] = basis.replace(
                    OLD_CONTRACT, policy["boundary_contract_version"], 1
                )

        path = GOLD_DIR / f"{doc_id}.json"
        new_bytes = _json_bytes(document)
        if path.read_bytes() != new_bytes:
            path.write_bytes(new_bytes)

    return {
        "before_hashes": before_hashes,
        "semantic_changed_documents": sorted(semantic_docs),
        "applied": sorted(
            applied, key=lambda item: (item["doc_id"], item["entity_id"])
        ),
    }


def _audit(
    policy: dict[str, Any], documents: dict[str, dict[str, Any]], application: dict[str, Any]
) -> dict[str, Any]:
    allowed_non_wildcard = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in policy["allowed_non_wildcard_cpe_surface_patterns"]
    ]
    revised = {
        (item["doc_id"], item["entity_id"]): item
        for item in application["applied"]
    }
    counts: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []

    for doc_id, document in sorted(documents.items()):
        text = document.get("text", "")
        entity_by_id = _entity_index(document)
        configuration_tails = {
            str(relation.get("tail"))
            for relation in document.get("relations", [])
            if relation.get("type") == "affects"
        }
        seen: set[tuple[str, int, int]] = set()
        for entity in document.get("entities", []):
            start, end = entity.get("start"), entity.get("end")
            exact = (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
                and text[start:end] == entity.get("text")
            )
            if not exact:
                blocking.append(
                    {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "invalid_span"}
                )
            key = (str(entity.get("type")), start, end)
            if key in seen:
                blocking.append(
                    {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "duplicate_type_span"}
                )
            seen.add(key)
            if entity.get("type") != "Configuration":
                continue

            counts["configuration_mentions"] += 1
            surface = str(entity.get("text") or "")
            cpe = str(entity.get("normalized_id") or "")
            version = _cpe_version(cpe)
            approved_generation_or_literal = any(
                pattern.search(surface) for pattern in allowed_non_wildcard
            )
            flags: list[str] = []
            if surface.casefold().startswith("cpe:2.3:"):
                flags.append("literal_cpe_surface")
            if re.search(r"\d", surface):
                flags.append("numeric_product_or_generation")
            if GENERIC_SUFFIX.search(surface):
                flags.append("generic_suffix_candidate")
                blocking.append(
                    {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "generic_suffix_residue", "surface": surface}
                )
            if (
                ORDINARY_RELEASE_SURFACE.search(surface)
                and not approved_generation_or_literal
            ):
                flags.append("ordinary_release_surface")
                blocking.append(
                    {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "ordinary_release_surface", "surface": surface}
                )
            if version not in {None, "*", "-"}:
                counts["non_wildcard_cpe_versions"] += 1
                allowed = approved_generation_or_literal
                if allowed:
                    flags.append("approved_non_wildcard_cpe")
                else:
                    blocking.append(
                        {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "unapproved_non_wildcard_cpe", "surface": surface, "cpe": cpe}
                    )
            if str(entity.get("id")) not in configuration_tails:
                blocking.append(
                    {"doc_id": doc_id, "entity_id": entity.get("id"), "reason": "orphan_configuration"}
                )

            decision = revised.get((doc_id, str(entity.get("id"))))
            items.append(
                {
                    "doc_id": doc_id,
                    "entity_id": entity.get("id"),
                    "decision": decision["decision"] if decision else "KEEP",
                    "surface": surface,
                    "start": start,
                    "end": end,
                    "normalized_id": cpe,
                    "cpe_version": version,
                    "flags": flags,
                    "basis": (
                        decision["basis"]
                        if decision
                        else "Retained after the full Configuration inventory audit under MCPU v2."
                    ),
                }
            )

        for relation in document.get("relations", []):
            head = entity_by_id.get(str(relation.get("head")))
            tail = entity_by_id.get(str(relation.get("tail")))
            start, end = relation.get("evidence_start"), relation.get("evidence_end")
            if not head or not tail:
                blocking.append(
                    {"doc_id": doc_id, "relation_id": relation.get("id"), "reason": "missing_endpoint"}
                )
                continue
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
                and text[start:end] == relation.get("evidence")
                and start <= head["start"] < head["end"] <= end
                and start <= tail["start"] < tail["end"] <= end
            ):
                blocking.append(
                    {"doc_id": doc_id, "relation_id": relation.get("id"), "reason": "invalid_relation_evidence_after_boundary_revision"}
                )

    counts.update(Counter(item["decision"] for item in items))
    if counts["configuration_mentions"] != 406:
        blocking.append(
            {"reason": "unexpected_configuration_count", "actual": counts["configuration_mentions"], "expected": 406}
        )
    return {
        "schema_version": "configuration-boundary-mcpu-audit-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_outputs_used": False,
        "annotation_protocol_version": policy["annotation_protocol_version"],
        "boundary_contract_version": policy["boundary_contract_version"],
        "policy_file": "config/configuration_boundary_mcpu_v2.json",
        "policy_sha256": _sha256(POLICY_FILE),
        "summary": {
            "documents": len(documents),
            **dict(sorted(counts.items())),
            "blocking_errors": len(blocking),
            "gate_status": "gate_passed" if not blocking else "gate_failed",
        },
        "blocking_errors": blocking,
        "items": items,
    }


def _write_review_tables(audit: dict[str, Any]) -> None:
    headers = [
        "doc_id", "entity_id", "decision", "surface", "start", "end",
        "normalized_id", "cpe_version", "flags", "basis",
    ]
    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for item in audit["items"]:
            row = dict(item)
            row["flags"] = "|".join(item["flags"])
            writer.writerow({key: row.get(key, "") for key in headers})

    revised = [item for item in audit["items"] if item["decision"] != "KEEP"]
    lines = [
        "# Configuration MCPU v2 全量复裁台账",
        "",
        "本台账覆盖全部 406 个 Configuration mention；规则制定与复裁未读取模型预测。",
        "",
        f"- 门禁：`{audit['summary']['gate_status']}`",
        f"- 跨度修正及/或 CPE 修正：{len(revised)} 项",
        f"- 非通配符 CPE 版本：{audit['summary'].get('non_wildcard_cpe_versions', 0)} 项",
        "",
        "| 文档 | 实体 | 裁决 | 新跨度 | 新 CPE | 依据 |",
        "|---|---|---|---|---|---|",
    ]
    for item in revised:
        basis = str(item["basis"]).replace("|", "\\|")
        lines.append(
            f"| {item['doc_id']} | {item['entity_id']} | {item['decision']} | "
            f"`{item['surface']}` | `{item['normalized_id']}` | {basis} |"
        )
    REVIEW_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    policy = _load(POLICY_FILE)
    paths = sorted(GOLD_DIR.glob("*.json"))
    documents = {path.stem: _load(path) for path in paths}
    if len(documents) != 105:
        raise ValueError(f"expected 105 Gold documents, found {len(documents)}")
    _preflight(policy, documents)
    application = _apply(policy, documents)
    documents = {
        path.stem: _load(path) for path in sorted(GOLD_DIR.glob("*.json"))
    }
    audit = _audit(policy, documents, application)
    AUDIT_FILE.write_bytes(_json_bytes(audit))
    _write_review_tables(audit)

    receipt = {
        "schema_version": "configuration-boundary-mcpu-receipt-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_outputs_used": False,
        "policy_file": "config/configuration_boundary_mcpu_v2.json",
        "policy_sha256": _sha256(POLICY_FILE),
        "audit_file": "results/configuration_boundary_mcpu_v2_audit.json",
        "audit_sha256": _sha256(AUDIT_FILE),
        "review_csv": "results/configuration_boundary_mcpu_v2_review.csv",
        "review_csv_sha256": _sha256(REVIEW_CSV),
        "review_md": "results/configuration_boundary_mcpu_v2_review.md",
        "review_md_sha256": _sha256(REVIEW_MD),
        "semantic_changed_documents": application["semantic_changed_documents"],
        "applied": application["applied"],
        "post_gold_sha256": {
            doc_id: _sha256(GOLD_DIR / f"{doc_id}.json")
            for doc_id in sorted(documents)
        },
        "gate_status": audit["summary"]["gate_status"],
    }
    RECEIPT_FILE.write_bytes(_json_bytes(receipt))
    print(
        json.dumps(
            {
                "documents": len(documents),
                "configuration_mentions": audit["summary"]["configuration_mentions"],
                "semantic_changed_documents": len(
                    application["semantic_changed_documents"]
                ),
                "revised_mentions": sum(
                    1 for item in audit["items"] if item["decision"] != "KEEP"
                ),
                "blocking_errors": audit["summary"]["blocking_errors"],
                "gate_status": audit["summary"]["gate_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["summary"]["gate_status"] == "gate_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
