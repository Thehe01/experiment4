"""Prediction-blind audit for Configuration CPE normalization in v5 Gold.

The script reads only Gold annotations and the repository's archived NVD CVE
feeds.  It never reads model predictions.  The archived feed blobs are bound
to a Git ref and their blob hashes are emitted into the evidence snapshot.

Typical use::

    python scripts/audit_cpe_gold.py --build-evidence

Applying adjudicated changes is intentionally a separate operation and
requires an explicit corrections file::

    python scripts/audit_cpe_gold.py --corrections config/cpe_gold_corrections_v1.json --apply
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


EXP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXP_DIR.parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_EVIDENCE = RESULTS_DIR / "cpe_gold_nvd_evidence_v1.json"
DEFAULT_AUDIT = RESULTS_DIR / "cpe_gold_audit_v1.json"
DEFAULT_PRE_AUDIT = RESULTS_DIR / "cpe_gold_audit_pre_v1.json"
DEFAULT_INVENTORY = RESULTS_DIR / "cpe_gold_inventory_v1.csv"
DEFAULT_RECEIPT = RESULTS_DIR / "cpe_gold_review_v1.json"
DEFAULT_REVIEW_CSV = RESULTS_DIR / "cpe_gold_review_v1.csv"
DEFAULT_REVIEW_MD = RESULTS_DIR / "cpe_gold_review_v1.md"
DEFAULT_QUARANTINE = (
    EXP_DIR / "data" / "annotations" / "quarantine"
    / "cpe-gold-audit-v1" / "rejected_items.json"
)
CPE_PREFIX = "cpe:2.3:"
CVE_RE = re.compile(r"CVE-(\d{4})-\d{4,}", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(args: list[str], repo: Path, *, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8", errors="replace").strip()


def _split_cpe(cpe: str) -> list[str]:
    """Split a formatted CPE 2.3 string on unescaped colons."""
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in cpe.strip():
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            current.append(char)
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def _cpe_family(cpe: str) -> tuple[str, str, str] | None:
    fields = _split_cpe(cpe.casefold())
    if len(fields) != 13 or fields[:2] != ["cpe", "2.3"]:
        return None
    if fields[2] not in {"a", "h", "o", "*", "-"}:
        return None
    if not fields[3] or not fields[4]:
        return None
    return fields[2], fields[3], fields[4]


def _walk_cpe_matches(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"criteria", "cpe23Uri"} and isinstance(child, str):
                if child.casefold().startswith(CPE_PREFIX):
                    yield child.casefold()
            else:
                yield from _walk_cpe_matches(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_cpe_matches(child)


def _iter_cve_records(feed: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # NVD API 2.0 feed shape.
    for wrapper in feed.get("vulnerabilities", []):
        if isinstance(wrapper, dict):
            cve = wrapper.get("cve", wrapper)
            if isinstance(cve, dict):
                yield cve
    # Legacy NVD 1.1 feed shape.
    for item in feed.get("CVE_Items", []):
        if isinstance(item, dict):
            yield item


def _record_cve_id(record: dict[str, Any]) -> str | None:
    value = record.get("id")
    if isinstance(value, str) and CVE_RE.fullmatch(value.strip()):
        return value.strip().upper()
    value = record.get("cve", {}).get("CVE_data_meta", {}).get("ID")
    if isinstance(value, str) and CVE_RE.fullmatch(value.strip()):
        return value.strip().upper()
    return None


def _record_configurations(record: dict[str, Any]) -> Any:
    if "configurations" in record:
        return record.get("configurations")
    return record.get("configurations", {})


@dataclass(frozen=True)
class GoldConfiguration:
    doc_id: str
    path: Path
    entity_id: str
    surface: str
    normalized_id: str
    start: int
    end: int
    relation_ids: tuple[str, ...]
    cves: tuple[str, ...]


def _load_gold(gold_dir: Path) -> tuple[list[GoldConfiguration], dict[str, dict[str, Any]]]:
    rows: list[GoldConfiguration] = []
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(gold_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(document.get("doc_id") or path.stem)
        documents[doc_id] = document
        entity_by_id = {str(e.get("id")): e for e in document.get("entities", [])}
        for entity in document.get("entities", []):
            if entity.get("type") != "Configuration":
                continue
            relation_ids: list[str] = []
            cves: set[str] = set()
            for relation in document.get("relations", []):
                if relation.get("type") != "affects" or str(relation.get("tail")) != str(entity.get("id")):
                    continue
                relation_ids.append(str(relation.get("id")))
                head = entity_by_id.get(str(relation.get("head")), {})
                normalized = str(head.get("normalized_id") or "").strip().upper()
                if CVE_RE.fullmatch(normalized):
                    cves.add(normalized)
            rows.append(GoldConfiguration(
                doc_id=doc_id,
                path=path,
                entity_id=str(entity.get("id")),
                surface=str(entity.get("text") or ""),
                normalized_id=str(entity.get("normalized_id") or "").casefold(),
                start=int(entity.get("start", -1)),
                end=int(entity.get("end", -1)),
                relation_ids=tuple(sorted(relation_ids)),
                cves=tuple(sorted(cves)),
            ))
    return rows, documents


def _feed_path(year: str) -> str:
    return f"data/raw/raw_CVE_{year}.json.gz"


def _load_archived_nvd_subset(
    repo: Path,
    git_ref: str,
    wanted_cves: set[str],
    wanted_families: set[tuple[str, str, str]],
) -> dict[str, Any]:
    by_year: dict[str, set[str]] = defaultdict(set)
    for cve in wanted_cves:
        match = CVE_RE.fullmatch(cve)
        if match:
            by_year[match.group(1)].add(cve)

    records: dict[str, dict[str, Any]] = {}
    feeds: dict[str, dict[str, Any]] = {}
    missing_feed_years: list[str] = []
    for year, year_cves in sorted(by_year.items()):
        path = _feed_path(year)
        try:
            blob_hash = str(_git(["rev-parse", f"{git_ref}:{path}"], repo))
            compressed = _git(["cat-file", "blob", f"{git_ref}:{path}"], repo, binary=True)
        except subprocess.CalledProcessError:
            missing_feed_years.append(year)
            continue
        assert isinstance(compressed, bytes)
        raw = gzip.decompress(compressed)
        feed = json.loads(raw)
        feeds[year] = {
            "path": path,
            "git_blob_sha1": blob_hash,
            "compressed_sha256": _sha256_bytes(compressed),
            "compressed_bytes": len(compressed),
            "decompressed_sha256": _sha256_bytes(raw),
            "feed_timestamp": feed.get("timestamp")
            or feed.get("CVE_data_timestamp")
            or feed.get("lastModifiedDate"),
            "format": feed.get("format") or feed.get("CVE_data_format"),
            "version": feed.get("version") or feed.get("CVE_data_version"),
        }
        remaining = set(year_cves)
        for record in _iter_cve_records(feed):
            cve_id = _record_cve_id(record)
            if cve_id not in remaining:
                continue
            criteria = sorted(set(_walk_cpe_matches(_record_configurations(record))))
            records[cve_id] = {
                "criteria": criteria,
                "families": sorted({":".join(family) for cpe in criteria if (family := _cpe_family(cpe))}),
                "source_identifier": record.get("sourceIdentifier"),
                "published": record.get("published"),
                "last_modified": record.get("lastModified"),
                "vuln_status": record.get("vulnStatus"),
            }
            remaining.remove(cve_id)
            if not remaining:
                break

    missing_cves = sorted(wanted_cves - set(records))
    family_presence, family_scan_incomplete_feeds = _scan_archived_family_presence(
        repo, git_ref, wanted_families
    )
    return {
        "schema_version": "cpe-gold-nvd-evidence-v1",
        "created_at_utc": _utc_now(),
        "source": "repository-archived NVD CVE JSON feeds",
        "prediction_outputs_used": False,
        "git_ref_requested": git_ref,
        "git_commit": str(_git(["rev-parse", git_ref], repo)),
        "feeds": feeds,
        "records": dict(sorted(records.items())),
        "family_presence": family_presence,
        "family_scan_incomplete_feeds": family_scan_incomplete_feeds,
        "missing_feed_years": missing_feed_years,
        "missing_cves": missing_cves,
    }


def _scan_archived_family_presence(
    repo: Path,
    git_ref: str,
    wanted_families: set[tuple[str, str, str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Locate Gold CPE families across all archived yearly NVD CVE feeds.

    This is a byte-level exact-prefix scan.  It is intentionally used only as
    corroborating evidence that a family occurs in archived NVD data; linked
    CVE validation still uses parsed configuration records.
    """
    names_raw = str(_git([
        "ls-tree", "-r", "--name-only", git_ref, "--", "data/raw"
    ], repo))
    paths = sorted(
        path for path in names_raw.splitlines()
        if re.fullmatch(r"data/raw/raw_CVE_\d{4}\.json\.gz", path)
    )
    prefixes = {
        family: f"cpe:2.3:{family[0]}:{family[1]}:{family[2]}:".encode("utf-8")
        for family in wanted_families
    }
    found: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    incomplete_feeds: list[str] = []
    maximum = max((len(prefix) for prefix in prefixes.values()), default=0)
    for path in paths:
        year_match = re.search(r"(\d{4})", path)
        year = year_match.group(1) if year_match else path
        compressed = _git(["cat-file", "blob", f"{git_ref}:{path}"], repo, binary=True)
        assert isinstance(compressed, bytes)
        remaining = {
            family: prefix
            for family, prefix in prefixes.items()
            if family not in found
        }
        if not remaining:
            break
        overlap = b""
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            try:
                while True:
                    chunk = stream.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    haystack = (overlap + chunk).lower()
                    for family, prefix in list(remaining.items()):
                        if prefix in haystack:
                            found[family].append(year)
                            del remaining[family]
                    overlap = haystack[-maximum:] if maximum else b""
                    if not remaining:
                        break
            except (EOFError, gzip.BadGzipFile):
                # Preserve positive matches from readable bytes, but disclose
                # that absence in this feed is not conclusive.
                incomplete_feeds.append(path)
    return ({
        _family_key(family): years
        for family, years in sorted(found.items())
    }, incomplete_feeds)


def _family_key(family: tuple[str, str, str]) -> str:
    return ":".join(family)


def _candidate_similarity(gold: tuple[str, str, str], candidate: tuple[str, str, str]) -> float:
    part_bonus = 0.12 if gold[0] == candidate[0] else 0.0
    vendor = SequenceMatcher(None, gold[1], candidate[1]).ratio()
    product = SequenceMatcher(None, gold[2], candidate[2]).ratio()
    return round(part_bonus + 0.28 * vendor + 0.60 * product, 4)


def _audit(
    rows: list[GoldConfiguration],
    evidence: dict[str, Any],
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = evidence.get("records", {})
    family_presence = evidence.get("family_presence", {})
    # Families parsed from any frozen linked-CVE record are also positive
    # canonical evidence.  This matters after a correction introduces a
    # family that was absent from the pre-audit Gold inventory and therefore
    # was not among the byte-scan targets used to build family_presence.
    parsed_evidence_families = {
        family
        for record in records.values()
        for cpe in record.get("criteria", [])
        if (family := _cpe_family(cpe))
    }
    decision_index = {
        (str(decision["doc_id"]), str(decision["entity_id"])): decision
        for decision in (decisions or [])
    }
    items: list[dict[str, Any]] = []
    counts = Counter()
    unresolved: list[dict[str, str]] = []
    blocking_statuses = {
        "invalid_cpe_syntax",
        "no_linked_cve",
        "no_archived_nvd_configuration",
        "deterministic_part_mismatch",
        "family_not_in_linked_cve",
    }
    for row in rows:
        gold_family = _cpe_family(row.normalized_id)
        official_cpes: set[str] = set()
        for cve in row.cves:
            official_cpes.update(records.get(cve, {}).get("criteria", []))
        official_families = sorted({family for cpe in official_cpes if (family := _cpe_family(cpe))})

        if gold_family is None:
            status = "invalid_cpe_syntax"
        elif not row.cves:
            status = "no_linked_cve"
        elif not official_cpes:
            status = "no_archived_nvd_configuration"
        elif gold_family in official_families:
            status = "validated_linked_cve_family"
        elif any(gold_family[1:] == candidate[1:] for candidate in official_families):
            status = "deterministic_part_mismatch"
        elif (
            _family_key(gold_family) in family_presence
            or gold_family in parsed_evidence_families
        ):
            status = "canonical_family_but_not_linked_cve"
        else:
            status = "family_not_in_linked_cve"
        raw_status = status
        decision = decision_index.get((row.doc_id, row.entity_id))
        if decision and decision.get("action") == "KEEP":
            status = "validated_by_adjudication"
        counts[status] += 1
        if status in blocking_statuses:
            unresolved.append({
                "doc_id": row.doc_id,
                "entity_id": row.entity_id,
                "status": status,
            })

        ranked = []
        if gold_family:
            ranked = sorted(
                (
                    {
                        "family": _family_key(candidate),
                        "similarity": _candidate_similarity(gold_family, candidate),
                    }
                    for candidate in official_families
                ),
                key=lambda value: (-value["similarity"], value["family"]),
            )[:12]
        items.append({
            "doc_id": row.doc_id,
            "entity_id": row.entity_id,
            "surface": row.surface,
            "start": row.start,
            "end": row.end,
            "gold_cpe": row.normalized_id,
            "gold_family": _family_key(gold_family) if gold_family else None,
            "gold_family_seen_in_archived_nvd_years": (
                family_presence.get(_family_key(gold_family), [])
                if gold_family else []
            ),
            "relation_ids": list(row.relation_ids),
            "linked_cves": list(row.cves),
            "status": status,
            "raw_status": raw_status,
            "adjudication": decision,
            "official_family_candidates": ranked,
        })

    unique_cpes = Counter(row.normalized_id for row in rows)
    unique_families = Counter(
        _family_key(family)
        for row in rows
        if (family := _cpe_family(row.normalized_id))
    )
    return {
        "schema_version": "cpe-gold-audit-v1",
        "created_at_utc": _utc_now(),
        "prediction_outputs_used": False,
        "evidence_sha256": _sha256_bytes(_json_bytes(evidence)),
        "summary": {
            "configuration_mentions": len(rows),
            "unique_cpes": len(unique_cpes),
            "unique_families": len(unique_families),
            "status_counts": dict(sorted(counts.items())),
            "unresolved_items": len(unresolved),
            "gate_status": "gate_passed" if not unresolved else "gate_failed",
        },
        "unresolved": unresolved,
        "items": items,
    }


def _csv_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return '"' + text.replace('"', '""') + '"'


def _write_inventory(path: Path, rows: list[GoldConfiguration], audit: dict[str, Any]) -> None:
    status_by_key = {
        (item["doc_id"], item["entity_id"]): item["status"]
        for item in audit["items"]
    }
    lines = [
        "doc_id,entity_id,surface,start,end,gold_cpe,cpe_family,relation_ids,linked_cves,audit_status"
    ]
    for row in rows:
        family = _cpe_family(row.normalized_id)
        values = [
            row.doc_id,
            row.entity_id,
            row.surface,
            row.start,
            row.end,
            row.normalized_id,
            _family_key(family) if family else "",
            "|".join(row.relation_ids),
            "|".join(row.cves),
            status_by_key[(row.doc_id, row.entity_id)],
        ]
        lines.append(",".join(_csv_escape(value) for value in values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _apply_corrections(
    corrections_path: Path,
    corrections: dict[str, Any],
    decisions: list[dict[str, Any]],
    gold_dir: Path,
    audit: dict[str, Any],
    quarantine_path: Path,
) -> dict[str, Any]:
    if corrections.get("prediction_outputs_used") is not False:
        raise ValueError("corrections file must explicitly record prediction_outputs_used=false")
    audit_index = {
        (item["doc_id"], item["entity_id"]): item
        for item in audit["items"]
    }
    changes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for decision in decisions:
        key = (str(decision["doc_id"]), str(decision["entity_id"]))
        if key in seen:
            raise ValueError(f"duplicate correction decision: {key}")
        seen.add(key)
        if key not in audit_index:
            raise ValueError(f"correction target absent from audit: {key}")
        if decision.get("action") not in {"KEEP", "REVISE_CPE", "REJECT"}:
            raise ValueError(f"unsupported action for {key}: {decision.get('action')}")
        changes_by_doc[key[0]].append(decision)

    applied: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    for doc_id, decisions in sorted(changes_by_doc.items()):
        path = gold_dir / f"{doc_id}.json"
        before_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        document = json.loads(path.read_text(encoding="utf-8"))
        entities = list(document.get("entities", []))
        relations = list(document.get("relations", []))
        entity_by_id = {str(entity.get("id")): entity for entity in entities}
        for decision in decisions:
            entity_id = str(decision["entity_id"])
            entity = entity_by_id.get(entity_id)
            if entity is None:
                raise ValueError(f"entity missing while applying correction: {(doc_id, entity_id)}")
            old_cpe = str(entity.get("normalized_id") or "")
            if old_cpe.casefold() != str(decision["old_cpe"]).casefold():
                raise ValueError(f"old_cpe precondition failed for {(doc_id, entity_id)}")
            if decision["action"] == "KEEP":
                new_cpe = old_cpe
                removed_relations = []
            elif decision["action"] == "REVISE_CPE":
                new_cpe = str(decision["new_cpe"]).casefold()
                if _cpe_family(new_cpe) is None:
                    raise ValueError(f"invalid replacement CPE for {(doc_id, entity_id)}")
                entity["normalized_id"] = new_cpe
                entity["notes"] = str(decision["adjudication_basis"])
                removed_relations: list[str] = []
            else:
                removed_relation_objects = [
                    relation
                    for relation in relations
                    if str(relation.get("head")) == entity_id or str(relation.get("tail")) == entity_id
                ]
                removed_relations = [
                    str(relation.get("id"))
                    for relation in removed_relation_objects
                ]
                relations = [
                    relation
                    for relation in relations
                    if str(relation.get("head")) != entity_id and str(relation.get("tail")) != entity_id
                ]
                entities = [candidate for candidate in entities if str(candidate.get("id")) != entity_id]
                rejected_items.append({
                    "doc_id": doc_id,
                    "entity": entity,
                    "relations": removed_relation_objects,
                    "reason": decision["adjudication_basis"],
                    "source_urls": decision.get("source_urls", []),
                })
            applied.append({
                "doc_id": doc_id,
                "entity_id": entity_id,
                "surface": entity.get("text"),
                "action": decision["action"],
                "old_cpe": old_cpe,
                "new_cpe": decision.get("new_cpe"),
                "removed_relation_ids": removed_relations,
                "adjudication_basis": decision["adjudication_basis"],
            })
        document["entities"] = entities
        document["relations"] = relations
        document.setdefault("gold_review", {})["cpe_audit"] = {
            "version": corrections.get("version"),
            "prediction_outputs_used": False,
            "decision_count": len(decisions),
        }
        path.write_bytes(_json_bytes(document))
        after_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in applied:
            if item["doc_id"] == doc_id:
                item["document_before_sha256"] = before_sha256
                item["document_after_sha256"] = after_sha256

    quarantine = {
        "schema_version": "cpe-gold-quarantine-v1",
        "created_at_utc": _utc_now(),
        "prediction_outputs_used": False,
        "source_corrections": str(corrections_path),
        "rejected_count": len(rejected_items),
        "items": rejected_items,
    }
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_bytes(_json_bytes(quarantine))
    return {
        "applied_count": len(applied),
        "action_counts": dict(sorted(Counter(item["action"] for item in applied).items())),
        "quarantine": str(quarantine_path),
        "applied": applied,
    }


def _expand_corrections(
    corrections: dict[str, Any],
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit = list(corrections.get("decisions", []))
    expanded = list(explicit)
    for rule in corrections.get("rules", []):
        match = dict(rule.get("match", {}))
        matched = 0
        for item in audit["items"]:
            if "old_cpe" in match and item["gold_cpe"].casefold() != str(match["old_cpe"]).casefold():
                continue
            if "doc_ids" in match and item["doc_id"] not in set(match["doc_ids"]):
                continue
            if "entity_ids" in match and item["entity_id"] not in set(match["entity_ids"]):
                continue
            if "surfaces" in match and item["surface"] not in set(match["surfaces"]):
                continue
            if "statuses" in match and item["status"] not in set(match["statuses"]):
                continue
            decision = {
                key: value for key, value in rule.items()
                if key not in {"match", "rule_id"}
            }
            decision.update({
                "doc_id": item["doc_id"],
                "entity_id": item["entity_id"],
                "surface": item["surface"],
                "old_cpe": item["gold_cpe"],
                "rule_id": rule.get("rule_id"),
            })
            expanded.append(decision)
            matched += 1
        expected = rule.get("expected_matches")
        if expected is not None and matched != int(expected):
            raise ValueError(
                f"correction rule {rule.get('rule_id')} expected {expected} matches, got {matched}"
            )
        if matched == 0:
            raise ValueError(f"correction rule matched no audit items: {rule.get('rule_id')}")
    keys = [(str(item["doc_id"]), str(item["entity_id"])) for item in expanded]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate expanded correction targets: {duplicates}")
    return sorted(expanded, key=lambda item: (item["doc_id"], item["entity_id"]))


def _load_verified_applied_decisions(
    receipt_path: Path,
    pre_audit_path: Path,
    corrections_path: Path,
    gold_dir: Path,
) -> list[dict[str, Any]] | None:
    """Reuse the frozen decision ledger when verifying an already migrated Gold."""
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("corrections_sha256") != _sha256_bytes(corrections_path.read_bytes()):
        return None
    applied = list(receipt.get("application", {}).get("applied", []))
    if not applied:
        return None
    documents: dict[str, dict[str, Any]] = {}
    for decision in applied:
        doc_id = str(decision["doc_id"])
        if doc_id not in documents:
            path = gold_dir / f"{doc_id}.json"
            if not path.is_file():
                return None
            documents[doc_id] = json.loads(path.read_text(encoding="utf-8"))
        entity_by_id = {
            str(entity.get("id")): entity
            for entity in documents[doc_id].get("entities", [])
        }
        entity = entity_by_id.get(str(decision["entity_id"]))
        action = decision.get("action")
        if action == "REJECT":
            if entity is not None:
                return None
        elif entity is None:
            return None
        elif action == "REVISE_CPE":
            if str(entity.get("normalized_id", "")).casefold() != str(
                decision.get("new_cpe", "")
            ).casefold():
                return None
        elif action == "KEEP":
            if str(entity.get("normalized_id", "")).casefold() != str(
                decision.get("old_cpe", "")
            ).casefold():
                return None
        else:
            return None
    if pre_audit_path.is_file():
        pre_audit = json.loads(pre_audit_path.read_text(encoding="utf-8"))
        adjudicated = [
            item["adjudication"]
            for item in pre_audit.get("items", [])
            if item.get("adjudication")
        ]
        applied_keys = {
            (str(item["doc_id"]), str(item["entity_id"])) for item in applied
        }
        adjudicated_keys = {
            (str(item["doc_id"]), str(item["entity_id"])) for item in adjudicated
        }
        if adjudicated_keys == applied_keys:
            return adjudicated
    return applied


def _preserve_audit_timestamp_if_unchanged(
    audit_path: Path,
    audit: dict[str, Any],
) -> None:
    """Avoid hash drift when a completed audit is verified repeatedly."""
    if not audit_path.is_file():
        return
    existing = json.loads(audit_path.read_text(encoding="utf-8"))
    existing_core = dict(existing)
    current_core = dict(audit)
    existing_core.pop("created_at_utc", None)
    current_core.pop("created_at_utc", None)
    if existing_core == current_core and existing.get("created_at_utc"):
        audit["created_at_utc"] = existing["created_at_utc"]


def _write_review_tables(
    csv_path: Path,
    md_path: Path,
    decisions: list[dict[str, Any]],
) -> None:
    headers = [
        "doc_id", "entity_id", "action", "old_cpe", "new_cpe",
        "adjudication_basis", "source_urls",
    ]
    csv_lines = [",".join(headers)]
    md_lines = [
        "# CPE Gold 复裁台账 v1",
        "",
        "本台账只依据 Gold 原文、仓库归档 NVD CVE 配置和当前 NVD CPE 页面；未查看 APO/P0 预测。",
        "",
        "| 文档 | 实体 | 裁决 | 原 CPE | 新 CPE | 依据 |",
        "|---|---|---|---|---|---|",
    ]
    for decision in decisions:
        row = {
            **decision,
            "source_urls": " | ".join(decision.get("source_urls", [])),
        }
        csv_lines.append(",".join(_csv_escape(row.get(header, "")) for header in headers))
        basis = str(decision.get("adjudication_basis", "")).replace("|", "\\|")
        md_lines.append(
            f"| {decision['doc_id']} | {decision['entity_id']} | {decision['action']} | "
            f"`{decision['old_cpe']}` | `{decision.get('new_cpe', '')}` | {basis} |"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DIR)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--git-ref", default="HEAD")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--pre-audit", type=Path, default=DEFAULT_PRE_AUDIT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--review-md", type=Path, default=DEFAULT_REVIEW_MD)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--build-evidence", action="store_true")
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, _ = _load_gold(args.gold_dir)
    wanted_cves = {cve for row in rows for cve in row.cves}
    wanted_families = {
        family for row in rows
        if (family := _cpe_family(row.normalized_id))
    }
    if args.build_evidence:
        evidence = _load_archived_nvd_subset(
            args.repo, args.git_ref, wanted_cves, wanted_families
        )
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_bytes(_json_bytes(evidence))
    else:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    corrections = None
    decisions: list[dict[str, Any]] = []
    if args.corrections:
        corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
    audit = _audit(rows, evidence)
    if corrections:
        decisions = _load_verified_applied_decisions(
            args.receipt, args.pre_audit, args.corrections, args.gold_dir
        ) or _expand_corrections(corrections, audit)
        audit = _audit(rows, evidence, decisions)
    if not args.apply:
        _preserve_audit_timestamp_if_unchanged(args.audit, audit)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    application = None
    if args.apply:
        if not args.corrections:
            raise ValueError("--apply requires --corrections")
        args.pre_audit.write_bytes(_json_bytes(audit))
        application = _apply_corrections(
            args.corrections, corrections, decisions, args.gold_dir, audit,
            args.quarantine
        )
        rows, _ = _load_gold(args.gold_dir)
        audit = _audit(rows, evidence, decisions)
        receipt = {
            "schema_version": "cpe-gold-review-receipt-v1",
            "created_at_utc": _utc_now(),
            "prediction_outputs_used": False,
            "corrections_file": str(args.corrections),
            "corrections_sha256": _sha256_bytes(args.corrections.read_bytes()),
            "evidence_file": str(args.evidence),
            "evidence_sha256": _sha256_bytes(args.evidence.read_bytes()),
            "pre_audit_file": str(args.pre_audit),
            "pre_audit_sha256": _sha256_bytes(args.pre_audit.read_bytes()),
            "post_audit_file": str(args.audit),
            "post_audit_summary": audit["summary"],
            "application": application,
        }
        args.audit.write_bytes(_json_bytes(audit))
        receipt["post_audit_sha256"] = _sha256_bytes(args.audit.read_bytes())
        args.receipt.write_bytes(_json_bytes(receipt))
        _write_review_tables(args.review_csv, args.review_md, decisions)
    args.audit.write_bytes(_json_bytes(audit))
    _write_inventory(args.inventory, rows, audit)
    print(json.dumps({
        "evidence": str(args.evidence),
        "audit": str(args.audit),
        "inventory": str(args.inventory),
        "summary": audit["summary"],
        "missing_cves": evidence.get("missing_cves", []),
        "decision_count": len(decisions),
        "decision_action_counts": dict(sorted(Counter(
            decision["action"] for decision in decisions
        ).items())),
        "application": application,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
