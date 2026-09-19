"""Freeze the repaired v5 Gold and bind all boundary-sync evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
MANIFEST_FILE = EXP_DIR / "data" / "dataset_freeze_manifest_v5.json"
BASE_MANIFEST = EXP_DIR.parent / "v4.5" / "data" / "dataset_freeze_manifest_v13.json"


SUPPORTING_EVIDENCE = {
    "review_receipt": "results/human_reannotation_gold_status.json",
    "audit_report": "results/gold_strategy_audit_v5_v7.json",
    "boundary_audit_report": "results/boundary_sync_audit_v1.json",
    "leakage_report": "results/split_leakage_audit_v5_v7.json",
    "paired_bootstrap": "results/paired_bootstrap_v5.json",
    "guideline": "ANNOTATION_GUIDELINE.md",
    "boundary_review_csv": "results/boundary_sync_review_v1.csv",
    "boundary_review_md": "results/boundary_sync_review_v1.md",
    "boundary_contract": "config/boundary_contract_v1.json",
    "boundary_quarantine": (
        "data/annotations/quarantine/chapter3-boundary-sync-v1/rejected_items.json"
    ),
    "cpe_corrections": "config/cpe_gold_corrections_v1.json",
    "cpe_nvd_evidence": "results/cpe_gold_nvd_evidence_v1.json",
    "cpe_audit_pre": "results/cpe_gold_audit_pre_v1.json",
    "cpe_audit": "results/cpe_gold_audit_v1.json",
    "cpe_review_receipt": "results/cpe_gold_review_v1.json",
    "cpe_review_csv": "results/cpe_gold_review_v1.csv",
    "cpe_review_md": "results/cpe_gold_review_v1.md",
    "cpe_quarantine": (
        "data/annotations/quarantine/cpe-gold-audit-v1/rejected_items.json"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    partition_stats: dict[str, dict] = {}
    aggregate = hashlib.sha256()

    for split_name in ("train", "dev", "test"):
        entity_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        relation_documents: Counter[str] = Counter()
        for doc_id in split[split_name]:
            path = GOLD_DIR / f"{doc_id}.json"
            digest = _sha256(path)
            hashes[doc_id] = digest
            document = json.loads(path.read_text(encoding="utf-8"))
            entity_counts.update(entity["type"] for entity in document["entities"])
            relation_counts.update(relation["type"] for relation in document["relations"])
            relation_documents.update(
                {relation["type"] for relation in document["relations"]}
            )
        partition_stats[split_name] = {
            "documents": len(split[split_name]),
            "entities": dict(sorted(entity_counts.items())),
            "entity_total": sum(entity_counts.values()),
            "relations": dict(sorted(relation_counts.items())),
            "relation_total": sum(relation_counts.values()),
            "relation_documents": dict(sorted(relation_documents.items())),
        }

    for doc_id in sorted(hashes):
        aggregate.update(f"{doc_id}\0{hashes[doc_id]}\n".encode("utf-8"))

    supporting = {}
    for name, relative in SUPPORTING_EVIDENCE.items():
        path = EXP_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing freeze evidence: {path}")
        supporting[name] = {"path": relative, "sha256": _sha256(path)}

    previous = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "chapter3-no-capec-v1",
        "dataset_version": "v5-v4.5-gold-v8-cpe-audit-v1",
        "annotation_protocol_version": "4.5-mention-fact-dual-layer-v1",
        "boundary_contract_version": "chapter3-boundary-sync-v1",
        "cpe_normalization_contract_version": "cpe-gold-audit-v1",
        "status": "frozen_cpe_gold_audit_v1_pending_human_iaa",
        "frozen_at_utc": now,
        "gate_status": "gate_passed",
        "controlled_test_rerun_ready": True,
        "formal_experiment_ready": False,
        "formal_experiment_blocker": "verifiable human IAA results are not yet available",
        "base_manifest": {
            "path": "../v4.5/data/dataset_freeze_manifest_v13.json",
            "sha256": _sha256(BASE_MANIFEST),
            "dataset_version": base["dataset_version"],
        },
        "split_file": "data/train_dev_test_split_v7.json",
        "split_sha256": _sha256(SPLIT_FILE),
        "split_membership_changed": previous.get("split_membership_changed", True),
        "partition_stats": partition_stats,
        "gold_directory": "data/annotations/gold",
        "gold_document_count": len(hashes),
        "gold_aggregate_sha256": aggregate.hexdigest(),
        "gold_document_sha256": dict(sorted(hashes.items())),
        "supporting_evidence": supporting,
        "provenance_refreshed_at_utc": now,
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset_version": manifest["dataset_version"],
        "gold_document_count": manifest["gold_document_count"],
        "gold_aggregate_sha256": manifest["gold_aggregate_sha256"],
        "supporting_evidence": sorted(supporting),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
