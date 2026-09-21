"""Freeze the MCPU-v2 repaired v6 Gold and bind its audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
MANIFEST_FILE = EXP_DIR / "data" / "dataset_freeze_manifest_v6.json"
BASE_MANIFEST = EXP_DIR / "data" / "dataset_freeze_manifest_v5.json"

sys.path.insert(0, str(EXP_DIR / "scripts"))
from llm_methods import (  # noqa: E402
    build_text_windows,
    window_split_version,
)


SUPPORTING_EVIDENCE = {
    "review_receipt": "results/human_reannotation_gold_status.json",
    "audit_report": "results/gold_strategy_audit_v6_v7.json",
    "boundary_audit_report": "results/boundary_sync_audit_v1.json",
    "leakage_report": "results/split_leakage_audit_v6_v7.json",
    "guideline": "ANNOTATION_GUIDELINE.md",
    "boundary_review_csv": "results/boundary_sync_review_v1.csv",
    "boundary_review_md": "results/boundary_sync_review_v1.md",
    "boundary_contract_v1": "config/boundary_contract_v1.json",
    "boundary_contract": "config/boundary_contract_v2.json",
    "configuration_mcpu_policy": "config/configuration_boundary_mcpu_v2.json",
    "configuration_mcpu_audit": "results/configuration_boundary_mcpu_v2_audit.json",
    "configuration_mcpu_receipt": "results/configuration_boundary_mcpu_v2_receipt.json",
    "configuration_mcpu_review_csv": "results/configuration_boundary_mcpu_v2_review.csv",
    "configuration_mcpu_review_md": "results/configuration_boundary_mcpu_v2_review.md",
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


def compute_window_inventory(
    split: dict,
    *,
    dense_run_split: bool = False,
    dense_min_ids: int | None = None,
    dense_min_span: int | None = None,
    dense_gap: int | None = None,
    dense_max_ids: int | None = None,
    dense_seam: int | None = None,
) -> dict:
    """计算窗口构造清单（确定性，只依赖 Gold 文本与构造参数）。

    默认记录 window-split-v1（dense 关闭）基线清单；构造参数批准后，
    用批准的参数重算并刷新冻结，清单哈希变化即证明窗 inventory 已变。
    """
    version = window_split_version(
        dense_run_split, dense_min_ids, dense_min_span,
        dense_gap, dense_max_ids, dense_seam,
    )
    window_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for split_name in ("train", "dev", "test"):
        count = 0
        for doc_id in split[split_name]:
            document = json.loads(
                (GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8")
            )
            windows = build_text_windows(
                document.get("text", ""),
                max_chars=3000,
                overlap=400,
                dense_run_split=dense_run_split,
                dense_min_ids=dense_min_ids,
                dense_min_span=dense_min_span,
                dense_gap=dense_gap,
                dense_max_ids=dense_max_ids,
                dense_seam=dense_seam,
            )
            count += len(windows)
            for win in windows:
                digest.update(
                    f"{doc_id}\0{win['start']}\0{win['end']}\0".encode("utf-8")
                )
                digest.update(
                    hashlib.sha256(win["text"].encode("utf-8")).digest()
                )
        window_counts[split_name] = count
    return {
        "version": version,
        "dense_run_split": bool(dense_run_split),
        "dense_params": {
            "dense_min_ids": dense_min_ids,
            "dense_min_span": dense_min_span,
            "dense_gap": dense_gap,
            "dense_max_ids": dense_max_ids,
            "dense_seam": dense_seam,
        },
        "max_chars": 3000,
        "overlap": 400,
        "inventory_sha256": digest.hexdigest(),
        "window_counts": window_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze dataset manifest v6")
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验现有冻结清单完整性，不重新生成或写入",
    )
    parser.add_argument(
        "--dense-run-split",
        action="store_true",
        help="用 WINDOW_SPLIT_V3 记录窗口清单（默认记录 v1 基线）",
    )
    parser.add_argument("--dense-min-ids", type=int, default=None)
    parser.add_argument("--dense-min-span", type=int, default=None)
    parser.add_argument("--dense-gap", type=int, default=None)
    parser.add_argument("--dense-max-ids", type=int, default=None)
    parser.add_argument("--dense-seam", type=int, default=None)
    parser.add_argument(
        "--params-final",
        action="store_true",
        help="声明本次记录的构造参数为已批准终值（默认暂定）",
    )
    args = parser.parse_args()

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
            if document.get("annotation_protocol_version") != (
                "4.6-mcpu-mention-fact-dual-layer-v1"
            ):
                raise ValueError(f"stale annotation protocol in {doc_id}")
            if document.get("boundary_contract_version") != (
                "chapter3-boundary-sync-v2"
            ):
                raise ValueError(f"stale boundary contract in {doc_id}")
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

    if args.check:
        if not MANIFEST_FILE.is_file():
            raise FileNotFoundError(f"缺少冻结清单文件：{MANIFEST_FILE}")
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if manifest.get("gold_aggregate_sha256") != aggregate.hexdigest():
            raise ValueError("Gold 聚合哈希与当前标注不一致，冻结校验失败！")
        if manifest.get("split_sha256") != _sha256(SPLIT_FILE):
            raise ValueError("划分文件哈希与冻结清单不一致，冻结校验失败！")
        base_rec = manifest.get("base_manifest", {})
        if _sha256(BASE_MANIFEST) != base_rec.get("sha256"):
            raise ValueError("基础清单哈希与冻结清单不一致，冻结校验失败！")
        for doc_id, digest in hashes.items():
            if manifest.get("gold_document_sha256", {}).get(doc_id) != digest:
                raise ValueError(f"Gold 文档哈希不一致 ({doc_id})，冻结校验失败！")
        for name, expected_rec in manifest.get("supporting_evidence", {}).items():
            ev_path = EXP_DIR / expected_rec["path"]
            if not ev_path.is_file() or _sha256(ev_path) != expected_rec["sha256"]:
                raise ValueError(f"支持证据哈希不一致 ({name})，冻结校验失败！")
        recorded_construction = manifest.get("window_construction")
        if not isinstance(recorded_construction, dict):
            raise ValueError("冻结清单缺少 window_construction 记录，冻结校验失败！")
        recorded_params = recorded_construction.get("dense_params") or {}
        live_inventory = compute_window_inventory(
            split,
            dense_run_split=bool(recorded_construction.get("dense_run_split")),
            dense_min_ids=recorded_params.get("dense_min_ids"),
            dense_min_span=recorded_params.get("dense_min_span"),
            dense_gap=recorded_params.get("dense_gap"),
            dense_max_ids=recorded_params.get("dense_max_ids"),
            dense_seam=recorded_params.get("dense_seam"),
        )
        for field in ("inventory_sha256", "window_counts", "version"):
            if live_inventory[field] != recorded_construction.get(field):
                raise ValueError(
                    f"窗口清单 {field} 与冻结记录不一致，冻结校验失败！"
                )
        print(json.dumps({
            "dataset_version": manifest["dataset_version"],
            "gold_document_count": manifest["gold_document_count"],
            "gold_aggregate_sha256": manifest["gold_aggregate_sha256"],
            "supporting_evidence": sorted(manifest.get("supporting_evidence", {})),
            "window_construction": recorded_construction.get("version"),
            "window_counts": recorded_construction.get("window_counts"),
            "check_status": "passed",
        }, ensure_ascii=False, indent=2))
        return

    previous = (
        json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if MANIFEST_FILE.is_file()
        else {}
    )
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    frozen_at = (
        previous.get("frozen_at_utc")
        if (
            previous.get("gold_aggregate_sha256") == aggregate.hexdigest()
            and previous.get("split_sha256") == _sha256(SPLIT_FILE)
            and previous.get("frozen_at_utc")
        )
        else now
    )
    manifest = {
        "schema_version": "chapter3-no-capec-v1",
        "dataset_version": "v6-v5-gold-v9-mcpu-v2",
        "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
        "boundary_contract_version": "chapter3-boundary-sync-v2",
        "cpe_normalization_contract_version": "cpe-gold-audit-v1",
        "status": "frozen_mcpu_v2_pending_human_iaa_and_new_final_test",
        "frozen_at_utc": frozen_at,
        "gate_status": "gate_passed",
        "controlled_test_rerun_ready": False,
        "formal_experiment_ready": False,
        "formal_experiment_blocker": (
            "verifiable human IAA results and a newly reserved unseen final test "
            "are not yet available"
        ),
        "window_construction": {
            **compute_window_inventory(
                split,
                dense_run_split=bool(args.dense_run_split),
                dense_min_ids=args.dense_min_ids,
                dense_min_span=args.dense_min_span,
                dense_gap=args.dense_gap,
                dense_max_ids=args.dense_max_ids,
                dense_seam=args.dense_seam,
            ),
            "computed_at_utc": now,
            "params_provisional": not bool(args.params_final),
        },
        "base_manifest": {
            "path": "data/dataset_freeze_manifest_v5.json",
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
        "window_construction": manifest["window_construction"]["version"],
        "window_counts": manifest["window_construction"]["window_counts"],
        "params_provisional": manifest["window_construction"][
            "params_provisional"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
