"""Export the Chapter 3 boundary-recertification queue from the audit report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = EXP_DIR / "results" / "boundary_sync_audit_v1.json"
DEFAULT_CSV = EXP_DIR / "results" / "boundary_sync_review_v1.csv"
DEFAULT_MD = EXP_DIR / "results" / "boundary_sync_review_v1.md"


def _row(document: dict, item: dict, severity: str) -> dict:
    return {
        "split": document.get("split"),
        "doc_id": document.get("doc_id"),
        "severity": severity,
        "kind": item.get("kind"),
        "item_id": item.get("entity_id") or item.get("relation_id"),
        "surface_or_technique": item.get("surface") or item.get("technique"),
        "normalized_id": item.get("normalized_id"),
        "evidence_preview": item.get("evidence_preview")
        or item.get("local_fact_block"),
        "previous_basis": item.get("previous_basis"),
        "decision": "",
        "adjudication_basis": "",
        "reviewer": "",
        "review_date": "",
    }


def export(audit_path: Path, csv_path: Path, md_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = []
    for document in audit.get("documents", []):
        rows.extend(
            _row(document, item, "blocking")
            for item in document.get("blocking_semantic_errors", [])
        )
        rows.extend(
            _row(document, item, "review")
            for item in document.get("review_candidates", [])
        )
    grouped = {}
    for row in rows:
        key = (row["split"], row["doc_id"], row["item_id"])
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = row
            continue
        previous["kind"] = ";".join(
            sorted(set(previous["kind"].split(";")) | {row["kind"]})
        )
        if row["severity"] == "blocking":
            previous["severity"] = "blocking"
        for field in (
            "surface_or_technique",
            "normalized_id",
            "evidence_preview",
            "previous_basis",
        ):
            if not previous.get(field) and row.get(field):
                previous[field] = row[field]
    rows = sorted(
        grouped.values(),
        key=lambda row: (
            str(row["split"]),
            str(row["doc_id"]),
            str(row["item_id"]),
        ),
    )
    fields = list(rows[0]) if rows else [
        "split",
        "doc_id",
        "severity",
        "kind",
        "item_id",
        "surface_or_technique",
        "normalized_id",
        "evidence_preview",
        "previous_basis",
        "decision",
        "adjudication_basis",
        "reviewer",
        "review_date",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_kind = Counter(
        kind
        for row in rows
        for kind in str(row["kind"]).split(";")
    )
    summary = audit.get("summary", {})
    lines = [
        "# 第三章边界同步复裁清单",
        "",
        f"契约版本：`{audit.get('boundary_contract_version')}`",
        "",
        "该清单由 Gold 与原文规则审计生成，不读取模型预测。旧 Gold 和旧 test 结果均未改写。",
        "",
        "## 门禁状态",
        "",
        f"- 审计文档：{summary.get('documents', 0)} 篇",
        f"- 阻断性语义问题：{summary.get('blocking_semantic_errors', 0)} 项",
        f"- 需重新裁决的 exploited_by：{summary.get('exploited_by_boundary_recertification_required', 0)} 条",
        f"- 去重后复裁对象：{len(rows)} 个",
        f"- 当前门禁：`{audit.get('gate_status')}`",
        "",
        "## 分类数量",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{kind}` | {count} |" for kind, count in sorted(by_kind.items()))
    lines.extend([
        "",
        "## 裁决要求",
        "",
        "1. Weakness 只有在同一事实块存在唯一对应的明示 CWE 时保留；否则移入候选区。",
        "2. Configuration 删除普通发布版本、补丁和 build，只保留最小可独立识别产品跨度；完整 CPE URI 与唯一规范化所需的约定代际标识除外。",
        "3. exploited_by 逐条判断 ATT&CK 技术是否直接承担该 CVE 的利用或入口行为；利用后活动不得保留，且不得使用固定 T 编号黑名单代替语义裁决。",
        "4. 每条保留关系的 adjudication_basis 写入 `chapter3-boundary-sync-v1` 和可回指的句、表格行或列表作用域。",
        "5. 完成 train/dev/test 全部复裁后重新审计、重新冻结；在此之前不运行 APO 或 test。",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"rows": len(rows), "by_kind": dict(sorted(by_kind.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    print(json.dumps(export(args.audit, args.csv, args.markdown), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
