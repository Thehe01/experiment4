"""v2 偏移误差特征分析（诊断专用，一次性）。

在 aa22-320a_w2（正常窗）与 aa24-207a_w5（A 类）上重跑 v2，
保存原始输出，并把每条发射记录与同类型最近 Gold 做比对。
只读 Gold、不改任何契约，只写诊断目录。
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "scripts"))
sys.path.insert(0, str(EXP_DIR))

from llm_methods import _loose_json, make_extractor  # noqa: E402
from protegi.evaluator import TaskEvaluator  # noqa: E402
from protegi.optimizer import prepare_stage1_window_samples  # noqa: E402
from protegi.output_contract_v2 import (  # noqa: E402
    build_v2_entity_prompt,
    parse_minimal_entity_mentions,
)

TARGETS = [("aa22-320a", 2), ("aa24-207a", 5)]
SYSTEM = (
    "You are a helpful cybersecurity intelligence annotation expert. "
    "Output valid JSON only."
)


def main() -> int:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault(
        "V5_OPENCODE_SESSION_ID", f"bron-v6-v2diag-{uuid.uuid4().hex[:12]}"
    )
    os.environ.setdefault("V5_OPENCODE_USER_AGENT", "bron-v2diag-research/1.0")

    out_dir = EXP_DIR / "results" / "output_contract_v2_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = EXP_DIR / "data" / "annotations" / "gold"
    docs = sorted({d for d, _ in TARGETS})
    samples = prepare_stage1_window_samples(
        docs, gold_dir, max_chars=3000, overlap=400
    )
    by_key = {(s["doc_id"], s["window_index"]): s for s in samples}
    client = make_extractor(
        model="hy3", temperature=0.0, thinking="disabled",
        max_tokens=4096, top_p=0.95, reasoning_effort="none",
    )
    evaluator = TaskEvaluator(task_client=client, max_workers=1)
    v2_prompt = build_v2_entity_prompt()

    for doc, idx in TARGETS:
        sample = by_key[(doc, idx)]
        raw = evaluator._call_task_model_once(
            stage="entity",
            prompt_content=v2_prompt.replace("{text}", sample["text"]),
            system_content=SYSTEM,
            sample_id=sample["sample_id"],
            full_prompt_for_hash=v2_prompt,
        )
        (out_dir / f"offset_case_{sample['sample_id']}_raw.txt").write_text(
            raw, encoding="utf-8"
        )
        parsed = _loose_json(raw)
        raws = parsed.get("entities", [])
        entities, diag = parse_minimal_entity_mentions(sample["text"], raws)
        gold_by_type: dict[str, list] = {}
        for g in sample["gold_entities"]:
            gold_by_type.setdefault(g["type"], []).append(g)
        rows = []
        for e in entities[:40]:
            cands = gold_by_type.get(e["type"], [])
            best = min(
                cands,
                key=lambda g: abs(g["start"] - e["start"]) + abs(g["end"] - e["end"]),
                default=None,
            )
            rows.append({
                "emit": [e["type"], e["start"], e["end"],
                         sample["text"][e["start"]:e["end"]][:40]],
                "nearest_gold": (
                    [best["start"], best["end"],
                     sample["text"][best["start"]:best["end"]][:40]]
                    if best else None
                ),
                "norm": e.get("normalized_id"),
            })
        # 发射记录原文 vs 切片一致性（v2 无 text 字段，检查长度分布）。
        lens = sorted({e["end"] - e["start"] for e in entities})
        analysis = {
            "sample_id": sample["sample_id"],
            "gold_count": len(sample["gold_entities"]),
            "emitted": len(entities),
            "diag": diag,
            "emitted_span_lengths_sorted_unique": lens[:20],
            "text_len": len(sample["text"]),
            "rows": rows,
        }
        (out_dir / f"offset_case_{sample['sample_id']}_analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{sample['sample_id']}: emitted={len(entities)} "
              f"lengths={lens[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
