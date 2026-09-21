"""WINDOW_SPLIT_V2 P0 mini-probe（诊断专用，未冻结参数）。

状态：WINDOW_SPLIT_V2_IMPLEMENTATION_APPROVED / NOT_READY_FOR_RUN9。

- 用推荐暂定参数（25/1000/120/20/100）重窗 aa24-207a（dev）与
  aa24-317a（train）全部窗口，跑冻结 v1 P0（run9 种子的硬要求）。
- 同 runtime：hy3 / temp 0.0 / disabled / none / top_p 0.95 /
  max_tokens 4096（无 escalation）/ 3000/400。
- 验收：budget exhaustion=0、JSON 可解析（evaluator 路径）、
  strict metrics sane（相对 legacy 同 doc 窗可比）。
- 只写 results/window_split_p0_probe/，不碰正式目录、不进 run9。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "scripts"))
sys.path.insert(0, str(EXP_DIR))

from llm_methods import make_extractor  # noqa: E402
from protegi.evaluator import TaskEvaluator  # noqa: E402
from protegi.metrics import (  # noqa: E402
    aggregate_micro_f1,
    calc_strict_entity_sample_counts,
)
from protegi.optimizer import prepare_stage1_window_samples  # noqa: E402
from protegi.prompts_p0 import ENTITY_PROMPT_P0  # noqa: E402
from protegi.search_stability import WindowBudgetExhaustedError  # noqa: E402
from probe_output_contract_v2 import TASK_RUNTIME  # noqa: E402

# 推荐暂定参数（敏感性分析结论，NOT FROZEN）。
DENSE = {
    "dense_run_split": True,
    "dense_min_ids": 25,
    "dense_min_span": 1000,
    "dense_gap": 120,
    "dense_max_ids": 20,
    "dense_seam": 100,
}
DOCS = ["aa24-207a", "aa24-317a"]


def main() -> int:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault(
        "V5_OPENCODE_SESSION_ID", f"bron-v6-wsplit-{uuid.uuid4().hex[:12]}"
    )
    os.environ.setdefault("V5_OPENCODE_USER_AGENT", "bron-wsplit-research/1.0")

    out_dir = EXP_DIR / "results" / "window_split_p0_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = EXP_DIR / "data" / "annotations" / "gold"
    split_file = EXP_DIR / "data" / "train_dev_test_split_v7.json"
    split = json.loads(split_file.read_text(encoding="utf-8"))
    doc_to_part = {}
    for part in ("train", "dev", "test"):
        for doc_id in split.get(part, []):
            doc_to_part[doc_id] = part
    freeze = json.loads(
        (EXP_DIR / "data" / "dataset_freeze_manifest_v6.json").read_text(
            encoding="utf-8"
        )
    )

    samples = prepare_stage1_window_samples(
        DOCS, gold_dir, max_chars=3000, overlap=400, **DENSE
    )
    print(f"重窗样本数: {len(samples)}", flush=True)
    client = make_extractor(
        model=TASK_RUNTIME["model"],
        temperature=TASK_RUNTIME["temperature"],
        thinking=TASK_RUNTIME["thinking"],
        max_tokens=TASK_RUNTIME["max_tokens"],
        top_p=TASK_RUNTIME["top_p"],
        reasoning_effort=TASK_RUNTIME["reasoning_effort"],
    )
    assert int(client.config.get("max_tokens")) == 4096
    assert int(client.config.get("max_escalated_tokens")) == 4096
    evaluator = TaskEvaluator(task_client=client, max_workers=1)

    records = []
    for sample in samples:
        try:
            entities = evaluator.predict_stage1_window(
                sample["text"], ENTITY_PROMPT_P0,
                sample_id=sample["sample_id"],
            )
            row = {"budget_exhausted": False, "pred_count": len(entities)}
        except WindowBudgetExhaustedError as exc:
            entities = []
            row = {"budget_exhausted": True,
                   "failure": str(exc)[:200], "pred_count": 0}
        tp, fp, fn = calc_strict_entity_sample_counts(
            entities, sample["gold_entities"]
        )
        metric = aggregate_micro_f1(tp, fp, fn).to_dict()
        row.update({
            "sample_id": sample["sample_id"],
            "doc_id": sample["doc_id"],
            "split_part": doc_to_part.get(sample["doc_id"], "unknown"),
            "window_chars": len(sample["text"]),
            "gold_local_count": len(sample["gold_entities"]),
            "strict": {"tp": tp, "fp": fp, "fn": fn,
                       "precision": metric["precision"],
                       "recall": metric["recall"], "f1": metric["f1"]},
        })
        records.append(row)
        status = ("BUDGET_EXHAUSTED" if row["budget_exhausted"]
                  else f"tp={tp} fp={fp} fn={fn} f1={metric['f1']:.3f}")
        print(f"{sample['sample_id']} (gold={len(sample['gold_entities'])}) "
              f"v1: {status}", flush=True)

    summary = {
        "diagnostic_status": "WINDOW_SPLIT_V2_IMPLEMENTATION_APPROVED",
        "run9_status": "NOT_READY_FOR_RUN9",
        "params_provisional": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_runtime": TASK_RUNTIME,
        "dense_params": DENSE,
        "prompt": "frozen_v1_P0",
        "window": {"max_chars": 3000, "overlap": 400},
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "gold_aggregate_sha256": freeze.get("gold_aggregate_sha256"),
        "task_model_calls": evaluator.call_count,
        "transient_api_retries": evaluator.transient_api_retries,
        "budget_exhaustion_retries": evaluator.budget_exhaustion_retries,
        "records": records,
        "acceptance": {
            "budget_exhaustions": sum(
                1 for r in records if r["budget_exhausted"]),
            "windows": len(records),
        },
    }
    (out_dir / "p0_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nP0 mini-probe 完成：budget="
          f"{summary['acceptance']['budget_exhaustions']}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
