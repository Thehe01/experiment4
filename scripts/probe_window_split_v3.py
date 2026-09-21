"""WINDOW_SPLIT_V3 定向 P0 mini-probe（诊断专用，未冻结运行）。

状态：NOT_READY_FOR_RUN10（转 FROZEN/READY 由放行决定）。

- 新拆 aa24-317a_w4 子窗：各重复 3 次（旧 w4 bistable，一次成功不足）。
- 207a / 317a 原 killer 子窗：各至少 1 次确认。
- 冻结 v1 P0，同 runtime：hy3 / 0.0 / disabled / none / 0.95 / 4096
  （无 escalation）/ 3000/400 + V3 批准参数。
- 验收：全部正常终止、budget=0、evaluator 路径解析成功、
  无 duplicate explosion。
- 只写 results/window_split_v3_probe/，不跑搜索、不进 run10。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections import Counter
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

DENSE = {
    "dense_run_split": True,
    "dense_min_ids": 25,
    "dense_min_span": 1000,
    "dense_gap": 120,
    "dense_max_ids": 20,
    "dense_seam": 100,
}
# (doc, 子窗全局起止下界/上界, 重复次数)
PLAN = [
    ("aa24-317a", 9020, 11898, 3),    # 新拆 w4 children
    ("aa24-207a", 11830, 14711, 1),   # 原 killer children
    ("aa24-317a", 11498, 14498, 1),   # 原 killer children
]


def main() -> int:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault(
        "V5_OPENCODE_SESSION_ID", f"bron-v6-wsplit3-{uuid.uuid4().hex[:12]}"
    )
    os.environ.setdefault("V5_OPENCODE_USER_AGENT", "bron-wsplit3-research/1.0")

    out_dir = EXP_DIR / "results" / "window_split_v3_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_dir = EXP_DIR / "data" / "annotations" / "gold"
    split_file = EXP_DIR / "data" / "train_dev_test_split_v7.json"
    freeze = json.loads(
        (EXP_DIR / "data" / "dataset_freeze_manifest_v6.json").read_text(
            encoding="utf-8"
        )
    )
    docs = sorted({doc for doc, _, _, _ in PLAN})
    samples = prepare_stage1_window_samples(
        docs, gold_dir, max_chars=3000, overlap=400, **DENSE
    )
    by_doc = {}
    for sample in samples:
        by_doc.setdefault(sample["doc_id"], []).append(sample)

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
    for doc, lo, hi, trials in PLAN:
        children = sorted(
            (s for s in by_doc[doc]
             if s["window_start"] >= lo and s["window_end"] <= hi),
            key=lambda s: s["window_start"],
        )
        print(f"{doc} [{lo},{hi}]: {len(children)} children x{trials}",
              flush=True)
        assert children, f"no children for {doc} [{lo},{hi}]"
        for sample in children:
            for trial in range(1, trials + 1):
                try:
                    entities = evaluator.predict_stage1_window(
                        sample["text"], ENTITY_PROMPT_P0,
                        sample_id=sample["sample_id"],
                    )
                    row = {"budget_exhausted": False,
                           "pred_count": len(entities)}
                except WindowBudgetExhaustedError as exc:
                    entities = []
                    row = {"budget_exhausted": True,
                           "failure": str(exc)[:200], "pred_count": 0}
                tp, fp, fn = calc_strict_entity_sample_counts(
                    entities, sample["gold_entities"]
                )
                metric = aggregate_micro_f1(tp, fp, fn).to_dict()
                spans = [(e.get("type"), e.get("start"), e.get("end"))
                         for e in entities]
                dup = sum(1 for _, c in Counter(spans).items() if c > 1)
                row.update({
                    "sample_id": sample["sample_id"],
                    "trial": trial,
                    "window_chars": len(sample["text"]),
                    "gold_local_count": len(sample["gold_entities"]),
                    "parse_method": (
                        "none" if row["budget_exhausted"]
                        else "evaluator_path"
                    ),
                    "duplicate_spans": dup,
                    "strict": {"tp": tp, "fp": fp, "fn": fn,
                               "precision": metric["precision"],
                               "recall": metric["recall"],
                               "f1": metric["f1"]},
                })
                records.append(row)
                status = ("BUDGET_EXHAUSTED" if row["budget_exhausted"]
                          else f"tp={tp} fp={fp} fn={fn} "
                          f"f1={metric['f1']:.3f} dup={dup}")
                print(f"  {sample['sample_id']} t{trial} "
                      f"(gold={len(sample['gold_entities'])}) v1: {status}",
                      flush=True)

    summary = {
        "diagnostic_status": "WINDOW_SPLIT_V3_PROBE",
        "run10_status": "NOT_READY_FOR_RUN10",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_runtime": TASK_RUNTIME,
        "dense_params": DENSE,
        "prompt": "frozen_v1_P0",
        "window": {"max_chars": 3000, "overlap": 400},
        "split_sha256": hashlib.sha256(
            split_file.read_bytes()).hexdigest(),
        "gold_aggregate_sha256": freeze.get("gold_aggregate_sha256"),
        "window_construction": (
            freeze.get("window_construction") or {}
        ).get("version"),
        "task_model_calls": evaluator.call_count,
        "transient_api_retries": evaluator.transient_api_retries,
        "budget_exhaustion_retries": evaluator.budget_exhaustion_retries,
        "records": records,
        "acceptance": {
            "budget_exhaustions": sum(
                1 for r in records if r["budget_exhausted"]),
            "calls": len(records),
            "unparseable": sum(
                1 for r in records
                if not r["budget_exhausted"]
                and r["parse_method"] not in ("evaluator_path",)),
            "total_duplicates": sum(
                r.get("duplicate_spans") or 0 for r in records),
            "max_pred_to_gold_ratio": max(
                (r["pred_count"] / r["gold_local_count"]
                 for r in records if r["gold_local_count"]), default=0),
        },
    }
    (out_dir / "v3_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nV3 probe 完成：budget="
          f"{summary['acceptance']['budget_exhaustions']}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
