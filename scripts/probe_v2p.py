"""输出范式 v2' 诊断探针（诊断专用，未冻结）。

状态：OUTPUT_CONTRACT_V2_DIAGNOSTIC / NOT_READY_FOR_RUN9。

v2' = v2 的禁枚举措辞 + 保留 text 接地证据（只省 id），
解析走冻结 parse_entity_mentions，与 v1 同窗 A/B 对比：
- budget exhaustion 是否为 0；
- strict entity metrics 相对 v1 是否不退化。

只跑 v2' 臂（v1 基线见 probe_output_contract_v2.py 产物）。
只写 results/output_contract_v2_diagnostic/，不碰正式目录。
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

from llm_methods import _loose_json, make_extractor, parse_entity_mentions  # noqa: E402
from protegi.evaluator import TaskEvaluator  # noqa: E402
from protegi.metrics import (  # noqa: E402
    aggregate_micro_f1,
    calc_strict_entity_sample_counts,
)
from protegi.optimizer import prepare_stage1_window_samples  # noqa: E402
from protegi.output_contract_v2 import (  # noqa: E402
    OUTPUT_PARADIGM_VERSION,
    build_v2p_entity_prompt,
)
from protegi.search_stability import WindowBudgetExhaustedError  # noqa: E402
from probe_output_contract_v2 import (  # noqa: E402
    DIAGNOSTIC_WINDOWS,
    TASK_RUNTIME,
    V1_SYSTEM,
    _parseability,
)

V2P_PARADIGM = OUTPUT_PARADIGM_VERSION + "-text-grounded-v1"


def _run_v2p(evaluator: TaskEvaluator, prompt: str, sample: dict) -> dict:
    try:
        raw = evaluator._call_task_model_once(
            stage="entity",
            prompt_content=prompt.replace("{text}", sample["text"]),
            system_content=V1_SYSTEM,
            sample_id=sample["sample_id"],
            full_prompt_for_hash=prompt,
        )
    except WindowBudgetExhaustedError as exc:
        return {
            "paradigm": V2P_PARADIGM,
            "budget_exhausted": True,
            "failure": str(exc)[:300],
            "entities": [],
            "parse_method": "none",
            "raw_chars": None,
        }
    parsed = _loose_json(raw)
    raw_entities = parsed.get("entities", [])
    entities, _ = parse_entity_mentions(sample["text"], raw_entities)
    from collections import Counter as _C

    spans = [(e.get("type"), e.get("start"), e.get("end")) for e in entities]
    dup = sum(1 for _, c in _C(spans).items() if c > 1)
    nested = 0
    keys = sorted(set(spans))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            t1, s1, e1 = keys[i]
            t2, s2, e2 = keys[j]
            if t1 != t2 or s2 >= e1:
                continue
            if s1 < e2 and s2 < e1:
                nested += 1
    return {
        "paradigm": V2P_PARADIGM,
        "budget_exhausted": False,
        "entities": entities,
        "parse_method": _parseability(raw),
        "raw_chars": len(raw),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "duplicate_spans": dup,
        "nested_overlap_pairs": nested,
    }


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
    split_file = EXP_DIR / "data" / "train_dev_test_split_v7.json"
    split = json.loads(split_file.read_text(encoding="utf-8"))
    doc_to_part = {}
    for part in ("train", "dev", "test"):
        for doc_id in split.get(part, []):
            doc_to_part[doc_id] = part

    docs = sorted({doc for _, doc, _ in DIAGNOSTIC_WINDOWS})
    samples = prepare_stage1_window_samples(
        docs, gold_dir, max_chars=3000, overlap=400
    )
    by_key = {(s["doc_id"], s["window_index"]): s for s in samples}
    freeze = json.loads(
        (EXP_DIR / "data" / "dataset_freeze_manifest_v6.json").read_text(
            encoding="utf-8"
        )
    )
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
    prompt = build_v2p_entity_prompt()

    records = []
    for cls, doc, idx in DIAGNOSTIC_WINDOWS:
        sample = by_key[(doc, idx)]
        result = _run_v2p(evaluator, prompt, sample)
        tp, fp, fn = calc_strict_entity_sample_counts(
            result["entities"], sample["gold_entities"]
        )
        metric = aggregate_micro_f1(tp, fp, fn).to_dict()
        result["strict"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1": metric["f1"],
        }
        result["pred_count"] = len(result["entities"])
        entry = {
            "class": cls,
            "sample_id": sample["sample_id"],
            "doc_id": doc,
            "split_part": doc_to_part.get(doc, "unknown"),
            "window_index": idx,
            "window_chars": len(sample["text"]),
            "gold_local_count": len(sample["gold_entities"]),
            "v2p": result,
        }
        records.append(entry)
        status = (
            "BUDGET_EXHAUSTED"
            if result["budget_exhausted"]
            else f"tp={tp} fp={fp} fn={fn} f1={metric['f1']:.3f} "
            f"dup={result['duplicate_spans']} nested={result['nested_overlap_pairs']}"
        )
        print(f"[{cls}] {sample['sample_id']} "
              f"(gold={len(sample['gold_entities'])}) v2p: {status}", flush=True)

    # 与 v1 基线（上一轮探针产物）同窗对比。
    v1_summary = json.loads(
        (out_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
    v1_by_id = {r["sample_id"]: r["calls"]["v1"] for r in v1_summary["records"]}
    comparison = []
    for entry in records:
        v1 = v1_by_id[entry["sample_id"]]
        v2p = entry["v2p"]
        comparison.append({
            "sample_id": entry["sample_id"],
            "class": entry["class"],
            "v1_budget": v1["budget_exhausted"],
            "v1_f1": v1["strict"]["f1"] if not v1["budget_exhausted"] else None,
            "v2p_budget": v2p["budget_exhausted"],
            "v2p_f1": v2p["strict"]["f1"] if not v2p["budget_exhausted"] else None,
            "v2p_dup": v2p.get("duplicate_spans"),
            "v2p_parse": v2p.get("parse_method"),
        })
    summary = {
        "diagnostic_status": "OUTPUT_CONTRACT_V2_DIAGNOSTIC",
        "run9_status": "NOT_READY_FOR_RUN9",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_runtime": TASK_RUNTIME,
        "v2p_paradigm": V2P_PARADIGM,
        "window": {"max_chars": 3000, "overlap": 400},
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "gold_aggregate_sha256": freeze.get("gold_aggregate_sha256"),
        "task_model_calls": evaluator.call_count,
        "transient_api_retries": evaluator.transient_api_retries,
        "budget_exhaustion_retries": evaluator.budget_exhaustion_retries,
        "records": records,
        "v1_comparison": comparison,
        "acceptance": {
            "v2p_budget_exhaustions": sum(
                1 for r in records if r["v2p"]["budget_exhausted"]
            ),
            "v2p_json_unparseable": sum(
                1 for r in records
                if not r["v2p"]["budget_exhausted"]
                and r["v2p"]["parse_method"] == "unparseable"
            ),
            "v2p_total_duplicates": sum(
                r["v2p"].get("duplicate_spans") or 0 for r in records
            ),
        },
    }
    (out_dir / "probe_v2p_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nv2p budget={summary['acceptance']['v2p_budget_exhaustions']}")
    print(f"汇总已写入: {out_dir / 'probe_v2p_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
