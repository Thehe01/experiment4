"""输出范式 v2 poison-window 诊断探针（诊断专用，未冻结）。

状态：OUTPUT_CONTRACT_V2_DIAGNOSTIC / NOT_READY_FOR_RUN9。

- 同 runtime 对比 v1（冻结 P0）与 v2（最小字段 + 显式禁枚举）：
  model=hy3, temperature=0.0, thinking=disabled, reasoning=none,
  top_p=0.95, max_tokens=4096（无 escalation）, window=3000/400。
- 诊断窗：A 类高密 x4 + B 类零 Gold x4 + 正常窗 x4，共 12 窗 x 2 范式。
- 验收：budget exhaustion=0、JSON 可解析、duplicate=0、
  strict entity metrics 不退化（相对同窗 v1）。
- 只写 results/output_contract_v2_diagnostic/，不碰正式 run 目录、
  不写检查点、不触发 promotion、不进 run9。

用法：
  python -X utf8 scripts/probe_output_contract_v2.py
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

from llm_methods import (  # noqa: E402
    _loose_json,
    make_extractor,
    parse_entity_mentions,
)
from protegi.evaluator import TaskEvaluator  # noqa: E402
from protegi.metrics import (  # noqa: E402
    aggregate_micro_f1,
    calc_strict_entity_sample_counts,
)
from protegi.optimizer import prepare_stage1_window_samples  # noqa: E402
from protegi.output_contract_v2 import (  # noqa: E402
    OUTPUT_PARADIGM_VERSION,
    build_v2_entity_prompt,
    parse_minimal_entity_mentions,
)
from protegi.prompts_p0 import ENTITY_PROMPT_P0  # noqa: E402
from protegi.search_stability import WindowBudgetExhaustedError  # noqa: E402

# 诊断窗：A(高密) x4 + B(零 Gold) x4 + 正常 x4。aa24-207a 属 dev，其余 train。
DIAGNOSTIC_WINDOWS = [
    ("A", "aa24-207a", 5),
    ("A", "aa23-215a", 4),
    ("A", "aa24-317a", 5),
    ("A", "aa24-317a", 4),
    ("B", "aa23-349a", 16),
    ("B", "aa23-278a", 18),
    ("B", "aa23-074a", 16),
    ("B", "aa22-277a", 7),
    ("N", "aa22-138b-vmware-cve-22954", 2),
    ("N", "aa22-320a", 2),
    ("N", "aa20-259a-iran-citrix-vpn-cve-19781", 0),
    ("N", "aa21-200b", 4),
]

TASK_RUNTIME = {
    "model": "hy3",
    "temperature": 0.0,
    "thinking": "disabled",
    "reasoning_effort": "none",
    "top_p": 0.95,
    "max_tokens": 4096,
    "window_chars": 3000,
    "window_overlap": 400,
}

V1_SYSTEM = (
    "You are a helpful cybersecurity intelligence annotation expert. "
    "Output valid JSON only."
)


def _parseability(raw: str) -> str:
    try:
        json.loads(raw)
        return "strict"
    except (json.JSONDecodeError, ValueError):
        pass
    import re as _re

    if _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, _re.DOTALL):
        try:
            json.loads(
                _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, _re.DOTALL).group(1)
            )
            return "fenced"
        except (json.JSONDecodeError, ValueError):
            pass
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            json.loads(raw[a:b + 1])
            return "brace_slice"
        except (json.JSONDecodeError, ValueError):
            pass
    return "unparseable"


def _run_v1(evaluator: TaskEvaluator, sample: dict) -> dict:
    try:
        entities = evaluator.predict_stage1_window(
            sample["text"],
            ENTITY_PROMPT_P0,
            sample_id=sample["sample_id"],
        )
    except WindowBudgetExhaustedError as exc:
        return {
            "paradigm": "v1_frozen_p0",
            "budget_exhausted": True,
            "failure": str(exc)[:300],
            "entities": [],
            "parse_method": "none",
            "raw_chars": None,
        }
    from collections import Counter as _C

    spans = [(e.get("type"), e.get("start"), e.get("end")) for e in entities]
    dup = sum(1 for _, c in _C(spans).items() if c > 1)
    return {
        "paradigm": "v1_frozen_p0",
        "budget_exhausted": False,
        "entities": entities,
        "parse_method": "evaluator_path",
        "raw_chars": None,
        "duplicate_spans": dup,
        "nested_overlap_pairs": None,
    }


def _run_v2(evaluator: TaskEvaluator, v2_prompt: str, sample: dict) -> dict:
    prompt_content = v2_prompt.replace("{text}", sample["text"])
    try:
        raw = evaluator._call_task_model_once(
            stage="entity",
            prompt_content=prompt_content,
            system_content=V1_SYSTEM,
            sample_id=sample["sample_id"],
            full_prompt_for_hash=v2_prompt,
        )
    except WindowBudgetExhaustedError as exc:
        return {
            "paradigm": OUTPUT_PARADIGM_VERSION,
            "budget_exhausted": True,
            "failure": str(exc)[:300],
            "entities": [],
            "parse_method": "none",
            "raw_chars": None,
        }
    parsed = _loose_json(raw)
    raw_entities = parsed.get("entities", [])
    entities, diag = parse_minimal_entity_mentions(sample["text"], raw_entities)
    return {
        "paradigm": OUTPUT_PARADIGM_VERSION,
        "budget_exhausted": False,
        "entities": entities,
        "parse_method": _parseability(raw),
        "raw_chars": len(raw),
        "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "duplicate_spans": diag["duplicates_collapsed"],
        "nested_overlap_pairs": diag["nested_overlap_pairs"],
        "invalid_dropped": diag["invalid_dropped"],
        "text_mismatches": diag["text_mismatches"],
        "raw_count": diag["raw_count"],
    }


def main() -> int:
    # 进程内代理白名单修复（no repo change 约定）：宿主 NO_PROXY 含
    # "[::1]" 会使 httpx 解析 mount 时抛 Invalid port；覆盖为安全值。
    # 远端 API 仍走 HTTP(S)_PROXY，与正式 run 一致。
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

    wanted = {(doc, idx) for _, doc, idx in DIAGNOSTIC_WINDOWS}
    needed_docs = sorted({doc for _, doc, _ in DIAGNOSTIC_WINDOWS})
    all_samples = prepare_stage1_window_samples(
        needed_docs, gold_dir, max_chars=3000, overlap=400
    )
    by_key = {(s["doc_id"], s["window_index"]): s for s in all_samples}
    missing = [
        f"{doc}_w{idx}" for doc, idx in sorted(wanted - set(by_key))
    ]
    if missing:
        raise RuntimeError(f"诊断窗构造缺失: {missing}")

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
    assert int(client.config.get("max_escalated_tokens")) == 4096, (
        "诊断禁止 budget escalation"
    )
    evaluator = TaskEvaluator(task_client=client, max_workers=1)
    v2_prompt = build_v2_entity_prompt()

    records = []
    for cls, doc, idx in DIAGNOSTIC_WINDOWS:
        sample = by_key[(doc, idx)]
        entry = {
            "class": cls,
            "sample_id": sample["sample_id"],
            "doc_id": doc,
            "split_part": doc_to_part.get(doc, "unknown"),
            "window_index": idx,
            "window_chars": len(sample["text"]),
            "gold_local_count": len(sample["gold_entities"]),
            "calls": {},
        }
        for name, fn in (
            ("v1", lambda: _run_v1(evaluator, sample)),
            ("v2", lambda: _run_v2(evaluator, v2_prompt, sample)),
        ):
            result = fn()
            tp, fp, fn_ = calc_strict_entity_sample_counts(
                result["entities"], sample["gold_entities"]
            )
            metric = aggregate_micro_f1(tp, fp, fn_).to_dict()
            result["strict"] = {
                "tp": tp, "fp": fp, "fn": fn_,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
            }
            result["pred_count"] = len(result["entities"])
            entry["calls"][name] = result
            status = (
                "BUDGET_EXHAUSTED"
                if result["budget_exhausted"]
                else f"tp={tp} fp={fp} fn={fn_} "
                f"f1={metric['f1']:.3f} dup={result.get('duplicate_spans')}"
            )
            print(
                f"[{cls}] {sample['sample_id']} "
                f"(gold={len(sample['gold_entities'])}) {name}: {status}",
                flush=True,
            )
        records.append(entry)

    summary = {
        "diagnostic_status": "OUTPUT_CONTRACT_V2_DIAGNOSTIC",
        "run9_status": "NOT_READY_FOR_RUN9",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_runtime": TASK_RUNTIME,
        "v2_paradigm": OUTPUT_PARADIGM_VERSION,
        "v1_paradigm": "frozen_p0",
        "window": {"max_chars": 3000, "overlap": 400},
        "split_sha256": hashlib.sha256(
            split_file.read_bytes()
        ).hexdigest(),
        "gold_aggregate_sha256": freeze.get("gold_aggregate_sha256"),
        "task_model_calls": evaluator.call_count,
        "transient_api_retries": evaluator.transient_api_retries,
        "budget_exhaustion_retries": evaluator.budget_exhaustion_retries,
        "records": records,
    }

    budget = {
        name: sum(1 for r in records if r["calls"][name]["budget_exhausted"])
        for name in ("v1", "v2")
    }
    summary["acceptance"] = {
        "v1_budget_exhaustions": budget["v1"],
        "v2_budget_exhaustions": budget["v2"],
        "v2_json_unparseable": sum(
            1 for r in records
            if not r["calls"]["v2"]["budget_exhausted"]
            and r["calls"]["v2"]["parse_method"] == "unparseable"
        ),
        "v2_total_duplicates": sum(
            r["calls"]["v2"].get("duplicate_spans") or 0 for r in records
        ),
    }
    (out_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n诊断完成：v1 budget={budget['v1']}, v2 budget={budget['v2']}")
    print(f"汇总已写入: {out_dir / 'probe_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
