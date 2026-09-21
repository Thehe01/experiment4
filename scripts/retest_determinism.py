"""runaway 确定性复测（诊断专用，一次性）。

- v1 在 aa24-207a_w5 / aa24-317a_w5 上重跑：runaway 是否稳定复现；
- v2' 在 aa21-200b_w4 上重跑：空输出（abstain）是否稳定。
只写诊断目录，不碰正式产物。
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "scripts"))
sys.path.insert(0, str(EXP_DIR))

from llm_methods import make_extractor, parse_entity_mentions  # noqa: E402
from protegi.evaluator import TaskEvaluator  # noqa: E402
from protegi.metrics import (  # noqa: E402
    aggregate_micro_f1,
    calc_strict_entity_sample_counts,
)
from protegi.optimizer import prepare_stage1_window_samples  # noqa: E402
from protegi.output_contract_v2 import build_v2p_entity_prompt  # noqa: E402
from protegi.prompts_p0 import ENTITY_PROMPT_P0  # noqa: E402
from protegi.search_stability import WindowBudgetExhaustedError  # noqa: E402
from probe_output_contract_v2 import TASK_RUNTIME, V1_SYSTEM  # noqa: E402

PLAN = [("v1", "aa24-207a", 5), ("v1", "aa24-317a", 5), ("v2p", "aa21-200b", 4)]


def main() -> int:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.setdefault(
        "V5_OPENCODE_SESSION_ID", f"bron-v6-v2diag-{uuid.uuid4().hex[:12]}"
    )
    os.environ.setdefault("V5_OPENCODE_USER_AGENT", "bron-v2diag-research/1.0")

    out_dir = EXP_DIR / "results" / "output_contract_v2_diagnostic"
    gold_dir = EXP_DIR / "data" / "annotations" / "gold"
    docs = sorted({d for _, d, _ in PLAN})
    samples = prepare_stage1_window_samples(
        docs, gold_dir, max_chars=3000, overlap=400
    )
    by_key = {(s["doc_id"], s["window_index"]): s for s in samples}
    client = make_extractor(
        model=TASK_RUNTIME["model"], temperature=TASK_RUNTIME["temperature"],
        thinking=TASK_RUNTIME["thinking"], max_tokens=TASK_RUNTIME["max_tokens"],
        top_p=TASK_RUNTIME["top_p"],
        reasoning_effort=TASK_RUNTIME["reasoning_effort"],
    )
    evaluator = TaskEvaluator(task_client=client, max_workers=1)
    v2p_prompt = build_v2p_entity_prompt()

    rows = []
    for arm, doc, idx in PLAN:
        sample = by_key[(doc, idx)]
        prompt = ENTITY_PROMPT_P0 if arm == "v1" else v2p_prompt
        prompt_content = (
            prompt.replace("{text}", sample["text"])
            if "{text}" in prompt else prompt
        )
        try:
            if arm == "v1":
                entities = evaluator.predict_stage1_window(
                    sample["text"], ENTITY_PROMPT_P0,
                    sample_id=sample["sample_id"],
                )
            else:
                raw = evaluator._call_task_model_once(
                    stage="entity", prompt_content=prompt_content,
                    system_content=V1_SYSTEM,
                    sample_id=sample["sample_id"],
                    full_prompt_for_hash=prompt,
                )
                from llm_methods import _loose_json as _lj

                entities, _ = parse_entity_mentions(
                    sample["text"], _lj(raw).get("entities", [])
                )
            tp, fp, fn = calc_strict_entity_sample_counts(
                entities, sample["gold_entities"]
            )
            m = aggregate_micro_f1(tp, fp, fn).to_dict()
            row = {"arm": arm, "sample_id": sample["sample_id"],
                   "budget_exhausted": False, "tp": tp, "fp": fp,
                   "fn": fn, "f1": m["f1"], "pred": len(entities)}
        except WindowBudgetExhaustedError as exc:
            row = {"arm": arm, "sample_id": sample["sample_id"],
                   "budget_exhausted": True, "failure": str(exc)[:200]}
        rows.append(row)
        print(row, flush=True)

    (out_dir / "determinism_retest.json").write_text(
        json.dumps({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "task_runtime": TASK_RUNTIME,
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
