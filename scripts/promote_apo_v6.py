"""Promote a validated non-P0 v6 APO artifact to the test-runnable state.

This is deliberately separate from the optimizer. It updates only the v6
review status after checking that the artifact was frozen on the current v7
development split and that no test Gold or predictions were used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from schema import ANNOTATION_PROTOCOL_VERSION, BOUNDARY_CONTRACT_VERSION


EXP_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
STATUS_FILE = EXP_DIR / "data" / "review_status.json"
DEFAULT_ARTIFACT = EXP_DIR / "results" / "apo_optimization_v6" / "final_prompt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote(artifact_path: Path) -> dict:
    if artifact_path.resolve() != DEFAULT_ARTIFACT.resolve():
        raise ValueError(
            "only the canonical v6 artifact can be promoted: "
            f"{DEFAULT_ARTIFACT}"
        )
    if not artifact_path.is_file():
        raise FileNotFoundError(f"APO artifact not found: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("selected_candidate") == "p0_manual":
        raise ValueError(
            "APO 只选中 P0 基线；不晋级，也不得进入 apo/apo_full 测试"
        )
    if not (
        str(artifact.get("stage1_guidance", "")).strip()
        or str(artifact.get("stage2_guidance", "")).strip()
    ):
        raise ValueError(
            "APO 产物 guidance 为空；当前只有 P0 基线，不进入 apo/apo_full 测试"
        )
    required = {
        "schema_version": "chapter3-no-capec-v1",
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "split": "dev",
        "split_file": SPLIT_FILE.name,
        "final_quality_gate_passed": True,
        "apo_candidate_selected": True,
        "frozen_for_test": True,
        "test_gold_loaded": False,
        "test_predictions_generated": False,
    }
    for key, expected in required.items():
        if artifact.get(key) != expected:
            raise ValueError(f"APO artifact gate failed: {key}={artifact.get(key)!r}")
    current_split_hash = _sha256(SPLIT_FILE)
    if artifact.get("split_sha256") != current_split_hash:
        raise ValueError("APO artifact is not bound to the current v7 split")

    status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    boundary = status.get("boundary_sync", {})
    if (
        boundary.get("contract_version") != BOUNDARY_CONTRACT_VERSION
        or boundary.get("complete") is not True
    ):
        raise ValueError(
            "Gold 尚未完成 chapter3-boundary-sync-v2 MCPU 复裁与重新冻结；"
            "不得晋级 APO 提示或运行 test"
        )
    status["apo_prompt_status"] = {
        "ready": True,
        "artifact": "results/apo_optimization_v6/final_prompt.json",
        "artifact_sha256": _sha256(artifact_path),
        "split": "dev",
        "split_sha256": current_split_hash,
        "reason": "A non-P0 APO candidate passed paired/full-dev quality gates and was frozen for controlled test rerun.",
    }
    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return status["apo_prompt_status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    print(json.dumps(promote(args.artifact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
