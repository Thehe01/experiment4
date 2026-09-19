"""离线核对并汇总 constrained / unconstrained ProTeGi pilot 结果。

本脚本不调用模型、不读取 Test；只有两臂运行均完成后才生成比较报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


EXPECTED_PAIR_ID = "protegi-prompt-scope-pilot-v1"
IGNORED_CONFIG_KEYS = {
    "prompt_scope",
    "_config_file",
    "_config_file_sha256",
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"结果文件顶层必须为对象: {path}")
    return data


def _load_arm(result_dir: Path, expected_scope: str) -> dict:
    summary = _load_json(result_dir / "summary.json")
    dev = _load_json(result_dir / "final_dev_evaluations.json")
    if summary.get("stage") != "entity" or dev.get("stage") != "entity":
        raise ValueError(f"{expected_scope} 不是 Stage 1 entity 结果")
    if summary.get("prompt_scope") != expected_scope:
        raise ValueError(
            f"{expected_scope} summary 的 prompt_scope={summary.get('prompt_scope')!r}"
        )
    if dev.get("prompt_scope") != expected_scope:
        raise ValueError(
            f"{expected_scope} Dev 记录的 prompt_scope={dev.get('prompt_scope')!r}"
        )
    if summary.get("experiment_pair_id") != EXPECTED_PAIR_ID:
        raise ValueError(f"{expected_scope} experiment_pair_id 不匹配")
    if dev.get("experiment_pair_id") != EXPECTED_PAIR_ID:
        raise ValueError(f"{expected_scope} Dev experiment_pair_id 不匹配")

    candidates = dev.get("candidates", [])
    winner_id = dev.get("winner_candidate_id")
    winner = next(
        (candidate for candidate in candidates if candidate.get("candidate_id") == winner_id),
        None,
    )
    if winner is None or winner.get("selected") is not True:
        raise ValueError(f"{expected_scope} 未找到唯一获胜 Dev 候选")
    p0 = next(
        (candidate for candidate in candidates if candidate.get("is_p0_baseline") is True),
        None,
    )
    if p0 is None:
        raise ValueError(f"{expected_scope} Dev 记录缺少 P0 baseline")

    prompt_file = result_dir / "final_entity_prompt.txt"
    if not prompt_file.is_file():
        raise FileNotFoundError(f"{expected_scope} 缺少 final_entity_prompt.txt")
    prompt_text = prompt_file.read_text(encoding="utf-8")
    canonical_prompt = prompt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    prompt_hash = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
    if prompt_hash != winner.get("prompt_sha256"):
        raise ValueError(f"{expected_scope} 最终提示文件与 Dev 获胜候选哈希不一致")
    if prompt_hash != summary.get("winner_prompt_sha256"):
        raise ValueError(f"{expected_scope} 最终提示文件与 summary 哈希不一致")
    return {
        "summary": summary,
        "dev": dev,
        "winner": winner,
        "p0": p0,
        "winner_prompt_length": len(prompt_text),
    }


def _comparable_config(config: Dict[str, Any]) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in IGNORED_CONFIG_KEYS
    }


def _winner_payload(arm: dict) -> dict:
    evaluation = arm["winner"].get("evaluation", {})
    details = evaluation.get("details", {})
    summary = arm["summary"]
    return {
        "candidate_id": arm["winner"].get("candidate_id"),
        "prompt_sha256": arm["winner"].get("prompt_sha256"),
        "prompt_length": arm["winner_prompt_length"],
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "by_type": details.get("by_type", {}),
        "normalization": details.get("normalization", {}),
        "prompt_scope_audit": arm["winner"].get("prompt_scope_audit", {}),
        "task_max_workers": summary.get("runtime", {}).get("task_max_workers"),
        "call_stats": summary.get("call_stats", {}),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _decision(constrained: dict, unconstrained: dict) -> dict:
    c_f1 = constrained.get("f1")
    u_f1 = unconstrained.get("f1")
    if not isinstance(c_f1, (int, float)) or not isinstance(u_f1, (int, float)):
        return {
            "status": "invalid_metrics",
            "provisional_choice": None,
            "reason": "一个或两个实验臂缺少可比较的 Dev F1",
        }
    unconstrained_ranks_first = u_f1 > c_f1
    ranking_reason = "无约束臂 Dev F1 更高"
    if u_f1 < c_f1:
        ranking_reason = "有约束臂 Dev F1 更高"
    if u_f1 == c_f1:
        c_precision = constrained.get("precision")
        u_precision = unconstrained.get("precision")
        if not isinstance(c_precision, (int, float)) or not isinstance(
            u_precision, (int, float)
        ):
            unconstrained_ranks_first = False
            ranking_reason = "两臂 Dev F1 相同且 Precision 不可比较"
        elif u_precision > c_precision:
            unconstrained_ranks_first = True
            ranking_reason = "两臂 Dev F1 相同，无约束臂 Precision 更高"
        elif u_precision == c_precision and unconstrained["prompt_length"] < constrained[
            "prompt_length"
        ]:
            unconstrained_ranks_first = True
            ranking_reason = "两臂 Dev F1 与 Precision 相同，无约束提示更短"
        else:
            unconstrained_ranks_first = False
            ranking_reason = "两臂 Dev F1 相同，按 Precision、提示长度和最终保守规则排序"

    if not unconstrained_ranks_first:
        return {
            "status": "provisional_complete",
            "provisional_choice": "constrained",
            "reason": f"{ranking_reason}；按预注册规则选择 constrained",
        }

    audit = unconstrained.get("prompt_scope_audit", {})
    if audit.get("frozen_contract_exact_match") is True:
        return {
            "status": "provisional_complete",
            "provisional_choice": "unconstrained",
            "reason": f"{ranking_reason}且最终提示仍精确匹配冻结契约",
        }
    return {
        "status": "pending_boundary_audit",
        "provisional_choice": None,
        "reason": (
            f"{ranking_reason}，但最终提示不再精确匹配冻结契约；"
            "必须先完成人工/协议边界漂移审计，不能自动晋级"
        ),
    }


def _markdown(report: dict) -> str:
    c = report["arms"]["constrained"]
    u = report["arms"]["unconstrained"]
    lines = [
        "# ProTeGi 小规模提示范围对比",
        "",
        f"- 配对标识：`{report['experiment_pair_id']}`",
        "- 范围：固定 pilot Train/Dev，仅用于流程与正向信号判断，不是正式 Test 结论。",
        "",
        "| 实验臂 | Precision | Recall | F1 | 最终提示长度 | Task 并发上限 | Task 调用 | Muse 调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| constrained | {_fmt(c['precision'])} | {_fmt(c['recall'])} | "
            f"{_fmt(c['f1'])} | {c['prompt_length']} | {c['task_max_workers']} | "
            f"{c['call_stats'].get('task_model_calls', 'N/A')} | "
            f"{c['call_stats'].get('optimizer_model_calls', 'N/A')} |"
        ),
        (
            f"| unconstrained | {_fmt(u['precision'])} | {_fmt(u['recall'])} | "
            f"{_fmt(u['f1'])} | {u['prompt_length']} | {u['task_max_workers']} | "
            f"{u['call_stats'].get('task_model_calls', 'N/A')} | "
            f"{u['call_stats'].get('optimizer_model_calls', 'N/A')} |"
        ),
        "",
        "## 分类型 F1",
        "",
        "| 实体类型 | constrained | unconstrained |",
        "|---|---:|---:|",
    ]
    all_types = sorted(set(c["by_type"]) | set(u["by_type"]))
    for entity_type in all_types:
        lines.append(
            f"| {entity_type} | "
            f"{_fmt(c['by_type'].get(entity_type, {}).get('f1'))} | "
            f"{_fmt(u['by_type'].get(entity_type, {}).get('f1'))} |"
        )
    lines.extend([
        "",
        "## 预注册晋级判断",
        "",
        f"- 状态：`{report['decision']['status']}`",
        f"- 暂定选择：`{report['decision']['provisional_choice']}`",
        f"- 依据：{report['decision']['reason']}",
        "",
        "> 本报告不读取 Test，也不证明统计显著性。无约束提示的运行接口有效不等于第三章边界一致。",
        "",
    ])
    return "\n".join(lines)


def compare(constrained_dir: Path, unconstrained_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"比较输出目录已存在，拒绝覆盖: {output_dir}")
    constrained = _load_arm(constrained_dir, "constrained")
    unconstrained = _load_arm(unconstrained_dir, "unconstrained")

    for scope, arm in (
        ("constrained", constrained),
        ("unconstrained", unconstrained),
    ):
        if arm["summary"].get("runtime", {}).get("task_max_workers") != 8:
            raise ValueError(f"{scope} 任务模型并发上限不是 8")

    if _comparable_config(constrained["summary"].get("config", {})) != _comparable_config(
        unconstrained["summary"].get("config", {})
    ):
        raise ValueError("两臂运行时配置除 prompt_scope/配置文件绑定外不一致")
    c_bindings = constrained["summary"].get("input_bindings", {})
    u_bindings = unconstrained["summary"].get("input_bindings", {})
    if c_bindings.get("split_file_sha256_raw_bytes") != u_bindings.get(
        "split_file_sha256_raw_bytes"
    ):
        raise ValueError("两臂 split 文件哈希不一致")
    if constrained["p0"].get("prompt_sha256") != unconstrained["p0"].get(
        "prompt_sha256"
    ):
        raise ValueError("两臂 P0 提示哈希不一致")

    split_path = Path(str(c_bindings.get("split_file", "")))
    pilot_split = _load_json(split_path)
    if pilot_split.get("test") != []:
        raise ValueError("pilot split 的 test 不为空，拒绝生成比较结论")

    c_payload = _winner_payload(constrained)
    u_payload = _winner_payload(unconstrained)
    report = {
        "experiment_pair_id": EXPECTED_PAIR_ID,
        "stage": "entity",
        "split_file": str(split_path),
        "split_file_sha256": c_bindings.get("split_file_sha256_raw_bytes"),
        "test_documents_loaded": False,
        "arms": {
            "constrained": c_payload,
            "unconstrained": u_payload,
        },
        "decision": _decision(c_payload, u_payload),
        "limitations": [
            "small fixed Train/Dev pilot",
            "no test evaluation",
            "no statistical significance claim",
            "unconstrained winner requires separate Chapter 3 boundary audit",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constrained-dir", type=Path, required=True)
    parser.add_argument("--unconstrained-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.constrained_dir.resolve(),
        args.unconstrained_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
