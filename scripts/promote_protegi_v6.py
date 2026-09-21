"""Promote a validated two-stage ProTeGi optimization to formal final artifact.

Reads:
- final_entity_prompt.txt
- final_relation_prompt.txt
- Stage 1 optimization summary (summary.json)
- Stage 2 optimization summary (summary.json)
- entity cache train manifest (entity_cache_train_manifest.json)
- entity cache dev manifest (entity_cache_dev_manifest.json)
- dataset_freeze_manifest_v6.json
- train_dev_test_split_v7.json

Enforces all boundary contracts, protocol versions, split & gold hashes,
and formal eligibility before generating:
results/protegi_final/protegi_final_artifact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from schema import (
    ANNOTATION_PROTOCOL_VERSION,
    BOUNDARY_CONTRACT_VERSION,
    SCHEMA_VERSION,
)
from protegi.contract_validator import PromptContractValidator
from protegi.entity_cache import compute_prompt_hash
from protegi.runtime_contract import TASK_RUNTIME_FIELDS, validate_task_runtime

DEFAULT_FREEZE_MANIFEST = EXP_DIR / "data" / "dataset_freeze_manifest_v6.json"
DEFAULT_SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
DEFAULT_OUTPUT_ARTIFACT = (
    EXP_DIR / "results" / "protegi_final" / "protegi_final_artifact.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EXP_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


_MISSING = object()


def _extract_task_runtime(summary: dict, label: str) -> dict:
    """从单个 Stage summary 的 recorded effective runtime 提取完整 Task Runtime。

    正式 artifact 只能冻结“运行时实际记录下来的 runtime”，而不是 promotion
    第二次推导出来的 runtime。因此优先读取
    ``summary["config"]["effective_task_runtime"]`` 并要求 11 字段齐全。
    formal_eligible=true 的新 ProTeGi summary 缺失该字段即 hard fail；
    仅非 formal 历史产物才允许走旧推导路径（测试 fixture 按需更新）。
    """
    config = summary.get("config", {}) or {}
    effective = config.get("effective_task_runtime")
    if effective is not None:
        validate_task_runtime(effective, label=f"{label} effective_task_runtime")
        return {
            "model": str(effective["model"]),
            "max_workers": int(effective["max_workers"]),
            "temperature": float(effective["temperature"]),
            "thinking": str(effective["thinking"]).strip().lower(),
            "reasoning_effort": str(effective["reasoning_effort"]).strip().lower(),
            "top_p": float(effective["top_p"]),
            "max_tokens": int(effective["max_tokens"]),
            "window_chars": int(effective["window_chars"]),
            "window_overlap": int(effective["window_overlap"]),
            "document_abbreviation_context": bool(
                effective["document_abbreviation_context"]
            ),
            "vulnerability_anchored_backfill": bool(
                effective["vulnerability_anchored_backfill"]
            ),
        }
    if summary.get("formal_eligible") is True:
        raise ValueError(
            f"{label} summary 缺少 config.effective_task_runtime 字段；"
            "正式 artifact 只能使用运行时记录的 effective runtime，"
            "禁止 promotion 重新推导默认值！"
        )
    runtime = summary.get("runtime", {}) or {}
    task_client_cfg = runtime.get("task_client", {}) or {}
    runtime_max_workers = runtime.get("task_max_workers")

    def _from_config_or_client(
        config_key: str,
        client_key: str | None = None,
        *,
        required: bool = True,
        default=_MISSING,
    ):
        if config_key in config and config[config_key] is not None:
            return config[config_key]
        if (
            client_key
            and isinstance(task_client_cfg, dict)
            and client_key in task_client_cfg
            and task_client_cfg[client_key] is not None
        ):
            return task_client_cfg[client_key]
        if config_key == "task_max_workers" and runtime_max_workers is not None:
            return runtime_max_workers
        if default is not _MISSING:
            return default
        if required:
            raise ValueError(
                f"{label} summary 缺少 Task Runtime 字段: config[{config_key}]"
                + (f"/runtime.task_client[{client_key}]" if client_key else "")
            )
        return None

    raw_model = _from_config_or_client("task_model", "model")
    raw_workers = _from_config_or_client("task_max_workers", None)
    raw_temp = _from_config_or_client("task_temperature", "temperature")
    raw_thinking = _from_config_or_client("task_thinking", "thinking")
    raw_effort = _from_config_or_client(
        "task_reasoning_effort", "reasoning_effort"
    )
    raw_top_p = _from_config_or_client("task_top_p", "top_p")
    raw_max_tokens = _from_config_or_client("task_max_tokens", "max_tokens")
    raw_window_chars = _from_config_or_client(
        "window_chars", "window_chars", required=False, default=_MISSING
    )
    if raw_window_chars is _MISSING:
        raw_window_chars = config.get(
            "window_max_chars", config.get("max_chars", 3000)
        )
    raw_window_overlap = _from_config_or_client(
        "window_overlap", "window_overlap", required=False, default=_MISSING
    )
    if raw_window_overlap is _MISSING:
        raw_window_overlap = config.get("overlap", 400)
    raw_abbrev = _from_config_or_client(
        "document_abbreviation_context", None, required=False, default=False
    )
    raw_backfill = _from_config_or_client(
        "vulnerability_anchored_backfill", None, required=False, default=False
    )

    try:
        task_runtime = {
            "model": str(raw_model),
            "max_workers": int(raw_workers),
            "temperature": float(raw_temp),
            "thinking": str(raw_thinking).strip().lower(),
            "reasoning_effort": str(raw_effort).strip().lower(),
            "top_p": float(raw_top_p),
            "max_tokens": int(raw_max_tokens),
            "window_chars": int(raw_window_chars),
            "window_overlap": int(raw_window_overlap),
            "document_abbreviation_context": bool(raw_abbrev),
            "vulnerability_anchored_backfill": bool(raw_backfill),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} summary Task Runtime 类型非法: {exc}") from exc
    if task_runtime["max_workers"] <= 0:
        raise ValueError(f"{label} summary max_workers 必须为正整数")
    if task_runtime["window_chars"] <= 0:
        raise ValueError(f"{label} summary window_chars 必须为正整数")
    if not 0 <= task_runtime["window_overlap"] < task_runtime["window_chars"]:
        raise ValueError(
            f"{label} summary 窗口参数非法: "
            f"window_chars={task_runtime['window_chars']}, "
            f"window_overlap={task_runtime['window_overlap']}"
        )
    return task_runtime


def promote_protegi(
    *,
    entity_prompt_path: Path,
    relation_prompt_path: Path,
    entity_summary_path: Path,
    relation_summary_path: Path,
    entity_cache_train_manifest_path: Path,
    entity_cache_dev_manifest_path: Path,
    prompt_scope: str = "constrained",
    output_artifact_path: Path = DEFAULT_OUTPUT_ARTIFACT,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
    split_file_path: Path = DEFAULT_SPLIT_FILE,
) -> dict:
    entity_prompt_path = Path(entity_prompt_path)
    relation_prompt_path = Path(relation_prompt_path)
    entity_summary_path = Path(entity_summary_path)
    relation_summary_path = Path(relation_summary_path)
    entity_cache_train_manifest_path = Path(entity_cache_train_manifest_path)
    entity_cache_dev_manifest_path = Path(entity_cache_dev_manifest_path)
    freeze_manifest_path = Path(freeze_manifest_path)
    split_file_path = Path(split_file_path)
    output_artifact_path = Path(output_artifact_path)

    # 1. 显式拒绝旧 APO 产物冒充 ProTeGi
    for p in (
        entity_prompt_path,
        relation_prompt_path,
        entity_summary_path,
        relation_summary_path,
    ):
        p_str = str(p).lower()
        if "apo_optimization" in p_str or p.name == "final_prompt.json":
            raise ValueError(
                f"检测到历史 APO 产物路径 {p}；历史 APO 产物不得晋级为正式 ProTeGi 产物！"
            )
        if p.is_file() and p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if (
                    "stage1_guidance" in data
                    or "stage2_guidance" in data
                    or "apo_candidate_selected" in data
                ):
                    raise ValueError(
                        f"检测到历史 APO 产物结构 {p}；禁止旧 APO 产物冒充正式 ProTeGi 产物！"
                    )
            except (json.JSONDecodeError, OSError):
                pass

    # 2. 检查所有必需输入文件是否存在
    required_files = {
        "entity_prompt": entity_prompt_path,
        "relation_prompt": relation_prompt_path,
        "entity_summary": entity_summary_path,
        "relation_summary": relation_summary_path,
        "entity_cache_train_manifest": entity_cache_train_manifest_path,
        "entity_cache_dev_manifest": entity_cache_dev_manifest_path,
        "freeze_manifest": freeze_manifest_path,
        "split_file": split_file_path,
    }
    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"晋级必需文件不存在 ({name}): {path}")

    # 3. 校验冻结清单
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    freeze_sha = _sha256(freeze_manifest_path)
    split_sha = _sha256(split_file_path)
    if freeze_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"冻结清单 schema_version ({freeze_manifest.get('schema_version')}) 与当前 ({SCHEMA_VERSION}) 不一致"
        )
    if freeze_manifest.get("annotation_protocol_version") != ANNOTATION_PROTOCOL_VERSION:
        raise ValueError(
            f"冻结清单 annotation_protocol_version ({freeze_manifest.get('annotation_protocol_version')}) "
            f"与当前 ({ANNOTATION_PROTOCOL_VERSION}) 不一致"
        )
    if freeze_manifest.get("boundary_contract_version") != BOUNDARY_CONTRACT_VERSION:
        raise ValueError(
            f"冻结清单 boundary_contract_version ({freeze_manifest.get('boundary_contract_version')}) "
            f"与当前 ({BOUNDARY_CONTRACT_VERSION}) 不一致"
        )
    if freeze_manifest.get("split_sha256") != split_sha:
        raise ValueError(
            f"冻结清单 split_sha256 ({freeze_manifest.get('split_sha256')}) 与当前切分文件哈希 ({split_sha}) 不一致"
        )
    gold_agg_sha = freeze_manifest.get("gold_aggregate_sha256")
    if not gold_agg_sha:
        raise ValueError("冻结清单缺少 gold_aggregate_sha256 字段")

    # 4. 校验优化摘要 (Stage 1 & Stage 2)
    entity_summary = json.loads(entity_summary_path.read_text(encoding="utf-8"))
    relation_summary = json.loads(relation_summary_path.read_text(encoding="utf-8"))

    if entity_summary.get("formal_eligible") is not True:
        raise ValueError(
            "Stage 1 优化产物未标记为 formal_eligible=true（使用了非正式自定义切分/dry-run或未通过门禁），禁止晋级！"
        )
    if relation_summary.get("formal_eligible") is not True:
        raise ValueError(
            "Stage 2 优化产物未标记为 formal_eligible=true（使用了非正式自定义切分/dry-run或未通过门禁），禁止晋级！"
        )
    if entity_summary.get("dry_run") is True or relation_summary.get("dry_run") is True:
        raise ValueError("检测到 dry-run 优化产物，禁止晋级为正式 ProTeGi 产物！")

    # 4a. 校验 summary.stage
    if entity_summary.get("stage") != "entity":
        raise ValueError(
            f"Stage 1 summary stage 必须为 'entity'，当前={entity_summary.get('stage')!r}"
        )
    if relation_summary.get("stage") != "relation":
        raise ValueError(
            f"Stage 2 summary stage 必须为 'relation'，当前={relation_summary.get('stage')!r}"
        )

    # 4b. 校验 Prompt Scope（三方一致）
    ent_scope = entity_summary.get("prompt_scope")
    rel_scope = relation_summary.get("prompt_scope")
    if ent_scope != prompt_scope:
        raise ValueError(
            f"Stage 1 summary prompt_scope ({ent_scope!r}) 与晋级参数 ({prompt_scope!r}) 不一致"
        )
    if rel_scope != prompt_scope:
        raise ValueError(
            f"Stage 2 summary prompt_scope ({rel_scope!r}) 与晋级参数 ({prompt_scope!r}) 不一致"
        )
    if ent_scope != rel_scope:
        raise ValueError(
            f"Stage 1 与 Stage 2 prompt_scope 不一致: {ent_scope!r} != {rel_scope!r}"
        )

    # 4c. 校验 experiment_pair_id（双方一致并写入 artifact）
    ent_pair = entity_summary.get("experiment_pair_id")
    rel_pair = relation_summary.get("experiment_pair_id")
    if not ent_pair or not rel_pair:
        raise ValueError(
            "Stage 1/Stage 2 summary 缺少 experiment_pair_id 字段，禁止晋级！"
        )
    if ent_pair != rel_pair:
        raise ValueError(
            f"Stage 1 与 Stage 2 experiment_pair_id 不一致: {ent_pair!r} != {rel_pair!r}"
        )
    experiment_pair_id = ent_pair

    # 5. 校验提示词内容及结构契约
    entity_prompt_text = entity_prompt_path.read_text(encoding="utf-8")
    relation_prompt_text = relation_prompt_path.read_text(encoding="utf-8")
    val_ent = PromptContractValidator.validate_candidate(
        "entity", entity_prompt_text, prompt_scope=prompt_scope
    )
    if not val_ent:
        raise ValueError(f"实体提示词未通过 {prompt_scope} 准入校验: {val_ent.error_message}")
    val_rel = PromptContractValidator.validate_candidate(
        "relation", relation_prompt_text, prompt_scope=prompt_scope
    )
    if not val_rel:
        raise ValueError(f"关系提示词未通过 {prompt_scope} 准入校验: {val_rel.error_message}")

    entity_prompt_sha = _sha256(entity_prompt_path)
    relation_prompt_sha = _sha256(relation_prompt_path)

    # 5a. 严格绑定 final prompt == summary winner（raw bytes + canonical 双哈希）
    ent_winner_raw = entity_summary.get("winner_prompt_sha256_raw_bytes")
    rel_winner_raw = relation_summary.get("winner_prompt_sha256_raw_bytes")
    if not ent_winner_raw or entity_prompt_sha != ent_winner_raw:
        raise ValueError(
            "final prompt does not match summary winner (entity raw bytes): "
            f"final={entity_prompt_sha} winner={ent_winner_raw}"
        )
    if not rel_winner_raw or relation_prompt_sha != rel_winner_raw:
        raise ValueError(
            "final prompt does not match summary winner (relation raw bytes): "
            f"final={relation_prompt_sha} winner={rel_winner_raw}"
        )
    ent_canonical = compute_prompt_hash(entity_prompt_text)
    rel_canonical = compute_prompt_hash(relation_prompt_text)
    ent_winner_canonical = entity_summary.get("winner_prompt_sha256")
    rel_winner_canonical = relation_summary.get("winner_prompt_sha256")
    if not ent_winner_canonical:
        raise ValueError(
            "Stage 1 summary 缺少 winner_prompt_sha256（canonical hash 为强制字段）"
        )
    if ent_canonical != ent_winner_canonical:
        raise ValueError(
            "final prompt does not match summary winner (entity canonical): "
            f"final={ent_canonical} winner={ent_winner_canonical}"
        )
    if not rel_winner_canonical:
        raise ValueError(
            "Stage 2 summary 缺少 winner_prompt_sha256（canonical hash 为强制字段）"
        )
    if rel_canonical != rel_winner_canonical:
        raise ValueError(
            "final prompt does not match summary winner (relation canonical): "
            f"final={rel_canonical} winner={rel_winner_canonical}"
        )

    # 5b. 校验 Summary 的 Split / Freeze / Config Input Binding
    ent_bind = entity_summary.get("input_bindings", {}) or {}
    rel_bind = relation_summary.get("input_bindings", {}) or {}
    for label, bind in (("Stage 1", ent_bind), ("Stage 2", rel_bind)):
        if bind.get("split_file_sha256_raw_bytes") != split_sha:
            raise ValueError(
                f"{label} summary split_file_sha256_raw_bytes "
                f"({bind.get('split_file_sha256_raw_bytes')}) 与当前切分 ({split_sha}) 不一致"
            )
        if bind.get("dataset_freeze_manifest_sha256_raw_bytes") != freeze_sha:
            raise ValueError(
                f"{label} summary dataset_freeze_manifest_sha256_raw_bytes "
                f"({bind.get('dataset_freeze_manifest_sha256_raw_bytes')}) 与当前冻结清单 ({freeze_sha}) 不一致"
            )
    if ent_bind.get("split_file_sha256_raw_bytes") != rel_bind.get(
        "split_file_sha256_raw_bytes"
    ):
        raise ValueError("Stage 1 与 Stage 2 split binding 不一致，禁止晋级！")
    if ent_bind.get("dataset_freeze_manifest_sha256_raw_bytes") != rel_bind.get(
        "dataset_freeze_manifest_sha256_raw_bytes"
    ):
        raise ValueError("Stage 1 与 Stage 2 freeze binding 不一致，禁止晋级！")
    entity_config_sha256 = ent_bind.get("config_file_sha256_raw_bytes")
    relation_config_sha256 = rel_bind.get("config_file_sha256_raw_bytes")
    if not entity_config_sha256 or not relation_config_sha256:
        raise ValueError(
            "Stage 1/Stage 2 summary 缺少 config_file_sha256_raw_bytes，"
            "禁止静默丢失 provenance！"
        )

    # 5c. Stage 1 与 Stage 2 Task Runtime 必须一致（禁止默认选其中一个）
    entity_task_runtime = _extract_task_runtime(entity_summary, "Stage 1")
    relation_task_runtime = _extract_task_runtime(relation_summary, "Stage 2")
    for field in TASK_RUNTIME_FIELDS:
        if entity_task_runtime[field] != relation_task_runtime[field]:
            raise ValueError(
                f"Stage 1 与 Stage 2 Task Runtime 不一致 (字段 {field}): "
                f"{entity_task_runtime[field]!r} != {relation_task_runtime[field]!r}；"
                "正式 P_E* 与 P_R* 必须在同一 Task Runtime 下产生。"
            )
    task_runtime = entity_task_runtime

    # 5b. 窗口构造一致性：两 summary 必须相同，且等于冻结清单记录版本。
    frozen_construction = (freeze_manifest.get("window_construction") or {}).get(
        "version"
    )
    ent_construction = entity_summary.get("window_construction")
    rel_construction = relation_summary.get("window_construction")
    for label, value in (
        ("Stage 1", ent_construction),
        ("Stage 2", rel_construction),
    ):
        if not value:
            raise ValueError(
                f"{label} summary 缺少 window_construction 字段，禁止晋级！"
            )
    if ent_construction != rel_construction:
        raise ValueError(
            f"Stage 1 与 Stage 2 窗口构造 (window_construction) 不一致: "
            f"{ent_construction!r} != {rel_construction!r}；"
            "正式 P_E* 与 P_R* 必须在同一窗口构造下产生。"
        )
    if ent_construction != frozen_construction:
        raise ValueError(
            f"晋级窗口构造 (window_construction={ent_construction!r}) "
            f"与冻结清单记录 ({frozen_construction!r}) 不一致，禁止晋级！"
        )

    # 6. 校验实体缓存清单深度绑定（含 chain binding：prompt_scope + frozen runtime）
    for cache_name, cache_manifest_path in (
        ("train", entity_cache_train_manifest_path),
        ("dev", entity_cache_dev_manifest_path),
    ):
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{cache_name} 实体缓存 schema_version ({cache_manifest.get('schema_version')}) 与当前模式不一致"
            )
        if cache_manifest.get("annotation_protocol_version") != ANNOTATION_PROTOCOL_VERSION:
            raise ValueError(
                f"{cache_name} 实体缓存 annotation_protocol_version ({cache_manifest.get('annotation_protocol_version')}) 不一致"
            )
        if cache_manifest.get("boundary_contract_version") != BOUNDARY_CONTRACT_VERSION:
            raise ValueError(
                f"{cache_name} 实体缓存 boundary_contract_version ({cache_manifest.get('boundary_contract_version')}) 不一致"
            )
        if cache_manifest.get("split_sha256") != split_sha:
            raise ValueError(
                f"{cache_name} 实体缓存 split_sha256 与当前划分文件不一致"
            )
        if cache_manifest.get("gold_aggregate_sha256") != gold_agg_sha:
            raise ValueError(
                f"{cache_name} 实体缓存 gold_aggregate_sha256 与当前冻结清单不一致"
            )
        if cache_manifest.get("dataset_freeze_manifest_sha256") != freeze_sha:
            raise ValueError(
                f"{cache_name} 实体缓存 dataset_freeze_manifest_sha256 与当前冻结清单不一致"
            )
        if cache_manifest.get("entity_prompt_sha256") != entity_prompt_sha:
            raise ValueError(
                f"{cache_name} 实体缓存 entity_prompt_sha256 ({cache_manifest.get('entity_prompt_sha256')}) "
                f"与待晋级实体提示词哈希 ({entity_prompt_sha}) 不一致"
            )
        if cache_manifest.get("prompt_scope") != prompt_scope:
            raise ValueError(
                f"{cache_name} 实体缓存 prompt_scope ({cache_manifest.get('prompt_scope')!r}) "
                f"与晋级 prompt_scope ({prompt_scope!r}) 不一致"
            )
        cached_runtime = cache_manifest.get("task_runtime")
        if not isinstance(cached_runtime, dict):
            raise ValueError(
                f"{cache_name} 实体缓存缺少 task_runtime 字段，"
                "禁止 fallback 到旧格式；请用新格式重建缓存。"
            )
        for field in TASK_RUNTIME_FIELDS:
            if cached_runtime.get(field) != task_runtime[field]:
                raise ValueError(
                    f"{cache_name} 实体缓存 task_runtime[{field}] "
                    f"({cached_runtime.get(field)!r}) 与最终冻结 Task Runtime "
                    f"({task_runtime[field]!r}) 不一致"
                )
        if cache_manifest.get("window_construction") != ent_construction:
            raise ValueError(
                f"{cache_name} 实体缓存 window_construction "
                f"({cache_manifest.get('window_construction')!r}) 与晋级窗口构造 "
                f"({ent_construction!r}) 不一致；窗口构造已变更，请重建缓存。"
            )

    task_model = task_runtime["model"]
    optimizer_model = (
        entity_summary.get("config", {}).get("optimizer_model")
        or relation_summary.get("config", {}).get("optimizer_model")
        or "muse-spark-1.3-contributor"
    )

    artifact = {
        "artifact_version": "protegi-final-v1",
        "schema_version": SCHEMA_VERSION,
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "experiment_pair_id": experiment_pair_id,
        "prompt_scope": prompt_scope,
        "entity_prompt_path": _rel_path(entity_prompt_path),
        "entity_prompt_sha256": entity_prompt_sha,
        "relation_prompt_path": _rel_path(relation_prompt_path),
        "relation_prompt_sha256": relation_prompt_sha,
        "entity_optimization_summary_path": _rel_path(entity_summary_path),
        "entity_optimization_summary_sha256": _sha256(entity_summary_path),
        "relation_optimization_summary_path": _rel_path(relation_summary_path),
        "relation_optimization_summary_sha256": _sha256(relation_summary_path),
        "task_runtime": task_runtime,
        "window_construction": ent_construction,
        "entity_config_sha256": entity_config_sha256,
        "relation_config_sha256": relation_config_sha256,
        "split_file": _rel_path(split_file_path),
        "split_sha256": split_sha,
        "dataset_version": freeze_manifest.get("dataset_version", "v6-v5-gold-v9-mcpu-v2"),
        "gold_aggregate_sha256": gold_agg_sha,
        "dataset_freeze_manifest": _rel_path(freeze_manifest_path),
        "dataset_freeze_manifest_sha256": freeze_sha,
        "entity_cache_train_manifest": _rel_path(entity_cache_train_manifest_path),
        "entity_cache_train_manifest_sha256": _sha256(entity_cache_train_manifest_path),
        "entity_cache_dev_manifest": _rel_path(entity_cache_dev_manifest_path),
        "entity_cache_dev_manifest_sha256": _sha256(entity_cache_dev_manifest_path),
        "task_model": task_model,
        "optimizer_model": optimizer_model,
        "test_gold_loaded": False,
        "test_predictions_generated": False,
        "frozen_for_test": True,
        "formal_eligible": True,
    }
    if entity_config_sha256 == relation_config_sha256:
        artifact["task_runtime_config_sha256"] = entity_config_sha256

    output_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    output_artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity-dir",
        type=Path,
        default=EXP_DIR / "results" / "protegi_optimization" / "entity_protegi",
        help="Stage 1 实体优化输出目录",
    )
    parser.add_argument(
        "--relation-dir",
        type=Path,
        default=EXP_DIR / "results" / "protegi_optimization" / "relation_protegi",
        help="Stage 2 关系优化输出目录",
    )
    parser.add_argument(
        "--entity-cache-dir",
        type=Path,
        default=EXP_DIR / "results" / "protegi_optimization" / "entity_cache",
        help="Stage 2 冻结实体预测缓存目录",
    )
    parser.add_argument(
        "--entity-prompt",
        type=Path,
        default=None,
        help="Stage 1 实体提示词文件 (优先于 --entity-dir)",
    )
    parser.add_argument(
        "--relation-prompt",
        type=Path,
        default=None,
        help="Stage 2 关系提示词文件 (优先于 --relation-dir)",
    )
    parser.add_argument(
        "--entity-summary",
        type=Path,
        default=None,
        help="Stage 1 优化摘要文件 (优先于 --entity-dir)",
    )
    parser.add_argument(
        "--relation-summary",
        type=Path,
        default=None,
        help="Stage 2 优化摘要文件 (优先于 --relation-dir)",
    )
    parser.add_argument(
        "--entity-cache-train-manifest",
        type=Path,
        default=None,
        help="Train 实体缓存清单文件 (优先于 --entity-cache-dir)",
    )
    parser.add_argument(
        "--entity-cache-dev-manifest",
        type=Path,
        default=None,
        help="Dev 实体缓存清单文件 (优先于 --entity-cache-dir)",
    )
    parser.add_argument(
        "--prompt-scope",
        type=str,
        default="constrained",
        choices=["constrained", "unconstrained"],
        help="提示词约束范围 (默认 constrained)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ARTIFACT,
        help="产物输出路径",
    )
    args = parser.parse_args()

    entity_prompt_path = args.entity_prompt or (args.entity_dir / "final_entity_prompt.txt")
    relation_prompt_path = args.relation_prompt or (args.relation_dir / "final_relation_prompt.txt")
    entity_summary_path = args.entity_summary or (args.entity_dir / "summary.json")
    relation_summary_path = args.relation_summary or (args.relation_dir / "summary.json")
    entity_cache_train_manifest_path = args.entity_cache_train_manifest or (
        args.entity_cache_dir / "entity_cache_train_manifest.json"
    )
    entity_cache_dev_manifest_path = args.entity_cache_dev_manifest or (
        args.entity_cache_dir / "entity_cache_dev_manifest.json"
    )

    artifact = promote_protegi(
        entity_prompt_path=entity_prompt_path,
        relation_prompt_path=relation_prompt_path,
        entity_summary_path=entity_summary_path,
        relation_summary_path=relation_summary_path,
        entity_cache_train_manifest_path=entity_cache_train_manifest_path,
        entity_cache_dev_manifest_path=entity_cache_dev_manifest_path,
        prompt_scope=args.prompt_scope,
        output_artifact_path=args.output,
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
