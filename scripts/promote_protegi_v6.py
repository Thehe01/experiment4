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

    # 6. 校验实体缓存清单深度绑定
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

    task_model = (
        entity_summary.get("config", {}).get("task_model")
        or relation_summary.get("config", {}).get("task_model")
        or "gpt-4o-mini"
    )
    optimizer_model = (
        entity_summary.get("config", {}).get("optimizer_model")
        or relation_summary.get("config", {}).get("optimizer_model")
        or "gpt-4o"
    )

    artifact = {
        "artifact_version": "protegi-final-v1",
        "schema_version": SCHEMA_VERSION,
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "prompt_scope": prompt_scope,
        "entity_prompt_path": _rel_path(entity_prompt_path),
        "entity_prompt_sha256": entity_prompt_sha,
        "relation_prompt_path": _rel_path(relation_prompt_path),
        "relation_prompt_sha256": relation_prompt_sha,
        "entity_optimization_summary_path": _rel_path(entity_summary_path),
        "entity_optimization_summary_sha256": _sha256(entity_summary_path),
        "relation_optimization_summary_path": _rel_path(relation_summary_path),
        "relation_optimization_summary_sha256": _sha256(relation_summary_path),
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
