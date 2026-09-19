"""ProTeGi 自动化提示词优化运行入口 (CLI Runner).

支持命令行执行 Stage 1 (实体)、构建冻结实体缓存、执行 Stage 2 (关系)，
以及 5 种基线与消融模式的统一调度。

严格遵循测试集隔离红线：在 P_E* 与 P_R* 均被正式锁定前，绝不接触或评估 Test 集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import os
from typing import Optional
import uuid

import yaml

# 保证 OpenCode Go 会话标识存在
os.environ.setdefault("V5_OPENCODE_SESSION_ID", f"bron-v6-protegi-{uuid.uuid4().hex[:12]}")
os.environ.setdefault("V5_OPENCODE_USER_AGENT", "bron-protegi-research/1.0")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protegi.entity_cache import EntityCacheManager, compute_prompt_hash
from protegi.contract_validator import PromptContractValidator
from protegi.evaluator import TaskEvaluator
from protegi.optimizer import (
    ProTeGiOptimizer,
    load_split_doc_ids,
    prepare_stage1_window_samples,
    prepare_stage2_window_samples,
)

DEFAULT_SPLIT_FILE = ROOT / "data" / "train_dev_test_split_v7.json"
DEFAULT_GOLD_DIR = ROOT / "data" / "annotations" / "gold"
DEFAULT_CONFIG_FORMAL = ROOT / "protegi" / "configs" / "protegi_formal_muse.yaml"
DEFAULT_CONFIG_DRYRUN = ROOT / "protegi" / "configs" / "protegi_dryrun.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "protegi_optimization"


def parse_args():
    parser = argparse.ArgumentParser(description="ProTeGi Prompt Optimization CLI Runner")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["entity", "relation", "build_entity_cache"],
        help="优化阶段：entity (Stage 1), relation (Stage 2), 或 build_entity_cache (构建冻结实体缓存)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="protegi",
        choices=["initial", "mc", "greedy_protegi", "protegi", "protegi_uniform"],
        help="搜索方法变体 (默认 protegi 主方法)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 配置文件路径 (默认根据 --dry-run 自动选择)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="开启联调冒烟模式 (使用微型批次与步数)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="实验产物输出目录",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
        help="数据划分 JSON 路径",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Gold 标注文件夹路径",
    )
    parser.add_argument(
        "--entity-cache-dir",
        type=Path,
        default=None,
        help="Stage 2 冻结实体预测缓存目录",
    )
    parser.add_argument(
        "--entity-prompt-file",
        type=Path,
        default=None,
        help="Stage 1 产出的 final_entity_prompt.txt 路径 (运行 relation 或 build_entity_cache 时需要)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 确定配置文件
    if args.config:
        config_path = args.config
    elif args.dry_run:
        config_path = DEFAULT_CONFIG_DRYRUN
    else:
        config_path = DEFAULT_CONFIG_FORMAL

    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required_model_keys = {
        "task_model",
        "task_max_workers",
        "task_temperature",
        "task_thinking",
        "task_reasoning_effort",
        "task_top_p",
        "task_max_tokens",
        "optimizer_model",
        "optimizer_temperature",
        "optimizer_thinking",
        "optimizer_reasoning_effort",
        "optimizer_top_p",
        "optimizer_max_tokens",
    }
    missing_model_keys = sorted(
        key for key in required_model_keys
        if key not in config or config.get(key) is None
    )
    if missing_model_keys:
        raise ValueError(f"配置文件缺少显式模型参数: {missing_model_keys}")
    prompt_scope = str(config.get("prompt_scope", "")).strip().lower()
    if prompt_scope not in {"constrained", "unconstrained"}:
        raise ValueError(
            "配置文件必须显式声明 prompt_scope: constrained 或 unconstrained"
        )
    config["_config_file"] = str(config_path.resolve())
    config["_config_file_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    include_document_abbreviations = bool(
        config.get("document_abbreviation_context", False)
    )

    def build_task_evaluator() -> TaskEvaluator:
        return TaskEvaluator(
            max_workers=int(config.get("task_max_workers", 8)),
            task_model=config.get("task_model"),
            task_temperature=float(config.get("task_temperature", 0.0)),
            task_thinking=config.get("task_thinking", "disabled"),
            task_max_tokens=(
                int(config["task_max_tokens"])
                if config.get("task_max_tokens")
                else None
            ),
            task_top_p=(
                float(config["task_top_p"])
                if config.get("task_top_p") is not None
                else None
            ),
            task_reasoning_effort=config.get("task_reasoning_effort", "none"),
            vulnerability_anchored_backfill=bool(
                config.get("vulnerability_anchored_backfill", False)
            ),
        )

    def relation_aware_dryrun_subset(samples: list[dict], limit: int = 2) -> list[dict]:
        """优先保留含关系 Gold 的窗口，避免构建不可用于 Stage 2 的冒烟缓存。"""
        positives = [sample for sample in samples if sample.get("gold_relations")]
        selected = positives[:limit]
        if len(selected) < limit:
            selected_ids = {sample.get("sample_id") for sample in selected}
            selected.extend(
                sample
                for sample in samples
                if sample.get("sample_id") not in selected_ids
            )
        return selected[:limit]

    # 2. 确定输出目录
    suffix = "_dryrun" if args.dry_run else ""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = DEFAULT_OUTPUT_ROOT / (
            f"{args.stage}_{args.method}_{prompt_scope}{suffix}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.entity_cache_dir or (
        DEFAULT_OUTPUT_ROOT / f"entity_cache_{prompt_scope}{suffix}"
    )

    # 3. 加载数据集切分
    split_ids = load_split_doc_ids(args.split_file)
    train_doc_ids = split_ids["train"]
    dev_doc_ids = split_ids["dev"]

    max_train_docs = config.get("max_docs_train")
    max_dev_docs = config.get("max_docs_dev")

    print(f"=== ProTeGi CLI Runner ===")
    print(f"Stage: {args.stage}, Method: {args.method}")
    print(f"Prompt Scope: {prompt_scope}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Train Docs: {len(train_doc_ids)} (cap: {max_train_docs}), Dev Docs: {len(dev_doc_ids)} (cap: {max_dev_docs})")

    # 4. 执行逻辑分支
    if args.stage == "build_entity_cache":
        if not args.entity_prompt_file or not args.entity_prompt_file.is_file():
            raise ValueError("构建实体缓存时必须指定合法的 --entity-prompt-file (P_E*)！")
        entity_prompt_text = args.entity_prompt_file.read_text(encoding="utf-8")
        prompt_validation = PromptContractValidator.validate_candidate(
            "entity",
            entity_prompt_text,
            prompt_scope=prompt_scope,
        )
        if not prompt_validation:
            raise ValueError(
                f"实体提示词未通过 {prompt_scope} 实验臂准入校验: "
                f"{prompt_validation.error_message}"
            )
        evaluator = build_task_evaluator()
        cache_manager = EntityCacheManager(cache_dir)

        print("正在为 Train 集生成冻结实体预测缓存...")
        train_samples = prepare_stage2_window_samples(
            train_doc_ids,
            args.gold_dir,
            max_docs=max_train_docs,
            include_document_abbreviations=include_document_abbreviations,
        )
        if args.dry_run:
            train_samples = relation_aware_dryrun_subset(train_samples)
        cache_manager.build_and_save_cache(
            evaluator,
            entity_prompt_text,
            train_samples,
            "train",
            prompt_scope=prompt_scope,
        )

        print("正在为 Dev 集生成冻结实体预测缓存...")
        dev_samples = prepare_stage2_window_samples(
            dev_doc_ids,
            args.gold_dir,
            max_docs=max_dev_docs,
            include_document_abbreviations=include_document_abbreviations,
        )
        if args.dry_run:
            dev_samples = relation_aware_dryrun_subset(dev_samples)
        cache_manager.build_and_save_cache(
            evaluator,
            entity_prompt_text,
            dev_samples,
            "dev",
            prompt_scope=prompt_scope,
        )
        print(f"实体缓存构建完成，保存在: {cache_dir}")
        return

    elif args.stage == "entity":
        train_samples = prepare_stage1_window_samples(
            train_doc_ids,
            args.gold_dir,
            max_docs=max_train_docs,
            include_document_abbreviations=include_document_abbreviations,
        )
        dev_samples = prepare_stage1_window_samples(
            dev_doc_ids,
            args.gold_dir,
            max_docs=max_dev_docs,
            include_document_abbreviations=include_document_abbreviations,
        )
        if args.dry_run:
            train_samples = train_samples[:2]
            dev_samples = dev_samples[:2]
        print(f"已生成 Stage 1 样本: Train={len(train_samples)} windows, Dev={len(dev_samples)} windows")

        optimizer = ProTeGiOptimizer(
            stage="entity",
            method=args.method,
            config=config,
            output_dir=output_dir,
            gold_dir=args.gold_dir,
            split_file=args.split_file,
            entity_cache_dir=cache_dir,
        )
        winner = optimizer.run_optimization(train_samples, dev_samples)
        print(f"\n[Stage 1 优化完成]")
        print(f"获胜 Candidate: {winner.candidate_id} (Round {winner.round_idx})")
        print(f"指标: {winner.metrics}")
        print(f"最优实体提示词保存在: {output_dir / 'final_entity_prompt.txt'}")

    elif args.stage == "relation":
        # Stage 2 必须绑定 P_E* 与其缓存；严禁失败后回退到 Gold 实体。
        if not args.entity_prompt_file or not args.entity_prompt_file.is_file():
            raise ValueError("运行 relation 时必须指定合法的 --entity-prompt-file (P_E*)！")
        entity_prompt_text = args.entity_prompt_file.read_text(encoding="utf-8")
        prompt_validation = PromptContractValidator.validate_candidate(
            "entity",
            entity_prompt_text,
            prompt_scope=prompt_scope,
        )
        if not prompt_validation:
            raise ValueError(
                f"实体提示词未通过 {prompt_scope} 实验臂准入校验: "
                f"{prompt_validation.error_message}"
            )
        cache_manager = EntityCacheManager(cache_dir)
        expected_hash = compute_prompt_hash(entity_prompt_text)
        train_samples = cache_manager.load_cache(
            "train",
            expected_prompt_hash=expected_hash,
            require_gold_relations=True,
            expected_task_model=config.get("task_model"),
            expected_prompt_scope=prompt_scope,
            expected_task_max_workers=int(config.get("task_max_workers", 8)),
        )
        dev_samples = cache_manager.load_cache(
            "dev",
            expected_prompt_hash=expected_hash,
            require_gold_relations=True,
            expected_task_model=config.get("task_model"),
            expected_prompt_scope=prompt_scope,
            expected_task_max_workers=int(config.get("task_max_workers", 8)),
        )
        print(f"成功加载上游冻结实体预测缓存: Train={len(train_samples)}, Dev={len(dev_samples)}")

        if args.dry_run:
            train_samples = relation_aware_dryrun_subset(train_samples)
            dev_samples = relation_aware_dryrun_subset(dev_samples)

        print(f"已生成 Stage 2 样本: Train={len(train_samples)} windows, Dev={len(dev_samples)} windows")

        optimizer = ProTeGiOptimizer(
            stage="relation",
            method=args.method,
            config=config,
            output_dir=output_dir,
            gold_dir=args.gold_dir,
            split_file=args.split_file,
            entity_cache_dir=cache_dir,
        )
        winner = optimizer.run_optimization(train_samples, dev_samples)
        print(f"\n[Stage 2 优化完成]")
        print(f"获胜 Candidate: {winner.candidate_id} (Round {winner.round_idx})")
        print(f"指标: {winner.metrics}")
        print(f"最优关系提示词保存在: {output_dir / 'final_relation_prompt.txt'}")


if __name__ == "__main__":
    main()
