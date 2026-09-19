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
from protegi.gold_integrity import verify_frozen_gold_integrity
from protegi.runtime_contract import build_effective_task_runtime
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


def compute_formal_eligibility(*, dry_run: bool, allow_custom_split: bool) -> bool:
    """纯函数：--dry-run 或 --allow-custom-split 任一出现即 non-formal。

    注意：路径是否恰好等于 canonical 不影响结果；flag 本身决定语义。
    抽出为纯函数以便无模型回归测试直接断言。
    """
    return (not dry_run) and (not allow_custom_split)


def normalize_config_with_effective_runtime(config: dict) -> dict:
    """规范化 Task 字段并写回 effective runtime（单一记录来源）。

    输入为 YAML 解析后的 config dict；返回同一 dict（就地更新），新增：
    window_chars / window_overlap / document_abbreviation_context /
    vulnerability_anchored_backfill（规范化后）与
    effective_task_runtime（11 字段，见 protegi.runtime_contract）。
    optimizer summary 的 "config" 即实际执行 runtime。
    不调用任何模型，可被离线测试直接断言。
    """
    include_document_abbreviations = bool(
        config.get("document_abbreviation_context", False)
    )
    vulnerability_backfill_flag = bool(
        config.get("vulnerability_anchored_backfill", False)
    )
    window_chars = int(
        config.get("window_chars", config.get("window_max_chars", 3000))
    )
    window_overlap = int(config.get("window_overlap", 400))
    if not window_chars > 0:
        raise ValueError(f"window_chars 必须为正整数，当前={window_chars!r}")
    if not 0 <= window_overlap < window_chars:
        raise ValueError(
            f"window_overlap 必须满足 0 <= overlap < window_chars，"
            f"当前 overlap={window_overlap!r}, chars={window_chars!r}"
        )
    config["task_model"] = str(config["task_model"])
    config["task_max_workers"] = int(config["task_max_workers"])
    config["task_temperature"] = float(config["task_temperature"])
    config["task_thinking"] = str(config["task_thinking"]).strip().lower()
    config["task_reasoning_effort"] = (
        str(config["task_reasoning_effort"]).strip().lower()
    )
    config["task_top_p"] = float(config["task_top_p"])
    config["task_max_tokens"] = int(config["task_max_tokens"])
    config["window_chars"] = int(window_chars)
    config["window_overlap"] = int(window_overlap)
    config["document_abbreviation_context"] = bool(include_document_abbreviations)
    config["vulnerability_anchored_backfill"] = bool(vulnerability_backfill_flag)
    config["effective_task_runtime"] = build_effective_task_runtime(
        model=config["task_model"],
        max_workers=config["task_max_workers"],
        temperature=config["task_temperature"],
        thinking=config["task_thinking"],
        reasoning_effort=config["task_reasoning_effort"],
        top_p=config["task_top_p"],
        max_tokens=config["task_max_tokens"],
        window_chars=config["window_chars"],
        window_overlap=config["window_overlap"],
        document_abbreviation_context=config["document_abbreviation_context"],
        vulnerability_anchored_backfill=config[
            "vulnerability_anchored_backfill"
        ],
    )
    return config


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
    parser.add_argument(
        "--allow-custom-split",
        action="store_true",
        help="允许使用自定义/非正式切分；产物将被标记为 formal_eligible=false 且禁止晋级",
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
    config = normalize_config_with_effective_runtime(config)
    include_document_abbreviations = bool(
        config["document_abbreviation_context"]
    )
    vulnerability_backfill_flag = bool(
        config["vulnerability_anchored_backfill"]
    )
    window_chars = int(config["window_chars"])
    window_overlap = int(config["window_overlap"])

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

    # 3. 加载数据集切分并严格校验正式门禁。
    # --allow-custom-split / --dry-run 任一出现即无条件 non-formal，
    # 与传入路径是否恰好等于 canonical 无关。
    split_ids = load_split_doc_ids(args.split_file)
    train_doc_ids = split_ids["train"]
    dev_doc_ids = split_ids["dev"]

    formal_eligible = compute_formal_eligibility(
        dry_run=bool(args.dry_run),
        allow_custom_split=bool(args.allow_custom_split),
    )
    freeze_manifest_file = ROOT / "data" / "dataset_freeze_manifest_v6.json"

    # canonical test 隔离：formal 与 diagnostic/custom 路径都要检查，
    # 但 diagnostic/custom 产物永远 formal_eligible=false。
    canonical_split_data = json.loads(DEFAULT_SPLIT_FILE.read_text(encoding="utf-8"))
    canonical_test_ids = set(canonical_split_data.get("test", []))
    custom_split_data = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    all_test_ids = canonical_test_ids | set(custom_split_data.get("test", []))
    train_test_leak = set(train_doc_ids) & all_test_ids
    dev_test_leak = set(dev_doc_ids) & all_test_ids
    if train_test_leak or dev_test_leak:
        raise RuntimeError(
            f"数据切分污染：optimizer 数据与 canonical test 存在重叠！"
            f"train ∩ test: {train_test_leak}, dev ∩ test: {dev_test_leak}"
        )

    if formal_eligible:
        if not freeze_manifest_file.is_file():
            raise RuntimeError(
                f"缺少数据集冻结清单 {freeze_manifest_file}；正式模式禁止使用未冻结数据集。"
            )
        freeze_manifest = json.loads(freeze_manifest_file.read_text(encoding="utf-8"))
        current_split_hash = hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest()
        manifest_split_hash = freeze_manifest.get("split_sha256")
        frozen_gold_dir = (ROOT / freeze_manifest.get("gold_directory", "data/annotations/gold")).resolve()
        actual_gold_dir = Path(args.gold_dir).resolve()

        if current_split_hash != manifest_split_hash:
            raise RuntimeError(
                f"正式 ProTeGi 优化必须绑定当前冻结划分："
                f"split_file 哈希 ({current_split_hash}) 与冻结清单 ({manifest_split_hash}) 不符。"
            )
        if actual_gold_dir != frozen_gold_dir:
            raise RuntimeError(
                f"正式 ProTeGi 优化必须绑定当前冻结 Gold 数据目录："
                f"gold_dir ({actual_gold_dir}) 与冻结目录 ({frozen_gold_dir}) 不符。"
            )

    config["formal_eligible"] = formal_eligible

    # Formal Gold 完整性门禁：唯一触发条件就是 formal_eligible。
    # 在读取任何 Gold 之前重新计算实际 Gold 内容。
    verified_gold_aggregate_sha256: str | None = None
    _freeze_manifest_for_gold = ROOT / "data" / "dataset_freeze_manifest_v6.json"
    if formal_eligible:
        ok, message = verify_frozen_gold_integrity(
            Path(args.gold_dir),
            _freeze_manifest_for_gold,
        )
        if not ok:
            raise RuntimeError(
                f"正式 ProTeGi Gold 完整性校验失败: {message}；"
                "已在模型调用前阻断 (optimizer/model call count=0)。"
            )
        verified_gold_aggregate_sha256 = json.loads(
            _freeze_manifest_for_gold.read_text(encoding="utf-8")
        ).get("gold_aggregate_sha256")

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
            max_chars=window_chars,
            overlap=window_overlap,
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
            freeze_manifest_path=(
                _freeze_manifest_for_gold
                if _freeze_manifest_for_gold.is_file()
                else None
            ),
            split_file_path=Path(args.split_file),
            gold_dir=Path(args.gold_dir),
            window_chars=window_chars,
            window_overlap=window_overlap,
            document_abbreviation_context=include_document_abbreviations,
            vulnerability_anchored_backfill=vulnerability_backfill_flag,
            verified_gold_aggregate_sha256=verified_gold_aggregate_sha256,
            validate_formal_gold=bool(formal_eligible),
        )

        print("正在为 Dev 集生成冻结实体预测缓存...")
        dev_samples = prepare_stage2_window_samples(
            dev_doc_ids,
            args.gold_dir,
            max_chars=window_chars,
            overlap=window_overlap,
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
            freeze_manifest_path=(
                _freeze_manifest_for_gold
                if _freeze_manifest_for_gold.is_file()
                else None
            ),
            split_file_path=Path(args.split_file),
            gold_dir=Path(args.gold_dir),
            window_chars=window_chars,
            window_overlap=window_overlap,
            document_abbreviation_context=include_document_abbreviations,
            vulnerability_anchored_backfill=vulnerability_backfill_flag,
            verified_gold_aggregate_sha256=verified_gold_aggregate_sha256,
            validate_formal_gold=bool(formal_eligible),
        )
        print(f"实体缓存构建完成，保存在: {cache_dir}")
        return

    elif args.stage == "entity":
        train_samples = prepare_stage1_window_samples(
            train_doc_ids,
            args.gold_dir,
            max_chars=window_chars,
            overlap=window_overlap,
            max_docs=max_train_docs,
            include_document_abbreviations=include_document_abbreviations,
        )
        dev_samples = prepare_stage1_window_samples(
            dev_doc_ids,
            args.gold_dir,
            max_chars=window_chars,
            overlap=window_overlap,
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
        # Stage 2 必须使用 run 阶段已记录的 effective runtime，不再现场拼装。
        expected_task_runtime = dict(config["effective_task_runtime"])
        train_samples = cache_manager.load_cache(
            "train",
            expected_prompt_hash=expected_hash,
            require_gold_relations=True,
            expected_task_model=config.get("task_model"),
            expected_prompt_scope=prompt_scope,
            expected_task_max_workers=int(config.get("task_max_workers", 8)),
            expected_task_runtime=expected_task_runtime,
        )
        dev_samples = cache_manager.load_cache(
            "dev",
            expected_prompt_hash=expected_hash,
            require_gold_relations=True,
            expected_task_model=config.get("task_model"),
            expected_prompt_scope=prompt_scope,
            expected_task_max_workers=int(config.get("task_max_workers", 8)),
            expected_task_runtime=expected_task_runtime,
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
