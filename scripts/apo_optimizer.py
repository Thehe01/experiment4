"""[LEGACY - DEPRECATED] v6 遗留 APO-v2 自动提示优化脚本。

警告：本脚本为上一代 APO-v2 机制（固定 P0 + 仅优化 guidance + 多目标门控 + beam=1），
已被忠实于 ProTeGi 的两阶段提示词优化框架（experiments/v6/protegi/ 及 scripts/run_protegi.py）
全面取代。保留本脚本仅为归档对照与历史实验可复现性验证，新实验请统一使用 run_protegi.py。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from provider_config import apply_formal_apo_runtime_defaults  # noqa: E402

apply_formal_apo_runtime_defaults()

import eval_metrics as EM  # noqa: E402
import apo_metrics as AM  # noqa: E402
from llm_methods import (  # noqa: E402
    LLM_MAX_WORKERS,
    STAGE1_PROMPT_VERSION,
    _append_apo_guidance,
    _call_with_retries,
    _ext,
    _loose_json,
    _offset_prediction,
    _relation_evidence_interval,
    apply_postprocess,
    build_text_windows,
    make_extractor,
    merge_window_predictions,
    parse_entity_mentions,
    predict_llm_multipass,
    runtime_config,
    stage1_runtime_config,
    validate_apo_guidance,
    validate_apo_textual_gradient,
)
from schema import (  # noqa: E402
    ANNOTATION_PROTOCOL_VERSION,
    BOUNDARY_CONTRACT_VERSION,
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
    EXTRACTION_RELATION_TYPES,
    SCHEMA_VERSION,
)
from prompts.multipass_prompts import (  # noqa: E402
    MULTIPASS_FEWSHOT,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
)

GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
REVIEW_STATUS_FILE = EXP_DIR / "data" / "review_status.json"
OUTPUT_ROOT = EXP_DIR / "results" / "apo_optimization_v6"
TASK_CACHE_ROOT = OUTPUT_ROOT / "_task_cache"


def _require_boundary_recertification_complete() -> None:
    status = json.loads(REVIEW_STATUS_FILE.read_text(encoding="utf-8"))
    boundary = status.get("boundary_sync", {})
    if (
        boundary.get("contract_version") != BOUNDARY_CONTRACT_VERSION
        or boundary.get("complete") is not True
    ):
        raise RuntimeError(
            "chapter3-boundary-sync-v2 MCPU Gold 复裁尚未完成；为避免在漂移的 dev "
            "标签上优化，当前只允许 --validate-only 或 --estimate-only，不调用模型"
        )

ENTITY_LABELS = (
    "Configuration",
    "Vulnerability",
    "Weakness",
    "AttackTechnique",
)
RELATION_LABELS = ("affects", "instantiates", "exploited_by")

ENTITY_WEIGHT = 0.4
RELATION_WEIGHT = 0.6
CRITIC_TEMPERATURE = float(os.environ.get("V3_APO_CRITIC_TEMPERATURE", "0.1"))
EDITOR_TEMPERATURE = float(os.environ.get("V3_APO_EDITOR_TEMPERATURE", "0.1"))
OPTIMIZER_TOP_P = float(os.environ.get("V3_APO_OPTIMIZER_TOP_P", "1.0"))
EDITOR_TOP_P = float(os.environ.get("V3_APO_EDITOR_TOP_P", "1.0"))
OPTIMIZER_MODEL = os.environ.get(
    "V3_APO_OPTIMIZER_MODEL", runtime_config()["model"]
).strip()
OPTIMIZER_THINKING = os.environ.get(
    "V3_APO_OPTIMIZER_THINKING", runtime_config()["thinking"]
).strip().lower()
# The shared runtime profile and provider defaults are covered by the frozen
# Rule/Multipass/Full manifest.  Keep them byte-identical.  OpenCode Go exposes
# `high` as the validated effort for the Muse Spark Contributor optimizer.
OPTIMIZER_REASONING_EFFORT = "high"
CRITIC_FALLBACK_THINKING = os.environ.get(
    "V3_APO_CRITIC_FALLBACK_THINKING", "disabled"
).strip().lower()
CRITIC_FALLBACK_REASONING_EFFORT = "high"
OPTIMIZER_MAX_TOKENS = int(
    os.environ.get("V3_APO_OPTIMIZER_MAX_TOKENS", "8192")
)
EDITOR_MODEL = os.environ.get("V3_APO_EDITOR_MODEL", OPTIMIZER_MODEL).strip()
EDITOR_THINKING = os.environ.get(
    "V3_APO_EDITOR_THINKING", "disabled"
).strip().lower()
EDITOR_REASONING_EFFORT = "high"
EDITOR_MAX_TOKENS = int(
    os.environ.get("V3_APO_EDITOR_MAX_TOKENS", "4096")
)
CANDIDATE_GENERATION_MAX_ATTEMPTS = max(
    1,
    int(os.environ.get("V3_APO_CANDIDATE_GENERATION_MAX_ATTEMPTS", "3")),
)


def _model_endpoint(model: str) -> str:
    """Return the endpoint selected by ``make_extractor`` for this model."""
    return (
        "/v1/responses"
        if re.fullmatch(r"muse-spark-1\.[23]-contributor", model)
        else "/v1/chat/completions"
    )


def make_apo_extractor(**kwargs):
    """Create a client and apply the APO-specific model endpoint contract.

    The frozen baseline factory predates Muse Spark 1.3.  APO-v2 keeps that
    baseline byte-identical and applies the new Responses endpoint only to
    Critic/Editor clients.
    """
    client = make_extractor(**kwargs)
    model = str(kwargs.get("model") or client.config.get("model") or "")
    client.config["endpoint"] = _model_endpoint(model)
    return client


MIN_EVIDENCE_VALIDITY = 0.95
TYPE_F1_MAX_DROP = float(os.environ.get("V3_APO_TYPE_F1_MAX_DROP", "0.03"))
TARGET_F1_MIN_GAIN = float(os.environ.get("V3_APO_TARGET_F1_MIN_GAIN", "0.005"))


def _optional_nonnegative_env(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


RELATION_F1_MAX_DROP = _optional_nonnegative_env(
    "V3_APO_RELATION_F1_MAX_DROP"
)
NORMALIZED_RELATION_F1_MAX_DROP = _optional_nonnegative_env(
    "V3_APO_NORMALIZED_RELATION_F1_MAX_DROP"
)
TARGET_PARENT_MAX_DROP = 0.0
RAW_QUALITY_MEAN_MAX_DROP = 0.01
RAW_QUALITY_REPEAT_LARGE_DROP = 0.03
RAW_QUALITY_MAX_LARGE_DROP_REPEATS = 1
PAIRED_POSITIVE_REPEAT_RATIO = 2 / 3
CONFIGURATION_STRATEGIES = (
    "recall_only",
    "boundary_only",
    "full_rewrite",
    "precision_control",
)
CONFIGURATION_CPE_STRATEGIES = (
    "mention_boundary",
    "canonical_cpe_fields",
    "parent_family_projection",
    "ambiguity_abstention",
)
CONFIGURATION_RECALL_STRATEGIES_BY_ROUND = {
    1: (
        "repeated_alias_occurrence",
        "structured_record_occurrence",
        "narrative_component_occurrence",
        "section_scope_occurrence",
    ),
    2: (
        "alias_parenthetical_boundary",
        "component_suffix_boundary",
        "version_qualifier_boundary",
        "coordinated_product_boundary",
    ),
}
EXPLOITED_BY_STRATEGIES = (
    "direct_exploit_trigger",
    "post_exploitation_exclusion",
    "local_pair_boundary",
    "precision_control_rewrite",
)
ERROR_EXAMPLES_PER_LABEL_AND_KIND = 4
CONFIGURATION_RECALL_ERROR_EXAMPLES = 12
ERROR_SNIPPET_CONTEXT_CHARS = 180
CONFIGURATION_RECALL_MIN_GAIN = 0.08
CONFIGURATION_RECALL_MIN_TP_GAIN = 5.0
PRIVATE_GRADIENT_ID_REPLACEMENTS = (
    (re.compile(r"\bCVE-\s*\d{4}-\d{4,7}\b", re.IGNORECASE), "<CVE_ID>"),
    (re.compile(r"\bCWE-\s*\d+\b", re.IGNORECASE), "<CWE_ID>"),
    (re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE), "<TECHNIQUE_ID>"),
    (re.compile(r"\bTA\d{4}\b", re.IGNORECASE), "<TACTIC_ID>"),
    (re.compile(r"\bKC-[A-Z0-9_-]+\b", re.IGNORECASE), "<PHASE_ID>"),
)
DEFAULT_PHASE_ROUNDS = {"entity": 2, "relation": 2, "joint": 1}
APO_PRESETS = {
    "apo_v2_relation_smoke": {
        # Lowest-cost real API gate: one train and one dev document both carry
        # all three relation labels, so every registered relation round runs.
        "candidate_count": 1,
        "beam_size": 1,
        "train_batch_size": 1,
        "selection_dev_size": 1,
        "train_doc_ids": ["aa23-339a"],
        "selection_dev_doc_ids": ["aa23-213a"],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 3, "joint": 0},
        "freeze_artifact": False,
    },
    "apo_v2_relation_pilot": {
        # Cheap end-to-end gate for the relation-only redesign.  Stage 1 is
        # byte-identical P0; each Stage-2 round targets one registered relation.
        "candidate_count": 2,
        "beam_size": 1,
        "train_batch_size": 6,
        "selection_dev_size": 4,
        "train_doc_ids": None,
        "selection_dev_doc_ids": [
            "aa24-109a",
            "aa23-213a",
            "aa24-242a",
            "aa25-239a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 3, "joint": 0},
        "freeze_artifact": False,
    },
    "apo_v2_relation_tuning_small": {
        # Low-cost APO development run.  This remains a fully automatic APO
        # search starting from P0; it is intended for tuning the optimizer's
        # registered meta-parameters, never for manually retaining or editing
        # an observed candidate prompt.  The six fixed dev documents cover all
        # three text-extraction relation types.
        "candidate_count": 2,
        "beam_size": 1,
        "train_batch_size": 6,
        "selection_dev_size": 6,
        "train_doc_ids": None,
        "selection_dev_doc_ids": [
            "aa24-109a",
            "aa23-213a",
            "aa24-242a",
            "aa25-239a",
            "aa24-249a",
            "aa22-074a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 3, "joint": 0},
        "freeze_artifact": False,
    },
    "apo_v2_cpe_stage1_relation_tuning_small": {
        # Gold-v8 tuning protocol.  One automatic Configuration/CPE Stage-1
        # round precedes the three registered relation rounds.  Historical
        # relation-only presets remain unchanged for exact reproducibility.
        "candidate_count": 2,
        "beam_size": 1,
        "train_batch_size": 6,
        "selection_dev_size": 6,
        "train_doc_ids": None,
        "selection_dev_doc_ids": [
            "aa24-109a",
            "aa23-213a",
            "aa24-242a",
            "aa25-239a",
            "aa24-249a",
            "aa22-074a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 1, "relation": 3, "joint": 0},
        "objective_focus": "configuration_cpe_then_relations",
        "freeze_artifact": False,
    },
    "apo_v2_relation_formal": {
        # Development-only APO-v2 protocol.  The 14-document screen contains
        # every positive exploited_by dev document and all positive
        # instantiates dev documents; promotion is still decided on all 21 dev
        # documents with paired repeats.  It never freezes or evaluates test.
        "candidate_count": 4,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": [
            "aa24-109a",
            "aa23-213a",
            "aa25-239a",
            "aa24-249a",
            "aa20-259a-iran-citrix-vpn-cve-19781",
            "aa22-074a",
            "aa24-242a",
            "aa22-228a",
            "aa25-022a",
            "accellion-fta-google-accellion-data-theft-extortion",
            "proxylogon-microsoft-hafnium-exchange",
            "confluence-nvd-confluence-26084",
            "aa22-257a",
            "aa23-187a",
        ],
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": True,
        "phase_rounds": {"entity": 0, "relation": 3, "joint": 0},
        "freeze_artifact": False,
    },
    "apo_v2_cpe_stage1_relation_formal": {
        # Formal development-only successor to the relation-only protocol.
        # Stage 1 is optimized once for Configuration/CPE before the same
        # affects -> instantiates -> exploited_by relation schedule.
        "candidate_count": 4,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": [
            "aa24-109a",
            "aa23-213a",
            "aa25-239a",
            "aa24-249a",
            "aa20-259a-iran-citrix-vpn-cve-19781",
            "aa22-074a",
            "aa24-242a",
            "aa22-228a",
            "aa25-022a",
            "accellion-fta-google-accellion-data-theft-extortion",
            "proxylogon-microsoft-hafnium-exchange",
            "confluence-nvd-confluence-26084",
            "aa22-257a",
            "aa23-187a",
        ],
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": True,
        "phase_rounds": {"entity": 1, "relation": 3, "joint": 0},
        "objective_focus": "configuration_cpe_then_relations",
        "freeze_artifact": False,
    },
    "pilot": {
        "candidate_count": 1,
        "beam_size": 1,
        "train_batch_size": 2,
        "selection_dev_size": 2,
        "train_doc_ids": [
            "aa23-339a-coldfusion-cve-26360",
            "aa22-187a",
        ],
        "selection_dev_doc_ids": [
            "confluence-nvd-confluence-26084",
            "aa22-074a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 1, "relation": 1, "joint": 0},
        "freeze_artifact": False,
    },
    "configuration_pilot": {
        "candidate_count": 3,
        "beam_size": 2,
        "train_batch_size": 6,
        "selection_dev_size": 4,
        "train_doc_ids": [
            "aa23-339a-coldfusion-cve-26360",
            "aa22-187a",
            "fortios-sslvpn-rapid7-fortinet-targeting",
            "accellion-fta-nvd-accellion-27101",
            "goanywhere-nvd-goanywhere-0669",
            "apache-httpd-41773-tenable-httpd-41773",
        ],
        "selection_dev_doc_ids": [
            "confluence-nvd-confluence-26084",
            "aa22-074a",
            "aa24-249a",
            "aa25-022a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        # Configuration 是实体类型；先优化实体识别，再优化 affects，最后联合校准。
        "phase_rounds": {"entity": 1, "relation": 1, "joint": 1},
        "objective_focus": "configuration_affects",
        "freeze_artifact": False,
    },
    "configuration_affects_atomic": {
        "candidate_count": 3,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 8,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        # The single-repeat selection pass is exploratory only.  Positive
        # target candidates are promoted before cross-label safety gates and
        # receive three complete-dev paired repeats.
        "dev_repeats": 1,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 2,
        "full_dev_recheck": True,
        "component_ablation": False,
        # 两轮先修 Configuration 候选，再分别修 affects 决策和缺失端点。
        "phase_rounds": {"entity": 2, "relation": 1, "joint": 1},
        "objective_focus": "configuration_affects",
        "freeze_artifact": False,
    },
    "configuration_base_probe": {
        "candidate_count": 1,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        # A clean three-repeat full-dev baseline without APO candidates or a
        # duplicate finalist/final-report pass. This separates base-prompt
        # redesign from the stochastic prompt search.
        "dev_repeats": 3,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 0, "joint": 0},
        "objective_focus": "configuration_affects",
        "freeze_artifact": False,
    },
    "weakness_model_probe": {
        "candidate_count": 1,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        # Model-only gate: evaluate the frozen Weakness prompt on the complete
        # development set without critic/editor calls or APO candidates.
        "dev_repeats": 3,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 0, "joint": 0},
        "objective_focus": "weakness_instantiates",
        "freeze_artifact": False,
    },
    "weakness_instantiates_atomic": {
        "candidate_count": 2,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 8,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 1,
        # Three paired complete-dev repeats are required because low-count
        # Weakness/instantiates and unrelated relation labels have high run
        # variance; two repeats can turn a single fluctuation into a hard gate.
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        # Keep both an entity winner and any later relation/joint descendant.
        # A single finalist can otherwise let a target-tied descendant displace
        # the shorter Weakness rule before complete-dev paired evaluation.
        "finalist_count": 2,
        "full_dev_recheck": True,
        "component_ablation": False,
        "phase_rounds": {"entity": 2, "relation": 1, "joint": 1},
        "objective_focus": "weakness_instantiates",
        "freeze_artifact": False,
    },
    "exploited_by_pilot": {
        "candidate_count": 3,
        "beam_size": 2,
        "train_batch_size": 6,
        "selection_dev_size": 4,
        # 只在已复核 train/dev 中选择 exploited_by 覆盖度高的文档。
        # 此专项目不改 Stage 1，只优化 Stage 2 的 exploited_by 决策。
        "train_doc_ids": [
            "aa24-241a",
            "aa24-109a",
            "aa22-321a",
            "aa22-321a-hive-exchange",
            "aa25-050a",
            "aa23-278a",
        ],
        "selection_dev_doc_ids": [
            "aa24-249a",
            "aa20-259a-iran-citrix-vpn-cve-19781",
            "aa23-187a",
            "aa25-239a",
        ],
        "dev_repeats": 1,
        "final_dev_repeats": 1,
        "final_report_repeats": 1,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 1, "joint": 0},
        "objective_focus": "exploited_by",
        "freeze_artifact": False,
    },
    "hotstart_configuration_affects_recall": {
        # One heterogeneous joint round, warm-started from an audited prompt.
        # Search uses 14 stratified dev documents x2; the decision uses all 21
        # dev documents x3 with a fresh paired final validation.
        "candidate_count": 4,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 0, "joint": 1},
        "objective_focus": "configuration_affects_recall",
        "freeze_artifact": False,
    },
    "hotstart_exploited_by_precision": {
        # Stage 1 remains frozen; only the exploited_by decision guidance is
        # edited.  The same two-tier 14-dev/21-dev paired protocol is retained.
        "candidate_count": 4,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": False,
        "phase_rounds": {"entity": 0, "relation": 1, "joint": 0},
        "objective_focus": "exploited_by_precision",
        "freeze_artifact": False,
    },
    "stage1_all_entity_recall_p0": {
        # Stage-1-only diagnostic: four atomic entity rounds, warm-started from
        # manual P0 when --initial-prompt-artifact is omitted.  Stage 2 and
        # relation closure are skipped so candidate-layer entity recall is not
        # obscured by downstream decisions.
        "candidate_count": 4,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": False,
        "phase_rounds": {"entity": 4, "relation": 0, "joint": 0},
        "objective_focus": "all_entity_recall",
        "freeze_artifact": False,
    },
    "configuration_recall_only_p0": {
        # Configuration-only Stage-1 screening.  The two rounds first expand
        # mention coverage and then repair exact spans.  It intentionally does
        # not spend complete-dev or test budget until the +0.08 recall gate is
        # met on the 14-document paired selection set.
        "candidate_count": 4,
        "beam_size": 1,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 1,
        "full_dev_recheck": False,
        "component_ablation": False,
        "phase_rounds": {"entity": 2, "relation": 0, "joint": 0},
        "objective_focus": "configuration_recall_only",
        "freeze_artifact": False,
    },
    "formal": {
        "candidate_count": 4,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 8,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 1,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 2,
        "full_dev_recheck": True,
        "component_ablation": True,
        "phase_rounds": {"entity": 2, "relation": 2, "joint": 0},
        "freeze_artifact": True,
    },
    "formal_stratified": {
        # Two-tier exploration-verification protocol:
        # Search phase uses 14 stratified documents and 2 repeats for stable exploration.
        # Complete dev decision uses 21 documents with 3 paired repeats on top 3 finalists.
        "candidate_count": 4,
        "beam_size": 2,
        "train_batch_size": 8,
        "selection_dev_size": 14,
        "train_doc_ids": None,
        "selection_dev_doc_ids": None,
        "dev_repeats": 2,
        "final_dev_repeats": 3,
        "final_report_repeats": 3,
        "finalist_count": 3,
        "full_dev_recheck": True,
        "component_ablation": False,
        "phase_rounds": {"entity": 2, "relation": 2, "joint": 1},
        "freeze_artifact": True,
    },
}
BASE_PROMPT_SHA256 = hashlib.sha256(
    "\0".join(
        (
            STAGE1_SYSTEM_PROMPT,
            STAGE1_USER_PROMPT,
            STAGE2_SYSTEM_PROMPT,
            STAGE2_USER_PROMPT,
            MULTIPASS_FEWSHOT,
        )
    ).encode("utf-8")
).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _aggregate_file_hash(paths: list[Path]) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(paths):
        aggregate.update(path.stem.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(hashlib.sha256(path.read_bytes()).digest())
    return aggregate.hexdigest()


def _task_cache_path(
    prompt_pair: dict,
    gold: dict,
    cache_namespace: str,
) -> Path:
    """为温度0的任务预测生成可跨运行复用、按重复轮次隔离的键。"""
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "base_prompt_sha256": BASE_PROMPT_SHA256,
        "prompt_id": prompt_pair["id"],
        "stage1_only": bool(prompt_pair.get("stage1_only")),
        "configuration_policy_override": bool(
            prompt_pair.get("configuration_policy_override")
        ),
        "text_sha256": hashlib.sha256(
            gold["text"].encode("utf-8")
        ).hexdigest(),
        "runtime_config": runtime_config(),
        "cache_namespace": cache_namespace,
    }
    digest = hashlib.sha256(
        json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return TASK_CACHE_ROOT / cache_namespace / prompt_pair["id"] / f"{digest}.json"


def _has_nonempty_model_response(value: object) -> bool:
    """Return whether a stage response is safe to reuse from cache."""
    return isinstance(value, str) and bool(value.strip())


def _prediction_cacheable(prediction: object) -> bool:
    """Reject cached predictions that contain a failed extraction stage.

    Evaluation records empty stage outputs to keep failure accounting honest,
    but a transient proxy failure must never become a reusable P0 baseline or
    a reusable candidate prediction.
    """
    if not isinstance(prediction, dict):
        return False
    trace = prediction.get("_trace")
    if not isinstance(trace, dict):
        return False
    windows = trace.get("windows")
    if not isinstance(windows, list) or not windows:
        return False
    return all(
        isinstance(window, dict)
        and (
            _has_nonempty_model_response(
                (window.get("stage1") or {}).get("raw_response")
            )
            if window.get("stage1_only") is True
            else all(
                _has_nonempty_model_response(
                    (window.get(stage) or {}).get("raw_response")
                )
                for stage in ("stage1", "stage2")
            )
        )
        for window in windows
    )


def _stage1_cache_is_ready(cache_dir: Path, expected_windows: int) -> bool:
    """Require successful Stage-1 responses before accepting a task cache."""
    if not cache_dir.exists():
        return False
    valid_responses = 0
    for cache_path in cache_dir.glob("*.json"):
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _has_nonempty_model_response(cached.get("raw_response")):
            valid_responses += 1
    return valid_responses >= expected_windows


def _configuration_recall_stage1_prompt(guidance: str) -> str:
    """Build a candidate prompt with P0's Configuration rules truly replaced.

    Empty guidance returns frozen P0 byte-for-byte.  A non-empty candidate is
    evaluated against a runtime copy in which only numbered Configuration
    selection, boundary, source and repetition rules are replaced.  The old
    clauses are removed from the model context rather than contradicted by a
    later natural-language override.
    """
    policy = guidance.strip()
    if not policy:
        return STAGE1_SYSTEM_PROMPT

    replacement_rules = {
        3: (
            "3. Return every explicit occurrence of CVE, CWE, and ATT&CK "
            "technique IDs. For Configuration, apply the active APO discovery "
            "hypothesis below to every distinct source occurrence that "
            "plausibly names a concrete affected product, software, component, "
            "device, or full CPE URI. Do not require a relation decision at "
            "Stage 1 and do not deduplicate occurrences by surface form or "
            "normalized ID."
        ),
        4: (
            "4. Configuration spans must be exact source substrings and follow "
            "the minimum independently identifiable product mention in that "
            "local occurrence. Do not use a global suffix blacklist: retain a "
            "component, role term, possessive, or parenthetical alias when it "
            "is part of the locally stated product name; omit version, patch, "
            "build, and deployment qualifiers only when they are outside that "
            "identity."
        ),
        7: (
            "7. Configuration candidate discovery is recall-first. Apply the "
            "single active discovery hypothesis below as a mandatory audit of "
            "its named occurrence class. A plausible product mention may be "
            "retained even when its final affects support is ambiguous; "
            "downstream relation and closure stages make that decision."
        ),
        8: (
            "8. Within the occurrence class named by the active hypothesis, "
            "treat every independently bounded occurrence as its own mention "
            "search unit. Preserve every distinct character offset; do not "
            "select one representative occurrence for a repeated product."
        ),
        9: (
            "9. At Stage 1, do not suppress a concrete Configuration candidate "
            "solely because the local relation wording is implicit, structural, "
            "or ambiguous. When uncertain between retaining and omitting a "
            "plausible affected-product mention, retain the candidate."
        ),
        12: (
            "12. Deduplicate only identical entries at the same type, start, "
            "and end. Every Configuration occurrence at a different source "
            "offset is a separate candidate, including a later abbreviation or "
            "short form of a product established earlier in the local section."
        ),
    }
    prompt = STAGE1_SYSTEM_PROMPT
    for rule_number, replacement in replacement_rules.items():
        pattern = re.compile(
            rf"(?ms)^{rule_number}\. .*?(?=^\d+\. |\n\nReturn one valid JSON object:)"
        )
        prompt, replacement_count = pattern.subn(replacement + "\n", prompt)
        if replacement_count != 1:
            raise RuntimeError(
                "frozen_stage1_configuration_rule_shape_changed:"
                f"{rule_number}:{replacement_count}"
            )

    return f"""{prompt}

## Active APO Configuration hypothesis
The numbered Configuration rules above are the complete runtime replacement;
the frozen P0 versions of Rules 3, 4, 7, 8, 9 and 12 are not active. Apply the
single hypothesis below without changing other entity types, JSON fields,
normalization requirements, or the exact 0-based substring contract.

{policy}

Before returning JSON, perform a second left-to-right Configuration coverage
audit for the active discovery hypothesis and verify every emitted
text/start/end triple against the source substring."""


def _predict_stage1_only(
    text: str,
    doc_id: str,
    *,
    stage1_guidance: str,
    examples_str: str = MULTIPASS_FEWSHOT,
    stage1_cache_dir: Path | str | None = None,
    configuration_policy_override: bool = False,
) -> dict:
    """Run the frozen Stage-1 contract without calling Stage 2.

    This executor is intentionally local to APO diagnostics.  Keeping it out
    of llm_methods.py preserves the frozen Rule/Multipass/Full dependency hash.
    It reuses the frozen Stage-1 prompt, parser, windowing and offset merger.
    """
    validate_apo_guidance(stage1_guidance, "")
    windows = build_text_windows(text)
    extractor = _ext()
    stage1_root = Path(stage1_cache_dir) if stage1_cache_dir else None

    def predict_window(window: dict) -> dict:
        cache_path = None
        cached = None
        if stage1_root is not None:
            fingerprint = {
                "prompt_version": STAGE1_PROMPT_VERSION,
                "runtime_config": stage1_runtime_config(),
                "examples_sha256": hashlib.sha256(
                    examples_str.encode("utf-8")
                ).hexdigest(),
                "stage1_guidance": stage1_guidance,
                "stage1_only": True,
                "configuration_policy_override": bool(
                    configuration_policy_override
                ),
                "window_start": window["start"],
                "window_end": window["end"],
                "window_text_sha256": hashlib.sha256(
                    window["text"].encode("utf-8")
                ).hexdigest(),
            }
            digest = hashlib.sha256(
                json.dumps(
                    fingerprint, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            cache_path = stage1_root / f"{digest}.json"
            if cache_path.exists():
                try:
                    candidate = json.loads(
                        cache_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and _has_nonempty_model_response(
                        candidate.get("raw_response")
                    )
                    and isinstance(candidate.get("trace"), dict)
                ):
                    cached = candidate

        if cached is not None:
            response = cached["raw_response"]
            stage_trace = dict(cached["trace"])
            stage_trace["cache_status"] = "hit"
            stage_trace["cache_path"] = str(cache_path)
        else:
            system_prompt = (
                _configuration_recall_stage1_prompt(stage1_guidance)
                if configuration_policy_override
                else _append_apo_guidance(
                    STAGE1_SYSTEM_PROMPT,
                    stage1_guidance,
                    "stage1",
                )
            )
            response, stage_trace = _call_with_retries(
                extractor,
                STAGE1_USER_PROMPT.format(
                    examples=examples_str,
                    text=window["text"],
                ),
                system_prompt,
            )
            stage_trace["cache_status"] = "miss"
            if cache_path is not None and _has_nonempty_model_response(response):
                _write_json(
                    cache_path,
                    {
                        "created_at_utc": _utc_now(),
                        "raw_response": response,
                        "trace": stage_trace,
                    },
                )
                stage_trace["cache_path"] = str(cache_path)

        parsed = _loose_json(response)
        entities, _ = parse_entity_mentions(
            window["text"], parsed.get("entities", [])
        )
        public_entities = [
            {
                key: value
                for key, value in entity.items()
                if not key.startswith("_")
            }
            for entity in entities
        ]
        local = {
            "entities": public_entities,
            "relations": [],
            "_trace": {
                "stage1_only": True,
                "configuration_policy_override": bool(
                    configuration_policy_override
                ),
                "stage1": {
                    **stage_trace,
                    "raw_response": response,
                    "parsed_nonempty": bool(public_entities),
                },
                "stage2": {
                    "skipped": True,
                    "attempts": 0,
                    "errors": [],
                    "parsed_nonempty": False,
                },
            },
        }
        print(
            f"[{doc_id}] stage1-only window {window['start']}:{window['end']} "
            f"cache_{stage_trace.get('cache_status', 'disabled')}",
            flush=True,
        )
        return _offset_prediction(local, window["start"])

    if len(windows) > 1 and LLM_MAX_WORKERS > 1:
        with ThreadPoolExecutor(
            max_workers=min(LLM_MAX_WORKERS, len(windows))
        ) as executor:
            predictions = list(executor.map(predict_window, windows))
    else:
        predictions = [predict_window(window) for window in windows]
    return merge_window_predictions(predictions)


def _guidance_key(stage1_guidance: str, stage2_guidance: str) -> str:
    return hashlib.sha256(
        json.dumps(
            [stage1_guidance.strip(), stage2_guidance.strip()],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _prompt_id(stage1_guidance: str, stage2_guidance: str) -> str:
    digest = _guidance_key(stage1_guidance, stage2_guidance)[:12]
    return f"p_{digest}"


def _append_guidance(current: str, delta: str) -> str:
    """Append an APO delta without allowing it to erase audited rules."""
    current = current.strip()
    delta = delta.strip()
    if not current:
        return delta
    if not delta or delta in current:
        return current
    return f"{current}\n{delta}"


def _apply_guidance_edit(current: str, delta: str, operation: str) -> str:
    """Edit only the mutable guidance layer, never the frozen base prompt."""
    operation = operation.strip().lower()
    if operation == "append":
        return _append_guidance(current, delta)
    if operation == "replace":
        return delta.strip()
    raise ValueError(f"unknown_guidance_edit_operation:{operation}")


def _validate_candidate_delta_scope(
    phase: str,
    target_labels: list[str],
    stage1_delta: str,
    stage2_delta: str,
) -> None:
    """Require atomic APO edits instead of cross-label prompt rewrites."""
    if not isinstance(target_labels, list) or not all(
        isinstance(label, str) for label in target_labels
    ):
        raise ValueError("target_labels_must_be_a_string_list")
    if len(set(target_labels)) != len(target_labels):
        raise ValueError("duplicate_target_labels")
    entity_targets = [label for label in target_labels if label in ENTITY_LABELS]
    relation_targets = [label for label in target_labels if label in RELATION_LABELS]
    unknown = sorted(set(target_labels) - set(ENTITY_LABELS) - set(RELATION_LABELS))
    if unknown:
        raise ValueError(f"unknown_target_labels:{unknown}")
    if phase == "entity" and not (
        len(entity_targets) == 1 and not relation_targets
    ):
        raise ValueError("entity_phase_requires_one_entity_target")
    if phase == "relation" and not (
        len(relation_targets) == 1 and not entity_targets
    ):
        raise ValueError("relation_phase_requires_one_relation_target")
    if phase == "joint" and not (
        len(entity_targets) == 1 and len(relation_targets) == 1
    ):
        raise ValueError("joint_phase_requires_one_entity_and_one_relation_target")
    broad_patterns = (
        r"\b(?:all|every)\s+(?:entity|entities|entity\s+types?)\b",
        r"\b(?:all|every)\s+(?:relation|relations|relation\s+types?)\b",
    )
    combined = f"{stage1_delta}\n{stage2_delta}"
    if any(re.search(pattern, combined, re.IGNORECASE) for pattern in broad_patterns):
        raise ValueError("delta_contains_broad_cross_label_rule")
    explicit_targets = {
        match.group(1).casefold()
        for match in re.finditer(
            r"\bfor\s+(Configuration|Vulnerability|Weakness|AttackTechnique|"
            r"affects|instantiates|exploited_by)\b",
            combined,
            re.IGNORECASE,
        )
    }
    allowed_targets = {label.casefold() for label in target_labels}
    if explicit_targets - allowed_targets:
        raise ValueError(
            f"delta_explicitly_edits_non_target_labels:"
            f"{sorted(explicit_targets - allowed_targets)}"
        )
    if relation_targets:
        relation_target = re.escape(relation_targets[0])
        ambiguous_exclusive_patterns = (
            rf"\b(?:output|emit|return|produce|include|retain)\s+"
            rf"(?:the\s+)?{relation_target}(?:\s+relations?)?\s+only\b",
            rf"\b(?:only|exclusively)\s+"
            rf"(?:output|emit|return|produce|include|retain)\s+"
            rf"(?:the\s+)?{relation_target}(?:\s+relations?)?\b",
            r"\b(?:suppress|omit|drop|exclude|ignore)\b[^.\n]{0,60}"
            r"\b(?:other|non-target)\s+relation(?:s|\s+types?)?\b",
        )
        if any(
            re.search(pattern, stage2_delta, re.IGNORECASE)
            for pattern in ambiguous_exclusive_patterns
        ):
            raise ValueError(
                "relation_delta_uses_ambiguous_exclusive_output"
            )
    # The program already fixes the atomic target and the editable stage for
    # every candidate.  Requiring the model to repeat the literal label in the
    # delta rejects otherwise valid rules such as "preserve locally supported
    # product mentions" or "require a local CVE-product assignment".  Keep the
    # explicit non-target check above, but do not confuse missing label words
    # with an out-of-scope edit.


def _validate_configuration_candidate_policy(
    stage1_delta: str,
    strategy: str,
) -> None:
    """Reject precision shortcuts that previously destroyed product recall."""
    if strategy not in CONFIGURATION_STRATEGIES:
        raise ValueError(f"unknown_configuration_strategy:{strategy}")
    text = " ".join(stage1_delta.split())
    lexical_deletion_patterns = (
        r"\b(?:remove|strip|drop|omit|cut)\b[^.\n]{0,70}"
        r"\b(?:words?|tokens?|suffix(?:es)?|descriptors?)\b[^.\n]{0,90}"
        r"\b(?:server|device|protocol|suite|component|application|platform|service)\b",
        r"\b(?:remove|strip|drop|omit|cut)\s+(?:the\s+)?"
        r"(?:server|device|protocol|suite|component|application|platform|service)"
        r"\s+(?:word|token|suffix|descriptor)\b",
        r"\bshortest\s+(?:base|core)\s+(?:product|component|name)\b",
        r"\bshortest\s+(?:vendor\s+and\s+product|canonical|product)\s+head\b",
        r"\bstop\s+before\b[^.\n]{0,90}\bdescriptive\s+tokens?\b",
        r"\bcut\b[^.\n]{0,90}\bdescriptive\s+component\s+phrases?\b",
        r"\b(?:server|device|protocol|suite|component|application|platform|service)"
        r"\s+(?:is|are)\s+(?:always\s+)?(?:a\s+)?(?:generic|removable|non-identifying)\b",
    )
    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in lexical_deletion_patterns
    ):
        raise ValueError("configuration_candidate_uses_generic_lexical_deletion")

    if re.search(
        r"\b(?:strip|remove|drop|omit)\b[^.\n]{0,100}"
        r"\b(?:leading|trailing)\b[^.\n]{0,80}"
        r"\b(?:adjectives?|generic\s+(?:category\s+)?nouns?|category\s+nouns?)\b",
        text,
        re.IGNORECASE,
    ):
        raise ValueError("configuration_candidate_uses_generic_lexical_deletion")

    direct_only_patterns = (
        r"\bonly\s+when\b[^.\n]{0,140}\b(?:same\s+(?:sentence|clause)|"
        r"explicit\s+(?:relation\s+)?(?:verb|marker|cue)|directly\s+binds?)\b",
        r"\brequire\b[^.\n]{0,120}\b(?:same\s+(?:sentence|clause)|"
        r"explicit\s+(?:relation\s+)?(?:verb|marker|cue))\b",
    )
    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in direct_only_patterns
    ):
        raise ValueError("configuration_stage1_improperly_requires_direct_relation")

    strategy_markers = {
        "recall_only": (
            r"\b(?:ordinary\s+prose|product\s+mention|affected\s+(?:object|product)|"
            r"vulnerability\s+(?:description|statement)|"
            r"local\b[^.\n]{0,50}\b(?:fact|context|mention|scope|statement)|"
            r"nearby\s+context|adjacent\s+clause|recover|retain|candidate)\b"
        ),
        "boundary_only": (
            r"\b(?:boundary|span|head|tail|phrase|interval|bullet|list|table|"
            r"row|CPE|structured|section)\b"
        ),
        "full_rewrite": (
            r"\b(?:local|mention|fact|scope|product|component|affected)\b"
        ),
        "precision_control": (
            r"\b(?:exclude|reject|context-only|navigation|detection|remediation|"
            r"reference|unrelated)\b"
        ),
    }
    if not re.search(strategy_markers[strategy], text, re.IGNORECASE):
        raise ValueError(
            f"configuration_candidate_content_mismatches_strategy:{strategy}"
        )

    if len(stage1_delta) > 900:
        raise ValueError("configuration_stage1_delta_exceeds_900_characters")


def _validate_configuration_cpe_candidate_policy(
    stage1_delta: str,
    strategy: str,
) -> None:
    """Keep the added Stage-1 round on generalizable CPE decisions."""
    if strategy not in CONFIGURATION_CPE_STRATEGIES:
        raise ValueError(f"unknown_configuration_cpe_strategy:{strategy}")
    text = " ".join(stage1_delta.split())
    if re.search(r"\bcpe:2\.3:[aho]:[^\s,;]+", text, re.IGNORECASE):
        raise ValueError("configuration_cpe_candidate_contains_concrete_cpe_uri")
    strategy_markers = {
        "mention_boundary": (
            r"\b(?:Configuration|mention|occurrence|span|boundary|offset|"
            r"product|component)\b"
        ),
        "canonical_cpe_fields": (
            r"\b(?:CPE|canonical)\b[^.\n]{0,180}"
            r"\b(?:part|vendor|product)\b|"
            r"\b(?:part|vendor|product)\b[^.\n]{0,180}\b(?:CPE|canonical)\b"
        ),
        "parent_family_projection": (
            r"\b(?:parent|family)\b[^.\n]{0,180}"
            r"\b(?:component|product|project|projection)\b|"
            r"\b(?:component|product|project|projection)\b[^.\n]{0,180}"
            r"\b(?:parent|family)\b"
        ),
        "ambiguity_abstention": (
            r"\b(?:ambiguous|ambiguity|multiple|non-equivalent)\b[^.\n]{0,180}"
            r"\b(?:abstain|quarantine|reject|omit)\b|"
            r"\b(?:abstain|quarantine|reject|omit)\b[^.\n]{0,180}"
            r"\b(?:ambiguous|ambiguity|multiple|non-equivalent)\b"
        ),
    }
    if not re.search(strategy_markers[strategy], text, re.IGNORECASE):
        raise ValueError(
            f"configuration_cpe_candidate_content_mismatches_strategy:{strategy}"
        )
    if len(stage1_delta) > 900:
        raise ValueError("configuration_stage1_delta_exceeds_900_characters")


def _validate_configuration_recall_candidate_policy(
    stage1_delta: str,
    strategy: str,
    phase_round: int,
) -> None:
    """Enforce one discovery or boundary hypothesis per candidate."""
    expected = CONFIGURATION_RECALL_STRATEGIES_BY_ROUND.get(
        phase_round,
        CONFIGURATION_RECALL_STRATEGIES_BY_ROUND[2],
    )
    if strategy not in expected:
        raise ValueError(
            f"unknown_configuration_recall_strategy_round_{phase_round}:{strategy}"
        )
    text = " ".join(stage1_delta.split())
    precision_patterns = (
        r"\b(?:precision|false\s+positive|abstain|filter\s+out|context-only|"
        r"unrelated)\b",
        r"\b(?:exclude|reject|suppress|discard)\b[^.\n]{0,80}"
        r"\b(?:tool|platform|context|candidate|product)\b",
        r"\brequire\b[^.\n]{0,100}\b(?:direct|explicit)\b[^.\n]{0,60}"
        r"\b(?:relation|affects|cue|verb|evidence)\b",
    )
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in precision_patterns):
        raise ValueError("configuration_recall_candidate_adds_precision_filter")

    strategy_markers = {
        "repeated_alias_occurrence": (
            r"\b(?:repeat|later\s+(?:mention|occurrence)|alias|abbreviation|"
            r"acronym|short\s+form|same-product)\b"
        ),
        "structured_record_occurrence": (
            r"\b(?:table|row|cell|bullet|list|CPE|line-oriented|structured)\b"
        ),
        "narrative_component_occurrence": (
            r"\b(?:narrative|ordinary\s+prose|possessive|parenthetical|"
            r"standalone\s+(?:product|component)|title)\b"
        ),
        "section_scope_occurrence": (
            r"\b(?:section|heading|fact-block|scope|govern|terminat)\w*\b"
        ),
        "alias_parenthetical_boundary": (
            r"\b(?:parenthetical|parenthes(?:is|es)|alias|abbreviation|acronym)\b"
        ),
        "component_suffix_boundary": (
            r"\b(?:component|role\s+term|suffix|product\s+head|descriptor)\b"
        ),
        "version_qualifier_boundary": (
            r"\b(?:version|patch|build|deployment|edition|qualifier)\b"
        ),
        "coordinated_product_boundary": (
            r"\b(?:coordinated|conjunction|shared\s+(?:base|head)|"
            r"separate\s+product|product\s+family)\b"
        ),
    }
    if not re.search(strategy_markers[strategy], text, re.IGNORECASE):
        raise ValueError(
            "configuration_recall_candidate_content_mismatches_strategy:"
            f"{strategy}"
        )
    cross_strategy = [
        other
        for other in expected
        if other != strategy
        and re.search(strategy_markers[other], text, re.IGNORECASE)
    ]
    if cross_strategy:
        raise ValueError(
            "configuration_recall_candidate_mixes_strategies:"
            f"{strategy}:{','.join(cross_strategy)}"
        )
    if phase_round == 1:
        if re.search(
            r"\b(?:stop\s+before|trim|truncate|shortest|overextend|"
            r"underextend|boundary|suffix|version\s+(?:number|qualifier))\b",
            text,
            re.IGNORECASE,
        ):
            raise ValueError("configuration_discovery_candidate_adds_boundary_rule")
    else:
        if re.search(
            r"\b(?:scan|rescan|coverage\s+audit|every\s+(?:offset|occurrence)|"
            r"repeat(?:ed)?\s+mention|table|row|cell|bullet|CPE\s+line|"
            r"ordinary\s+prose|heading|section\s+scope)\b",
            text,
            re.IGNORECASE,
        ):
            raise ValueError("configuration_boundary_candidate_adds_discovery_rule")
        if not re.search(
            r"\b(?:exact|verbatim|source)\b[^.\n]{0,80}"
            r"\b(?:span|offset|start|end|substring)\b|"
            r"\b(?:span|offset|start|end)\b[^.\n]{0,80}"
            r"\b(?:exact|verbatim|source)\b",
            text,
            re.IGNORECASE,
        ):
            raise ValueError("configuration_boundary_candidate_missing_exact_span_rule")
    if len(stage1_delta) > 900:
        raise ValueError("configuration_stage1_delta_exceeds_900_characters")


def _validate_configuration_recall_gradient_policy(
    gradient: str,
    phase_round: int,
) -> None:
    """Keep the critic diagnosis inside the active recall-error route."""
    text = " ".join(gradient.split())
    precision_pattern = re.compile(
        r"\b(?:precision|false\s+positive|abstain|context-only|unrelated)\b|"
        r"\b(?:exclude|reject|suppress|discard)\b[^.\n]{0,80}"
        r"\b(?:tool|platform|context|candidate|product)\b",
        re.IGNORECASE,
    )
    if precision_pattern.search(text):
        raise ValueError("configuration_recall_gradient_adds_precision_route")
    if phase_round == 1:
        forbidden = re.compile(
            r"\b(?:boundary|trim|truncate|truncation|overextend|underextend|"
            r"stop\s+before|product-head\s+endpoint|version\s+qualifier|"
            r"role\s+suffix|trailing\s+punctuation)\b",
            re.IGNORECASE,
        )
        required = re.compile(
            r"\b(?:occurrence|repeat|alias|abbreviation|acronym|structured|"
            r"table|row|cell|narrative|prose|heading|section|scope|missed)\b",
            re.IGNORECASE,
        )
        if forbidden.search(text):
            raise ValueError(
                "configuration_round1_gradient_leaks_boundary_route"
            )
        if not required.search(text):
            raise ValueError(
                "configuration_round1_gradient_missing_discovery_route"
            )
        return
    forbidden = re.compile(
        r"\b(?:scan|rescan|coverage\s+audit|every\s+(?:offset|occurrence)|"
        r"repeat(?:ed)?\s+(?:mention|occurrence)|table|row|cell|bullet|"
        r"CPE\s+line|ordinary\s+prose|heading|section\s+scope|new\s+source)\b",
        re.IGNORECASE,
    )
    required = re.compile(
        r"\b(?:boundary|span|overextended|underextended|product\s+head|suffix|"
        r"qualifier|parenthetical|version|coordinated|conjunction)\b",
        re.IGNORECASE,
    )
    if forbidden.search(text):
        raise ValueError("configuration_round2_gradient_leaks_discovery_route")
    if not required.search(text):
        raise ValueError("configuration_round2_gradient_missing_boundary_route")


def _validate_exploited_by_candidate_policy(
    stage2_delta: str,
    strategy: str,
) -> None:
    """Keep the precision round heterogeneous and tied to direct exploitation."""
    if strategy not in EXPLOITED_BY_STRATEGIES:
        raise ValueError(f"unknown_exploited_by_strategy:{strategy}")
    text = " ".join(stage2_delta.split())
    strategy_markers = {
        "direct_exploit_trigger": (
            r"\b(?:direct|exploit|exploitation|initial\s+access|entry|trigger|"
            r"vulnerability-specific)\b"
        ),
        "post_exploitation_exclusion": (
            r"\b(?:post-exploit|post-exploitation|after|subsequent|execution|"
            r"persistence|privilege|credential|lateral|command|control|exclude|reject)\b"
        ),
        "local_pair_boundary": (
            r"\b(?:local|interval|boundary|same\s+(?:row|item|fact|clause)|"
            r"nearest|pair|endpoint|scope)\b"
        ),
        "precision_control_rewrite": (
            r"\b(?:direct|explicit|abstain|reject|exclude|evidence|trigger)\b"
        ),
    }
    if not re.search(strategy_markers[strategy], text, re.IGNORECASE):
        raise ValueError(
            f"exploited_by_candidate_content_mismatches_strategy:{strategy}"
        )


def _pair(
    stage1_guidance: str,
    stage2_guidance: str,
    *,
    id: str | None = None,
    parent_id: str | None = None,
    phase: str = "initial",
    round_index: int = 0,
    rationale: str = "",
    stage1_only: bool = False,
    configuration_policy_override: bool = False,
) -> dict:
    validate_apo_guidance(stage1_guidance, stage2_guidance)
    return {
        "id": id or _prompt_id(stage1_guidance, stage2_guidance),
        "stage1_guidance": stage1_guidance.strip(),
        "stage2_guidance": stage2_guidance.strip(),
        "parent_id": parent_id,
        "phase": phase,
        "round": round_index,
        "rationale": rationale.strip(),
        "stage1_only": bool(stage1_only),
        "configuration_policy_override": bool(
            configuration_policy_override
        ),
    }


def _initial_pair_from_artifact(path: Path | None) -> dict:
    """Load a prior audited prompt pair as the next APO run's seed."""
    if path is None:
        initial = _pair("", "")
        initial["id"] = "p0_manual"
        return initial
    artifact = json.loads(path.read_text(encoding="utf-8"))
    stage1_guidance = artifact.get("stage1_guidance")
    stage2_guidance = artifact.get("stage2_guidance")
    if not isinstance(stage1_guidance, str) or not isinstance(
        stage2_guidance,
        str,
    ):
        raise ValueError(
            "初始提示产物必须包含字符串 stage1_guidance/stage2_guidance"
        )
    return _pair(
        stage1_guidance,
        stage2_guidance,
        phase="continuation_seed",
        rationale=f"continued from {path.name}",
    )


def compose_score(entity_macro_f1: float, relation_macro_f1: float) -> float:
    """预注册主目标：关系抽取权重更高，且两部分均为分类型宏平均。"""
    return round(
        ENTITY_WEIGHT * entity_macro_f1
        + RELATION_WEIGHT * relation_macro_f1,
        6,
    )


def _filter_extraction(annotation: dict) -> tuple[list[dict], list[dict]]:
    entities = [
        entity
        for entity in annotation.get("entities", [])
        if entity.get("type") in EXTRACTION_ENTITY_TYPES
    ]
    entity_ids = {entity.get("id") for entity in entities}
    relations = [
        relation
        for relation in annotation.get("relations", [])
        if relation.get("type") in EXTRACTION_RELATION_TYPES
        and relation.get("head") in entity_ids
        and relation.get("tail") in entity_ids
    ]
    return entities, relations


def _macro_f1(metrics_by_type: dict, labels: tuple[str, ...]) -> float:
    return round(
        sum(metrics_by_type.get(label, {}).get("f1", 0.0) for label in labels)
        / len(labels),
        6,
    )


def _macro_fbeta(
    metrics_by_type: dict,
    labels: tuple[str, ...],
    beta: float,
) -> float:
    return round(
        sum(_fbeta(metrics_by_type.get(label, {}), beta) for label in labels)
        / len(labels),
        6,
    )


def _macro_recall(metrics_by_type: dict, labels: tuple[str, ...]) -> float:
    return round(
        sum(
            float(metrics_by_type.get(label, {}).get("recall", 0.0))
            for label in labels
        )
        / len(labels),
        6,
    )


def _raw_quality(prediction: dict, source_text: str) -> dict:
    """在评价投影之前审计非法输出和证据，防止非法项被静默丢弃。"""
    entities = prediction.get("entities", [])
    relations = prediction.get("relations", [])
    entity_by_id = {entity.get("id"): entity for entity in entities}

    valid_entity_ids = set()
    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        if (
            entity.get("type") in EXTRACTION_ENTITY_TYPES
            and isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(source_text)
            and source_text[start:end] == entity.get("text")
        ):
            valid_entity_ids.add(entity.get("id"))

    relation_schema_valid = 0
    evidence_valid = 0
    for relation in relations:
        head = entity_by_id.get(relation.get("head"))
        tail = entity_by_id.get(relation.get("tail"))
        expected = EXTRACTION_RELATION_ARGUMENT_TYPES.get(relation.get("type"))
        schema_valid = (
            head is not None
            and tail is not None
            and relation.get("head") in valid_entity_ids
            and relation.get("tail") in valid_entity_ids
            and expected == (head.get("type"), tail.get("type"))
        )
        if schema_valid:
            relation_schema_valid += 1
        evidence = relation.get("evidence")
        evidence_interval = _relation_evidence_interval(
            source_text,
            relation,
            head,
            tail,
        )
        if (
            schema_valid
            and isinstance(evidence, str)
            and evidence
            and evidence in source_text
            and evidence_interval is not None
        ):
            evidence_valid += 1

    entity_total = len(entities)
    relation_total = len(relations)
    overall_total = entity_total + relation_total
    overall_valid = len(valid_entity_ids) + relation_schema_valid
    return {
        "entity_consistency": (
            len(valid_entity_ids) / entity_total if entity_total else 1.0
        ),
        "relation_consistency": (
            relation_schema_valid / relation_total if relation_total else 1.0
        ),
        "overall_consistency": (
            overall_valid / overall_total if overall_total else 1.0
        ),
        "evidence_validity": (
            evidence_valid / relation_total if relation_total else 1.0
        ),
        "entity_total": entity_total,
        "entity_valid": len(valid_entity_ids),
        "relation_total": relation_total,
        "relation_schema_valid": relation_schema_valid,
        "relation_evidence_valid": evidence_valid,
    }


def _trace_attempts(prediction: dict) -> tuple[int, int, int]:
    attempts = 0
    replayed_attempts = 0
    failed_stages = 0
    windows = (prediction.get("_trace") or {}).get("windows", [])
    for window_trace in windows:
        if not isinstance(window_trace, dict):
            continue
        for stage in ("stage1", "stage2"):
            trace = window_trace.get(stage) or {}
            stage_attempts = int(trace.get("attempts") or 0)
            if (
                window_trace.get("cache_status") == "hit"
                or trace.get("cache_status") == "hit"
            ):
                replayed_attempts += stage_attempts
            else:
                attempts += stage_attempts
            if trace.get("errors") and not trace.get("raw_response"):
                failed_stages += 1
    return attempts, replayed_attempts, failed_stages


def evaluate_prompt_pair(
    prompt_pair: dict,
    gold_paths: list[Path],
    prediction_dir: Path,
    cache_namespace: str = "repeat_01",
) -> dict:
    """严格评价一个提示对；调用失败按该阶段空输出保留，不跳过文档。"""
    prediction_dir.mkdir(parents=True, exist_ok=True)
    entity_metrics = []
    overlap_entity_metrics = []
    candidate_entity_metrics = []
    candidate_overlap_entity_metrics = []
    relation_metrics = []
    normalized_relation_metrics = []
    entity_by_doc = []
    overlap_entity_by_doc = []
    candidate_entity_by_doc = []
    candidate_overlap_entity_by_doc = []
    relation_by_doc = []
    conditional_relation_by_doc = []
    normalized_relation_by_doc = []
    na_metrics = []
    candidate_na_metrics = []
    raw_totals = Counter()
    final_totals = Counter()
    attempts = 0
    replayed_attempts = 0
    failed_stages = 0
    cache_hits = 0
    cache_misses = 0
    started = time.perf_counter()
    stage1_only = bool(prompt_pair.get("stage1_only"))

    def run_prediction(
        gold: dict,
        doc_id: str,
        window_cache_dir: Path,
        stage1_cache_dir: Path,
    ) -> dict:
        if stage1_only:
            return _predict_stage1_only(
                gold["text"],
                doc_id,
                stage1_guidance=prompt_pair["stage1_guidance"],
                stage1_cache_dir=stage1_cache_dir,
                configuration_policy_override=bool(
                    prompt_pair.get("configuration_policy_override")
                ),
            )
        return predict_llm_multipass(
            gold["text"],
            doc_id,
            stage1_guidance=prompt_pair["stage1_guidance"],
            stage2_guidance=prompt_pair["stage2_guidance"],
            window_cache_dir=window_cache_dir,
            stage1_cache_dir=stage1_cache_dir,
        )

    for gold_path in gold_paths:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        cache_path = _task_cache_path(prompt_pair, gold, cache_namespace)
        window_cache_dir = (
            TASK_CACHE_ROOT
            / "_windows"
            / cache_namespace
            / prompt_pair["id"]
            / gold_path.stem
        )
        stage1_guidance_id = hashlib.sha256(
            prompt_pair["stage1_guidance"].encode("utf-8")
        ).hexdigest()[:12]
        stage1_cache_dir = (
            TASK_CACHE_ROOT
            / "_stage1"
            / cache_namespace
            / stage1_guidance_id
            / gold_path.stem
        )
        expected_windows = len(build_text_windows(gold["text"]))
        stage1_cache_ready = _stage1_cache_is_ready(
            stage1_cache_dir,
            expected_windows,
        )
        if cache_path.exists() and stage1_cache_ready:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_prediction = cached["prediction"]
            except (json.JSONDecodeError, KeyError, OSError):
                cached_prediction = None
            if _prediction_cacheable(cached_prediction):
                prediction = cached_prediction
                cache_hits += 1
                cache_status = "hit"
            else:
                prediction = run_prediction(
                    gold,
                    gold_path.stem,
                    window_cache_dir,
                    stage1_cache_dir,
                )
                if _prediction_cacheable(prediction):
                    _write_json(
                        cache_path,
                        {
                            "created_at_utc": _utc_now(),
                            "cache_namespace": cache_namespace,
                            "prompt_id": prompt_pair["id"],
                            "doc_id": gold_path.stem,
                            "prediction": prediction,
                        },
                    )
                    cache_status = "miss"
                else:
                    cache_status = "miss_unstable"
                cache_misses += 1
        else:
            prediction = run_prediction(
                gold,
                gold_path.stem,
                window_cache_dir,
                stage1_cache_dir,
            )
            if _prediction_cacheable(prediction):
                _write_json(
                    cache_path,
                    {
                        "created_at_utc": _utc_now(),
                        "cache_namespace": cache_namespace,
                        "prompt_id": prompt_pair["id"],
                        "doc_id": gold_path.stem,
                        "prediction": prediction,
                    },
                )
                cache_status = "miss"
            else:
                cache_status = "miss_unstable"
            cache_misses += 1
        raw_record = {
            "doc_id": gold_path.stem,
            "text": gold["text"],
            "schema_version": SCHEMA_VERSION,
            "prompt_id": prompt_pair["id"],
            "entities": prediction.get("entities", []),
            "relations": prediction.get("relations", []),
            "trace": prediction.get("_trace"),
            "cache": {
                "namespace": cache_namespace,
                "status": cache_status,
                "path": str(cache_path),
            },
        }
        if stage1_only:
            processed = {
                "candidate_entities": [
                    dict(entity) for entity in raw_record["entities"]
                ],
                "entities": [dict(entity) for entity in raw_record["entities"]],
                "relations": [],
                "_postprocess": {
                    "version": "stage1-only-no-closure-v1",
                    "input_entities": len(raw_record["entities"]),
                    "candidate_entities": len(raw_record["entities"]),
                    "output_entities": len(raw_record["entities"]),
                    "input_relations": 0,
                    "output_relations": 0,
                    "rejected": {},
                    "stage2_skipped": True,
                },
            }
        else:
            processed = apply_postprocess(raw_record)
        record = {
            **raw_record,
            "candidate_entities": processed["candidate_entities"],
            "entities": processed["entities"],
            "relations": processed["relations"],
            "_postprocess": processed["_postprocess"],
            "raw_prediction": {
                "entities": raw_record["entities"],
                "relations": raw_record["relations"],
            },
        }
        _write_json(prediction_dir / gold_path.name, record)
        print(
            f"[{prompt_pair['id']}] {gold_path.stem}: cache_{cache_status}",
            flush=True,
        )

        pred_entities, pred_relations = _filter_extraction(record)
        candidate_entities = [
            entity
            for entity in record.get("candidate_entities", [])
            if entity.get("type") in ENTITY_LABELS
        ]
        gold_entities, gold_relations = _filter_extraction(gold)
        entity_metrics.append(
            EM.calc_entity_metrics(pred_entities, gold_entities)
        )
        overlap_entity_metrics.append(
            EM.calc_overlap_entity_metrics(pred_entities, gold_entities)
        )
        candidate_entity_metrics.append(
            EM.calc_entity_metrics(candidate_entities, gold_entities)
        )
        candidate_overlap_entity_metrics.append(
            EM.calc_overlap_entity_metrics(candidate_entities, gold_entities)
        )
        relation_metrics.append(
            EM.calc_relation_metrics(
                pred_entities,
                pred_relations,
                gold_entities,
                gold_relations,
            )
        )
        normalized_relation_metrics.append(
            EM.calc_normalized_relation_metrics(
                pred_entities,
                pred_relations,
                gold_entities,
                gold_relations,
            )
        )
        entity_by_doc.append(
            EM.calc_entity_metrics_by_type(pred_entities, gold_entities)
        )
        overlap_entity_by_doc.append(
            EM.calc_overlap_entity_metrics_by_type(
                pred_entities, gold_entities
            )
        )
        candidate_entity_by_doc.append(
            EM.calc_entity_metrics_by_type(candidate_entities, gold_entities)
        )
        candidate_overlap_entity_by_doc.append(
            EM.calc_overlap_entity_metrics_by_type(
                candidate_entities, gold_entities
            )
        )
        relation_by_doc.append(
            EM.calc_relation_metrics_by_type(
                pred_entities,
                pred_relations,
                gold_entities,
                gold_relations,
            )
        )
        conditional_relation_by_doc.append(
            AM.conditional_relation_recall_by_type(
                pred_entities,
                pred_relations,
                gold_entities,
                gold_relations,
                endpoint_entities=candidate_entities,
            )
        )
        normalized_relation_by_doc.append(
            EM.calc_normalized_relation_metrics_by_type(
                pred_entities,
                pred_relations,
                gold_entities,
                gold_relations,
            )
        )
        na_metrics.append(EM.calc_na(pred_entities, gold_entities))
        candidate_na_metrics.append(
            EM.calc_na(candidate_entities, gold_entities)
        )
        quality = _raw_quality(raw_record, gold["text"])
        final_quality = _raw_quality(record, gold["text"])
        for key in (
            "entity_total",
            "entity_valid",
            "relation_total",
            "relation_schema_valid",
            "relation_evidence_valid",
        ):
            raw_totals[key] += quality[key]
            final_totals[key] += final_quality[key]
        doc_attempts, doc_replayed, doc_failed = _trace_attempts(prediction)
        if cache_status == "hit":
            replayed_attempts += doc_attempts + doc_replayed
        else:
            attempts += doc_attempts
            replayed_attempts += doc_replayed
        failed_stages += doc_failed

    entity_by_type = EM.aggregate_metrics_by_type(entity_by_doc)
    overlap_entity_by_type = EM.aggregate_metrics_by_type(
        overlap_entity_by_doc
    )
    candidate_entity_by_type = EM.aggregate_metrics_by_type(
        candidate_entity_by_doc
    )
    candidate_overlap_entity_by_type = EM.aggregate_metrics_by_type(
        candidate_overlap_entity_by_doc
    )
    relation_by_type = EM.aggregate_metrics_by_type(relation_by_doc)
    conditional_relation_by_type = (
        AM.aggregate_conditional_relation_recall_by_type(
            conditional_relation_by_doc
        )
    )
    normalized_relation_by_type = EM.aggregate_metrics_by_type(
        normalized_relation_by_doc
    )
    entity_macro = _macro_f1(entity_by_type, ENTITY_LABELS)
    candidate_entity_macro = _macro_f1(
        candidate_entity_by_type, ENTITY_LABELS
    )
    candidate_entity_macro_f2 = _macro_fbeta(
        candidate_entity_by_type, ENTITY_LABELS, 2.0
    )
    candidate_entity_macro_recall = _macro_recall(
        candidate_entity_by_type, ENTITY_LABELS
    )
    relation_macro = _macro_f1(relation_by_type, RELATION_LABELS)
    overall_total = raw_totals["entity_total"] + raw_totals["relation_total"]
    overall_valid = (
        raw_totals["entity_valid"] + raw_totals["relation_schema_valid"]
    )
    raw_quality = {
        **dict(raw_totals),
        "overall_consistency": round(
            overall_valid / overall_total if overall_total else 1.0,
            6,
        ),
        "evidence_validity": round(
            raw_totals["relation_evidence_valid"]
            / raw_totals["relation_total"]
            if raw_totals["relation_total"]
            else 1.0,
            6,
        ),
    }
    final_overall_total = (
        final_totals["entity_total"] + final_totals["relation_total"]
    )
    final_overall_valid = (
        final_totals["entity_valid"]
        + final_totals["relation_schema_valid"]
    )
    final_quality = {
        **dict(final_totals),
        "overall_consistency": round(
            final_overall_valid / final_overall_total
            if final_overall_total
            else 1.0,
            6,
        ),
        "evidence_validity": round(
            final_totals["relation_evidence_valid"]
            / final_totals["relation_total"]
            if final_totals["relation_total"]
            else 1.0,
            6,
        ),
    }
    return {
        "prompt_id": prompt_pair["id"],
        "documents": len(gold_paths),
        "entity": EM.aggregate_metrics(entity_metrics),
        "overlap_entity": EM.aggregate_metrics(overlap_entity_metrics),
        "candidate_entity": EM.aggregate_metrics(candidate_entity_metrics),
        "candidate_overlap_entity": EM.aggregate_metrics(
            candidate_overlap_entity_metrics
        ),
        "relation": EM.aggregate_metrics(relation_metrics),
        "normalized_relation": EM.aggregate_metrics(
            normalized_relation_metrics
        ),
        "entity_by_type": entity_by_type,
        "overlap_entity_by_type": overlap_entity_by_type,
        "candidate_entity_by_type": candidate_entity_by_type,
        "candidate_overlap_entity_by_type": candidate_overlap_entity_by_type,
        "relation_by_type": relation_by_type,
        "conditional_relation_recall_by_type": conditional_relation_by_type,
        "normalized_relation_by_type": normalized_relation_by_type,
        "entity_macro_f1": entity_macro,
        "candidate_entity_macro_f1": candidate_entity_macro,
        "candidate_entity_macro_f2": candidate_entity_macro_f2,
        "candidate_entity_macro_recall": candidate_entity_macro_recall,
        "relation_macro_f1": relation_macro,
        "exploited_by_f1": relation_by_type.get("exploited_by", {}).get(
            "f1", 0.0
        ),
        "conditional_affects_recall": conditional_relation_by_type.get(
            "affects", {}
        ).get("conditional_recall", 0.0),
        "score": compose_score(entity_macro, relation_macro),
        "raw_quality": raw_quality,
        "final_quality": final_quality,
        "na": EM.aggregate_na(na_metrics),
        "candidate_na": EM.aggregate_na(candidate_na_metrics),
        "api_call_attempts": attempts,
        "replayed_api_call_attempts": replayed_attempts,
        "cache_hit_documents": cache_hits,
        "cache_miss_documents": cache_misses,
        "failed_stage_calls": failed_stages,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prediction_dir": str(prediction_dir),
        "cache_namespace": cache_namespace,
        "stage1_only": stage1_only,
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 6)


def _mean_metric_dict(runs: list[dict], key: str) -> dict:
    fields = ("precision", "recall", "f1", "tp", "fp", "fn")
    return {
        field: _mean([float(run[key].get(field, 0.0)) for run in runs])
        for field in fields
    }


def _mean_by_type(
    runs: list[dict],
    key: str,
    labels: tuple[str, ...],
) -> dict:
    fields = ("precision", "recall", "f1", "tp", "fp", "fn")
    return {
        label: {
            field: _mean(
                [
                    float(
                        run[key]
                        .get(label, {})
                        .get(field, 0.0)
                    )
                    for run in runs
                ]
            )
            for field in fields
        }
        for label in labels
    }


def _mean_na(runs: list[dict], key: str) -> dict:
    """Preserve per-type normalization accuracy across paired repeats."""
    labels = (*ENTITY_LABELS, "overall")
    fields = ("na", "correct", "total")
    return {
        label: {
            field: _mean(
                [
                    float(
                        run.get(key, {})
                        .get(label, {})
                        .get(field, 0.0)
                    )
                    for run in runs
                ]
            )
            for field in fields
        }
        for label in labels
    }


def _mean_conditional_by_type(runs: list[dict]) -> dict:
    fields = (
        "conditional_recall",
        "tp",
        "fn",
        "eligible_gold",
        "endpoint_missing",
        "gold_total",
    )
    return {
        label: {
            field: _mean(
                [
                    float(
                        run.get("conditional_relation_recall_by_type", {})
                        .get(label, {})
                        .get(field, 0.0)
                    )
                    for run in runs
                ]
            )
            for field in fields
        }
        for label in RELATION_LABELS
    }


def evaluate_prompt_pair_repeated(
    prompt_pair: dict,
    gold_paths: list[Path],
    prediction_dir: Path,
    repeats: int,
    cache_namespace_prefix: str = "",
) -> dict:
    """开发集重复运行并按均值选优，显式记录随机波动。"""
    if repeats < 1:
        raise ValueError("dev_repeats 必须大于 0")
    namespace_prefix = (
        f"{cache_namespace_prefix}_" if cache_namespace_prefix else ""
    )
    runs = [
        evaluate_prompt_pair(
            prompt_pair,
            gold_paths,
            prediction_dir / f"repeat_{repeat_index:02d}",
            cache_namespace=(
                f"{namespace_prefix}repeat_{repeat_index:02d}"
            ),
        )
        for repeat_index in range(1, repeats + 1)
    ]
    if repeats == 1:
        result = dict(runs[0])
        result["repeat_count"] = 1
        result["score_std"] = 0.0
        result["repeat_metrics"] = runs
        return result

    entity_macro = _mean([run["entity_macro_f1"] for run in runs])
    candidate_entity_macro = _mean(
        [run["candidate_entity_macro_f1"] for run in runs]
    )
    candidate_entity_macro_f2 = _mean(
        [run["candidate_entity_macro_f2"] for run in runs]
    )
    candidate_entity_macro_recall = _mean(
        [run["candidate_entity_macro_recall"] for run in runs]
    )
    relation_macro = _mean([run["relation_macro_f1"] for run in runs])
    raw_fields = (
        "entity_total",
        "entity_valid",
        "relation_total",
        "relation_schema_valid",
        "relation_evidence_valid",
        "overall_consistency",
        "evidence_validity",
    )
    result = {
        "prompt_id": prompt_pair["id"],
        "documents": len(gold_paths),
        "entity": _mean_metric_dict(runs, "entity"),
        "overlap_entity": _mean_metric_dict(runs, "overlap_entity"),
        "candidate_entity": _mean_metric_dict(runs, "candidate_entity"),
        "candidate_overlap_entity": _mean_metric_dict(
            runs, "candidate_overlap_entity"
        ),
        "relation": _mean_metric_dict(runs, "relation"),
        "normalized_relation": _mean_metric_dict(
            runs, "normalized_relation"
        ),
        "entity_by_type": _mean_by_type(runs, "entity_by_type", ENTITY_LABELS),
        "overlap_entity_by_type": _mean_by_type(
            runs,
            "overlap_entity_by_type",
            ENTITY_LABELS,
        ),
        "candidate_entity_by_type": _mean_by_type(
            runs,
            "candidate_entity_by_type",
            ENTITY_LABELS,
        ),
        "candidate_overlap_entity_by_type": _mean_by_type(
            runs,
            "candidate_overlap_entity_by_type",
            ENTITY_LABELS,
        ),
        "relation_by_type": _mean_by_type(
            runs,
            "relation_by_type",
            RELATION_LABELS,
        ),
        "conditional_relation_recall_by_type": _mean_conditional_by_type(
            runs
        ),
        "normalized_relation_by_type": _mean_by_type(
            runs,
            "normalized_relation_by_type",
            RELATION_LABELS,
        ),
        "entity_macro_f1": entity_macro,
        "candidate_entity_macro_f1": candidate_entity_macro,
        "candidate_entity_macro_f2": candidate_entity_macro_f2,
        "candidate_entity_macro_recall": candidate_entity_macro_recall,
        "relation_macro_f1": relation_macro,
        "exploited_by_f1": _mean(
            [run["exploited_by_f1"] for run in runs]
        ),
        "conditional_affects_recall": _mean(
            [run["conditional_affects_recall"] for run in runs]
        ),
        "score": compose_score(entity_macro, relation_macro),
        "score_std": round(
            statistics.pstdev(run["score"] for run in runs),
            6,
        ),
        "raw_quality": {
            field: _mean(
                [float(run["raw_quality"].get(field, 0.0)) for run in runs]
            )
            for field in raw_fields
        },
        "final_quality": {
            field: _mean(
                [
                    float(run["final_quality"].get(field, 0.0))
                    for run in runs
                ]
            )
            for field in raw_fields
        },
        "na": _mean_na(runs, "na"),
        "candidate_na": _mean_na(runs, "candidate_na"),
        "api_call_attempts": sum(run["api_call_attempts"] for run in runs),
        "replayed_api_call_attempts": sum(
            run["replayed_api_call_attempts"] for run in runs
        ),
        "cache_hit_documents": sum(
            run["cache_hit_documents"] for run in runs
        ),
        "cache_miss_documents": sum(
            run["cache_miss_documents"] for run in runs
        ),
        "failed_stage_calls": sum(
            run["failed_stage_calls"] for run in runs
        ),
        "elapsed_seconds": round(
            sum(run["elapsed_seconds"] for run in runs),
            3,
        ),
        "prediction_dir": str(prediction_dir),
        "cache_namespaces": [run["cache_namespace"] for run in runs],
        "repeat_count": repeats,
        "repeat_metrics": runs,
        "stage1_only": bool(prompt_pair.get("stage1_only")),
    }
    return result


def _evaluate_repeated_with_failure_recovery(
    prompt_pair: dict,
    gold_paths: list[Path],
    prediction_dir: Path,
    repeats: int,
    cache_namespace_prefix: str,
    max_recovery_passes: int = 1,
) -> dict:
    """Retry only non-cacheable failed stages before exploratory gating."""
    metrics = evaluate_prompt_pair_repeated(
        prompt_pair,
        gold_paths,
        prediction_dir,
        repeats,
        cache_namespace_prefix=cache_namespace_prefix,
    )
    history = [{
        "pass": 0,
        "failed_stage_calls": int(metrics.get("failed_stage_calls", 0)),
        "api_call_attempts": int(metrics.get("api_call_attempts", 0)),
    }]
    total_attempts = int(metrics.get("api_call_attempts", 0))
    total_replayed = int(metrics.get("replayed_api_call_attempts", 0))
    total_elapsed = float(metrics.get("elapsed_seconds", 0.0))
    for recovery_pass in range(1, max_recovery_passes + 1):
        if int(metrics.get("failed_stage_calls", 0)) == 0:
            break
        metrics = evaluate_prompt_pair_repeated(
            prompt_pair,
            gold_paths,
            prediction_dir,
            repeats,
            cache_namespace_prefix=cache_namespace_prefix,
        )
        pass_attempts = int(metrics.get("api_call_attempts", 0))
        total_attempts += pass_attempts
        total_replayed += int(metrics.get("replayed_api_call_attempts", 0))
        total_elapsed += float(metrics.get("elapsed_seconds", 0.0))
        history.append({
            "pass": recovery_pass,
            "failed_stage_calls": int(metrics.get("failed_stage_calls", 0)),
            "api_call_attempts": pass_attempts,
        })
    metrics["api_call_attempts"] = total_attempts
    metrics["replayed_api_call_attempts"] = total_replayed
    metrics["elapsed_seconds"] = round(total_elapsed, 3)
    metrics["failure_recovery"] = {
        "attempted": len(history) > 1,
        "passes": history,
        "recovered": history[0]["failed_stage_calls"] > 0
        and history[-1]["failed_stage_calls"] == 0,
    }
    return metrics


def _evaluate_once_with_failure_recovery(
    prompt_pair: dict,
    gold_paths: list[Path],
    prediction_dir: Path,
    cache_namespace: str,
    max_recovery_passes: int = 1,
) -> dict:
    """Recover transient failures in one train/dev evaluation via its cache."""
    metrics = evaluate_prompt_pair(
        prompt_pair,
        gold_paths,
        prediction_dir,
        cache_namespace=cache_namespace,
    )
    history = [{
        "pass": 0,
        "failed_stage_calls": int(metrics.get("failed_stage_calls", 0)),
        "api_call_attempts": int(metrics.get("api_call_attempts", 0)),
    }]
    total_attempts = int(metrics.get("api_call_attempts", 0))
    total_replayed = int(metrics.get("replayed_api_call_attempts", 0))
    total_elapsed = float(metrics.get("elapsed_seconds", 0.0))
    for recovery_pass in range(1, max_recovery_passes + 1):
        if int(metrics.get("failed_stage_calls", 0)) == 0:
            break
        metrics = evaluate_prompt_pair(
            prompt_pair,
            gold_paths,
            prediction_dir,
            cache_namespace=cache_namespace,
        )
        pass_attempts = int(metrics.get("api_call_attempts", 0))
        total_attempts += pass_attempts
        total_replayed += int(metrics.get("replayed_api_call_attempts", 0))
        total_elapsed += float(metrics.get("elapsed_seconds", 0.0))
        history.append({
            "pass": recovery_pass,
            "failed_stage_calls": int(metrics.get("failed_stage_calls", 0)),
            "api_call_attempts": pass_attempts,
        })
    metrics["api_call_attempts"] = total_attempts
    metrics["replayed_api_call_attempts"] = total_replayed
    metrics["elapsed_seconds"] = round(total_elapsed, 3)
    metrics["failure_recovery"] = {
        "attempted": len(history) > 1,
        "passes": history,
        "recovered": history[0]["failed_stage_calls"] > 0
        and history[-1]["failed_stage_calls"] == 0,
    }
    return metrics


def _repeat_independence(metrics: dict) -> dict:
    """Audit whether every reported repeat was a fresh, successful execution."""
    runs = list(metrics.get("repeat_metrics") or [])
    namespaces = [run.get("cache_namespace") for run in runs]
    cache_hits = [int(run.get("cache_hit_documents", 0)) for run in runs]
    replayed = [int(run.get("replayed_api_call_attempts", 0)) for run in runs]
    failed = [int(run.get("failed_stage_calls", 0)) for run in runs]
    passed = bool(runs) and (
        len(namespaces) == len(set(namespaces))
        and all(namespaces)
        and not any(cache_hits)
        and not any(replayed)
        and not any(failed)
        and all(int(run.get("api_call_attempts", 0)) > 0 for run in runs)
    )
    return {
        "required": True,
        "repeat_count": len(runs),
        "unique_namespaces": len(namespaces) == len(set(namespaces)),
        "namespaces": namespaces,
        "cache_hit_documents": cache_hits,
        "replayed_api_call_attempts": replayed,
        "failed_stage_calls": failed,
        "passed": passed,
    }


def _annotation_features(gold: dict) -> set[str]:
    entities, relations = _filter_extraction(gold)
    features = {f"entity:{entity['type']}" for entity in entities}
    features.update(f"relation:{relation['type']}" for relation in relations)
    if not relations:
        features.add("negative-relations")
    surfaces = Counter(
        (entity.get("text") or "").casefold()
        for entity in entities
        if entity.get("text")
    )
    if any(count > 1 for count in surfaces.values()):
        features.add("repeated-mention")
    if sum(e.get("type") == "Vulnerability" for e in entities) > 1:
        features.add("multi-vulnerability")
    if sum(e.get("type") == "AttackTechnique" for e in entities) > 1:
        features.add("multi-technique")
    return features


def select_training_batch(
    train_paths: list[Path],
    round_index: int,
    batch_size: int,
    seed: int,
) -> list[Path]:
    """确定性覆盖采样：优先覆盖全部类型、负样本和重复提及场景。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    rng = random.Random(seed + round_index)
    shuffled = list(train_paths)
    rng.shuffle(shuffled)
    features = {
        path: _annotation_features(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in shuffled
    }
    uncovered = set().union(*(features[path] for path in shuffled))
    selected = []
    remaining = list(shuffled)
    while remaining and len(selected) < min(batch_size, len(train_paths)):
        best = max(
            remaining,
            key=lambda path: (
                len(features[path] & uncovered),
                len(features[path]),
                -shuffled.index(path),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered -= features[best]
    return selected


def select_training_batch_for_target(
    train_paths: list[Path],
    round_index: int,
    batch_size: int,
    seed: int,
    target_labels: tuple[str, ...],
) -> list[Path]:
    """Prefer target-bearing documents while retaining hard negatives.

    At most three quarters of a batch are reserved for positives.  A stable
    cyclic offset gives successive rounds new target documents before the
    eligible pool is reused; the remaining slots preserve broader structural
    coverage and negative examples.
    """
    if not target_labels:
        return select_training_batch(train_paths, round_index, batch_size, seed)
    target_features = {
        (f"entity:{label}" if label in ENTITY_LABELS else f"relation:{label}")
        for label in target_labels
    }
    features = {
        path: _annotation_features(json.loads(path.read_text(encoding="utf-8")))
        for path in train_paths
    }
    positives = [
        path for path in train_paths if features[path] & target_features
    ]
    rng = random.Random(seed + sum(ord(ch) for ch in "|".join(target_labels)))
    rng.shuffle(positives)
    positive_quota = min(
        len(positives),
        max(1, (min(batch_size, len(train_paths)) * 3 + 3) // 4),
    )
    selected: list[Path] = []
    if positives:
        start = ((round_index - 1) * positive_quota) % len(positives)
        selected = [
            positives[(start + offset) % len(positives)]
            for offset in range(positive_quota)
        ]
    fill_pool = [path for path in train_paths if path not in selected]
    remaining = min(batch_size, len(train_paths)) - len(selected)
    if remaining > 0:
        selected.extend(
            select_training_batch(
                fill_pool,
                round_index,
                remaining,
                seed + 7919,
            )
        )
    return selected


def select_stratified_evaluation_batch(
    dev_paths: list[Path],
    batch_size: int,
    seed: int,
    minimum_feature_coverage: int = 2,
) -> list[Path]:
    """Select a reproducible dev subset with repeated structural coverage.

    Training feedback benefits from rotating broad coverage.  Candidate
    selection instead needs a stable subset in which a single dense document
    cannot be the only representative of a relation or document structure.
    This selector therefore tries to cover every available annotation feature
    at least ``minimum_feature_coverage`` times before using density as a
    tie-breaker.
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if minimum_feature_coverage <= 0:
        raise ValueError("minimum_feature_coverage 必须大于 0")
    rng = random.Random(seed)
    shuffled = list(dev_paths)
    rng.shuffle(shuffled)
    features = {
        path: _annotation_features(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in shuffled
    }
    available = set().union(*(features[path] for path in shuffled))
    needed = {
        feature: min(
            minimum_feature_coverage,
            sum(feature in features[path] for path in shuffled),
        )
        for feature in available
    }
    selected = []
    remaining = list(shuffled)
    while remaining and len(selected) < min(batch_size, len(dev_paths)):
        best = max(
            remaining,
            key=lambda path: (
                sum(needed[feature] > 0 for feature in features[path]),
                sum(needed[feature] for feature in features[path]),
                len(features[path]),
                -shuffled.index(path),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for feature in features[best]:
            needed[feature] = max(0, needed[feature] - 1)
    return selected


def select_target_stratified_evaluation_batch(
    dev_paths: list[Path],
    batch_size: int,
    seed: int,
    target_labels: tuple[str, ...],
) -> list[Path]:
    """Build a stable target-rich dev subset with structural negatives."""
    if not target_labels:
        return select_stratified_evaluation_batch(dev_paths, batch_size, seed)
    target_features = {
        (f"entity:{label}" if label in ENTITY_LABELS else f"relation:{label}")
        for label in target_labels
    }
    features = {
        path: _annotation_features(json.loads(path.read_text(encoding="utf-8")))
        for path in dev_paths
    }
    complete = [
        path for path in dev_paths if target_features.issubset(features[path])
    ]
    partial = [
        path
        for path in dev_paths
        if path not in complete and features[path] & target_features
    ]
    target_quota = min(
        len(complete) + len(partial),
        max(1, (min(batch_size, len(dev_paths)) * 3 + 3) // 4),
    )
    selected: list[Path] = []
    for pool, pool_seed in ((complete, seed), (partial, seed + 1)):
        remaining = target_quota - len(selected)
        if remaining <= 0 or not pool:
            continue
        selected.extend(
            select_stratified_evaluation_batch(
                pool,
                min(remaining, len(pool)),
                pool_seed,
                minimum_feature_coverage=1,
            )
        )
    remaining = min(batch_size, len(dev_paths)) - len(selected)
    if remaining > 0:
        fill_pool = [path for path in dev_paths if path not in selected]
        selected.extend(
            select_stratified_evaluation_batch(
                fill_pool,
                remaining,
                seed + 2,
                minimum_feature_coverage=1,
            )
        )
    return selected


def _relation_tuples(entities: list[dict], relations: list[dict]) -> set:
    return EM.extract_relation_tuples(entities, relations)


def _error_entity_view(entity: dict, source_text: str | None = None) -> dict:
    """Keep only grounded fields needed by the private APO critic."""
    view = {
        key: entity.get(key)
        for key in ("type", "text", "start", "end", "normalized_id")
        if entity.get(key) is not None
    }
    if "text" not in view and source_text is not None:
        start = entity.get("start")
        end = entity.get("end")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(source_text)
        ):
            view["text"] = source_text[start:end]
    return view


def _error_contexts(source_text: str, entities: list[dict]) -> list[dict]:
    """Return bounded local contexts without joining distant fact blocks."""
    intervals = []
    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        intervals.append(
            (
                max(0, start - ERROR_SNIPPET_CONTEXT_CHARS),
                min(len(source_text), end + ERROR_SNIPPET_CONTEXT_CHARS),
            )
        )
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        {"start": start, "end": end, "text": source_text[start:end]}
        for start, end in merged[:2]
    ]


def _append_error_example(
    examples: dict,
    label: str,
    kind: str,
    example: dict,
) -> None:
    bucket = examples.setdefault(label, {}).setdefault(kind, [])
    bucket.append(example)


def _select_diverse_error_examples(
    examples: dict,
    limit: int = ERROR_EXAMPLES_PER_LABEL_AND_KIND,
) -> dict:
    """Select one example per document before taking additional examples."""
    selected = {}
    for label, by_kind in examples.items():
        selected[label] = {}
        for kind, bucket in by_kind.items():
            unique_docs = []
            deferred = []
            seen_documents = set()
            for example in bucket:
                document_id = example.get("document_id")
                if document_id not in seen_documents:
                    seen_documents.add(document_id)
                    unique_docs.append(example)
                else:
                    deferred.append(example)
            selected[label][kind] = (
                unique_docs + deferred
            )[:limit]
    return selected


def _merge_phase_error_examples(
    error_summary: dict,
    routes: tuple[str, ...],
) -> dict:
    """Merge routed examples without erasing their causal repair route."""
    merged = {}
    source = error_summary.get("training_error_examples", {})
    for route in routes:
        for label, by_kind in source.get(route, {}).items():
            for kind, bucket in by_kind.items():
                routed_kind = f"{route}_{kind}"
                for example in bucket:
                    tagged = dict(example)
                    tagged["repair_route"] = route
                    _append_error_example(
                        merged, label, routed_kind, tagged
                    )
    return _select_diverse_error_examples(merged)


def _evidence_diagnostic_counts(
    prediction: dict,
    source_text: str,
) -> Counter:
    """细分关系证据失败原因，供训练集文本梯度使用。"""
    counts = Counter()
    entity_by_id = {
        entity.get("id"): entity
        for entity in prediction.get("entities", [])
    }
    for relation in prediction.get("relations", []):
        if relation.get("type") not in EXTRACTION_RELATION_TYPES:
            continue
        head = entity_by_id.get(relation.get("head"))
        tail = entity_by_id.get(relation.get("tail"))
        evidence = relation.get("evidence")
        if head is None or tail is None:
            counts["invalid_relation_evidence_missing_endpoint"] += 1
            continue
        if not isinstance(evidence, str) or not evidence:
            counts["invalid_relation_evidence_missing_text"] += 1
            continue
        if evidence not in source_text:
            counts["invalid_relation_evidence_not_verbatim"] += 1
        if str(head.get("text", "")) not in evidence:
            counts["invalid_relation_evidence_missing_head"] += 1
        if str(tail.get("text", "")) not in evidence:
            counts["invalid_relation_evidence_missing_tail"] += 1

        evidence_start = relation.get("evidence_start")
        evidence_end = relation.get("evidence_end")
        if evidence_start is not None or evidence_end is not None:
            if not (
                isinstance(evidence_start, int)
                and not isinstance(evidence_start, bool)
                and isinstance(evidence_end, int)
                and not isinstance(evidence_end, bool)
                and 0 <= evidence_start < evidence_end <= len(source_text)
                and source_text[evidence_start:evidence_end] == evidence
            ):
                counts["invalid_relation_evidence_offsets"] += 1
        if _relation_evidence_interval(
            source_text,
            relation,
            head,
            tail,
        ) is None:
            counts["invalid_relation_evidence_wrong_mention"] += 1
    return counts


def build_error_summary(
    gold_paths: list[Path],
    prediction_dir: Path,
    *,
    example_limit: int = ERROR_EXAMPLES_PER_LABEL_AND_KIND,
) -> dict:
    """Summarize counts and bounded concrete TRAIN errors for ProTeGi."""
    errors = Counter()
    entity_fn = Counter()
    entity_fp = Counter()
    relation_fn = Counter()
    relation_fp = Counter()
    entity_examples = {}
    relation_examples = {}
    joint_examples = {}
    routed_relation = {
        "relation_prompt": Counter(),
        "joint_prompt": Counter(),
        "postprocess": Counter(),
        "blocked_prerequisite": Counter(),
    }

    for gold_path in gold_paths:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        pred = json.loads(
            (prediction_dir / gold_path.name).read_text(encoding="utf-8")
        )
        gold_entities, gold_relations = _filter_extraction(gold)
        pred_entities, pred_relations = _filter_extraction(pred)
        candidate_record = {
            "entities": pred.get("candidate_entities", pred_entities),
            "relations": pred_relations,
        }
        candidate_entities, _ = _filter_extraction(candidate_record)
        raw_pred = pred.get("raw_prediction") or pred
        raw_entities, raw_relations = _filter_extraction(raw_pred)
        gold_spans = {
            (e["start"], e["end"], e["type"]): e for e in gold_entities
        }
        pred_spans = {
            (e["start"], e["end"], e["type"]): e
            for e in candidate_entities
        }
        raw_spans = {
            (e["start"], e["end"], e["type"]): e for e in raw_entities
            if all(key in e for key in ("start", "end", "type"))
        }
        for key in gold_spans.keys() - pred_spans.keys():
            entity_type = key[2]
            entity_fn[entity_type] += 1
            start, end, _ = key
            overlapping = [
                (pstart, pend)
                for pstart, pend, ptype in pred_spans
                if ptype == entity_type
                and max(start, pstart) < min(end, pend)
            ]
            if overlapping:
                errors[f"boundary_mismatch:{entity_type}"] += 1
                if any(
                    pstart <= start and pend >= end
                    and (pstart < start or pend > end)
                    for pstart, pend in overlapping
                ):
                    errors[f"boundary_overextended:{entity_type}"] += 1
                if any(
                    start <= pstart and end >= pend
                    and (start < pstart or end > pend)
                    for pstart, pend in overlapping
                ):
                    errors[f"boundary_underextended:{entity_type}"] += 1
            surface = (gold_spans[key].get("text") or "").casefold()
            if (
                surface
                and sum(
                    (entity.get("text") or "").casefold() == surface
                    for entity in gold_entities
                )
                > 1
            ):
                errors[f"missed_repeated_mention:{entity_type}"] += 1
            if entity_type == "Configuration" and not overlapping:
                errors["configuration_completely_missed"] += 1
            gold_entity = gold_spans[key]
            overlap_entities = [
                pred_spans[pred_key]
                for pred_key in pred_spans
                if pred_key[2] == entity_type
                and max(start, pred_key[0]) < min(end, pred_key[1])
            ]
            failure_modes = []
            if not overlapping:
                failure_modes.append("completely_missed")
            else:
                failure_modes.append("boundary_mismatch")
                if any(
                    pstart <= start and pend >= end
                    and (pstart < start or pend > end)
                    for pstart, pend in overlapping
                ):
                    failure_modes.append("boundary_overextended")
                if any(
                    start <= pstart and end >= pend
                    and (start < pstart or end > pend)
                    for pstart, pend in overlapping
                ):
                    failure_modes.append("boundary_underextended")
            if (
                surface
                and sum(
                    (entity.get("text") or "").casefold() == surface
                    for entity in gold_entities
                )
                > 1
            ):
                failure_modes.append("missed_repeated_mention")
            _append_error_example(
                entity_examples,
                entity_type,
                "false_negative",
                {
                    "document_id": gold_path.stem,
                    "failure_modes": failure_modes,
                    "gold": _error_entity_view(gold_entity, gold["text"]),
                    "overlapping_predictions": [
                        _error_entity_view(entity, gold["text"])
                        for entity in overlap_entities
                    ],
                    "contexts": _error_contexts(gold["text"], [gold_entity]),
                },
            )
        for key in gold_spans.keys() & pred_spans.keys():
            if key[2] != "Configuration":
                continue
            gold_entity = gold_spans[key]
            pred_entity = pred_spans[key]
            gold_normalized_id = str(
                gold_entity.get("normalized_id") or ""
            ).strip()
            pred_normalized_id = str(
                pred_entity.get("normalized_id") or ""
            ).strip()
            if not gold_normalized_id or pred_normalized_id == gold_normalized_id:
                continue
            errors["normalization_mismatch:Configuration"] += 1
            _append_error_example(
                entity_examples,
                "Configuration",
                "normalization_mismatch",
                {
                    "document_id": gold_path.stem,
                    "failure_modes": ["normalization_mismatch"],
                    "gold": _error_entity_view(gold_entity, gold["text"]),
                    "prediction": _error_entity_view(
                        pred_entity, gold["text"]
                    ),
                    "contexts": _error_contexts(
                        gold["text"], [gold_entity]
                    ),
                },
            )
        for key in pred_spans.keys() - gold_spans.keys():
            entity_fp[key[2]] += 1
            if key[2] == "Configuration":
                start, end, _ = key
                overlaps_gold_configuration = any(
                    gold_type == "Configuration"
                    and max(start, gold_start) < min(end, gold_end)
                    for gold_start, gold_end, gold_type in gold_spans
                )
                if not overlaps_gold_configuration:
                    errors["configuration_non_gold_tail"] += 1
            pred_entity = pred_spans[key]
            start, end, entity_type = key
            overlap_entities = [
                gold_spans[gold_key]
                for gold_key in gold_spans
                if gold_key[2] == entity_type
                and max(start, gold_key[0]) < min(end, gold_key[1])
            ]
            _append_error_example(
                entity_examples,
                entity_type,
                "false_positive",
                {
                    "document_id": gold_path.stem,
                    "prediction": _error_entity_view(pred_entity, gold["text"]),
                    "overlapping_gold": [
                        _error_entity_view(entity, gold["text"])
                        for entity in overlap_entities
                    ],
                    "contexts": _error_contexts(gold["text"], [pred_entity]),
                },
            )

        gold_tuples = _relation_tuples(gold_entities, gold_relations)
        pred_tuples = _relation_tuples(pred_entities, pred_relations)
        raw_tuples = _relation_tuples(raw_entities, raw_relations)
        for relation in gold_tuples - pred_tuples:
            relation_fn[relation[1]] += 1
            head, relation_type, tail = relation
            head_present = head in raw_spans
            tail_present = tail in raw_spans
            if relation in raw_tuples:
                route = "postprocess"
                routed_relation[route][relation_type] += 1
                errors[f"relation_postprocess_dropped:{relation_type}"] += 1
            elif head_present and tail_present:
                route = "relation_prompt"
                routed_relation[route][relation_type] += 1
                errors[
                    f"relation_endpoints_present_but_unlinked:{relation_type}"
                ] += 1
            elif head_present ^ tail_present:
                route = "joint_prompt"
                routed_relation[route][relation_type] += 1
                missing_role = "tail" if head_present else "head"
                errors[
                    f"relation_missing_{missing_role}_endpoint:{relation_type}"
                ] += 1
            else:
                route = "blocked_prerequisite"
                routed_relation[route][relation_type] += 1
                errors[f"relation_both_endpoints_missing:{relation_type}"] += 1
            head_entity = gold_spans.get(head)
            tail_entity = gold_spans.get(tail)
            if head_entity is not None and tail_entity is not None:
                target_examples = (
                    relation_examples
                    if route == "relation_prompt"
                    else joint_examples
                    if route == "joint_prompt"
                    else None
                )
                if target_examples is None:
                    continue
                _append_error_example(
                    target_examples,
                    relation_type,
                    "false_negative",
                    {
                        "document_id": gold_path.stem,
                        "route": route,
                        "raw_endpoint_status": {
                            "head_present": head_present,
                            "tail_present": tail_present,
                        },
                        "gold": {
                            "type": relation_type,
                            "head": _error_entity_view(head_entity, gold["text"]),
                            "tail": _error_entity_view(tail_entity, gold["text"]),
                        },
                        "contexts": _error_contexts(
                            gold["text"], [head_entity, tail_entity]
                        ),
                    },
                )
        for relation in pred_tuples - gold_tuples:
            relation_fp[relation[1]] += 1
            head, relation_type, tail = relation
            routed_relation["relation_prompt"][relation_type] += 1
            if relation_type == "affects":
                gold_affects = {
                    item for item in gold_tuples if item[1] == "affects"
                }
                if (
                    any(item[0] == head for item in gold_affects)
                    and any(item[2] == tail for item in gold_affects)
                ):
                    errors["affects_cross_fact_pairing"] += 1
            head_entity = pred_spans.get(head)
            tail_entity = pred_spans.get(tail)
            if head_entity is not None and tail_entity is not None:
                _append_error_example(
                    relation_examples,
                    relation_type,
                    "false_positive",
                    {
                        "document_id": gold_path.stem,
                        "prediction": {
                            "type": relation_type,
                            "head": _error_entity_view(head_entity, gold["text"]),
                            "tail": _error_entity_view(tail_entity, gold["text"]),
                        },
                        "contexts": _error_contexts(
                            gold["text"], [head_entity, tail_entity]
                        ),
                    },
                )
        if not gold_relations:
            errors["negative_relation_documents"] += 1
            if pred_relations:
                errors["negative_documents_with_relation_prediction"] += 1

        quality = _raw_quality(raw_pred, gold["text"])
        errors["invalid_entity_outputs"] += (
            quality["entity_total"] - quality["entity_valid"]
        )
        errors["invalid_relation_schema_outputs"] += (
            quality["relation_total"] - quality["relation_schema_valid"]
        )
        errors["invalid_relation_evidence_outputs"] += (
            quality["relation_total"] - quality["relation_evidence_valid"]
        )
        errors.update(_evidence_diagnostic_counts(raw_pred, gold["text"]))

    return {
        "documents": len(gold_paths),
        "entity_false_negative_by_type": dict(sorted(entity_fn.items())),
        "entity_false_positive_by_type": dict(sorted(entity_fp.items())),
        "relation_false_negative_by_type": dict(sorted(relation_fn.items())),
        "relation_false_positive_by_type": dict(sorted(relation_fp.items())),
        "diagnostic_counts": dict(sorted(errors.items())),
        "training_error_examples": {
            "entity": _select_diverse_error_examples(
                entity_examples, example_limit
            ),
            "relation": _select_diverse_error_examples(
                relation_examples, example_limit
            ),
            "joint": _select_diverse_error_examples(
                joint_examples, example_limit
            ),
        },
        "routes": {
            route: dict(sorted(counts.items()))
            for route, counts in routed_relation.items()
        },
        "routing_totals": {
            "relation_actionable": sum(
                routed_relation["relation_prompt"].values()
            ),
            "relation_blocked_by_entity": sum(
                routed_relation["joint_prompt"].values()
            ) + sum(routed_relation["blocked_prerequisite"].values()),
            "postprocess_only": sum(
                routed_relation["postprocess"].values()
            ),
        },
        "data_boundary_note": (
            "Examples are bounded local excerpts from the current TRAIN "
            "mini-batch only; dev and test text are never included."
        ),
    }


def _phase_error_summary(error_summary: dict, phase: str) -> dict:
    """Expose only TRAIN counts/examples repairable by the active stage."""
    if phase not in {"entity", "relation", "joint"}:
        raise ValueError(f"未知 APO 阶段：{phase}")
    if phase == "joint":
        projected = dict(error_summary)
        # A joint critic must see both ordinary Stage-1/Stage-2 errors and
        # endpoint-blocked relation errors.  Keeping the route in the kind
        # name prevents an entity miss from being mistaken for a relation
        # decision error.
        projected["training_error_examples"] = _merge_phase_error_examples(
            error_summary, ("entity", "relation", "joint")
        )
        return projected

    prefix = "entity" if phase == "entity" else "relation"
    projected = {
        "documents": error_summary.get("documents", 0),
        f"{prefix}_false_negative_by_type": error_summary.get(
            f"{prefix}_false_negative_by_type", {}
        ),
        f"{prefix}_false_positive_by_type": error_summary.get(
            f"{prefix}_false_positive_by_type", {}
        ),
    }
    if phase == "relation":
        actionable = error_summary.get("routes", {}).get(
            "relation_prompt", {}
        )
        relation_fp_counts = error_summary.get(
            "relation_false_positive_by_type", {}
        )
        projected["relation_false_negative_by_type"] = {
            label: max(
                0,
                int(actionable.get(label, 0))
                - int(relation_fp_counts.get(label, 0)),
            )
            for label in set(actionable) | set(relation_fp_counts)
        }
    diagnostic_counts = error_summary.get("diagnostic_counts", {})
    if prefix == "entity":
        allowed_diagnostics = {
            key: value
            for key, value in diagnostic_counts.items()
            if key.startswith("invalid_entity")
            or key.startswith("boundary_")
            or key.startswith("normalization_")
            or key.startswith("missed_repeated_mention:")
            or key.startswith("configuration_")
        }
    else:
        allowed_diagnostics = {
            key: value
            for key, value in diagnostic_counts.items()
            if (
                key.startswith("invalid_relation")
                or key.startswith("negative_relation")
                or key.startswith("negative_documents_with_relation")
                or key.startswith("relation_")
                or key.startswith("affects_")
            )
            and not key.startswith("relation_missing_")
            and not key.startswith("relation_postprocess_")
        }
    projected["diagnostic_counts"] = allowed_diagnostics
    projected["training_error_examples"] = error_summary.get(
        "training_error_examples", {}
    ).get(prefix, {})
    projected["data_boundary_note"] = error_summary.get(
        "data_boundary_note", ""
    )
    return projected


def _phase_contract(phase: str) -> str:
    contracts = {
        "entity": (
            "Stage 1 alone extracts entity mentions. Stage 2 is frozen. "
            "Diagnose only entity selection, typing, repeated mentions, exact "
            "boundaries and normalization; do not recommend relation rules."
        ),
        "relation": (
            "Stage 2 receives a fixed entity list from Stage 1 and cannot add, "
            "remove, retype or respan an entity. Stage 1 is frozen. Diagnose "
            "only relation decisions, endpoint choice, evidence and abstention."
        ),
        "joint": (
            "Both stages are editable, but preserve the boundary: Stage 1 "
            "extracts entities and Stage 2 decides relations over fixed IDs."
        ),
    }
    try:
        return contracts[phase]
    except KeyError as exc:
        raise ValueError(f"未知 APO 阶段：{phase}") from exc


def _atomic_target_for_round(
    phase: str,
    local_round: int,
    objective_focus: str | None = None,
) -> tuple[str, ...]:
    """Return a pre-registered target; optimizer models never choose labels."""
    if local_round < 1:
        raise ValueError("local_round must be positive")
    if objective_focus == "configuration_cpe_then_relations":
        if phase == "entity":
            return ("Configuration",)
        if phase == "relation":
            schedule = (
                ("affects",),
                ("instantiates",),
                ("exploited_by",),
            )
            return schedule[(local_round - 1) % len(schedule)]
        if phase == "joint":
            return ("Configuration", "affects")
        raise ValueError(f"未知 APO 阶段：{phase}")
    if objective_focus == "all_entity_recall":
        if phase == "entity":
            schedule = (
                ("Configuration",),
                ("Weakness",),
                ("Vulnerability",),
                ("AttackTechnique",),
            )
            return schedule[(local_round - 1) % len(schedule)]
        if phase == "joint":
            return ENTITY_LABELS
        if phase == "relation":
            return RELATION_LABELS
        raise ValueError(f"未知 APO 阶段：{phase}")
    focused = {
        "configuration_recall_only": {
            "entity": ("Configuration",),
            "relation": ("affects",),
            "joint": ("Configuration",),
        },
        "configuration_affects": {
            "entity": ("Configuration",),
            "relation": ("affects",),
            "joint": ("Configuration", "affects"),
        },
        "configuration_affects_recall": {
            "entity": ("Configuration",),
            "relation": ("affects",),
            "joint": ("Configuration", "affects"),
        },
        "exploited_by": {
            "entity": ("AttackTechnique",),
            "relation": ("exploited_by",),
            "joint": ("AttackTechnique", "exploited_by"),
        },
        "exploited_by_precision": {
            "entity": ("AttackTechnique",),
            "relation": ("exploited_by",),
            "joint": ("AttackTechnique", "exploited_by"),
        },
        "weakness_instantiates": {
            "entity": ("Weakness",),
            "relation": ("instantiates",),
            "joint": ("Weakness", "instantiates"),
        },
    }
    if objective_focus in focused:
        return focused[objective_focus][phase]
    schedule = {
        "entity": (
            ("Configuration",),
            ("Weakness",),
            ("Vulnerability",),
            ("AttackTechnique",),
        ),
        "relation": (
            ("affects",),
            ("instantiates",),
            ("exploited_by",),
        ),
        "joint": (
            ("Configuration", "affects"),
            ("Weakness", "instantiates"),
            ("AttackTechnique", "exploited_by"),
        ),
    }
    try:
        phase_schedule = schedule[phase]
    except KeyError as exc:
        raise ValueError(f"未知 APO 阶段：{phase}") from exc
    return phase_schedule[(local_round - 1) % len(phase_schedule)]


def _atomic_error_summary(
    error_summary: dict,
    phase: str,
    target_labels: tuple[str, ...],
) -> dict:
    """Expose only the selected labels and stage-actionable error families."""
    projected = _phase_error_summary(error_summary, phase)
    target_set = set(target_labels)
    for key in (
        "entity_false_negative_by_type",
        "entity_false_positive_by_type",
        "relation_false_negative_by_type",
        "relation_false_positive_by_type",
    ):
        if key in projected:
            projected[key] = {
                label: count
                for label, count in projected[key].items()
                if label in target_set
            }
    examples = projected.get("training_error_examples", {})
    projected["training_error_examples"] = {
        label: value for label, value in examples.items() if label in target_set
    }
    diagnostics = projected.get("diagnostic_counts", {})
    known_labels = set(ENTITY_LABELS) | set(RELATION_LABELS)

    def belongs_to_target(key: str) -> bool:
        mentioned = {
            label
            for label in known_labels
            if re.search(rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])", key,
                         re.IGNORECASE)
        }
        return not mentioned or bool(mentioned & target_set)

    projected["diagnostic_counts"] = {
        key: value
        for key, value in diagnostics.items()
        if belongs_to_target(key)
    }
    projected["program_selected_target_labels"] = list(target_labels)
    return projected


def _frozen_base_rules(
    phase: str,
    objective_focus: str | None = None,
) -> str:
    if objective_focus == "configuration_recall_only":
        entity_rules = (
            "Stage 1 keeps exactly four entity types, exact 0-based source "
            "spans, canonical normalized identifiers and JSON-only output. "
            "Vulnerability, Weakness and AttackTechnique policies are frozen. "
            "For this APO run only, the Configuration selection, repetition "
            "and boundary policy is the editable object: a candidate may "
            "replace that entire policy while preserving the Configuration "
            "entity definition and exact-span contract. Precision and other "
            "labels are not optimization targets."
        )
    else:
        entity_rules = (
        "Stage 1 already defines Vulnerability as an explicit CVE identifier; "
        "Weakness as a flaw mechanism or explicit CWE rather than an impact; "
        "Configuration as a concrete affected product, software, component or "
        "device (the label does not mean settings, parameters, flags or values); "
        "and AttackTechnique as an explicit ATT&CK technique ID, or an "
        "unambiguous technique name only when no adjacent technique ID is "
        "present. It also requires exact 0-based spans, separate repeated "
        "mentions, relation-anchored Configuration selection, concise "
        "product/component spans that omit versions and deployment qualifiers, "
        "full literal CPE spans when explicitly affected, canonical CPE 2.3 "
        "normalized_id values for every Configuration, excludes tactic IDs, "
        "and requires JSON-only output. Stage 1 preserves locally supportable "
        "Configuration mention candidates; Stage 2 and graph closure determine "
        "which of them are retained as the tail of a supported affects fact. Repeated "
        "mentions are retained separately only when they locally anchor distinct "
        "CVE-product facts; the first occurrence is never selected by default. "
        "These definitions are "
            "immutable and editable guidance must not narrow, broaden or "
            "reinterpret them."
        )
    relation_rules = (
        "Stage 2 already requires only the supplied entity IDs and legal "
        "directions; evidence_start/evidence_end selecting one continuous "
        "source interval containing both endpoint surface texts and the "
        "relation wording; no relation from co-occurrence; direct "
        "vulnerability-exploitation semantics for exploited_by; and abstention "
        "when evidence is insufficient. A labeled CVE-to-Vendor/Product row "
        "is direct structural affects evidence when it contains an identifiable "
        "product; vendor-only rows are insufficient, explicit unrelated/not "
        "affected wording blocks affects, and actor non-exploitation does not "
        "negate a separately stated CVE-product assignment."
    )
    if phase == "entity":
        return entity_rules
    if phase == "relation":
        return relation_rules
    if phase == "joint":
        return entity_rules + " " + relation_rules
    raise ValueError(f"未知 APO 阶段：{phase}")


def _optimizer_call(
    extractor,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict, str, dict]:
    raw_response, trace = _call_with_retries(
        extractor,
        user_prompt,
        system_prompt,
        max_retries=1,
    )
    if not raw_response.strip() and trace.get("errors"):
        last_error = str(trace["errors"][-1])
        if "responses_api_http_error" in last_error:
            raise RuntimeError(last_error)
    return _loose_json(raw_response), raw_response, trace


def _sanitize_private_gradient(text: str) -> str:
    """Replace concrete TRAIN identifiers before an APO gradient is persisted."""
    sanitized = text
    for pattern, replacement in PRIVATE_GRADIENT_ID_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _gradient_from_response(parsed: dict, raw_response: str) -> str | None:
    """Recover a critic's prose when a gateway drops the requested JSON shell."""
    candidate = parsed.get("textual_gradient")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    raw = raw_response.strip()
    if not raw:
        return None
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    reparsed = _loose_json(raw)
    candidate = reparsed.get("textual_gradient")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    # A textual gradient is private diagnostic prose, not an injected prompt.
    # Accept plain prose after the same schema/identifier validation below.
    if not raw.startswith(("{", "[")):
        return raw
    return None


def _candidate_items_from_response(parsed: dict) -> list:
    """Accept the requested list plus harmless single-candidate JSON shells."""
    candidates = parsed.get("candidates")
    if isinstance(candidates, list):
        return candidates
    candidate = parsed.get("candidate")
    if isinstance(candidate, dict):
        return [candidate]
    if any(key in parsed for key in ("stage1_delta", "stage2_delta")):
        return [parsed]
    return []


def _is_terminal_optimizer_error(exc: Exception) -> bool:
    """Do not hide authentication, permission, or exhausted-credit failures."""
    message = str(exc).lower()
    terminal_markers = (
        "终止性鉴权或余额错误",
        "authentication",
        "unauthorized",
        "forbidden",
        "insufficient_quota",
        "insufficient balance",
        "credit balance",
        "status 401",
        "status 402",
        "status 403",
        "error code: 401",
        "error code: 402",
        "error code: 403",
    )
    return any(marker in message for marker in terminal_markers)


def generate_textual_gradient(
    extractor,
    prompt_pair: dict,
    error_summary: dict,
    phase: str,
    target_labels: tuple[str, ...] | None = None,
    fallback_extractor=None,
    objective_focus: str | None = None,
    phase_round: int = 1,
) -> dict:
    target_labels = target_labels or _atomic_target_for_round(phase, 1)
    phase_errors = _atomic_error_summary(
        error_summary, phase, tuple(target_labels)
    )
    recall_focus_contract = ""
    cpe_focus_contract = ""
    if (
        objective_focus == "configuration_cpe_then_relations"
        and phase == "entity"
        and "Configuration" in target_labels
    ):
        cpe_focus_contract = """

## Configuration/CPE Stage-1 objective
This single entity round maximizes candidate-layer end-to-end Configuration
normalization accuracy. A prediction is correct only when the exact mention
span and its canonical CPE 2.3 normalized_id both match Gold. Diagnose missed
mentions, boundary errors and wrong canonical identifiers as separate error
routes. Infer only general rules from the bounded TRAIN examples: never copy a
product name, vulnerability identifier, document identifier or concrete CPE
URI into the gradient. Keep relation decisions and Stage 2 outside this round.
"""
    if objective_focus == "configuration_recall_only":
        phase_errors["entity_false_positive_by_type"] = {}
        configuration_examples = (
            phase_errors.get("training_error_examples", {})
            .get("Configuration", {})
        )
        routed_modes = (
            {"completely_missed", "missed_repeated_mention"}
            if phase_round == 1
            else {
                "boundary_mismatch",
                "boundary_overextended",
                "boundary_underextended",
            }
        )
        phase_errors["training_error_examples"] = {
            "Configuration": {
                key: [
                    example
                    for example in value
                    if set(example.get("failure_modes", ())) & routed_modes
                ]
                for key, value in configuration_examples.items()
                if "false_negative" in key
            }
        }
        diagnostic_prefixes = (
            ("configuration_completely_missed", "missed_repeated_mention")
            if phase_round == 1
            else ("boundary_mismatch", "boundary_overextended", "boundary_underextended")
        )
        phase_errors["diagnostic_counts"] = {
            key: value
            for key, value in phase_errors.get("diagnostic_counts", {}).items()
            if key.startswith(diagnostic_prefixes)
        }
        round_task = (
            "mention-position discovery: recover completely missed occurrences "
            "and repeated/aliased offsets; optimize overlap recall and do not "
            "propose boundary trimming"
            if phase_round == 1
            else "strict boundary conversion: diagnose only overlapping but "
            "non-exact spans; preserve the parent's mention coverage and do not "
            "add a new discovery source"
        )
        recall_focus_contract = f"""

## Configuration-only recall objective
This is entity round {phase_round}: {round_task}.
Diagnose false negatives only. Precision, false positives, relations and all
non-target entity metrics are outside this search objective. The editable
Configuration policy is evaluated under a true runtime replacement of P0's
Configuration clauses. In round 1 propose one mention-discovery rule per error
route. In round 2 propose one boundary rule that is appended to the selected
round-1 parent; never restate or replace its coverage policy.
"""
    system_prompt = (
        "You are a computational linguistics assistant evaluating an academic "
        "information extraction benchmark on standard cybersecurity technical documentation. "
        "Diagnose counts and concrete bounded TRAIN errors without changing "
        "the frozen task schema. Return valid JSON only."
    )
    user_prompt = f"""## Frozen task
The task extracts exactly four entity types (Configuration, Vulnerability,
Weakness, AttackTechnique) and three directed relations (affects,
instantiates, exploited_by). Exact text spans and verbatim relation evidence
are immutable requirements. Upper-layer mapping is outside this task.

## Optimization phase
{phase}

## Program-selected atomic target
{', '.join(target_labels)}
The target is immutable. Diagnose only this target and do not select or propose
another entity or relation label.

## Stage contract
{_phase_contract(phase)}

## Rules already present in the frozen base prompt
{_frozen_base_rules(phase, objective_focus)}
Do not diagnose their absence or merely propose restating them. Identify
additional operational distinctions that the editable guidance can add. First
separate errors that violate an existing frozen rule from errors that require a
genuinely new rule. Infer no dominant cause before inspecting the supplied
counts and examples, and do not propose a rule unsupported by a bounded
example. For Weakness, distinguish whether errors stem from omitted core flaw
mechanism heads (e.g. injection, traversal, overflow, file write) or from including
superfluous pre-authentication/privilege/vector modifiers. For Configuration,
diagnose false negatives and false positives separately, checking whether vendor
or specific component head spans were omitted or truncated, without mechanically
shortening product mentions or moving a relation decision into Stage 1.
{cpe_focus_contract}
{recall_focus_contract}


## Current editable guidance
Stage 1:
{prompt_pair['stage1_guidance'] or '(empty)'}

Stage 2:
{prompt_pair['stage2_guidance'] or '(empty)'}

## Error counts and bounded examples from a TRAINING mini-batch
{json.dumps(phase_errors, ensure_ascii=False, indent=2)}

## Required output
Return:
{{"textual_gradient":"A concise diagnosis of why the editable guidance causes
these errors and what decision rules should change."}}

Use the concrete examples to infer general decision rules, but do not copy any
concrete CVE, CWE, ATT&CK, product, organization or document identifier into the
textual_gradient. Do not propose a new entity type, relation type, mapping layer,
parser, windowing rule, example set or post-processing step. In the relation
phase, refer only to relation labels and source/target endpoint roles; do not
repeat entity type names or any upper-layer relation word. Do not restate
argument directions, grammatical subject/object roles, type hierarchies, or
who acts on whom; those semantics are frozen in the base prompt."""
    raw_responses = []
    traces = []
    total_input_chars = 0
    total_output_chars = 0
    gradient = None
    semantic_rejections = []
    for semantic_attempt in range(1, 4):
        attempt_prompt = user_prompt
        if semantic_attempt > 1:
            repair_reason = (
                semantic_rejections[-1]
                if semantic_rejections
                else "missing or empty textual_gradient"
            )
            configuration_route_repair = ""
            if objective_focus == "configuration_recall_only":
                configuration_route_repair = (
                    "Revise only mention discovery and remove every boundary, "
                    "precision, trimming or qualifier proposal."
                    if phase_round == 1
                    else "Revise only span-boundary conversion and remove every "
                    "scanning, repetition, coverage or new-source proposal."
                )
            attempt_prompt += f"""

## Semantic/format repair
The previous response was rejected: {repair_reason}
Return exactly one JSON object with the key textual_gradient. Do not add a
code fence, explanation, alternative key or candidate list. Preserve the stage
contract. In the relation phase, use source/target endpoint roles and relation
labels only; do not name entity types or upper-layer relations, and do not
describe argument direction, grammatical roles, or who acts on whom.
{configuration_route_repair}"""
        attempt_extractor = (
            extractor
            if semantic_attempt == 1 or fallback_extractor is None
            else fallback_extractor
        )
        parsed, raw_response, trace = _optimizer_call(
            attempt_extractor,
            system_prompt,
            attempt_prompt,
        )
        trace["critic_runtime"] = (
            "primary"
            if attempt_extractor is extractor
            else "fallback_non_thinking"
        )
        raw_responses.append(raw_response)
        traces.append(trace)
        total_input_chars += len(system_prompt) + len(attempt_prompt)
        total_output_chars += len(raw_response)
        candidate_gradient = _gradient_from_response(parsed, raw_response)
        if candidate_gradient:
            candidate_gradient = _sanitize_private_gradient(
                candidate_gradient.strip()
            )
            try:
                validate_apo_textual_gradient(candidate_gradient, phase)
                if objective_focus == "configuration_recall_only":
                    _validate_configuration_recall_gradient_policy(
                        candidate_gradient,
                        phase_round,
                    )
            except (TypeError, ValueError) as exc:
                semantic_rejections.append(f"{type(exc).__name__}: {exc}")
            else:
                gradient = candidate_gradient
                break
        else:
            semantic_rejections.append("missing_or_empty_textual_gradient")
    if gradient is None:
        transport_errors = [
            error
            for trace in traces
            for error in trace.get("errors", [])
        ]
        raise RuntimeError(
            "优化模型连续3次未返回有效 textual_gradient；"
            f"raw_response_lengths={[len(item) for item in raw_responses]}；"
            f"semantic_rejections={semantic_rejections}；"
            f"transport_errors={transport_errors}"
        )
    combined_trace = {
        "attempts": sum(int(trace.get("attempts", 0)) for trace in traces),
        "errors": [
            error
            for trace in traces
            for error in trace.get("errors", [])
        ],
        "calls": traces,
        "semantic_attempts": len(traces),
        "semantic_rejections": semantic_rejections,
    }
    return {
        "textual_gradient": gradient,
        "raw_response": raw_responses[-1],
        "raw_responses": raw_responses,
        "trace": combined_trace,
        "input_chars": total_input_chars,
        "output_chars": total_output_chars,
    }


def generate_candidates(
    extractor,
    prompt_pair: dict,
    textual_gradient: str,
    phase: str,
    candidate_count: int,
    round_index: int,
    target_labels: tuple[str, ...] | None = None,
    id_factory: Any = None,
    objective_focus: str | None = None,
    phase_round: int = 1,
) -> dict:
    if phase not in {"entity", "relation", "joint"}:
        raise ValueError(f"未知 APO 阶段：{phase}")
    target_labels = tuple(
        target_labels or _atomic_target_for_round(phase, 1)
    )
    editable = {
        "entity": "Add a stage1 delta only; stage2 remains unchanged.",
        "relation": "Add a stage2 delta only; stage1 remains unchanged.",
        "joint": "Add separate deltas to both guidance fields.",
    }[phase]
    configuration_strategy_contract = ""
    configuration_recall_strategy_contract = ""
    configuration_cpe_strategy_contract = ""
    configuration_strategies: tuple[str, ...] = ()
    if (
        objective_focus == "configuration_cpe_then_relations"
        and "Configuration" in target_labels
        and phase == "entity"
    ):
        if candidate_count > len(CONFIGURATION_CPE_STRATEGIES):
            raise ValueError(
                "configuration_cpe_candidate_count_exceeds_registered_strategies"
            )
        configuration_strategies = CONFIGURATION_CPE_STRATEGIES[:candidate_count]
        strategy_descriptions = {
            "mention_boundary": (
                "recover and localize the minimal independently identifiable "
                "Configuration mention at its exact source offsets"
            ),
            "canonical_cpe_fields": (
                "choose canonical CPE 2.3 part, vendor and product fields from "
                "the local mention without a product-specific lookup list"
            ),
            "parent_family_projection": (
                "project a named component to an official parent product family "
                "only when the local evidence supports one unique family"
            ),
            "ambiguity_abstention": (
                "omit the Configuration candidate when local evidence permits "
                "multiple non-equivalent CPE product families rather than "
                "emitting an entity without a canonical normalized_id"
            ),
        }
        numbered = "\n".join(
            f"{index}. {strategy}: {strategy_descriptions[strategy]}."
            for index, strategy in enumerate(configuration_strategies, start=1)
        )
        configuration_cpe_strategy_contract = f"""

## Required Configuration/CPE strategies
Return exactly one candidate for each registered strategy below, in this order:
{numbered}
Every candidate must set `stage1_operation` and `stage2_operation` to `append`,
and `stage2_delta` to an empty string. Each Stage-1 delta implements exactly
one general decision hypothesis. Do not copy a concrete product, CVE, document
identifier or CPE URI from TRAIN feedback, and do not add a product dictionary.
For ambiguity_abstention, omission applies to the Configuration candidate; do
not emit an entity with an empty, free-form or knowingly ambiguous normalized_id.
"""
    elif (
        objective_focus == "configuration_recall_only"
        and "Configuration" in target_labels
        and phase == "entity"
    ):
        configuration_strategies = (
            CONFIGURATION_RECALL_STRATEGIES_BY_ROUND.get(
                phase_round,
                CONFIGURATION_RECALL_STRATEGIES_BY_ROUND[2],
            )
        )
        if phase_round == 1:
            strategy_descriptions = (
                "recover every later acronym, abbreviation, short form and "
                "same-product occurrence at its own source offset",
                "make each table row, cell, bullet, list item, CPE line and "
                "line-oriented record a mandatory independent occurrence audit",
                "make standalone products and identifiable components in "
                "ordinary narrative prose, titles and parentheticals a "
                "mandatory occurrence audit",
                "make a local heading, section or multi-line fact-block scope "
                "govern a mandatory occurrence audit until that scope terminates",
            )
        else:
            strategy_descriptions = (
                "decide whether a locally written parenthetical alias belongs "
                "inside the exact product span",
                "retain or remove a component/role suffix according to whether "
                "it is required for the independently identifiable product",
                "remove version, patch, build and deployment qualifiers without "
                "truncating the product identity",
                "select the shared product base versus distinct coordinated "
                "product heads without merging separate mentions",
            )
        numbered = "\n".join(
            f"{index}. {strategy}: {description}."
            for index, (strategy, description) in enumerate(
                zip(configuration_strategies, strategy_descriptions),
                start=1,
            )
        )
        configuration_recall_strategy_contract = f"""

## Required Configuration-only recall strategies for entity round {phase_round}
Return exactly one candidate for each strategy below, in this order:
{numbered}
Every candidate must set `stage1_operation` to `{'replace' if phase_round == 1 else 'append'}`.
Round 1 deltas add exactly one mention-discovery hypothesis to the common
runtime Configuration policy. Round 2 deltas append exactly one boundary rule
to the selected round-1 parent and must not restate or replace its coverage.
Set `stage2_operation` to `append` and `stage2_delta` to an empty string.
Do not add precision-control or exclusion-list hypotheses. Do not mix another
listed strategy into the same candidate. The immutable runtime policy already
enforces exact 0-based spans and distinct source offsets.
"""
    elif "Configuration" in target_labels and phase in {"entity", "joint"}:
        configuration_strategies = CONFIGURATION_STRATEGIES
        configuration_strategy_contract = f"""

## Required Configuration strategy diversity
Return a `strategy` field on every candidate. Use distinct strategies until
all requested slots are covered; a rejected slot must be repaired with the
same missing strategy rather than shifting later labels:
1. {CONFIGURATION_STRATEGIES[0]}: a recall-only hypothesis that recovers
   locally supportable product/component mentions without requiring a
   same-sentence relation verb at Stage 1.
2. {CONFIGURATION_STRATEGIES[1]}: a boundary-only hypothesis that changes
   mention/evidence span and local fact-block boundaries without adding a new
   lexical exclusion list.
3. {CONFIGURATION_STRATEGIES[2]}: a concise full rewrite of both editable
   guidance fields; set both editable operations to `replace` and preserve the
   frozen high-recall-then-closure cascade.
4. {CONFIGURATION_STRATEGIES[3]}: a precision-control hypothesis that excludes
   context-only tools/platforms by discourse role while retaining locally
   supportable candidates and complete identifiable product phrases.
Do not collapse these into paraphrases of one another. Stage 1 favors recall of
locally supportable Configuration candidates; Stage 2 and graph closure decide
which candidates participate in a retained affects fact.
"""
    exploited_by_strategy_contract = ""
    if "exploited_by" in target_labels and phase in {"relation", "joint"}:
        exploited_by_strategy_contract = f"""

## Required exploited_by strategy diversity
Return a `strategy` field on every candidate. Use distinct strategies until
all requested slots are covered:
1. {EXPLOITED_BY_STRATEGIES[0]}: require an explicit vulnerability-specific
   direct exploitation or entry trigger while retaining direct-use recall.
2. {EXPLOITED_BY_STRATEGIES[1]}: reject techniques described only as actions
   after access, such as execution, persistence, privilege, credential access,
   lateral movement or command-and-control behavior.
3. {EXPLOITED_BY_STRATEGIES[2]}: resolve the correct local source/target pair
   and evidence interval inside one bounded fact, row, item or clause.
4. {EXPLOITED_BY_STRATEGIES[3]}: fully rewrite the editable Stage-2 guidance;
   set `stage2_operation` to `replace`, preserve direct-exploitation recall,
   and use explicit evidence plus abstention for precision control.
Non-target relation types remain governed by the frozen rules.
"""
    configuration_selection_constraints = (
        (
            "- Round 1 optimizes Configuration mention-position overlap recall. "
            "Each candidate changes one discovery route only and must not add a "
            "boundary-trimming rule.\n"
            "- Round 2 optimizes strict Configuration recall. Preserve the "
            "parent's discovery coverage, append one boundary decision only, "
            "and do not introduce scanning, repetition or source rules.\n"
        )
        if objective_focus == "configuration_recall_only"
        else (
            "- Stage 1 may retain a locally supportable Configuration mention "
            "candidate even when the binding relation is expressed structurally "
            "or in an adjacent clause. Stage 2 and graph closure retain it only "
            "when an affects fact is supported. Reject unrelated tools/platforms, "
            "navigation, tags and reference URLs by contextual role, not by a "
            "product-word blacklist.\n"
            "- Keep the frozen minimal-but-identifiable span rule: omit a "
            "version, edition or deployment qualifier only when the remaining "
            "exact phrase still uniquely identifies the mentioned product/"
            "component. Never mechanically remove server, device, protocol, "
            "suite, component, application, platform or service.\n"
        )
    )
    configuration_repeat_constraint = (
        (
            "- During this Stage-1 recall search, every plausible "
            "Configuration occurrence at a distinct source offset is a "
            "separate candidate, including repeated aliases and short forms. "
            "Do not require a distinct local CVE-product relation before "
            "emitting the occurrence; downstream stages make that decision."
        )
        if objective_focus == "configuration_recall_only"
        else (
            "- Every explicit identifier mention is separate. Repeated "
            "Configuration mentions are separate only when they locally "
            "anchor distinct CVE-product facts; never select the first "
            "occurrence by default."
        )
    )
    system_prompt = (
        "You are a computational linguistics assistant evaluating an academic "
        "information extraction benchmark on standard cybersecurity technical documentation. "
        "You edit only the explicitly editable guidance of a frozen "
        "cybersecurity extraction prompt. Return valid JSON only."
    )
    user_prompt = f"""## Phase and edit boundary
{phase}: {editable}

## Stage contract
{_phase_contract(phase)}

## Rules already present in the frozen base prompt
{_frozen_base_rules(phase, objective_focus)}
Do not use a restatement of these rules as the candidate's only change. Add
short operational decision rules that implement the textual gradient. Reject
any textual-gradient suggestion that conflicts with the frozen definitions.
The frozen base prompt is immutable. The editable guidance may be appended to,
or replaced when an earlier editable rule conflicts with the new hypothesis.
Replacement applies only to the current editable guidance field and never to
the frozen base prompt.

## Atomic edit scope
- The program-selected target is exactly: {', '.join(target_labels)}.
- Do not choose or return target_labels; the program attaches this immutable
  field after validating each edit.
- Name each target label in its corresponding delta. Do not add a decision
  rule for any non-target label and do not use broad "all entities/relations"
  rules.
- For a relation target, scope the condition as `For <target>, accept/reject
  ...`; never write the ambiguous form `Output <target> only ...` or suppress
  other relation outputs. State that non-target relation types remain governed
  by the frozen rules.
- Across the returned candidates, implement distinct decision hypotheses for
  the same program-selected target.
- Return `stage1_operation` and `stage2_operation` as `append` or `replace` for
  each editable field. Use `append` for a non-editable field. A replace delta is
  the complete new value of that editable guidance field.
{configuration_strategy_contract}
{configuration_recall_strategy_contract}
{configuration_cpe_strategy_contract}
{exploited_by_strategy_contract}

## Current guidance
stage1_guidance:
{prompt_pair['stage1_guidance'] or '(empty)'}

stage2_guidance:
{prompt_pair['stage2_guidance'] or '(empty)'}

## Textual gradient
{textual_gradient}

## Immutable constraints
- Exactly Configuration, Vulnerability, Weakness and AttackTechnique.
- Exactly affects (Vulnerability->Configuration), instantiates
  (Vulnerability->Weakness), exploited_by
  (Vulnerability->AttackTechnique).
- Exact source spans, 0-based exclusive offsets, verbatim evidence and JSON
  output remain unchanged.
{configuration_repeat_constraint}
- The four entity definitions are immutable. In particular, Configuration
  denotes an affected product/software/component/device, never a setting,
  parameter, flag, option or value.
{configuration_selection_constraints}
- Configuration normalized_id is always a canonical CPE 2.3 URI. Prefer an
  explicit affected CPE span over a descriptive duplicate of the same fact;
  never use a free-form product name as normalized_id.
- stage2_guidance may use relation labels and source/target endpoint roles, but
  must not repeat entity type names or redefine relation argument directions.
- stage2_guidance must not describe direction, grammatical subject/object
  roles, type hierarchies, or who acts on whom. Add only positive/negative
  evidence cues, structural context exclusions, interval-boundary rules and
  abstention conditions; frozen relation semantics remain untouched.
- No mapping-layer entity or relation and no CAPEC/attack-pattern layer.
- The editable guidance must not contain any of these literal tokens, even as
  negative examples: CAPEC, AttackPattern, AttackTactic, KillChainPhase,
  leverages, realizes, implies, belongs_to_phase.
- No concrete CVE/CWE/ATT&CK identifier, product dictionary or training
  document wording may be inserted.
- Do not add code, regex, post-processing, examples or new output fields.

Generate {candidate_count} distinct, generalizable candidates. Each delta must
contain at most 75 words and the rationale at most 25 words. Use no more than
two sentences per delta. Do not repeat the frozen schema or current guidance.

Return:
{{"candidates":[
  {{"strategy":"... or null", "stage1_operation":"append|replace",
    "stage2_operation":"append|replace", "stage1_delta":"...",
    "stage2_delta":"...", "rationale":"..."}}
]}}"""
    candidates = []
    seen = set()
    seen_configuration_strategies = set()
    seen_exploited_by_strategies = set()
    rejections = []
    raw_responses = []
    traces = []
    total_input_chars = 0
    total_output_chars = 0
    repair_context = ""
    for generation_attempt in range(1, CANDIDATE_GENERATION_MAX_ATTEMPTS + 1):
        remaining = candidate_count - len(candidates)
        if remaining <= 0:
            break
        attempt_prompt = user_prompt
        if repair_context:
            missing_strategy_note = ""
            if (
                configuration_strategy_contract
                or configuration_recall_strategy_contract
                or configuration_cpe_strategy_contract
            ):
                missing = [
                    strategy
                    for strategy in configuration_strategies
                    if strategy not in seen_configuration_strategies
                ]
                missing_strategy_note = (
                    "\nMissing Configuration strategies, in required repair "
                    f"order: {', '.join(missing[:remaining])}."
                )
            elif exploited_by_strategy_contract:
                missing = [
                    strategy
                    for strategy in EXPLOITED_BY_STRATEGIES
                    if strategy not in seen_exploited_by_strategies
                ]
                missing_strategy_note = (
                    "\nMissing exploited_by strategies, in required repair "
                    f"order: {', '.join(missing[:remaining])}."
                )
            attempt_prompt += f"""

## Repair request
The previous response produced rejected candidates:
{repair_context}
{missing_strategy_note}
Return {remaining} replacement candidate(s). Remove the forbidden content;
do not merely explain the error and do not repeat an accepted candidate."""
        parsed, raw_response, trace = _optimizer_call(
            extractor,
            system_prompt,
            attempt_prompt,
        )
        raw_responses.append(raw_response)
        traces.append(trace)
        total_input_chars += len(system_prompt) + len(attempt_prompt)
        total_output_chars += len(raw_response)
        raw_candidates = _candidate_items_from_response(parsed)
        if not raw_candidates:
            rejections.append(
                {
                    "generation_attempt": generation_attempt,
                    "candidate_index": None,
                    "reason": "response_has_no_candidates_list",
                }
            )

        before_count = len(rejections)
        for candidate_index, item in enumerate(raw_candidates):
            reason = None
            if not isinstance(item, dict):
                reason = "candidate_is_not_an_object"
            else:
                stage1_delta = str(item.get("stage1_delta", "")).strip()
                stage2_delta = str(item.get("stage2_delta", "")).strip()
                rationale = str(item.get("rationale", "")).strip()
                stage1_operation = str(
                    item.get("stage1_operation", "append")
                ).strip().lower()
                stage2_operation = str(
                    item.get("stage2_operation", "append")
                ).strip().lower()
                model_targets = item.get("target_labels")
                configuration_strategy = (
                    str(item.get("strategy", "")).strip()
                    if (
                        configuration_strategy_contract
                        or configuration_recall_strategy_contract
                        or configuration_cpe_strategy_contract
                    )
                    else None
                )
                exploited_by_strategy = (
                    str(item.get("strategy", "")).strip()
                    if exploited_by_strategy_contract
                    else None
                )
                stage1 = prompt_pair["stage1_guidance"]
                stage2 = prompt_pair["stage2_guidance"]
                if phase in {"entity", "joint"}:
                    try:
                        stage1 = _apply_guidance_edit(
                            stage1, stage1_delta, stage1_operation
                        )
                    except ValueError:
                        reason = (
                            "invalid_stage1_operation:"
                            f"{stage1_operation}"
                        )
                if phase in {"relation", "joint"}:
                    try:
                        stage2 = _apply_guidance_edit(
                            stage2, stage2_delta, stage2_operation
                        )
                    except ValueError:
                        reason = (
                            "invalid_stage2_operation:"
                            f"{stage2_operation}"
                        )
                if (
                    not reason
                    and phase == "relation"
                    and stage1_operation != "append"
                ):
                    reason = "noneditable_stage1_operation_must_be_append"
                elif (
                    not reason
                    and phase == "entity"
                    and stage2_operation != "append"
                ):
                    reason = "noneditable_stage2_operation_must_be_append"
                if reason:
                    pass
                elif model_targets not in (None, [], list(target_labels)):
                    reason = "model_attempted_to_override_program_target"
                elif (
                    (
                        configuration_strategy_contract
                        or configuration_recall_strategy_contract
                        or configuration_cpe_strategy_contract
                    )
                    and not configuration_strategy
                ):
                    reason = "missing_configuration_strategy"
                elif (
                    configuration_strategy
                    in seen_configuration_strategies
                    and len(seen_configuration_strategies)
                    < len(configuration_strategies)
                ):
                    reason = (
                        "duplicate_configuration_strategy:"
                        f"{configuration_strategy}"
                    )
                elif (
                    exploited_by_strategy_contract
                    and not exploited_by_strategy
                ):
                    reason = "missing_exploited_by_strategy"
                elif (
                    exploited_by_strategy
                    in seen_exploited_by_strategies
                    and len(seen_exploited_by_strategies)
                    < len(EXPLOITED_BY_STRATEGIES)
                ):
                    reason = (
                        "duplicate_exploited_by_strategy:"
                        f"{exploited_by_strategy}"
                    )
                elif (
                    configuration_recall_strategy_contract
                    and (
                        stage1_operation
                        != ("replace" if phase_round == 1 else "append")
                        or stage2_operation != "append"
                        or bool(stage2_delta)
                    )
                ):
                    reason = (
                        "configuration_recall_candidate_requires_"
                        f"stage1_{'replace' if phase_round == 1 else 'append'}_"
                        "and_empty_stage2_append"
                    )
                elif (
                    configuration_cpe_strategy_contract
                    and (
                        stage1_operation != "append"
                        or stage2_operation != "append"
                        or bool(stage2_delta)
                    )
                ):
                    reason = (
                        "configuration_cpe_candidate_requires_stage1_append_"
                        "and_empty_stage2_append"
                    )
                elif (
                    configuration_strategy == "full_rewrite"
                    and (
                        stage1_operation != "replace"
                        or (phase == "joint" and stage2_operation != "replace")
                    )
                ):
                    reason = "configuration_full_rewrite_requires_replace"
                elif (
                    exploited_by_strategy == "precision_control_rewrite"
                    and stage2_operation != "replace"
                ):
                    reason = "exploited_by_precision_rewrite_requires_replace"
                elif phase in {"entity", "joint"} and not stage1_delta:
                    reason = "empty_stage1_delta"
                elif phase in {"relation", "joint"} and not stage2_delta:
                    reason = "empty_stage2_delta"
                elif (
                    len(stage1_delta) > 900
                    or len(stage2_delta) > 900
                    or len(rationale) > 400
                ):
                    reason = "candidate_delta_or_rationale_too_long"
                elif len(stage1) > 5000 or len(stage2) > 5000:
                    reason = "guidance_exceeds_5000_characters"
                else:
                    try:
                        _validate_candidate_delta_scope(
                            phase,
                            list(target_labels),
                            stage1_delta,
                            stage2_delta,
                        )
                        if configuration_recall_strategy_contract:
                            _validate_configuration_recall_candidate_policy(
                                stage1_delta,
                                configuration_strategy,
                                phase_round,
                            )
                        elif configuration_cpe_strategy_contract:
                            _validate_configuration_cpe_candidate_policy(
                                stage1_delta,
                                configuration_strategy,
                            )
                        elif configuration_strategy is not None:
                            _validate_configuration_candidate_policy(
                                stage1_delta,
                                configuration_strategy,
                            )
                        if exploited_by_strategy is not None:
                            _validate_exploited_by_candidate_policy(
                                stage2_delta,
                                exploited_by_strategy,
                            )
                        cid = id_factory(stage1, stage2) if id_factory else None
                        candidate = _pair(
                            stage1,
                            stage2,
                            id=cid,
                            parent_id=prompt_pair["id"],
                            phase=phase,
                            round_index=round_index,
                            rationale=rationale,
                            stage1_only=bool(prompt_pair.get("stage1_only")),
                            configuration_policy_override=bool(
                                prompt_pair.get("configuration_policy_override")
                            ),
                        )
                    except (TypeError, ValueError) as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                    else:
                        candidate["edit_target_labels"] = list(target_labels)
                        candidate["target_labels"] = list(target_labels)
                        candidate["stage1_edit_operation"] = stage1_operation
                        candidate["stage2_edit_operation"] = stage2_operation
                        candidate["stage1_delta"] = stage1_delta
                        candidate["stage2_delta"] = stage2_delta
                        if configuration_strategy is not None:
                            candidate["strategy"] = configuration_strategy
                        if configuration_recall_strategy_contract:
                            candidate["configuration_recall_round"] = phase_round
                        if exploited_by_strategy is not None:
                            candidate["strategy"] = exploited_by_strategy
                        if candidate["id"] == prompt_pair["id"]:
                            reason = "candidate_equals_parent"
                        elif candidate["id"] in seen:
                            reason = "duplicate_candidate"
                        else:
                            seen.add(candidate["id"])
                            if configuration_strategy is not None:
                                seen_configuration_strategies.add(
                                    configuration_strategy
                                )
                            if exploited_by_strategy is not None:
                                seen_exploited_by_strategies.add(
                                    exploited_by_strategy
                                )
                            candidates.append(candidate)
            if reason:
                rejections.append(
                    {
                        "generation_attempt": generation_attempt,
                        "candidate_index": candidate_index,
                        "reason": reason,
                    }
                )
            if len(candidates) >= candidate_count:
                break

        recent = rejections[before_count:]
        repair_context = json.dumps(
            recent or [{"reason": "not_enough_distinct_candidates"}],
            ensure_ascii=False,
            indent=2,
        )
    if configuration_strategies:
        strategy_order = {
            strategy: index
            for index, strategy in enumerate(configuration_strategies)
        }
        candidates.sort(
            key=lambda candidate: strategy_order.get(
                candidate.get("strategy"), len(strategy_order)
            )
        )
    combined_trace = {
        "attempts": sum(int(trace.get("attempts", 0)) for trace in traces),
        "errors": [
            error
            for trace in traces
            for error in trace.get("errors", [])
        ],
        "calls": traces,
    }
    return {
        "candidates": candidates,
        "raw_response": raw_responses[-1] if raw_responses else "",
        "raw_responses": raw_responses,
        "rejections": rejections,
        "trace": combined_trace,
        "input_chars": total_input_chars,
        "output_chars": total_output_chars,
    }


def _fbeta(metric: dict, beta: float) -> float:
    precision = float(metric.get("precision", 0.0))
    recall = float(metric.get("recall", 0.0))
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return round((1.0 + beta_sq) * precision * recall / denominator, 6)


def _configuration_candidate_na(metrics: dict) -> float:
    """End-to-end Configuration NA before Stage-2/closure filtering."""
    return float(
        metrics.get("candidate_na", {})
        .get("Configuration", {})
        .get("na", 0.0)
    )


def _objective_description(objective_focus: str | None) -> str:
    descriptions = {
        "configuration_cpe_then_relations": (
            "entity round: maximize candidate-layer Configuration end-to-end "
            "CPE normalization accuracy; relation rounds retain the "
            "pre-registered affects, instantiates and exploited_by atomic F1 "
            "schedule; final selection uses the general composite objective"
        ),
        "configuration_recall_only": (
            "Stage 1 only: round 1 maximizes candidate-layer Configuration "
            "mention-position overlap recall; round 2 maximizes strict recall "
            "subject to no loss of the selected parent's overlap recall; final "
            "selection uses strict recall. Precision, relations and non-target "
            "labels do not affect candidate ranking; Stage 2 and relation "
            "closure are skipped"
        ),
        "all_entity_recall": (
            "Stage 1 only: maximize candidate-layer type-macro strict F2 "
            "across Configuration, Vulnerability, Weakness, and "
            "AttackTechnique; Stage 2 and relation closure are skipped"
        ),
        "configuration_affects_recall": (
            "entity phase: candidate-layer Configuration strict F2; relation "
            "phase: affects strict F2; joint/final selection: 0.5 * delivered "
            "Configuration strict F1 + 0.5 * affects strict F1"
        ),
        "configuration_affects": (
            "entity phase: candidate-layer Configuration strict recall; "
            "relation phase: affects strict F1; joint phase: 0.5 * delivered "
            "Configuration strict F1 + 0.5 * affects strict F1; final "
            "selection uses the same joint objective"
        ),
        "exploited_by_precision": (
            "relation phase and final shortlist: exploited_by strict F0.5; "
            "Stage 1 is frozen"
        ),
        "exploited_by": (
            "relation phase and final shortlist: exploited_by strict F1; "
            "Stage 1 is frozen"
        ),
        "weakness_instantiates": (
            "entity phase: Weakness strict F1; relation phase: instantiates "
            "strict F1; joint phase: 0.5 * Weakness strict F1 + 0.5 * "
            "instantiates strict F1"
        ),
        None: (
            "0.4 * entity type-macro strict F1 + 0.6 * relation type-macro "
            "strict F1"
        ),
    }
    return descriptions[objective_focus]


def _phase_score(
    record: dict,
    phase: str,
    objective_focus: str | None = None,
    phase_round: int | None = None,
) -> float:
    metrics = record["metrics"]
    if objective_focus == "configuration_cpe_then_relations":
        if phase == "entity":
            return _configuration_candidate_na(metrics)
        if phase == "relation":
            return float(metrics["relation_macro_f1"])
        return float(metrics["score"])
    if objective_focus == "configuration_recall_only":
        metric_family = (
            "candidate_overlap_entity_by_type"
            if phase_round == 1
            else "candidate_entity_by_type"
        )
        return float(
            metrics.get(
                metric_family,
                metrics.get(
                    "candidate_entity_by_type", metrics.get("entity_by_type", {})
                ),
            )
            .get("Configuration", {})
            .get("recall", 0.0)
        )
    if objective_focus == "all_entity_recall":
        return float(metrics.get("candidate_entity_macro_f2", 0.0))
    if objective_focus in {
        "configuration_affects",
        "configuration_affects_recall",
    }:
        candidate_entity_by_type = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
        configuration_recall = candidate_entity_by_type.get(
            "Configuration", {}
        ).get("recall", 0.0)
        configuration_final_f1 = metrics.get("entity_by_type", {}).get(
            "Configuration", {}
        ).get("f1", 0.0)
        affects_f1 = metrics["relation_by_type"].get("affects", {}).get(
            "f1", 0.0
        )
        if phase == "entity":
            if objective_focus == "configuration_affects_recall":
                return _fbeta(
                    candidate_entity_by_type.get("Configuration", {}), 2.0
                )
            return configuration_recall
        if phase == "relation":
            if objective_focus == "configuration_affects_recall":
                return _fbeta(
                    metrics["relation_by_type"].get("affects", {}), 2.0
                )
            return affects_f1
        return round(0.5 * configuration_final_f1 + 0.5 * affects_f1, 6)
    if objective_focus in {"exploited_by", "exploited_by_precision"}:
        exploited_by_metrics = metrics["relation_by_type"].get(
            "exploited_by", {}
        )
        exploited_by_f1 = exploited_by_metrics.get("f1", 0.0)
        # This dedicated preset keeps Stage 1 frozen and runs only relation
        # optimization.  Even its final shortlist must therefore not mix an
        # unrelated entity macro score into the exploitation-relation target.
        if phase == "entity":
            return metrics["entity_macro_f1"]
        if objective_focus == "exploited_by_precision":
            return _fbeta(exploited_by_metrics, 0.5)
        return exploited_by_f1
    if objective_focus == "weakness_instantiates":
        candidate_entity_by_type = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
        weakness_f1 = candidate_entity_by_type.get("Weakness", {}).get(
            "f1", 0.0
        )
        instantiates_f1 = metrics["relation_by_type"].get(
            "instantiates", {}
        ).get("f1", 0.0)
        if phase == "entity":
            return weakness_f1
        if phase == "relation":
            return instantiates_f1
        return round(0.5 * weakness_f1 + 0.5 * instantiates_f1, 6)
    if phase == "entity":
        return metrics.get(
            "candidate_entity_macro_f1", metrics["entity_macro_f1"]
        )
    if phase == "relation":
        return metrics["relation_macro_f1"]
    return metrics["score"]


def _label_f1(metrics: dict, label: str) -> float:
    if label in ENTITY_LABELS:
        observed = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
    elif label in RELATION_LABELS:
        observed = metrics.get("relation_by_type", {})
    else:
        raise ValueError(f"unknown_target_label:{label}")
    return float(observed.get(label, {}).get("f1", 0.0))


def _target_value(
    metrics: dict,
    label: str,
    objective_focus: str | None = None,
    phase: str = "joint",
    phase_round: int | None = None,
) -> tuple[float, str]:
    """Return the preregistered metric optimized for one atomic label."""
    if (
        objective_focus == "configuration_cpe_then_relations"
        and label == "Configuration"
    ):
        return _configuration_candidate_na(metrics), "candidate_cpe_na"
    if objective_focus == "configuration_recall_only" and label == "Configuration":
        if phase_round == 1:
            observed = metrics.get("candidate_overlap_entity_by_type", {})
            return (
                float(observed.get(label, {}).get("recall", 0.0)),
                "overlap_recall",
            )
        observed = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
        return float(observed.get(label, {}).get("recall", 0.0)), "strict_recall"
    if objective_focus == "all_entity_recall" and label in ENTITY_LABELS:
        observed = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
        return _fbeta(observed.get(label, {}), 2.0), "candidate_f2"
    if objective_focus in {
        "configuration_affects",
        "configuration_affects_recall",
    } and label == "Configuration":
        if phase == "entity":
            observed = metrics.get(
                "candidate_entity_by_type", metrics.get("entity_by_type", {})
            )
            if objective_focus == "configuration_affects_recall":
                return _fbeta(observed.get(label, {}), 2.0), "candidate_f2"
            return float(observed.get(label, {}).get("recall", 0.0)), "recall"
        observed = metrics.get("entity_by_type", {})
        return float(observed.get(label, {}).get("f1", 0.0)), "final_f1"
    if objective_focus == "configuration_affects_recall" and label == "affects":
        observed = metrics.get("relation_by_type", {}).get(label, {})
        if phase == "relation":
            return _fbeta(observed, 2.0), "f2"
        return float(observed.get("f1", 0.0)), "f1"
    if objective_focus == "exploited_by_precision" and label == "exploited_by":
        observed = metrics.get("relation_by_type", {}).get(label, {})
        return _fbeta(observed, 0.5), "f0.5"
    return _label_f1(metrics, label), "f1"


def _attach_target_performance(
    record: dict,
    reference: dict,
    field: str = "target_performance",
    objective_focus: str | None = None,
    evaluation_phase: str | None = None,
) -> None:
    """Record whether an atomic edit improved the labels it claims to target."""
    labels = list(record.get("target_labels") or [])
    if not labels:
        record.pop(field, None)
        return
    per_label = {}
    phase = str(evaluation_phase or record.get("phase") or "joint")
    phase_round = None
    if evaluation_phase is None and record.get("configuration_recall_round") is not None:
        phase_round = int(record["configuration_recall_round"])
    for label in labels:
        candidate_value, metric_name = _target_value(
            record["metrics"], label, objective_focus, phase, phase_round
        )
        reference_value, _ = _target_value(
            reference["metrics"], label, objective_focus, phase, phase_round
        )
        per_label[label] = {
            "metric": metric_name,
            "candidate_value": candidate_value,
            "reference_value": reference_value,
            # Backward-compatible names remain for existing audit readers.
            "candidate_f1": candidate_value,
            "reference_f1": reference_value,
            "gain": round(candidate_value - reference_value, 6),
        }
    record[field] = {
        "reference_id": reference["id"],
        "per_label": per_label,
        "mean_gain": round(
            statistics.mean(item["gain"] for item in per_label.values()),
            6,
        ),
        "minimum_gain": min(item["gain"] for item in per_label.values()),
    }


def _attach_paired_performance(
    record: dict,
    reference: dict,
    objective_focus: str | None = None,
    evaluation_phase: str | None = None,
) -> None:
    """Compare matching repeat indices instead of unrelated aggregate runs."""
    candidate_runs = list(record["metrics"].get("repeat_metrics") or [])
    reference_runs = list(reference["metrics"].get("repeat_metrics") or [])
    if not candidate_runs:
        candidate_runs = [record["metrics"]]
    if not reference_runs:
        reference_runs = [reference["metrics"]]
    repeat_count = min(len(candidate_runs), len(reference_runs))
    labels = list(record.get("target_labels") or [])
    target = {}
    phase = str(evaluation_phase or record.get("phase") or "joint")
    phase_round = None
    if evaluation_phase is None and record.get("configuration_recall_round") is not None:
        phase_round = int(record["configuration_recall_round"])
    for label in labels:
        gains = []
        metric_name = None
        for index in range(repeat_count):
            candidate_value, metric_name = _target_value(
                candidate_runs[index], label, objective_focus, phase, phase_round
            )
            reference_value, _ = _target_value(
                reference_runs[index], label, objective_focus, phase, phase_round
            )
            gains.append(round(candidate_value - reference_value, 6))
        target[label] = {
            "metric": metric_name,
            "per_repeat_gain": gains,
            "mean_gain": round(statistics.fmean(gains), 6) if gains else 0.0,
            "positive_repeats": sum(gain > 0.0 for gain in gains),
        }
    quality_fields = ("overall_consistency", "evidence_validity")
    quality = {}
    for quality_field in quality_fields:
        gains = [
            round(
                float(candidate_runs[index]["raw_quality"][quality_field])
                - float(reference_runs[index]["raw_quality"][quality_field]),
                6,
            )
            for index in range(repeat_count)
        ]
        quality[quality_field] = {
            "per_repeat_gain": gains,
            "mean_gain": round(statistics.fmean(gains), 6) if gains else 0.0,
            "large_drop_repeats": sum(
                gain < -RAW_QUALITY_REPEAT_LARGE_DROP - 1e-12
                for gain in gains
            ),
        }
    record["paired_performance"] = {
        "reference_id": reference["id"],
        "repeat_count": repeat_count,
        "targets": target,
        "raw_quality": quality,
    }


def _passes_target_f1_gate(record: dict, guardrails: dict) -> bool:
    """Reject an APO edit that wins by improving only non-target labels."""
    labels = list(record.get("target_labels") or [])
    if not labels:
        return True
    target = record.get("target_performance")
    if not isinstance(target, dict):
        return False
    required = float(
        guardrails.get("target_f1_min_gain", TARGET_F1_MIN_GAIN)
    )
    def required_gain(label: str) -> float:
        if guardrails.get("objective_focus") == "all_entity_recall":
            # Each atomic entity edit may tie its own P0 F2.  The overall
            # candidate is still selected by macro F2 and must preserve every
            # type's P0 recall through a separate hard guardrail.
            return 0.0
        if (
            guardrails.get("objective_focus") in {
                "configuration_affects",
                "configuration_affects_recall",
            }
            and label == "Configuration"
        ):
            return 0.0
        return required

    global_pass = all(
        float(item.get("gain", float("-inf"))) + 1e-12
        >= required_gain(label)
        for label, item in target.get("per_label", {}).items()
    ) and len(target.get("per_label", {})) == len(labels)
    local = record.get("local_target_performance")
    if not isinstance(local, dict):
        local_pass = True
    else:
        local_items = local.get("per_label", {})
        local_pass = all(
            float(item.get("gain", float("-inf"))) + TARGET_PARENT_MAX_DROP
            + 1e-12 >= 0.0
            for item in local_items.values()
        ) and len(local_items) == len(labels)
    if not (global_pass and local_pass):
        return False
    paired = record.get("paired_performance", {}).get("targets", {})
    if not paired:
        return True
    repeat_count = int(record["paired_performance"].get("repeat_count", 0))
    required_positive = max(
        1, int((repeat_count * PAIRED_POSITIVE_REPEAT_RATIO) + 0.999999)
    )
    for label in labels:
        item = paired.get(label, {})
        gains = list(item.get("per_repeat_gain") or [])
        threshold = required_gain(label)
        if float(item.get("mean_gain", float("-inf"))) + 1e-12 < threshold:
            return False
        if threshold == 0.0:
            positive = sum(gain >= -1e-12 for gain in gains)
        else:
            positive = sum(gain > 0.0 for gain in gains)
        if positive < required_positive:
            return False
    return True


def _passes_paired_quality_gate(record: dict) -> bool:
    paired = record.get("paired_performance", {}).get("raw_quality", {})
    if not paired:
        return True
    return all(
        float(item.get("mean_gain", float("-inf")))
        + RAW_QUALITY_MEAN_MAX_DROP + 1e-12 >= 0.0
        and int(item.get("large_drop_repeats", 0))
        <= RAW_QUALITY_MAX_LARGE_DROP_REPEATS
        for item in paired.values()
    )


def _rank_key(
    record: dict,
    phase: str,
    objective_focus: str | None = None,
    *,
    phase_round: int | None = None,
    target_first: bool = False,
) -> tuple:
    metrics = record["metrics"]
    prompt_length = len(record["stage1_guidance"]) + len(
        record["stage2_guidance"]
    )
    if objective_focus == "configuration_recall_only":
        strict = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        ).get("Configuration", {})
        overlap = metrics.get("candidate_overlap_entity_by_type", {}).get(
            "Configuration", {}
        )
        if phase_round == 1:
            base_key = (
                float(overlap.get("recall", 0.0)),
                float(strict.get("recall", 0.0)),
                float(overlap.get("tp", 0.0)),
                -prompt_length,
            )
        else:
            base_key = (
                float(strict.get("recall", 0.0)),
                float(overlap.get("recall", 0.0)),
                float(strict.get("tp", 0.0)),
                -prompt_length,
            )
        if target_first and record.get("target_performance"):
            target = record["target_performance"]
            return (
                float(target.get("minimum_gain", float("-inf"))),
                float(target.get("mean_gain", float("-inf"))),
                *base_key,
            )
        if target_first:
            return (0.0, 0.0, *base_key)
        return base_key
    if objective_focus == "configuration_cpe_then_relations":
        if phase == "entity":
            configuration_metric = metrics.get(
                "candidate_entity_by_type", metrics.get("entity_by_type", {})
            ).get("Configuration", {})
            secondary = float(configuration_metric.get("f1", 0.0))
            tertiary = float(configuration_metric.get("recall", 0.0))
        else:
            secondary = float(metrics["relation_macro_f1"])
            tertiary = _configuration_candidate_na(metrics)
    elif objective_focus in {
        "configuration_affects",
        "configuration_affects_recall",
    }:
        if phase == "entity":
            configuration_metric = metrics.get(
                "candidate_entity_by_type", metrics.get("entity_by_type", {})
            ).get("Configuration", {})
            secondary = (
                _fbeta(configuration_metric, 2.0)
                if objective_focus == "configuration_affects_recall"
                else configuration_metric.get("recall", 0.0)
            )
        else:
            secondary = metrics.get("entity_by_type", {}).get(
                "Configuration", {}
            ).get("f1", 0.0)
        affects_metric = metrics["relation_by_type"].get("affects", {})
        tertiary = (
            _fbeta(affects_metric, 2.0)
            if objective_focus == "configuration_affects_recall"
            else affects_metric.get("f1", 0.0)
        )
    elif objective_focus == "all_entity_recall":
        candidate_entity_by_type = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        )
        secondary = float(
            metrics.get(
                "candidate_entity_macro_recall",
                _macro_recall(candidate_entity_by_type, ENTITY_LABELS),
            )
        )
        tertiary = float(
            metrics.get(
                "candidate_entity_macro_f1",
                _macro_f1(candidate_entity_by_type, ENTITY_LABELS),
            )
        )
    elif objective_focus in {"exploited_by", "exploited_by_precision"}:
        exploited_metric = metrics["relation_by_type"].get(
            "exploited_by", {}
        )
        secondary = (
            _fbeta(exploited_metric, 0.5)
            if objective_focus == "exploited_by_precision"
            else exploited_metric.get("f1", 0.0)
        )
        tertiary = metrics["entity_macro_f1"]
    elif objective_focus == "weakness_instantiates":
        secondary = metrics.get(
            "candidate_entity_by_type", metrics.get("entity_by_type", {})
        ).get("Weakness", {}).get("f1", 0.0)
        tertiary = metrics["relation_by_type"].get(
            "instantiates", {}
        ).get("f1", 0.0)
    else:
        secondary = metrics["relation_macro_f1"]
        tertiary = metrics["exploited_by_f1"]
    if objective_focus == "weakness_instantiates":
        # Do not let stochastic, non-target relation scores break a tie in the
        # focused objective and displace a shorter Weakness candidate from the
        # complete-dev shortlist.
        base_key = (
            _phase_score(record, phase, objective_focus),
            secondary,
            tertiary,
            -prompt_length,
            metrics["relation_macro_f1"],
            metrics["exploited_by_f1"],
        )
    else:
        base_key = (
            _phase_score(record, phase, objective_focus),
            secondary,
            tertiary,
            metrics["relation_macro_f1"],
            metrics["exploited_by_f1"],
            -prompt_length,
        )
    if target_first and record.get("target_performance"):
        target = record["target_performance"]
        return (
            float(target.get("minimum_gain", float("-inf"))),
            float(target.get("mean_gain", float("-inf"))),
            *base_key,
        )
    if target_first:
        return (0.0, 0.0, *base_key)
    return base_key


def _type_f1_baselines(metrics: dict) -> dict:
    candidate_entity_by_type = metrics.get(
        "candidate_entity_by_type", metrics.get("entity_by_type", {})
    )
    return {
        "candidate_entity_by_type_f1": {
            label: float(
                candidate_entity_by_type.get(label, {}).get(
                    "f1", 0.0
                )
            )
            for label in ENTITY_LABELS
        },
        "entity_by_type_f1": {
            label: float(metrics["entity_by_type"].get(label, {}).get("f1", 0.0))
            for label in ENTITY_LABELS
        },
        "relation_by_type_f1": {
            label: float(metrics["relation_by_type"].get(label, {}).get("f1", 0.0))
            for label in RELATION_LABELS
        },
    }


def _passes_type_f1_guardrails(
    metrics: dict,
    guardrails: dict,
    phase: str,
) -> bool:
    """Prevent aggregate gains obtained by sacrificing an extraction label."""
    if guardrails.get("objective_focus") in {
        "all_entity_recall",
        "configuration_recall_only",
    }:
        # This diagnostic deliberately permits precision/F1 trade-offs.  It
        # uses candidate macro F2 plus per-type recall floors instead.
        return True
    tolerance = float(guardrails.get("type_f1_max_drop", 1.0))
    groups = ["candidate_entity_by_type_f1"]
    if phase != "entity":
        groups.extend(("entity_by_type_f1", "relation_by_type_f1"))
    for group in groups:
        metric_key = group.removesuffix("_f1")
        if metric_key == "candidate_entity_by_type":
            observed = metrics.get(
                metric_key, metrics.get("entity_by_type", {})
            )
        else:
            observed = metrics.get(metric_key, {})
        for label, baseline in guardrails.get(group, {}).items():
            candidate_f1 = float(observed.get(label, {}).get("f1", 0.0))
            if candidate_f1 + tolerance + 1e-12 < float(baseline):
                return False
    return True


def _passes_configuration_recall_guardrail(
    metrics: dict,
    guardrails: dict,
) -> bool:
    minimum = guardrails.get("configuration_candidate_recall_min")
    if minimum is None:
        return True
    observed = metrics.get(
        "candidate_entity_by_type", metrics.get("entity_by_type", {})
    ).get("Configuration", {}).get("recall", 0.0)
    return float(observed) + 1e-12 >= float(minimum)


def _passes_all_entity_recall_guardrails(
    metrics: dict,
    guardrails: dict,
) -> bool:
    """Require every Stage-1 entity type to preserve its paired P0 recall."""
    minima = guardrails.get("candidate_entity_recall_min")
    if not minima:
        return True
    observed = metrics.get(
        "candidate_entity_by_type", metrics.get("entity_by_type", {})
    )
    return all(
        float(observed.get(label, {}).get("recall", 0.0)) + 1e-12
        >= float(minimum)
        for label, minimum in minima.items()
    )


def _passes_focused_precision_recall_guardrails(
    metrics: dict,
    guardrails: dict,
) -> bool:
    relation_by_type = metrics.get("relation_by_type", {})
    affects_precision_min = guardrails.get("affects_precision_min")
    if affects_precision_min is not None:
        observed = float(
            relation_by_type.get("affects", {}).get("precision", 0.0)
        )
        if observed + 1e-12 < float(affects_precision_min):
            return False
    exploited_by_recall_min = guardrails.get("exploited_by_recall_min")
    if exploited_by_recall_min is not None:
        observed = float(
            relation_by_type.get("exploited_by", {}).get("recall", 0.0)
        )
        if observed + 1e-12 < float(exploited_by_recall_min):
            return False
    return True


def _aggregate_relation_guardrail_baselines(metrics: dict) -> dict:
    """Build optional micro-F1 floors from the paired P0 metrics."""
    floors = {}
    if RELATION_F1_MAX_DROP is not None:
        floors["relation_f1_min"] = max(
            0.0,
            float(metrics.get("relation", {}).get("f1", 0.0))
            - RELATION_F1_MAX_DROP,
        )
    if NORMALIZED_RELATION_F1_MAX_DROP is not None:
        floors["normalized_relation_f1_min"] = max(
            0.0,
            float(metrics.get("normalized_relation", {}).get("f1", 0.0))
            - NORMALIZED_RELATION_F1_MAX_DROP,
        )
    return floors


def _passes_aggregate_relation_guardrails(
    metrics: dict,
    guardrails: dict,
) -> bool:
    checks = (
        ("relation", "relation_f1_min"),
        ("normalized_relation", "normalized_relation_f1_min"),
    )
    for metric_key, floor_key in checks:
        minimum = guardrails.get(floor_key)
        if minimum is None:
            continue
        observed = float(metrics.get(metric_key, {}).get("f1", 0.0))
        if observed + 1e-12 < float(minimum):
            return False
    return True


def _feasible(
    metrics: dict,
    guardrails: dict,
    phase: str = "joint",
    *,
    enforce_type_f1: bool = True,
) -> bool:
    if int(metrics.get("failed_stage_calls", 0)) > 0:
        return False
    quality = metrics["raw_quality"]
    mean_tolerance = float(
        guardrails.get("raw_quality_mean_max_drop", RAW_QUALITY_MEAN_MAX_DROP)
    )
    mean_quality_pass = bool(guardrails.get("skip_absolute_raw_quality")) or (
        quality["overall_consistency"] + mean_tolerance + 1e-12
        >= guardrails["overall_consistency"]
        and quality["evidence_validity"] + mean_tolerance + 1e-12
        >= guardrails["evidence_validity"]
    )
    large_drop_threshold = float(
        guardrails.get(
            "raw_quality_repeat_large_drop",
            RAW_QUALITY_REPEAT_LARGE_DROP,
        )
    )
    maximum_large_drop_repeats = int(
        guardrails.get(
            "raw_quality_max_large_drop_repeats",
            RAW_QUALITY_MAX_LARGE_DROP_REPEATS,
        )
    )
    repeated_quality_pass = True
    runs = list(metrics.get("repeat_metrics") or [])
    if len(runs) > 1:
        for field in ("overall_consistency", "evidence_validity"):
            baseline = float(guardrails[field])
            large_drops = sum(
                baseline
                - float(run.get("raw_quality", {}).get(field, 0.0))
                > large_drop_threshold + 1e-12
                for run in runs
            )
            if large_drops > maximum_large_drop_repeats:
                repeated_quality_pass = False
                break
    return (
        mean_quality_pass
        and repeated_quality_pass
        and (
            not enforce_type_f1
            or _passes_type_f1_guardrails(metrics, guardrails, phase)
        )
        and _passes_configuration_recall_guardrail(metrics, guardrails)
        and _passes_all_entity_recall_guardrails(metrics, guardrails)
        and _passes_focused_precision_recall_guardrails(metrics, guardrails)
        and _passes_aggregate_relation_guardrails(metrics, guardrails)
    )


def _final_output_feasible(metrics: dict, raw_guardrails: dict) -> bool:
    """Require stable raw compliance and perfect delivered evidence."""
    if not _feasible(metrics, raw_guardrails):
        return False
    quality = metrics.get("final_quality") or {}
    return (
        quality.get("overall_consistency", 0.0) + 1e-12 >= 1.0
        and quality.get("evidence_validity", 0.0) + 1e-12 >= 1.0
    )


def _delivered_output_feasible(metrics: dict) -> bool:
    """Hard gate for the postprocessed artifact, independent of raw noise."""
    quality = metrics.get("final_quality") or {}
    return (
        int(metrics.get("failed_stage_calls", 0)) == 0
        and quality.get("overall_consistency", 0.0) + 1e-12 >= 1.0
        and quality.get("evidence_validity", 0.0) + 1e-12 >= 1.0
    )


def _configuration_overlap_recall(metrics: dict) -> float:
    return float(
        metrics.get("candidate_overlap_entity_by_type", {})
        .get("Configuration", {})
        .get("recall", 0.0)
    )


def _passes_configuration_overlap_parent_guardrail(record: dict) -> bool:
    """Round 2 may repair boundaries but must preserve round-1 discovery."""
    floor = record.get("configuration_overlap_recall_floor")
    if floor is None:
        return True
    observed = record.get("configuration_overlap_recall_observed")
    if observed is None:
        observed = _configuration_overlap_recall(record.get("metrics", {}))
    return float(observed) + 1e-12 >= float(floor)


def _eligible(record: dict, guardrails: dict) -> bool:
    """P0始终作为安全回退；自动候选必须满足全部硬约束。"""
    return record.get("id") == "p0_manual" or (
        _feasible(record["metrics"], guardrails)
        and _passes_target_f1_gate(record, guardrails)
        and _passes_paired_quality_gate(record)
        and _passes_configuration_overlap_parent_guardrail(record)
    )


def _phase_rejection_reasons(
    record: dict,
    guardrails: dict,
    phase: str,
) -> list[str]:
    """Return structural reasons that prevent candidate from entering search beam.

    Search phase allows exploration based on composite objective score.
    Only hard structural failures (API crashes, invalid output format, broken schema)
    are filtered out at this stage. Strict multi-repeat target gains and per-type safety gates
    are evaluated during complete development set verification.
    """
    if record.get("id") == "p0_manual":
        return []
    reasons = []
    metrics = record.get("metrics", {})
    if int(metrics.get("failed_stage_calls", 0)) > 0:
        reasons.append("failed_stage_calls")
    if metrics.get("final_quality") and not _delivered_output_feasible(metrics):
        reasons.append("delivered_output_quality_gate")
    quality = metrics.get("raw_quality", {})
    entity_total = max(int(quality.get("entity_total", 0)), 1)
    entity_validity = float(quality.get("entity_valid", 0)) / entity_total
    if entity_validity + 1e-12 < float(guardrails.get("entity_validity", 1.0)):
        reasons.append("entity_schema_validity_gate")
    if not _passes_configuration_overlap_parent_guardrail(record):
        reasons.append("configuration_overlap_recall_parent_gate")
    return reasons


def _eligible_for_phase(record: dict, guardrails: dict, phase: str) -> bool:
    """Check structural eligibility for search phase exploration."""
    return not _phase_rejection_reasons(record, guardrails, phase)


def _eligible_for_full_dev_promotion(
    record: dict,
    guardrails: dict,
) -> bool:
    """Promote structurally valid candidates to complete dev evaluation.

    The hard statistical verification (target gain, composite gain, quality gates)
    is performed on the complete 21-document development set with 3 paired repeats.
    """
    if record.get("id") == "p0_manual":
        return True
    metrics = record.get("metrics", {})
    if int(metrics.get("failed_stage_calls", 0)) > 0:
        return False
    if metrics.get("final_quality") and not _delivered_output_feasible(metrics):
        return False
    quality = metrics.get("raw_quality", {})
    entity_total = max(int(quality.get("entity_total", 0)), 1)
    entity_validity = float(quality.get("entity_valid", 0)) / entity_total
    if entity_validity + 1e-12 < float(guardrails.get("entity_validity", 1.0)):
        return False
    return (
        _passes_all_entity_recall_guardrails(metrics, guardrails)
        and _passes_configuration_overlap_parent_guardrail(record)
    )


def _build_phase_pool(
    initial: dict,
    beam: list[dict],
    generated: list[dict],
    guardrails: dict,
    phase: str,
) -> dict[str, dict]:
    """Construct a phase pool while preserving P0 as a zero-cost fallback."""
    return {
        record["id"]: record
        for record in [initial, *beam, *generated]
        if _eligible_for_phase(record, guardrails, phase)
    }


def _joint_recombination_pairs(
    entity_records: list[dict],
    relation_records: list[dict],
    limit: int,
    target_labels: tuple[str, ...],
    id_factory: Any = None,
) -> list[dict]:
    """Create auditable cross-component seeds for the joint phase."""
    combined = []
    seen = set()
    for entity_record in entity_records:
        stage1 = str(entity_record.get("stage1_guidance", "")).strip()
        if not stage1:
            continue
        for relation_record in relation_records:
            stage2 = str(relation_record.get("stage2_guidance", "")).strip()
            if not stage2:
                continue
            cid = id_factory(stage1, stage2) if id_factory else None
            candidate = _pair(
                stage1,
                stage2,
                id=cid,
                parent_id=relation_record.get("id"),
                phase="joint",
                rationale="Deterministic entity/relation beam recombination.",
            )
            if candidate["id"] in seen or candidate["id"] in {
                entity_record.get("id"), relation_record.get("id")
            }:
                continue
            seen.add(candidate["id"])
            candidate["component_parent_ids"] = [
                entity_record.get("id"),
                relation_record.get("id"),
            ]
            candidate["edit_target_labels"] = list(target_labels)
            candidate["target_labels"] = list(target_labels)
            combined.append(candidate)
            if len(combined) >= limit:
                return combined
    return combined


def _load_split() -> tuple[list[Path], list[Path], dict]:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("固定数据划分与当前 schema_version 不一致")
    train_ids = split.get("train", [])
    dev_ids = split.get("dev", [])
    test_ids = split.get("test", [])
    if set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids) or set(
        dev_ids
    ) & set(test_ids):
        raise ValueError("train/dev/test 存在文档重叠")
    train_paths = [GOLD_DIR / f"{doc_id}.json" for doc_id in train_ids]
    dev_paths = [GOLD_DIR / f"{doc_id}.json" for doc_id in dev_ids]
    missing = [str(path) for path in train_paths + dev_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"训练或开发标注缺失：{missing}")
    return train_paths, dev_paths, split


def optimize(
    *,
    candidate_count: int = 4,
    beam_size: int = 2,
    train_batch_size: int = 8,
    selection_dev_size: int = 8,
    train_doc_ids: list[str] | None = None,
    selection_dev_doc_ids: list[str] | None = None,
    structural_patience: int = 2,
    dev_repeats: int = 1,
    final_dev_repeats: int = 2,
    final_report_repeats: int | None = None,
    finalist_count: int = 2,
    full_dev_recheck: bool = True,
    component_ablation: bool = True,
    seed: int = 20260730,
    phase_rounds: dict[str, int] | None = None,
    objective_focus: str | None = None,
    run_kind: str = "formal",
    freeze_artifact: bool = True,
    initial_prompt_artifact: Path | None = None,
) -> dict:
    """运行 APO 并写入完整审计轨迹；本函数从不读取测试 Gold。"""
    if candidate_count < 1 or beam_size < 1:
        raise ValueError("candidate_count 和 beam_size 必须大于 0")
    if selection_dev_size < 1 or finalist_count < 1:
        raise ValueError("selection_dev_size 和 finalist_count 必须大于 0")
    if structural_patience < 1:
        raise ValueError("structural_patience 必须大于 0")
    final_report_repeats = final_report_repeats or final_dev_repeats
    if (
        dev_repeats < 1
        or final_dev_repeats < 1
        or final_report_repeats < 1
    ):
        raise ValueError("开发集重复次数必须大于 0")
    if component_ablation and not full_dev_recheck:
        raise ValueError("组件消融要求启用完整开发集复核")
    if objective_focus not in {
        None,
        "all_entity_recall",
        "configuration_recall_only",
        "configuration_cpe_then_relations",
        "configuration_affects",
        "configuration_affects_recall",
        "weakness_instantiates",
        "exploited_by",
        "exploited_by_precision",
    }:
        raise ValueError(f"未知 APO 专项目标：{objective_focus}")
    phase_rounds = dict(phase_rounds or DEFAULT_PHASE_ROUNDS)
    train_paths, dev_paths, split = _load_split()
    train_by_id = {path.stem: path for path in train_paths}
    dev_by_id = {path.stem: path for path in dev_paths}
    if train_doc_ids:
        missing_train = sorted(set(train_doc_ids) - set(train_by_id))
        if missing_train:
            raise ValueError(f"pilot训练文档不在train划分：{missing_train}")
        optimization_train_paths = [train_by_id[item] for item in train_doc_ids]
    else:
        optimization_train_paths = train_paths
    if selection_dev_doc_ids:
        missing_dev = sorted(set(selection_dev_doc_ids) - set(dev_by_id))
        if missing_dev:
            raise ValueError(f"pilot开发文档不在dev划分：{missing_dev}")
        selection_dev_paths = [dev_by_id[item] for item in selection_dev_doc_ids]
    else:
        selection_target_labels = (
            _atomic_target_for_round("joint", 1, objective_focus)
            if objective_focus is not None
            else ()
        )
        selection_dev_paths = select_target_stratified_evaluation_batch(
            dev_paths,
            batch_size=min(selection_dev_size, len(dev_paths)),
            seed=seed + 100_000,
            target_labels=selection_target_labels,
        )
    if run_kind not in APO_PRESETS:
        raise ValueError(f"未知 APO run_kind：{run_kind}")
    run_name = datetime.now(timezone.utc).strftime(
        f"{run_kind}_%Y%m%dT%H%M%SZ"
    )
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    selection_cache_namespace = f"{run_name}_selection"
    full_dev_cache_namespace = f"{run_name}_full_finalists"
    final_report_cache_namespace = f"{run_name}_final_report"
    training_cache_namespace = f"{run_name}_training_feedback"

    manifest = {
        "algorithm": "ProTeGi-inspired textual-gradient beam-search APO",
        "run_kind": run_kind,
        "freeze_artifact": freeze_artifact,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "split_file": str(SPLIT_FILE),
        "split_sha256": hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
        "feedback_split": "train",
        "selection_split": "dev",
        "train_documents": len(train_paths),
        "optimization_train_pool_documents": len(optimization_train_paths),
        "optimization_train_pool_ids": [
            path.stem for path in optimization_train_paths
        ],
        "dev_documents": len(dev_paths),
        "selection_dev_documents": len(selection_dev_paths),
        "selection_dev_ids": [path.stem for path in selection_dev_paths],
        "selection_dev_strategy": (
            "deterministic target-rich stratified coverage (up to 75% target "
            "documents) plus structural negatives; generic runs retain broad "
            "entity/relation, repeated-mention, and multi-entity coverage"
        ),
        "training_batch_strategy": (
            "program-target-aware rotating positives (up to 75%) plus "
            "structurally diverse negatives; target labels are fixed before "
            "training feedback is generated"
        ),
        "full_dev_recheck": full_dev_recheck,
        "train_gold_aggregate_sha256": _aggregate_file_hash(train_paths),
        "dev_gold_aggregate_sha256": _aggregate_file_hash(dev_paths),
        "held_out_test_documents": len(split.get("test", [])),
        "test_gold_loaded": False,
        "test_predictions_generated": False,
        "task_runtime_config": runtime_config(),
        "optimizer_model": OPTIMIZER_MODEL,
        "optimizer_thinking": OPTIMIZER_THINKING,
        "optimizer_reasoning_effort": OPTIMIZER_REASONING_EFFORT,
        "critic_fallback_thinking": CRITIC_FALLBACK_THINKING,
        "critic_fallback_reasoning_effort": CRITIC_FALLBACK_REASONING_EFFORT,
        "optimizer_max_tokens": OPTIMIZER_MAX_TOKENS,
        "optimizer_endpoint": _model_endpoint(OPTIMIZER_MODEL),
        "optimizer_top_p": OPTIMIZER_TOP_P,
        "optimizer_transport_max_retries": 0,
        "optimizer_semantic_retry_attempts": 3,
        "optimizer_max_workers": 1,
        "critic_temperature": CRITIC_TEMPERATURE,
        "editor_model": EDITOR_MODEL,
        "editor_thinking": EDITOR_THINKING,
        "editor_reasoning_effort": EDITOR_REASONING_EFFORT,
        "editor_max_tokens": EDITOR_MAX_TOKENS,
        "editor_endpoint": _model_endpoint(EDITOR_MODEL),
        "editor_top_p": EDITOR_TOP_P,
        "editor_temperature": EDITOR_TEMPERATURE,
        "editor_transport_max_retries": 0,
        "editor_format_retry_attempts": 3,
        "editor_max_workers": 1,
        "base_prompt_sha256": BASE_PROMPT_SHA256,
        "initial_prompt_artifact": (
            str(initial_prompt_artifact)
            if initial_prompt_artifact is not None
            else None
        ),
        "initial_prompt_artifact_sha256": (
            hashlib.sha256(initial_prompt_artifact.read_bytes()).hexdigest()
            if initial_prompt_artifact is not None
            else None
        ),
        "candidate_count": candidate_count,
        "beam_size": beam_size,
        "train_batch_size": train_batch_size,
        "selection_dev_size": selection_dev_size,
        "structural_patience": structural_patience,
        "early_stopping_policy": (
            "fixed_round_budget; score-based early stopping disabled; stop a "
            "phase after structural_patience consecutive rounds without a feasible "
            "unique new candidate, or immediately when every optimizer call in a "
            "round fails after recovery"
        ),
        "score_early_stopping_enabled": False,
        "dev_repeats": dev_repeats,
        "final_dev_repeats": final_dev_repeats,
        "final_report_repeats": final_report_repeats,
        "finalist_count": finalist_count,
        "full_dev_recheck": full_dev_recheck,
        "component_ablation": component_ablation,
        "seed": seed,
        "phase_rounds": phase_rounds,
        "objective_focus": objective_focus,
        "atomic_target_policy": (
            "program-selected deterministic target per phase/round; critic and "
            "editor cannot choose or override target_labels"
        ),
        "type_f1_max_drop": TYPE_F1_MAX_DROP,
        "relation_f1_max_drop": RELATION_F1_MAX_DROP,
        "normalized_relation_f1_max_drop": NORMALIZED_RELATION_F1_MAX_DROP,
        "target_f1_min_gain": 0.0,
        "final_target_f1_min_gain": TARGET_F1_MIN_GAIN,
        "raw_quality_mean_max_drop": RAW_QUALITY_MEAN_MAX_DROP,
        "raw_quality_repeat_large_drop": RAW_QUALITY_REPEAT_LARGE_DROP,
        "raw_quality_max_large_drop_repeats": (
            RAW_QUALITY_MAX_LARGE_DROP_REPEATS
        ),
        "training_error_feedback": (
            "bounded concrete TRAIN excerpts plus exact gold/predicted spans "
            "and relations; dev and test examples excluded"
        ),
        "cache_namespaces": {
            "selection_prefix": selection_cache_namespace,
            "full_finalists_prefix": full_dev_cache_namespace,
            "final_report_prefix": final_report_cache_namespace,
            "training_feedback": training_cache_namespace,
            "policy": (
                "run-scoped; Stage-1-only runs require a non-empty Stage-1 "
                "response; two-stage runs require non-empty Stage-1 and "
                "Stage-2 responses"
            ),
        },
        "objective": _objective_description(objective_focus),
        "guardrails": (
            "during beam search, repeated target gains and stage-actionable "
            "schema/evidence constraints determine exploratory parents; every "
            "entity and relation type F1 is checked on complete-dev finalists "
            "and may drop by at most "
            f"{TYPE_F1_MAX_DROP:.2f}; editable guidance may append or replace "
            "earlier editable rules; when configured, strict and normalized "
            "relation micro-F1 must also stay within their registered P0 "
            "non-degradation margins; "
            "candidates cannot edit the frozen base and "
            "cannot enter exploratory search unless every declared target "
            "label has a positive paired point estimate; complete-dev "
            "finalists must then improve by at least "
            f"{TARGET_F1_MIN_GAIN:.3f} against the paired initial prompt, "
            "except Configuration candidate recall, which may not decrease; "
            "at least two of three full-dev repeats must have the required "
            "sign; "
            "final delivered predictions must reach schema consistency and "
            "evidence validity = 1.0 before the prompt can be frozen for "
            "test; raw-model compliance is compared to P0 by matching repeat "
            "index, its paired mean may drop by at most 0.01 and no more than "
            "one repeat may drop by over 0.03; the initial prompt "
            "remains a fallback; failed calls remain as empty predictions and "
            "are never skipped"
        ),
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    initial = _initial_pair_from_artifact(initial_prompt_artifact)
    if objective_focus in {
        "all_entity_recall",
        "configuration_recall_only",
    }:
        # This preset evaluates a runtime copy of P0 as a pure Stage-1 system.
        # It does not modify the canonical frozen prompt or apply downstream
        # relation-closure pruning to candidate entities.
        initial["stage1_only"] = True
    if objective_focus == "configuration_recall_only":
        # Empty P0 guidance still resolves to the byte-identical frozen prompt;
        # non-empty children receive a runtime copy with the conflicting P0
        # Configuration clauses removed and replaced before their hypothesis.
        initial["configuration_policy_override"] = True
    initial["metrics"] = _evaluate_repeated_with_failure_recovery(
        initial,
        selection_dev_paths,
        run_dir / "selection_dev_predictions" / initial["id"],
        dev_repeats,
        cache_namespace_prefix=selection_cache_namespace,
    )
    if int(initial["metrics"].get("failed_stage_calls", 0)) > 0:
        _write_json(
            run_dir / "baseline_failure.json",
            {
                "reason": (
                    "P0 baseline contains failed extraction stages and cannot "
                    "be used to define APO guardrails"
                ),
                "failed_stage_calls": initial["metrics"]["failed_stage_calls"],
                "metrics": initial["metrics"],
            },
        )
        raise RuntimeError(
            "P0 开发集基线包含失败调用；已拒绝以不完整预测设定 APO "
            "guardrail，请在接口稳定后重新运行。"
        )
    initial_raw_quality = dict(initial["metrics"]["raw_quality"])
    guardrails = {
        "entity_validity": (
            initial["metrics"]["raw_quality"]["entity_valid"]
            / max(initial["metrics"]["raw_quality"]["entity_total"], 1)
        ),
        "overall_consistency": initial["metrics"]["raw_quality"][
            "overall_consistency"
        ],
        "evidence_validity": initial["metrics"]["raw_quality"][
            "evidence_validity"
        ],
        "type_f1_max_drop": TYPE_F1_MAX_DROP,
        # A one-repeat target result is a screen, not a final estimate.  Use a
        # zero mean threshold here (the paired sign check still requires a
        # positive relation gain) and restore 0.005 on complete dev.
        "target_f1_min_gain": 0.0,
        "objective_focus": objective_focus,
        "skip_absolute_raw_quality": True,
        "configuration_candidate_recall_min": (
            initial["metrics"]
            .get("candidate_entity_by_type", {})
            .get("Configuration", {})
            .get("recall", 0.0)
            if objective_focus in {
                "configuration_recall_only",
                "configuration_cpe_then_relations",
                "configuration_affects",
                "configuration_affects_recall",
            }
            else None
        ),
        "candidate_entity_recall_min": (
            {
                label: float(
                    initial["metrics"]
                    .get("candidate_entity_by_type", {})
                    .get(label, {})
                    .get("recall", 0.0)
                )
                for label in ENTITY_LABELS
            }
            if objective_focus == "all_entity_recall"
            else None
        ),
        "affects_precision_min": (
            max(
                0.0,
                float(
                    initial["metrics"]
                    .get("relation_by_type", {})
                    .get("affects", {})
                    .get("precision", 0.0)
                ) - 0.03,
            )
            if objective_focus == "configuration_affects_recall"
            else None
        ),
        "exploited_by_recall_min": (
            max(
                0.0,
                float(
                    initial["metrics"]
                    .get("relation_by_type", {})
                    .get("exploited_by", {})
                    .get("recall", 0.0)
                ) - 0.03,
            )
            if objective_focus == "exploited_by_precision"
            else None
        ),
        **_aggregate_relation_guardrail_baselines(initial["metrics"]),
        **_type_f1_baselines(initial["metrics"]),
    }
    _write_json(run_dir / "prompts" / f"{initial['id']}.json", initial)

    critic = make_apo_extractor(
        temperature=CRITIC_TEMPERATURE,
        model=OPTIMIZER_MODEL,
        thinking=OPTIMIZER_THINKING,
        reasoning_effort=OPTIMIZER_REASONING_EFFORT,
        max_tokens=OPTIMIZER_MAX_TOKENS,
        top_p=OPTIMIZER_TOP_P,
    )
    critic_fallback = make_apo_extractor(
        temperature=CRITIC_TEMPERATURE,
        model=OPTIMIZER_MODEL,
        thinking=CRITIC_FALLBACK_THINKING,
        reasoning_effort=CRITIC_FALLBACK_REASONING_EFFORT,
        max_tokens=OPTIMIZER_MAX_TOKENS,
        top_p=OPTIMIZER_TOP_P,
    )
    editor = make_apo_extractor(
        temperature=EDITOR_TEMPERATURE,
        model=EDITOR_MODEL,
        thinking=EDITOR_THINKING,
        reasoning_effort=EDITOR_REASONING_EFFORT,
        max_tokens=EDITOR_MAX_TOKENS,
        top_p=EDITOR_TOP_P,
    )
    all_evaluated = {initial["id"]: initial}
    candidate_counter = 0
    candidate_id_map: dict[str, str] = {
        _guidance_key(initial["stage1_guidance"], initial["stage2_guidance"]): initial["id"]
    }

    def _next_candidate_id(stage1: str, stage2: str) -> str:
        nonlocal candidate_counter
        key = _guidance_key(stage1, stage2)
        if key in candidate_id_map:
            return candidate_id_map[key]
        candidate_counter += 1
        cid = f"p{candidate_counter}"
        candidate_id_map[key] = cid
        return cid

    # Preserve the surviving parents across phase boundaries.  Resetting to a
    # single top-1 prompt made the configured beam width ineffective and
    # prevented relation/joint edits from building on entity candidates.
    beam = [initial]
    phase_beams = {}
    joint_recombination_ids = []
    round_logs = []
    global_round = 0
    optimizer_attempts = 0
    optimizer_input_chars = 0
    optimizer_output_chars = 0
    optimizer_failures = []
    training_task_attempts = 0
    training_task_seconds = 0.0

    for phase in ("entity", "relation", "joint"):
        rounds = int(phase_rounds.get(phase, 0))
        if rounds <= 0:
            continue
        if phase == "joint":
            joint_targets = _atomic_target_for_round(
                "joint", 1, objective_focus
            )
            recombined = _joint_recombination_pairs(
                phase_beams.get("entity", []),
                phase_beams.get("relation", beam),
                beam_size,
                joint_targets,
                id_factory=_next_candidate_id,
            )
            for candidate in recombined:
                if candidate["id"] in all_evaluated:
                    continue
                candidate["metrics"] = _evaluate_repeated_with_failure_recovery(
                    candidate,
                    selection_dev_paths,
                    run_dir / "selection_dev_predictions" / candidate["id"],
                    dev_repeats,
                    cache_namespace_prefix=selection_cache_namespace,
                )
                _attach_target_performance(
                    candidate, initial, objective_focus=objective_focus
                )
                _attach_paired_performance(
                    candidate, initial, objective_focus
                )
                candidate["feasible"] = _eligible_for_phase(
                    candidate, guardrails, "joint"
                )
                candidate["phase_rejection_reasons"] = (
                    _phase_rejection_reasons(
                        candidate, guardrails, "joint"
                    )
                )
                all_evaluated[candidate["id"]] = candidate
                joint_recombination_ids.append(candidate["id"])
                _write_json(
                    run_dir / "prompts" / f"{candidate['id']}.json",
                    candidate,
                )
            joint_seed_pool = _build_phase_pool(
                initial,
                beam,
                [
                    all_evaluated[candidate_id]
                    for candidate_id in joint_recombination_ids
                ],
                guardrails,
                "joint",
            )
            beam = sorted(
                joint_seed_pool.values(),
                key=lambda record: _rank_key(
                    record, "joint", objective_focus
                ),
                reverse=True,
            )[:beam_size]
        structural_stale_rounds = 0
        for local_round in range(1, rounds + 1):
            round_start_best = max(
                _phase_score(
                    record,
                    phase,
                    objective_focus,
                    phase_round=local_round,
                )
                for record in beam
            )
            global_round += 1
            atomic_target_labels = _atomic_target_for_round(
                phase, local_round, objective_focus
            )
            batch_paths = select_training_batch_for_target(
                optimization_train_paths,
                global_round,
                train_batch_size,
                seed,
                atomic_target_labels,
            )
            generated = []
            parent_audits = []
            for parent in beam:
                train_pred_dir = (
                    run_dir
                    / "train_feedback"
                    / f"round_{global_round:02d}_{phase}"
                    / parent["id"]
                )
                train_metrics = _evaluate_once_with_failure_recovery(
                    parent,
                    batch_paths,
                    train_pred_dir,
                    cache_namespace=training_cache_namespace,
                )
                training_task_attempts += train_metrics["api_call_attempts"]
                training_task_seconds += train_metrics["elapsed_seconds"]
                errors = build_error_summary(
                    batch_paths,
                    train_pred_dir,
                    example_limit=(
                        CONFIGURATION_RECALL_ERROR_EXAMPLES
                        if objective_focus == "configuration_recall_only"
                        else ERROR_EXAMPLES_PER_LABEL_AND_KIND
                    ),
                )
                parent_audit = {
                    "parent_id": parent["id"],
                    "training_batch_ids": [path.stem for path in batch_paths],
                    "training_metrics": train_metrics,
                    "error_summary": errors,
                }
                try:
                    gradient_record = generate_textual_gradient(
                        critic,
                        parent,
                        errors,
                        phase,
                        atomic_target_labels,
                        critic_fallback,
                        objective_focus=objective_focus,
                        phase_round=local_round,
                    )
                    optimizer_attempts += int(
                        gradient_record["trace"].get("attempts", 0)
                    )
                    optimizer_input_chars += gradient_record["input_chars"]
                    optimizer_output_chars += gradient_record["output_chars"]
                    candidate_record = generate_candidates(
                        editor,
                        parent,
                        gradient_record["textual_gradient"],
                        phase,
                        candidate_count,
                        global_round,
                        atomic_target_labels,
                        id_factory=_next_candidate_id,
                        objective_focus=objective_focus,
                        phase_round=local_round,
                    )
                    optimizer_attempts += int(
                        candidate_record["trace"].get("attempts", 0)
                    )
                    optimizer_input_chars += candidate_record["input_chars"]
                    optimizer_output_chars += candidate_record["output_chars"]
                except Exception as exc:
                    if _is_terminal_optimizer_error(exc):
                        raise
                    failure = {
                        "global_round": global_round,
                        "phase": phase,
                        "parent_id": parent["id"],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    optimizer_failures.append(failure)
                    parent_audit["optimizer_failure"] = failure
                    parent_audits.append(parent_audit)
                    continue
                parent_audits.append(
                    {
                        **parent_audit,
                        "textual_gradient": gradient_record,
                        "candidate_generation": {
                            key: value
                            for key, value in candidate_record.items()
                            if key != "candidates"
                        },
                        "candidate_ids": [
                            candidate["id"]
                            for candidate in candidate_record["candidates"]
                        ],
                    }
                )
                generated.extend(candidate_record["candidates"])

            unique_generated = {
                candidate["id"]: candidate for candidate in generated
            }
            for candidate in unique_generated.values():
                candidate["metrics"] = _evaluate_repeated_with_failure_recovery(
                    candidate,
                    selection_dev_paths,
                    run_dir / "selection_dev_predictions" / candidate["id"],
                    dev_repeats,
                    cache_namespace_prefix=selection_cache_namespace,
                )
                _attach_target_performance(
                    candidate, initial, objective_focus=objective_focus
                )
                _attach_paired_performance(
                    candidate, initial, objective_focus
                )
                parent_reference = next(
                    (
                        record
                        for record in beam
                        if record["id"] == candidate.get("parent_id")
                    ),
                    initial,
                )
                if (
                    objective_focus == "configuration_recall_only"
                    and local_round >= 2
                ):
                    candidate["configuration_overlap_recall_floor"] = (
                        _configuration_overlap_recall(parent_reference["metrics"])
                    )
                    candidate["configuration_overlap_recall_observed"] = (
                        _configuration_overlap_recall(candidate["metrics"])
                    )
                _attach_target_performance(
                    candidate,
                    parent_reference,
                    field="local_target_performance",
                    objective_focus=objective_focus,
                )
                candidate["feasible"] = _eligible_for_phase(
                    candidate,
                    guardrails,
                    phase,
                )
                candidate["phase_rejection_reasons"] = (
                    _phase_rejection_reasons(
                        candidate, guardrails, phase
                    )
                )
                all_evaluated[candidate["id"]] = candidate
                _write_json(
                    run_dir / "prompts" / f"{candidate['id']}.json",
                    candidate,
                )

            feasible_new_candidates = [
                candidate
                for candidate in unique_generated.values()
                if candidate.get("feasible") is True
            ]
            optimizer_failure_count = sum(
                1 for audit in parent_audits if "optimizer_failure" in audit
            )
            if feasible_new_candidates:
                structural_stale_rounds = 0
            else:
                structural_stale_rounds += 1

            structural_stop_reason = None
            if not unique_generated and optimizer_failure_count == len(beam):
                structural_stop_reason = "all_optimizer_calls_failed"
            elif structural_stale_rounds >= structural_patience:
                structural_stop_reason = (
                    "no_feasible_unique_new_candidate_for_"
                    f"{structural_patience}_consecutive_rounds"
                )

            pool = _build_phase_pool(
                initial,
                beam,
                list(unique_generated.values()),
                guardrails,
                phase,
            )
            if not pool:
                raise RuntimeError("本轮没有满足不可劣化 guardrail 的提示")
            ranked = sorted(
                pool.values(),
                key=lambda record: _rank_key(
                    record,
                    phase,
                    objective_focus,
                    phase_round=local_round,
                    target_first=False,
                ),
                reverse=True,
            )
            beam = ranked[:beam_size]
            new_best = _phase_score(
                beam[0],
                phase,
                objective_focus,
                phase_round=local_round,
            )
            improvement = new_best - round_start_best
            # The formal protocol uses a fixed round budget.  Do not stop on
            # small score changes measured on the 8-document search subset;
            # those changes are too noisy to define convergence.

            round_log = {
                "global_round": global_round,
                "phase": phase,
                "phase_round": local_round,
                "atomic_target_labels": list(atomic_target_labels),
                "parents": parent_audits,
                "candidate_metrics": {
                    candidate_id: record["metrics"]
                    for candidate_id, record in unique_generated.items()
                },
                "candidate_feasible": {
                    candidate_id: record.get("feasible", False)
                    for candidate_id, record in unique_generated.items()
                },
                "candidate_rejection_reasons": {
                    candidate_id: record.get(
                        "phase_rejection_reasons", []
                    )
                    for candidate_id, record in unique_generated.items()
                },
                "selected_beam": [record["id"] for record in beam],
                "phase_best_score": new_best,
                "improvement": round(improvement, 6),
                "structural_stale_rounds": structural_stale_rounds,
                "new_candidate_count": len(unique_generated),
                "feasible_new_candidate_count": len(feasible_new_candidates),
                "optimizer_failure_count": optimizer_failure_count,
            }
            if structural_stop_reason:
                round_log["stopping_reason"] = structural_stop_reason
            round_logs.append(round_log)
            _write_json(
                run_dir / f"round_{global_round:02d}_{phase}.json",
                round_log,
            )
            if structural_stop_reason:
                _write_json(
                    run_dir / f"round_{global_round:02d}_{phase}.json",
                    round_log,
                )
                break
        phase_beams[phase] = list(beam)
    # 搜索阶段只使用覆盖型开发子集。候选生成结束后，将子集排名靠前的候选
    # 连同人工初始提示放回完整开发集复核；最终冻结决策只依据完整开发集。
    feasible_so_far = [
        record
        for record in all_evaluated.values()
        if _eligible(record, guardrails)
    ]
    promotable_so_far = [
        record
        for record in all_evaluated.values()
        if _eligible_for_full_dev_promotion(record, guardrails)
    ]
    for record in all_evaluated.values():
        if record["id"] == initial["id"]:
            record["full_dev_promotion_rejection_reasons"] = []
            continue
        reasons = []
        if int(record.get("metrics", {}).get("failed_stage_calls", 0)) > 0:
            reasons.append("failed_stage_calls")
        if not _passes_target_f1_gate(record, guardrails):
            reasons.append("target_gain_gate")
        if not _passes_configuration_recall_guardrail(
            record.get("metrics", {}), guardrails
        ):
            reasons.append("configuration_candidate_recall_gate")
        if not _passes_all_entity_recall_guardrails(
            record.get("metrics", {}), guardrails
        ):
            reasons.append("all_entity_candidate_recall_gate")
        if not _passes_configuration_overlap_parent_guardrail(record):
            reasons.append("configuration_overlap_recall_parent_gate")
        if (
            record.get("metrics", {}).get("final_quality")
            and not _delivered_output_feasible(record["metrics"])
        ):
            reasons.append("delivered_output_quality_gate")
        record["full_dev_promotion_rejection_reasons"] = reasons
    ranked_selection = sorted(
        promotable_so_far,
        key=lambda record: _rank_key(record, "joint", objective_focus),
        reverse=True,
    )
    shortlist = [initial]
    shortlist.extend(
        record
        for record in ranked_selection
        if record["id"] != initial["id"]
    )
    shortlist = shortlist[: finalist_count + 1]

    full_dev_records = {}
    if full_dev_recheck:
        for record in shortlist:
            checked = dict(record)
            checked["selection_metrics"] = record["metrics"]
            checked["metrics"] = _evaluate_repeated_with_failure_recovery(
                record,
                dev_paths,
                run_dir / "full_dev_predictions" / record["id"],
                final_dev_repeats,
                cache_namespace_prefix=full_dev_cache_namespace,
            )
            full_dev_records[record["id"]] = checked

        full_initial = full_dev_records[initial["id"]]
        full_guardrails = {
            "overall_consistency": full_initial["metrics"]["raw_quality"][
                "overall_consistency"
            ],
            "evidence_validity": full_initial["metrics"]["raw_quality"][
                "evidence_validity"
            ],
            "type_f1_max_drop": TYPE_F1_MAX_DROP,
            "target_f1_min_gain": TARGET_F1_MIN_GAIN,
            "objective_focus": objective_focus,
            "skip_absolute_raw_quality": True,
            "configuration_candidate_recall_min": (
                full_initial["metrics"]
                .get("candidate_entity_by_type", {})
                .get("Configuration", {})
                .get("recall", 0.0)
                if objective_focus in {
                    "configuration_recall_only",
                    "configuration_cpe_then_relations",
                    "configuration_affects",
                    "configuration_affects_recall",
                }
                else None
            ),
            "candidate_entity_recall_min": (
                {
                    label: float(
                        full_initial["metrics"]
                        .get("candidate_entity_by_type", {})
                        .get(label, {})
                        .get("recall", 0.0)
                    )
                    for label in ENTITY_LABELS
                }
                if objective_focus == "all_entity_recall"
                else None
            ),
            "affects_precision_min": (
                max(
                    0.0,
                    float(
                        full_initial["metrics"]
                        .get("relation_by_type", {})
                        .get("affects", {})
                        .get("precision", 0.0)
                    ) - 0.03,
                )
                if objective_focus == "configuration_affects_recall"
                else None
            ),
            "exploited_by_recall_min": (
                max(
                    0.0,
                    float(
                        full_initial["metrics"]
                        .get("relation_by_type", {})
                        .get("exploited_by", {})
                        .get("recall", 0.0)
                    ) - 0.03,
                )
                if objective_focus == "exploited_by_precision"
                else None
            ),
            **_aggregate_relation_guardrail_baselines(full_initial["metrics"]),
            **_type_f1_baselines(full_initial["metrics"]),
        }
        for checked in full_dev_records.values():
            _attach_target_performance(
                checked,
                full_initial,
                objective_focus=objective_focus,
                evaluation_phase="joint",
            )
            if checked["id"] != full_initial["id"]:
                _attach_paired_performance(
                    checked,
                    full_initial,
                    objective_focus,
                    evaluation_phase="joint",
                )
            checked["feasible"] = _eligible(checked, full_guardrails)
            full_reasons = []
            if checked["id"] != full_initial["id"]:
                if not _feasible(checked["metrics"], full_guardrails):
                    full_reasons.append("full_dev_safety_guardrails")
                if not _passes_target_f1_gate(checked, full_guardrails):
                    full_reasons.append("full_dev_target_gain_gate")
                if not _passes_paired_quality_gate(checked):
                    full_reasons.append("full_dev_paired_raw_quality_gate")
                if not _passes_configuration_overlap_parent_guardrail(checked):
                    full_reasons.append(
                        "full_dev_configuration_overlap_recall_parent_gate"
                    )
            checked["full_dev_rejection_reasons"] = full_reasons

        full_dev_feasible = [
            record
            for record in full_dev_records.values()
            if record["feasible"]
        ]
        if not full_dev_feasible:
            raise RuntimeError(
                "完整开发集复核后没有满足不可劣化 guardrail 的提示"
            )
        final = sorted(
            full_dev_feasible,
            key=lambda record: _rank_key(record, "joint", objective_focus),
            reverse=True,
        )[0]
    else:
        # Without the complete-dev stage, this is the only automatic final
        # decision.  Promote the registered final target threshold here rather
        # than leaving the exploratory zero-gain screen in force.
        selection_dev_guardrails = {
            **guardrails,
            "target_f1_min_gain": TARGET_F1_MIN_GAIN,
        }
        full_guardrails = selection_dev_guardrails
        # Small-scale development runs still select automatically, but they
        # must not accept an edit merely because an unrelated label happened
        # to improve.  Apply the same registered target/safety gates before
        # returning a selection-dev winner.  P0 is always eligible, so this
        # list cannot be empty and remains the deterministic fallback.
        selection_dev_feasible = [
            record
            for record in ranked_selection
            if _eligible(record, selection_dev_guardrails)
        ]
        final = dict(selection_dev_feasible[0])
        final["selection_metrics"] = final["metrics"]
        final["feasible"] = _eligible(final, selection_dev_guardrails)

    final_validation_rejected = None
    if full_dev_recheck:
        selected_before_final_validation = final
        fresh_reference = dict(initial)
        fresh_reference["metrics"] = evaluate_prompt_pair_repeated(
            initial,
            dev_paths,
            run_dir / "final_report_predictions" / initial["id"],
            final_report_repeats,
            cache_namespace_prefix=final_report_cache_namespace,
        )
        if selected_before_final_validation["id"] == initial["id"]:
            final = fresh_reference
        else:
            final = dict(selected_before_final_validation)
            final["finalist_metrics"] = final["metrics"]
            final["metrics"] = evaluate_prompt_pair_repeated(
                final,
                dev_paths,
                run_dir / "final_report_predictions" / final["id"],
                final_report_repeats,
                cache_namespace_prefix=final_report_cache_namespace,
            )
            _attach_target_performance(
                final,
                fresh_reference,
                objective_focus=objective_focus,
                evaluation_phase="joint",
            )
            _attach_paired_performance(
                final,
                fresh_reference,
                objective_focus,
                evaluation_phase="joint",
            )
            fresh_candidate_pass = (
                _eligible(final, full_guardrails)
                and _final_output_feasible(final["metrics"], full_guardrails)
            )
            if not fresh_candidate_pass:
                final_validation_rejected = {
                    "candidate_id": final["id"],
                    "paired_performance": final.get("paired_performance"),
                    "target_performance": final.get("target_performance"),
                    "reason": "fresh_paired_final_validation_failed",
                }
                final = fresh_reference
        repeat_independence = _repeat_independence(final["metrics"])
    else:
        repeat_independence = {
            "required": False,
            "passed": False,
            "reason": "full_dev_recheck_disabled",
        }

    quality_reference = (
        full_dev_records[initial["id"]]["metrics"]
        if full_dev_recheck
        else initial["metrics"]
    )
    final_quality_gate = {
        "raw_overall_consistency": full_guardrails[
            "overall_consistency"
        ],
        "raw_evidence_validity_non_degradation": full_guardrails[
            "evidence_validity"
        ],
        "final_overall_consistency": 1.0,
        "final_evidence_validity": 1.0,
        "raw_diagnostic_target": MIN_EVIDENCE_VALIDITY,
    }
    final_quality_gate_passed = _delivered_output_feasible(
        final["metrics"]
    ) and (
        final["id"] == initial["id"] or (
            _final_output_feasible(
                final["metrics"],
                {**full_guardrails, "skip_absolute_raw_quality": True},
            )
            and
            _passes_target_f1_gate(final, full_guardrails)
            and _passes_paired_quality_gate(final)
        )
    )
    frozen_for_test = bool(
        freeze_artifact
        and final_quality_gate_passed
        and repeat_independence.get("passed", False)
    )
    # P0 is a safety baseline, not an APO result.  A run that cannot select a
    # non-empty candidate must never produce a test-runnable APO artifact.
    apo_candidate_selected = bool(
        final.get("id") != "p0_manual"
        and (
            str(final.get("stage1_guidance", "")).strip()
            or str(final.get("stage2_guidance", "")).strip()
        )
    )
    frozen_for_test = bool(frozen_for_test and apo_candidate_selected)

    # 组件级组合用于区分实体指导、关系指导和联合指导的贡献。关系指导是在
    # 预测实体条件下优化的；这里将其与初始实体指导重组，仅作为消融评价，
    # 不参与最终提示选择。
    entity_component_pool = [
        record
        for record in feasible_so_far
        if not record["stage2_guidance"]
    ]
    best_entity_component = (
        sorted(
            entity_component_pool,
            key=lambda record: _rank_key(record, "entity", objective_focus),
            reverse=True,
        )[0]
        if entity_component_pool
        else initial
    )
    relation_component_pool = [
        record for record in feasible_so_far if record["stage2_guidance"]
    ]
    best_relation_component = (
        sorted(
            relation_component_pool,
            key=lambda record: _rank_key(record, "relation", objective_focus),
            reverse=True,
        )[0]
        if relation_component_pool
        else initial
    )
    component_specs = {
        "p0_entity+p0_relation": ("", ""),
        "apo_entity+p0_relation": (
            best_entity_component["stage1_guidance"],
            "",
        ),
        "p0_entity+apo_relation": (
            "",
            best_relation_component["stage2_guidance"],
        ),
        "apo_entity+apo_relation": (
            best_entity_component["stage1_guidance"],
            best_relation_component["stage2_guidance"],
        ),
    }
    component_records = {}
    if component_ablation:
        for label, (stage1_guidance, stage2_guidance) in component_specs.items():
            if not stage1_guidance and not stage2_guidance:
                component = initial
            else:
                component = _pair(
                    stage1_guidance,
                    stage2_guidance,
                    id=_next_candidate_id(stage1_guidance, stage2_guidance),
                    phase="component-recombination",
                    round_index=global_round,
                    rationale=f"Deterministic component ablation: {label}",
                )
            if component["id"] not in full_dev_records:
                checked = dict(component)
                selection_record = all_evaluated.get(component["id"])
                checked["selection_metrics"] = (
                    selection_record.get("metrics") if selection_record else None
                )
                checked["metrics"] = _evaluate_repeated_with_failure_recovery(
                    component,
                    dev_paths,
                    run_dir / "full_dev_predictions" / component["id"],
                    final_dev_repeats,
                    cache_namespace_prefix=full_dev_cache_namespace,
                )
                checked["feasible"] = _feasible(
                    checked["metrics"],
                    full_guardrails,
                )
                full_dev_records[component["id"]] = checked
                _write_json(
                    run_dir / "prompts" / f"{component['id']}.json",
                    checked,
                )
            checked = full_dev_records[component["id"]]
            component_records[label] = {
                "prompt_id": checked["id"],
                "metrics": checked["metrics"],
                "feasible": checked["feasible"],
            }
    else:
        component_records["status"] = "skipped_by_preset"
    final_artifact = {
        "algorithm": manifest["algorithm"],
        "schema_version": SCHEMA_VERSION,
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "split": "dev" if full_dev_recheck else "selection_dev",
        "split_file": SPLIT_FILE.name,
        "split_sha256": manifest["split_sha256"],
        "frozen_at_utc": _utc_now(),
        "stage1_guidance": final["stage1_guidance"],
        "stage2_guidance": final["stage2_guidance"],
        "score": _phase_score(final, "joint", objective_focus),
        "general_composite_score": final["metrics"]["score"],
        "metrics": final["metrics"],
        "selected_round": final.get("round", 0),
        "selected_candidate": final["id"],
        "apo_candidate_selected": apo_candidate_selected,
        "selected_phase": final.get("phase"),
        "parent_id": final.get("parent_id"),
        "selection_metrics": final.get("selection_metrics"),
        "selection_shortlist_ids": [record["id"] for record in shortlist],
        "selection_promotion_rejection_reasons": {
            record_id: record.get(
                "full_dev_promotion_rejection_reasons", []
            )
            for record_id, record in all_evaluated.items()
        },
        "full_dev_finalist_ids": (
            [record["id"] for record in shortlist]
            if full_dev_recheck
            else []
        ),
        "full_dev_rejection_reasons": {
            record_id: record.get("full_dev_rejection_reasons", [])
            for record_id, record in full_dev_records.items()
        },
        "objective_weights": {
            "round_1_Configuration_candidate_overlap_recall": 1.0,
            "round_2_and_final_Configuration_candidate_strict_recall": 1.0,
            "round_2_parent_overlap_recall_minimum_gain": 0.0,
        } if objective_focus == "configuration_recall_only" else {
            "candidate_entity_type_macro_f2": 1.0,
        } if objective_focus == "all_entity_recall" else {
            "entity_round_Configuration_candidate_CPE_NA": 1.0,
            "relation_round_atomic_target_F1": 1.0,
            "final_general_composite_score": 1.0,
        } if objective_focus == "configuration_cpe_then_relations" else {
            "Configuration_final_strict_f1": 0.5,
            "affects_f1": 0.5,
        } if objective_focus in {
            "configuration_affects",
            "configuration_affects_recall",
        } else {
            "exploited_by_f0.5": 1.0,
        } if objective_focus == "exploited_by_precision" else {
            "exploited_by_f1": 1.0,
        } if objective_focus == "exploited_by" else {
            "Weakness_f1": 0.5,
            "instantiates_f1": 0.5,
        } if objective_focus == "weakness_instantiates" else {
            "entity_macro_f1": ENTITY_WEIGHT,
            "relation_macro_f1": RELATION_WEIGHT,
        },
        "objective_focus": objective_focus,
        "stage1_only": bool(final.get("stage1_only", False)),
        "configuration_policy_override": bool(
            final.get("configuration_policy_override", False)
        ),
        "configuration_recall_promotion_gate": (
            {
                "minimum_strict_recall_gain": CONFIGURATION_RECALL_MIN_GAIN,
                "minimum_mean_true_positive_gain": CONFIGURATION_RECALL_MIN_TP_GAIN,
                "screening_only": True,
            }
            if objective_focus == "configuration_recall_only"
            else None
        ),
        "selection_guardrail_baseline": guardrails,
        "guardrail_baseline": full_guardrails,
        "final_quality_gate_baseline": final_quality_gate,
        "final_quality_gate_passed": final_quality_gate_passed,
        "repeat_independence": repeat_independence,
        "final_validation_rejected": final_validation_rejected,
        "guardrail_reference_observed": initial_raw_quality,
        "component_ablation": component_records,
        "joint_recombination_ids": joint_recombination_ids,
        "optimization_budget": {
            "rounds_completed": global_round,
            "candidates_evaluated": len(all_evaluated) - 1,
            "selection_task_model_call_attempts": sum(
                record["metrics"]["api_call_attempts"]
                for record in all_evaluated.values()
            ),
            "full_dev_task_model_call_attempts": sum(
                record["metrics"]["api_call_attempts"]
                for record in full_dev_records.values()
            ),
            "final_report_task_model_call_attempts": (
                final["metrics"]["api_call_attempts"]
                if full_dev_recheck
                else 0
            ),
            "task_model_call_attempts": (
                sum(
                    record["metrics"]["api_call_attempts"]
                    for record in all_evaluated.values()
                )
                + sum(
                    record["metrics"]["api_call_attempts"]
                    for record in full_dev_records.values()
                )
                + (
                    final["metrics"]["api_call_attempts"]
                    if full_dev_recheck
                    else 0
                )
                + training_task_attempts
            ),
            "training_feedback_task_call_attempts": training_task_attempts,
            "training_feedback_elapsed_seconds": round(
                training_task_seconds,
                3,
            ),
            "optimizer_call_attempts": optimizer_attempts,
            "optimizer_input_chars": optimizer_input_chars,
            "optimizer_output_chars": optimizer_output_chars,
            "recoverable_optimizer_failures": optimizer_failures,
        },
        "run_dir": str(run_dir),
        "run_kind": run_kind,
        "initial_prompt_artifact": manifest["initial_prompt_artifact"],
        "initial_prompt_artifact_sha256": manifest[
            "initial_prompt_artifact_sha256"
        ],
        "frozen_for_test": frozen_for_test,
        "test_gold_loaded": False,
        "test_predictions_generated": False,
    }
    _write_json(run_dir / "final_prompt.json", final_artifact)
    if frozen_for_test:
        _write_json(OUTPUT_ROOT / "final_prompt.json", final_artifact)
    _write_json(
        run_dir / "optimization_summary.json",
        {
            "manifest": manifest,
            "guardrails": guardrails,
            "rounds": round_logs,
            "final_prompt": final_artifact,
        },
    )
    return final_artifact


def _window_count(paths: list[Path]) -> int:
    return sum(
        len(
            build_text_windows(
                json.loads(path.read_text(encoding="utf-8"))["text"]
            )
        )
        for path in paths
    )


def estimate_budget(
    *,
    candidate_count: int,
    beam_size: int,
    train_batch_size: int,
    selection_dev_size: int,
    train_doc_ids: list[str] | None,
    selection_dev_doc_ids: list[str] | None,
    dev_repeats: int,
    final_dev_repeats: int,
    final_report_repeats: int,
    finalist_count: int,
    full_dev_recheck: bool,
    component_ablation: bool,
    phase_rounds: dict[str, int],
    objective_focus: str | None = None,
    seed: int = 20260730,
) -> dict:
    """估算APO的请求上界；只读取train/dev，不调用模型。"""
    train_paths, dev_paths, split = _load_split()
    train_by_id = {path.stem: path for path in train_paths}
    dev_by_id = {path.stem: path for path in dev_paths}
    optimization_train_paths = (
        [train_by_id[item] for item in train_doc_ids]
        if train_doc_ids
        else train_paths
    )
    dev_windows = _window_count(dev_paths)
    selection_dev_paths = (
        [dev_by_id[item] for item in selection_dev_doc_ids]
        if selection_dev_doc_ids
        else select_target_stratified_evaluation_batch(
            dev_paths,
            batch_size=min(selection_dev_size, len(dev_paths)),
            seed=seed + 100_000,
            target_labels=(
                _atomic_target_for_round("joint", 1, objective_focus)
                if objective_focus is not None
                else ()
            ),
        )
    )
    selection_dev_windows = _window_count(selection_dev_paths)
    stage_call_multiplier = (
        1
        if objective_focus in {
            "all_entity_recall",
            "configuration_recall_only",
        }
        else 2
    )
    task_requests = (
        selection_dev_windows * stage_call_multiplier * dev_repeats
    )
    optimizer_requests = 0
    round_details = []
    global_round = 0
    joint_seed_requests = (
        beam_size
        * selection_dev_windows
        * stage_call_multiplier
        * dev_repeats
        if int(phase_rounds.get("joint", 0)) > 0
        and int(phase_rounds.get("entity", 0)) > 0
        and int(phase_rounds.get("relation", 0)) > 0
        else 0
    )
    task_requests += joint_seed_requests
    for phase in ("entity", "relation", "joint"):
        rounds = int(phase_rounds.get(phase, 0))
        for local_round in range(1, rounds + 1):
            global_round += 1
            parents = 1 if global_round == 1 else beam_size
            atomic_target_labels = _atomic_target_for_round(
                phase, local_round, objective_focus
            )
            batch_paths = select_training_batch_for_target(
                optimization_train_paths,
                global_round,
                train_batch_size,
                seed,
                atomic_target_labels,
            )
            train_windows = _window_count(batch_paths)
            training_requests = parents * train_windows * stage_call_multiplier
            candidate_requests = (
                parents
                * candidate_count
                * selection_dev_windows
                * stage_call_multiplier
                * dev_repeats
            )
            round_optimizer_requests = parents * 2
            task_requests += training_requests + candidate_requests
            optimizer_requests += round_optimizer_requests
            round_details.append(
                {
                    "phase": phase,
                    "round": local_round,
                    "parents_upper_bound": parents,
                    "training_batch_documents": len(batch_paths),
                    "training_batch_windows": train_windows,
                    "task_requests_nominal": (
                        training_requests + candidate_requests
                    ),
                    "optimizer_requests_nominal": round_optimizer_requests,
                }
            )

    full_dev_finalist_requests = (
        (finalist_count + 1)
        * dev_windows
        * stage_call_multiplier
        * final_dev_repeats
        if full_dev_recheck
        else 0
    )
    # 四个组件中，P0已经进入完整开发集复核；其余三个均按最坏情况估算。
    component_requests = (
        3 * dev_windows * stage_call_multiplier * final_dev_repeats
        if component_ablation and full_dev_recheck
        else 0
    )
    final_report_requests = (
        # Fresh P0 reference plus a non-P0 winner in the worst case.
        2 * dev_windows * stage_call_multiplier * final_report_repeats
        if full_dev_recheck
        else 0
    )
    task_requests += (
        full_dev_finalist_requests
        + component_requests
        + final_report_requests
    )
    total_nominal = task_requests + optimizer_requests
    return {
        "schema_version": SCHEMA_VERSION,
        "train_documents_loaded": len(train_paths),
        "optimization_train_pool_documents": len(
            optimization_train_paths
        ),
        "optimization_train_pool_ids": [
            path.stem for path in optimization_train_paths
        ],
        "dev_documents_loaded": len(dev_paths),
        "test_documents_loaded": 0,
        "held_out_test_documents": len(split.get("test", [])),
        "dev_windows": dev_windows,
        "selection_dev_documents": len(selection_dev_paths),
        "selection_dev_ids": [path.stem for path in selection_dev_paths],
        "selection_dev_windows": selection_dev_windows,
        "stage_model_calls_per_window": stage_call_multiplier,
        "rounds": round_details,
        "joint_search_recombination_task_requests_upper_bound": (
            joint_seed_requests
        ),
        "full_dev_finalist_task_requests_upper_bound": (
            full_dev_finalist_requests
        ),
        "component_recombination_task_requests_upper_bound": (
            component_requests
        ),
        "final_report_task_requests_upper_bound": final_report_requests,
        "task_model_requests_nominal_upper_bound": task_requests,
        "optimizer_model_requests_nominal_upper_bound": optimizer_requests,
        "all_model_requests_nominal_upper_bound": total_nominal,
        "all_model_requests_with_three_attempt_retries_upper_bound": (
            total_nominal * 3
        ),
        "note": (
            "This is a request-count upper bound, not a price estimate. "
            "Structural stopping and duplicate/invalid candidates can reduce it."
        ),
    }


def validate_setup(train_batch_size: int = 10, seed: int = 20260730) -> dict:
    """不调用模型，仅验证数据边界、模式与首轮训练批次覆盖。"""
    train_paths, dev_paths, split = _load_split()
    first_batch = select_training_batch(
        train_paths,
        round_index=1,
        batch_size=train_batch_size,
        seed=seed,
    )
    covered_features = sorted(
        set().union(
            *(
                _annotation_features(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                for path in first_batch
            )
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "split_file": str(SPLIT_FILE),
        "split_sha256": hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
        "train_documents_loaded": len(train_paths),
        "dev_documents_loaded": len(dev_paths),
        "test_documents_loaded": 0,
        "held_out_test_documents": len(split.get("test", [])),
        "first_training_batch": [path.stem for path in first_batch],
        "first_training_batch_features": covered_features,
        "base_prompt_sha256": BASE_PROMPT_SHA256,
        "runtime_config": runtime_config(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在固定 train/dev 划分上运行无 CAPEC 的两阶段 APO"
    )
    parser.add_argument(
        "--preset",
        choices=tuple(APO_PRESETS),
        default="formal_stratified",
        help="pilot只验证闭环；formal为历史正式配置；formal_stratified为当前预注册的分层五轮正式搜索",
    )
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--selection-dev-size", type=int)
    parser.add_argument(
        "--structural-patience",
        type=int,
        default=2,
        help="连续多少轮没有可行新候选时停止当前阶段；不用于分数早停",
    )
    parser.add_argument("--dev-repeats", type=int)
    parser.add_argument("--final-dev-repeats", type=int)
    parser.add_argument("--finalist-count", type=int)
    parser.add_argument(
        "--skip-component-ablation",
        action="store_true",
        help="跳过开发集组件重组消融；pilot默认跳过",
    )
    parser.add_argument(
        "--skip-full-dev-recheck",
        action="store_true",
        help="跳过全量开发集重复复核，仅在 selection-dev 完成优化与选优",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--entity-rounds", type=int)
    parser.add_argument("--relation-rounds", type=int)
    parser.add_argument("--joint-rounds", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证划分与批次覆盖，不调用任何模型",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="估算模型请求数，不调用任何模型",
    )
    parser.add_argument(
        "--initial-prompt-artifact",
        type=Path,
        help="从既有 final_prompt.json 继续 APO；仍重新评价且不读取测试集",
    )
    args = parser.parse_args()
    preset = dict(APO_PRESETS[args.preset])
    phase_rounds = dict(preset["phase_rounds"])
    candidate_count = args.candidates or preset["candidate_count"]
    beam_size = args.beam_size or preset["beam_size"]
    train_batch_size = args.train_batch_size or preset["train_batch_size"]
    selection_dev_size = (
        args.selection_dev_size or preset["selection_dev_size"]
    )
    dev_repeats = args.dev_repeats or preset["dev_repeats"]
    final_dev_repeats = (
        args.final_dev_repeats or preset["final_dev_repeats"]
    )
    final_report_repeats = preset.get(
        "final_report_repeats", final_dev_repeats
    )
    finalist_count = args.finalist_count or preset["finalist_count"]
    component_ablation = bool(preset["component_ablation"]) and not (
        args.skip_component_ablation
    )
    full_dev_recheck = bool(preset["full_dev_recheck"]) and not (
        args.skip_full_dev_recheck
    )

    train_doc_ids = preset.get("train_doc_ids")
    selection_dev_doc_ids = preset.get("selection_dev_doc_ids")
    objective_focus = preset.get("objective_focus")
    phase_rounds.update(
        {
            key: value
            for key, value in {
                "entity": args.entity_rounds,
                "relation": args.relation_rounds,
                "joint": args.joint_rounds,
            }.items()
            if value is not None
        }
    )
    if args.validate_only:
        print(
            json.dumps(
                validate_setup(train_batch_size, args.seed),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.estimate_only:
        print(
            json.dumps(
                estimate_budget(
                    candidate_count=candidate_count,
                    beam_size=beam_size,
                    train_batch_size=train_batch_size,
                    selection_dev_size=selection_dev_size,
                    train_doc_ids=train_doc_ids,
                    selection_dev_doc_ids=selection_dev_doc_ids,
                    dev_repeats=dev_repeats,
                    final_dev_repeats=final_dev_repeats,
                    final_report_repeats=final_report_repeats,
                    finalist_count=finalist_count,
                    full_dev_recheck=full_dev_recheck,
                    component_ablation=component_ablation,
                    phase_rounds=phase_rounds,
                    objective_focus=objective_focus,
                    seed=args.seed,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    _require_boundary_recertification_complete()
    final = optimize(
        candidate_count=candidate_count,
        beam_size=beam_size,
        train_batch_size=train_batch_size,
        selection_dev_size=selection_dev_size,
        train_doc_ids=train_doc_ids,
        selection_dev_doc_ids=selection_dev_doc_ids,
        structural_patience=args.structural_patience,
        dev_repeats=dev_repeats,
        final_dev_repeats=final_dev_repeats,
        final_report_repeats=final_report_repeats,
        finalist_count=finalist_count,
        full_dev_recheck=full_dev_recheck,
        component_ablation=component_ablation,
        seed=args.seed,
        phase_rounds=phase_rounds,
        objective_focus=objective_focus,
        run_kind=args.preset,
        freeze_artifact=bool(preset["freeze_artifact"]),
        initial_prompt_artifact=args.initial_prompt_artifact,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

