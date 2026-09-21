"""ProTeGi Stage 1 运行稳定性模块：错误分类、检查点恢复与评估缓存。

职责（只做运行稳定性，不改数据/协议/runtime/搜索超参数）：
1. 异常分类：budget exhaustion / transient HTTP-network / program bug，
   禁止任何 ``except Exception: continue`` 式静默吞错。
2. 检查点：在 transient 重试耗尽时保存可恢复的搜索状态，并支持
   ``--resume`` 从中断轮次继续；恢复时严格校验
   config/split/freeze/Gold/runtime/implementation 绑定。
3. 候选评估缓存：已成功评估的 (candidate, batch) 直接复用，避免
   重启后重复调用 task model；key 绑定 stage/prompt/sample/runtime/freeze。

本模块为纯逻辑，不发起任何模型调用，可被纯离线测试直接覆盖。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from protegi.entity_cache import compute_prompt_hash
from protegi.runtime_contract import TASK_RUNTIME_FIELDS

INVALID_BUDGET_EXHAUSTED = "invalid_budget_exhausted"
INVALID_OUTPUT_AMPLIFICATION = "invalid_output_amplification"
EVALUATION_STATUS_VALID = "valid"

CHECKPOINT_VERSION = "protegi-search-checkpoint-v1"
CHECKPOINT_FILENAME = "search_checkpoint.json"
EVAL_CACHE_FILENAME = "eval_cache.json"
EVAL_CACHE_VERSION = "protegi-eval-cache-v1"
EVAL_CACHE_MAX_ENTRIES = 5000

# 检查点绑定的实现文件集合（与 optimizer summary 的实现清单保持一致，
# 另加本模块与相关契约文件）。
IMPLEMENTATION_FILES = (
    "protegi/optimizer.py",
    "protegi/evaluator.py",
    "protegi/models.py",
    "protegi/selectors.py",
    "protegi/mutators.py",
    "protegi/retry_utils.py",
    "protegi/search_stability.py",
    "protegi/output_expansion_guard.py",
    "protegi/lineage.py",
    "protegi/logging_utils.py",
    "protegi/entity_cache.py",
    "protegi/runtime_contract.py",
    "protegi/gold_integrity.py",
    "scripts/run_protegi.py",
    "scripts/llm_methods.py",
    "scripts/llm_extractor.py",
)


def _is_budget_exhausted_error(exc: BaseException) -> bool:
    """判断是否为输出预算耗尽类错误（不依赖具体 import 是否成功）。"""
    if isinstance(exc, (WindowBudgetExhaustedError, CandidateBudgetExhaustedError)):
        return True
    if type(exc).__name__ == "ModelOutputBudgetExhaustedError":
        return True
    try:
        import sys as _sys

        for mod_name in ("llm_extractor", "scripts.llm_extractor"):
            mod = _sys.modules.get(mod_name)
            cls = getattr(mod, "ModelOutputBudgetExhaustedError", None) if mod else None
            if cls is not None and isinstance(exc, cls):
                return True
    except Exception:
        pass
    return False


def classify_search_error(exc: BaseException) -> str:
    """将搜索路径异常分为三类，调用方据此分支，禁止吞错。

    Returns:
        "budget"     输出预算耗尽 → 候选记 INVALID，不伪造 F1。
        "transient"  HTTP 500/502/503/429/timeout/网络抖动 → 重试/检查点。
        "program"    ValueError/assert/类型错误/未知错误 → 直接 hard fail。
    """
    if _is_budget_exhausted_error(exc):
        return "budget"
    try:
        from protegi.retry_utils import is_retryable_error

        if is_retryable_error(exc):
            return "transient"
    except Exception:
        pass
    return "program"


class WindowBudgetExhaustedError(Exception):
    """Evaluator 层：同一窗口在完全相同 runtime 下重试 1 次后仍超限。"""

    def __init__(
        self,
        *,
        stage: str,
        prompt_hash: str,
        sample_id: str,
        max_tokens: Any,
        retries_used: int = 1,
    ):
        super().__init__(
            f"window budget exhausted (stage={stage}, sample={sample_id}, "
            f"max_tokens={max_tokens}, retries_used={retries_used})"
        )
        self.stage = stage
        self.prompt_hash = prompt_hash
        self.sample_id = sample_id
        self.max_tokens = max_tokens
        self.retries_used = retries_used


class CandidateBudgetExhaustedError(Exception):
    """Optimizer 层：候选因预算耗尽被标记 INVALID，不参与 UCB/beam/winner。"""

    def __init__(
        self,
        *,
        candidate_id: str,
        stage: str,
        prompt_hash: str,
        sample_ids: List[str],
        failure_reason: str,
    ):
        super().__init__(
            f"candidate {candidate_id} invalid: budget exhausted "
            f"on samples {sample_ids[:4]}"
        )
        self.candidate_id = candidate_id
        self.stage = stage
        self.prompt_hash = prompt_hash
        self.sample_ids = list(sample_ids)
        self.failure_reason = failure_reason


class RoundSelectionAborted(Exception):
    """选择轮次因候选中途 INVALID 而提前终止，保留 incumbent beam。"""

    def __init__(self, *, invalid_candidate_id: str, round_idx: int):
        super().__init__(
            f"round {round_idx} selection aborted: "
            f"candidate {invalid_candidate_id} invalid"
        )
        self.invalid_candidate_id = invalid_candidate_id
        self.round_idx = round_idx


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def sample_ids_hash(sample_ids: List[str]) -> str:
    return _sha256_text("\n".join(str(s) for s in sample_ids))


def task_runtime_hash(task_runtime: Dict[str, Any]) -> str:
    ordered = {field: task_runtime[field] for field in TASK_RUNTIME_FIELDS}
    return _sha256_text(_canonical_json(ordered))


def eval_cache_key(
    *,
    stage: str,
    prompt_hash: str,
    sample_ids: List[str],
    task_runtime: Dict[str, Any],
    split_sha256: Optional[str],
    gold_aggregate_sha256: Optional[str],
    freeze_manifest_sha256: Optional[str],
    collect_errors: bool,
    capture_predictions: bool,
) -> str:
    """评估缓存 key：绑定 stage/prompt/样本/runtime/freeze，缺一即 miss。"""
    payload = {
        "version": EVAL_CACHE_VERSION,
        "stage": stage,
        "prompt_hash": prompt_hash,
        "sample_ids": [str(s) for s in sample_ids],
        "sample_ids_sha256": sample_ids_hash([str(s) for s in sample_ids]),
        "task_runtime_sha256": task_runtime_hash(task_runtime),
        "split_sha256": split_sha256,
        "gold_aggregate_sha256": gold_aggregate_sha256,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "collect_errors": bool(collect_errors),
        "capture_predictions": bool(capture_predictions),
    }
    return _sha256_text(_canonical_json(payload))


class CandidateEvalCache:
    """候选评估结果缓存：只存成功评估，重启后复用，零重复 task 调用。"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.file = self.output_dir / EVAL_CACHE_FILENAME
        self.entries: Dict[str, dict] = {}
        self.hits = 0
        if self.file.is_file():
            try:
                data = json.loads(self.file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == EVAL_CACHE_VERSION:
                    entries = data.get("entries", {})
                    if isinstance(entries, dict):
                        self.entries = entries
            except (OSError, json.JSONDecodeError, ValueError):
                self.entries = {}

    def lookup(self, key: str) -> Optional[dict]:
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.hits += 1
        return entry

    def store(self, key: str, evaluation: dict, errors: List[dict]) -> None:
        if key in self.entries:
            return
        if len(self.entries) >= EVAL_CACHE_MAX_ENTRIES:
            oldest = next(iter(self.entries))
            del self.entries[oldest]
        self.entries[key] = {"evaluation": evaluation, "errors": errors}
        self.save()

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"version": EVAL_CACHE_VERSION, "entries": self.entries},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.file)


def implementation_hashes(repo_root: Path) -> Dict[str, Optional[str]]:
    """计算检查点绑定的实现文件哈希；缺失记 None（恢复时同样比对）。"""
    result: Dict[str, Optional[str]] = {}
    for rel in IMPLEMENTATION_FILES:
        path = Path(repo_root) / rel
        if path.is_file():
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[rel] = digest.hexdigest()
        else:
            result[rel] = None
    return result


def rng_state_to_json(state: Any) -> dict:
    version, internal, gauss = state
    return {
        "version": int(version),
        "state": [int(x) for x in internal],
        "gauss": None if gauss is None else float(gauss),
    }


def rng_state_from_json(data: dict) -> Any:
    version = int(data["version"])
    internal = tuple(int(x) for x in data["state"])
    if len(internal) != 625:
        raise ValueError(
            f"RNG state 长度非法: {len(internal)} != 625，拒绝恢复"
        )
    gauss = data.get("gauss")
    return (version, internal, None if gauss is None else float(gauss))


def save_search_checkpoint(output_dir: Path, payload: dict) -> Path:
    """原子写入搜索检查点（含版本与绑定），供 --resume 使用。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = dict(payload)
    full["checkpoint_version"] = CHECKPOINT_VERSION
    target = output_dir / CHECKPOINT_FILENAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(target)
    return target


def load_search_checkpoint(output_dir: Path) -> dict:
    target = Path(output_dir) / CHECKPOINT_FILENAME
    if not target.is_file():
        raise FileNotFoundError(f"恢复目录缺少检查点文件: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"检查点版本不兼容: {data.get('checkpoint_version')!r} "
            f"!= {CHECKPOINT_VERSION!r}，拒绝恢复"
        )
    return data


def validate_checkpoint_bindings(checkpoint: dict, expected: dict) -> None:
    """严格比对恢复绑定；任一不一致即拒绝恢复并列出差异字段。"""
    mismatches: List[str] = []
    for field in (
        "stage",
        "method",
        "prompt_scope",
        "experiment_pair_id",
        "config_file_sha256",
        "split_sha256",
        "gold_aggregate_sha256",
        "freeze_manifest_sha256",
        "train_sample_ids_sha256",
        "dev_sample_ids_sha256",
    ):
        if checkpoint.get(field) != expected.get(field):
            mismatches.append(field)
    if checkpoint.get("effective_task_runtime") != expected.get(
        "effective_task_runtime"
    ):
        mismatches.append("effective_task_runtime")
    if checkpoint.get("implementation") != expected.get("implementation"):
        mismatches.append("implementation")
    if mismatches:
        raise ValueError(
            "检查点绑定与当前运行不一致，拒绝恢复 "
            f"(mismatched={sorted(mismatches)})：请用完全相同的 "
            "config/split/freeze/Gold/runtime/实现版本重试，或换新输出目录重跑。"
        )


def new_stability_counters() -> Dict[str, Any]:
    return {
        "budget_exhausted_samples": [],
        "budget_exhaustion_retries": 0,
        "candidates_valid": 0,
        "candidates_invalid_budget_exhausted": 0,
        "candidates_invalid_output_amplification": 0,
        "transient_api_retries": 0,
        "evaluation_cache_hits": 0,
        "resume_count": 0,
        "failure_log": [],
    }


__all__ = [
    "INVALID_BUDGET_EXHAUSTED",
    "INVALID_OUTPUT_AMPLIFICATION",
    "EVALUATION_STATUS_VALID",
    "CHECKPOINT_VERSION",
    "CHECKPOINT_FILENAME",
    "EVAL_CACHE_FILENAME",
    "IMPLEMENTATION_FILES",
    "WindowBudgetExhaustedError",
    "CandidateBudgetExhaustedError",
    "RoundSelectionAborted",
    "classify_search_error",
    "compute_prompt_hash",
    "sample_ids_hash",
    "task_runtime_hash",
    "eval_cache_key",
    "CandidateEvalCache",
    "implementation_hashes",
    "rng_state_to_json",
    "rng_state_from_json",
    "save_search_checkpoint",
    "load_search_checkpoint",
    "validate_checkpoint_bindings",
    "new_stability_counters",
]
