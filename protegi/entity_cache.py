"""Stage 2 冻结实体预测缓存模块 (Frozen Entity Cache)。

在 Stage 1 产出最优冻结实体提示词 P_E* 后，提前对 Train 与 Dev 数据生成
确定性实体预测并固化，确保 Stage 2 关系提示词搜索中的所有 Candidate
在完全相同且不变的实体输入上公平竞争，杜绝上游抽取波动造成的方差。
缓存同时保留对应 Gold 实体/关系，并绑定内容哈希与样本顺序；Stage 2
加载时 fail-closed，不允许缺失 Gold 或回退为 Gold 实体输入。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from schema import (
    ANNOTATION_PROTOCOL_VERSION,
    BOUNDARY_CONTRACT_VERSION,
    SCHEMA_VERSION,
)

DEFAULT_FREEZE_MANIFEST = ROOT / "data" / "dataset_freeze_manifest_v6.json"
DEFAULT_SPLIT_FILE = ROOT / "data" / "train_dev_test_split_v7.json"


def compute_prompt_hash(prompt_text: str) -> str:
    """计算跨平台稳定的提示词 SHA-256（统一 LF 并去除首尾空白）。"""
    canonical = prompt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_ids_hash(sample_ids: List[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EntityCacheManager:
    """实体预测缓存管理器。"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_and_save_cache(
        self,
        evaluator,
        final_entity_prompt: str,
        samples: List[dict],
        split_name: str,
        prompt_scope: str = "constrained",
        freeze_manifest_path: Optional[Path] = None,
        split_file_path: Optional[Path] = None,
        schema_version: Optional[str] = None,
        annotation_protocol_version: Optional[str] = None,
        boundary_contract_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
        gold_aggregate_sha256: Optional[str] = None,
        split_sha256: Optional[str] = None,
        dataset_freeze_manifest_sha256: Optional[str] = None,
    ) -> Path:
        """使用冻结的 P_E* 离线生成并持久化指定切分的实体预测缓存。"""
        prompt_hash = compute_prompt_hash(final_entity_prompt)
        cache_file = self.cache_dir / f"entity_cache_{split_name}.jsonl"
        manifest_file = self.cache_dir / f"entity_cache_{split_name}_manifest.json"
        if cache_file.exists() or manifest_file.exists():
            raise FileExistsError(
                f"{split_name} 缓存目标已存在，拒绝覆盖: {self.cache_dir}。"
                "请指定新的 --entity-cache-dir。"
            )

        freeze_manifest_p = freeze_manifest_path or DEFAULT_FREEZE_MANIFEST
        split_file_p = split_file_path or DEFAULT_SPLIT_FILE
        freeze_sha = None
        gold_agg = None
        s_hash = None
        s_version = SCHEMA_VERSION
        a_proto = ANNOTATION_PROTOCOL_VERSION
        b_contract = BOUNDARY_CONTRACT_VERSION
        d_version = "v6-v5-gold-v9-mcpu-v2"
        if freeze_manifest_p.is_file():
            fm_data = json.loads(freeze_manifest_p.read_text(encoding="utf-8"))
            freeze_sha = _sha256_file(freeze_manifest_p)
            gold_agg = fm_data.get("gold_aggregate_sha256")
            s_hash = fm_data.get("split_sha256")
            s_version = fm_data.get("schema_version", s_version)
            a_proto = fm_data.get("annotation_protocol_version", a_proto)
            b_contract = fm_data.get("boundary_contract_version", b_contract)
            d_version = fm_data.get("dataset_version", d_version)

        if split_file_p.is_file():
            file_s_hash = _sha256_file(split_file_p)
            s_hash = s_hash or file_s_hash
            split_file_str = "data/" + split_file_p.name
        else:
            split_file_str = "data/train_dev_test_split_v7.json"

        schema_version = schema_version or s_version
        annotation_protocol_version = annotation_protocol_version or a_proto
        boundary_contract_version = boundary_contract_version or b_contract
        dataset_version = dataset_version or d_version
        split_sha256 = split_sha256 or s_hash
        gold_aggregate_sha256 = gold_aggregate_sha256 or gold_agg
        dataset_freeze_manifest_sha256 = dataset_freeze_manifest_sha256 or freeze_sha

        texts = [sample["text"] for sample in samples]
        abbreviation_contexts = None
        if any("document_abbreviations" in sample for sample in samples):
            abbreviation_contexts = [
                str(sample.get("document_abbreviations", "(none detected)"))
                for sample in samples
            ]
        if hasattr(evaluator, "predict_stage1_texts"):
            predictions = evaluator.predict_stage1_texts(
                texts,
                final_entity_prompt,
                document_abbreviations=abbreviation_contexts,
            )
        else:
            # 保留纯逻辑测试与外部 evaluator 的兼容路径。
            predictions = [
                evaluator.predict_stage1_window(
                    text,
                    final_entity_prompt,
                    abbreviation_contexts[index],
                )
                if abbreviation_contexts is not None
                else evaluator.predict_stage1_window(text, final_entity_prompt)
                for index, text in enumerate(texts)
            ]
        if len(predictions) != len(samples):
            raise RuntimeError(
                "实体批量预测数量与输入样本数不一致，拒绝写入不完整缓存"
            )

        cached_samples = []
        with open(cache_file, "w", encoding="utf-8", newline="\n") as f:
            for s, pred_entities in zip(samples, predictions):
                sample_id = s.get("sample_id") or s.get("id") or "unknown"
                text = s["text"]

                record = {
                    "sample_id": sample_id,
                    "text": text,
                    "fixed_entities": pred_entities,
                    "gold_entities": s.get("gold_entities", s.get("entities", [])),
                    "gold_relations": s.get("gold_relations", s.get("relations", [])),
                }
                if "document_abbreviations" in s:
                    record["document_abbreviations"] = s["document_abbreviations"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                cached_samples.append(record)

        sample_ids = [str(record["sample_id"]) for record in cached_samples]
        gold_relation_count = sum(
            len(record.get("gold_relations", [])) for record in cached_samples
        )
        manifest = {
            "split_name": split_name,
            "entity_prompt_sha256": prompt_hash,
            "prompt_scope": prompt_scope,
            "task_max_workers": getattr(evaluator, "max_workers", None),
            "num_samples": len(cached_samples),
            "sample_ids_sha256": _sample_ids_hash(sample_ids),
            "gold_relation_count": gold_relation_count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_file": str(cache_file.name),
            "cache_file_sha256": _sha256_file(cache_file),
            "schema_version": schema_version,
            "annotation_protocol_version": annotation_protocol_version,
            "boundary_contract_version": boundary_contract_version,
            "dataset_version": dataset_version,
            "split_file": split_file_str,
            "split_sha256": split_sha256,
            "gold_aggregate_sha256": gold_aggregate_sha256,
            "dataset_freeze_manifest_sha256": dataset_freeze_manifest_sha256,
            "task_model_config": {
                key: value
                for key, value in getattr(
                    getattr(evaluator, "client", None), "config", {}
                ).items()
                if key not in {"api_key", "authorization", "headers"}
            },
        }
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return cache_file

    def load_cache(
        self,
        split_name: str,
        expected_prompt_hash: Optional[str] = None,
        require_gold_relations: bool = False,
        expected_task_model: Optional[str] = None,
        expected_prompt_scope: Optional[str] = None,
        expected_task_max_workers: Optional[int] = None,
        expected_schema_version: Optional[str] = None,
        expected_annotation_protocol_version: Optional[str] = None,
        expected_boundary_contract_version: Optional[str] = None,
        expected_dataset_version: Optional[str] = None,
        expected_split_sha256: Optional[str] = None,
        expected_gold_aggregate_sha256: Optional[str] = None,
        expected_freeze_manifest_sha256: Optional[str] = None,
        freeze_manifest_path: Optional[Path] = None,
        split_file_path: Optional[Path] = None,
        validate_freeze_binding: bool = True,
    ) -> List[dict]:
        """加载固化的实体预测缓存，并在哈希不匹配或冻结绑定失效时严格阻断。"""
        cache_file = self.cache_dir / f"entity_cache_{split_name}.jsonl"
        manifest_file = self.cache_dir / f"entity_cache_{split_name}_manifest.json"

        if not cache_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError(f"缺少 {split_name} 实体预测缓存，请先完成 Stage 1 并执行实体缓存构建！")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if expected_prompt_hash and manifest.get("entity_prompt_sha256") != expected_prompt_hash:
            raise ValueError(
                f"实体缓存哈希 ({manifest.get('entity_prompt_sha256')}) 与期望的 P_E* 哈希 ({expected_prompt_hash}) 不一致！"
            )
        if (
            expected_prompt_scope
            and manifest.get("prompt_scope") != expected_prompt_scope
        ):
            raise ValueError(
                f"实体缓存 prompt_scope ({manifest.get('prompt_scope')}) 与期望实验臂 "
                f"({expected_prompt_scope}) 不一致"
            )
        if (
            expected_task_max_workers is not None
            and manifest.get("task_max_workers") != expected_task_max_workers
        ):
            raise ValueError(
                f"实体缓存任务模型并发上限 ({manifest.get('task_max_workers')}) "
                f"与期望值 ({expected_task_max_workers}) 不一致"
            )
        cached_task_model = manifest.get("task_model_config", {}).get("model")
        if expected_task_model and cached_task_model != expected_task_model:
            raise ValueError(
                f"实体缓存任务模型 ({cached_task_model}) 与期望模型 "
                f"({expected_task_model}) 不一致"
            )

        if validate_freeze_binding:
            freeze_p = freeze_manifest_path or DEFAULT_FREEZE_MANIFEST
            split_p = split_file_path or DEFAULT_SPLIT_FILE
            if not freeze_p.is_file():
                raise FileNotFoundError(f"冻结清单文件不存在：{freeze_p}")
            if not split_p.is_file():
                raise FileNotFoundError(f"划分文件不存在：{split_p}")
            fm_data = json.loads(freeze_p.read_text(encoding="utf-8"))
            ref_freeze_sha = _sha256_file(freeze_p)
            ref_gold_agg = fm_data.get("gold_aggregate_sha256")
            ref_split_sha = fm_data.get("split_sha256")
            ref_schema = fm_data.get("schema_version", SCHEMA_VERSION)
            ref_proto = fm_data.get("annotation_protocol_version", ANNOTATION_PROTOCOL_VERSION)
            ref_boundary = fm_data.get("boundary_contract_version", BOUNDARY_CONTRACT_VERSION)
            ref_dataset = fm_data.get("dataset_version", "v6-v5-gold-v9-mcpu-v2")

            req_schema = expected_schema_version or ref_schema
            req_proto = expected_annotation_protocol_version or ref_proto
            req_boundary = expected_boundary_contract_version or ref_boundary
            req_dataset = expected_dataset_version or ref_dataset
            req_split_sha = expected_split_sha256 or ref_split_sha
            req_gold_agg = expected_gold_aggregate_sha256 or ref_gold_agg
            req_freeze_sha = expected_freeze_manifest_sha256 or ref_freeze_sha

            if req_schema and manifest.get("schema_version") != req_schema:
                raise ValueError(
                    f"实体缓存 schema_version ({manifest.get('schema_version')}) 与期望值 ({req_schema}) 不一致！"
                )
            if req_proto and manifest.get("annotation_protocol_version") != req_proto:
                raise ValueError(
                    f"实体缓存 annotation_protocol_version ({manifest.get('annotation_protocol_version')}) 与期望值 ({req_proto}) 不一致！"
                )
            if req_boundary and manifest.get("boundary_contract_version") != req_boundary:
                raise ValueError(
                    f"实体缓存 boundary_contract_version ({manifest.get('boundary_contract_version')}) 与期望值 ({req_boundary}) 不一致！"
                )
            if req_dataset and manifest.get("dataset_version") != req_dataset:
                raise ValueError(
                    f"实体缓存 dataset_version ({manifest.get('dataset_version')}) 与期望值 ({req_dataset}) 不一致！"
                )
            if req_split_sha and manifest.get("split_sha256") != req_split_sha:
                raise ValueError(
                    f"实体缓存 split_sha256 ({manifest.get('split_sha256')}) 与当前划分哈希 ({req_split_sha}) 不一致！划分已变更，缓存自动失效。"
                )
            if req_gold_agg and manifest.get("gold_aggregate_sha256") != req_gold_agg:
                raise ValueError(
                    f"实体缓存 gold_aggregate_sha256 ({manifest.get('gold_aggregate_sha256')}) 与当前 Gold 聚合哈希 ({req_gold_agg}) 不一致！Gold 已变更，缓存自动失效。"
                )
            if req_freeze_sha and manifest.get("dataset_freeze_manifest_sha256") != req_freeze_sha:
                raise ValueError(
                    f"实体缓存 dataset_freeze_manifest_sha256 ({manifest.get('dataset_freeze_manifest_sha256')}) 与当前冻结清单哈希 ({req_freeze_sha}) 不一致！"
                )

        actual_cache_hash = _sha256_file(cache_file)
        if manifest.get("cache_file_sha256") != actual_cache_hash:
            raise ValueError(
                f"{split_name} 实体缓存内容哈希不匹配；缓存可能被修改或未完整写入"
            )

        samples = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if manifest.get("num_samples") != len(samples):
            raise ValueError(
                f"{split_name} 实体缓存样本数与 manifest 不一致: "
                f"{len(samples)} != {manifest.get('num_samples')}"
            )
        sample_ids = [str(sample.get("sample_id")) for sample in samples]
        if manifest.get("sample_ids_sha256") != _sample_ids_hash(sample_ids):
            raise ValueError(f"{split_name} 实体缓存 sample_id 顺序/集合与 manifest 不一致")
        gold_relation_count = sum(
            len(sample.get("gold_relations", [])) for sample in samples
        )
        if manifest.get("gold_relation_count") != gold_relation_count:
            raise ValueError(f"{split_name} 实体缓存关系 Gold 计数与 manifest 不一致")
        if require_gold_relations and gold_relation_count <= 0:
            raise ValueError(
                f"{split_name} 实体缓存未携带任何 gold_relations，禁止用于 Stage 2 评估"
            )
        return samples
