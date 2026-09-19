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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    ) -> List[dict]:
        """加载固化的实体预测缓存，并在哈希不匹配时严格阻断。"""
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
