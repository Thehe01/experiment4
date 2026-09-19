"""Formal ProTeGi Gold 完整性校验 helper（可复用）。

职责：在 formal 模式读取任何 Gold 之前，针对当前实际
``data/annotations/gold/`` 重新计算全部 Gold 文档 SHA-256 与聚合 SHA，
并与 ``dataset_freeze_manifest_v6.json`` 中的冻结值比较。

聚合算法必须与 ``scripts/freeze_dataset_manifest_v6.py`` 完全一致：
    aggregate.update(f"{doc_id}\\0{doc_hash}\\n".encode("utf-8"))
按 doc_id 排序后依次喂入。

非 Gold 文件（``_manifest.json``、下划线前缀等）一律排除。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_gold_aggregate_sha256(gold_document_sha256: dict) -> str:
    """按冻结脚本完全相同的算法重新计算 Gold 聚合 SHA。"""
    aggregate = hashlib.sha256()
    for doc_id in sorted(gold_document_sha256):
        aggregate.update(f"{doc_id}\0{gold_document_sha256[doc_id]}\n".encode("utf-8"))
    return aggregate.hexdigest()


def _iter_actual_gold_files(gold_dir: Path) -> dict[str, Path]:
    """返回实际 Gold 文档集合 {doc_id: path}，排除非 Gold 文件。"""
    result: dict[str, Path] = {}
    if not gold_dir.is_dir():
        return result
    for path in gold_dir.glob("*.json"):
        name = path.name
        # 排除 _manifest.json 等非 Gold 文件
        if name.startswith("_") or name == "_manifest.json":
            continue
        result[path.stem] = path
    return result


def verify_frozen_gold_integrity(
    gold_dir: Path,
    freeze_manifest_path: Path,
) -> Tuple[bool, str]:
    """重新计算实际 Gold 内容并与冻结清单比较。

    Returns:
        (True, "ok") 表示全部一致；
        (False, reason) 表示任一不一致，调用方必须 hard fail。
    此函数本身不抛异常（除文件缺失等读取错误返回 False），
    便于测试断言与 formal preflight 统一转为 RuntimeError。
    """
    gold_dir = Path(gold_dir)
    freeze_manifest_path = Path(freeze_manifest_path)
    if not freeze_manifest_path.is_file():
        return False, f"冻结清单不存在: {freeze_manifest_path}"
    if not gold_dir.is_dir():
        return False, f"Gold 目录不存在: {gold_dir}"
    try:
        manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"冻结清单读取失败: {exc}"

    expected_doc_hashes = manifest.get("gold_document_sha256")
    expected_aggregate = manifest.get("gold_aggregate_sha256")
    expected_count = manifest.get("gold_document_count")
    if not isinstance(expected_doc_hashes, dict) or not expected_doc_hashes:
        return False, "冻结清单缺少 gold_document_sha256 字段"
    if not expected_aggregate:
        return False, "冻结清单缺少 gold_aggregate_sha256 字段"
    if expected_count is not None and int(expected_count) != len(expected_doc_hashes):
        return (
            False,
            f"冻结清单 gold_document_count ({expected_count}) 与 "
            f"gold_document_sha256 条目数 ({len(expected_doc_hashes)}) 不一致",
        )

    actual_files = _iter_actual_gold_files(gold_dir)
    expected_ids = set(expected_doc_hashes.keys())
    actual_ids = set(actual_files.keys())

    missing = sorted(expected_ids - actual_ids)
    if missing:
        return False, f"Gold 文档缺失 {len(missing)} 篇，示例: {missing[:5]}"
    extra = sorted(actual_ids - expected_ids)
    if extra:
        return False, f"Gold 目录存在未冻结的额外文件 {len(extra)} 个，示例: {extra[:5]}"

    actual_hashes: dict[str, str] = {}
    for doc_id in sorted(actual_ids):
        try:
            actual_hashes[doc_id] = _sha256_file(actual_files[doc_id])
        except OSError as exc:
            return False, f"Gold 文档读取失败 ({doc_id}): {exc}"
        expected_hash = expected_doc_hashes.get(doc_id)
        if actual_hashes[doc_id] != expected_hash:
            return (
                False,
                f"Gold 文档内容已变更 ({doc_id}): "
                f"实际={actual_hashes[doc_id][:16]}... "
                f"冻结={str(expected_hash)[:16]}...",
            )

    recomputed_aggregate = compute_gold_aggregate_sha256(actual_hashes)
    if recomputed_aggregate != expected_aggregate:
        return (
            False,
            f"Gold 聚合哈希不一致: 实际={recomputed_aggregate} "
            f"冻结={expected_aggregate}",
        )
    return True, "ok"


def assert_frozen_gold_integrity(
    gold_dir: Path,
    freeze_manifest_path: Path,
) -> str:
    """严格断言 Gold 完整性，不一致时抛 RuntimeError。

    Returns:
        已验证的 gold_aggregate_sha256（即冻结清单中的值，此时已证明
        等于实际重算值），供调用方直接写入 cache/产物，避免“抄清单冒充验证”。
    """
    ok, message = verify_frozen_gold_integrity(gold_dir, freeze_manifest_path)
    if not ok:
        raise RuntimeError(
            f"正式 ProTeGi Gold 完整性校验失败: {message}；"
            "已在模型调用前阻断 (optimizer/model call count=0)。"
        )
    manifest = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    return str(manifest["gold_aggregate_sha256"])
