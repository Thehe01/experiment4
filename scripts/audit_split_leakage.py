"""审计正式切分的文档组、完全重复及高相似文本泄漏。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_stratified_split import group_id


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD = EXP_DIR / "data" / "annotations" / "gold"
DEFAULT_SPLIT = EXP_DIR / "data" / "train_dev_test_split_v7.json"


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[A-Za-z0-9_.:/-]+", text.lower())
    return {
        tuple(tokens[index:index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path, help="保存审计 JSON 报告的路径")
    args = parser.parse_args()

    split_path = Path(args.split)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    partition_names = [
        name for name in ("train", "dev", "test") if name in split
    ]
    partitions = {name: split[name] for name in partition_names}
    errors = []

    for left_index, left_name in enumerate(partition_names):
        for right_name in partition_names[left_index + 1:]:
            overlap = sorted(
                set(partitions[left_name]) & set(partitions[right_name])
            )
            if overlap:
                errors.append(
                    f"{left_name}/{right_name} 文档 ID 重叠: {overlap}"
                )
            group_overlap = sorted(
                {group_id(doc_id) for doc_id in partitions[left_name]}
                & {group_id(doc_id) for doc_id in partitions[right_name]}
            )
            if group_overlap:
                errors.append(
                    f"{left_name}/{right_name} AA 文档组跨集合: {group_overlap}"
                )

    texts = {}
    fingerprints = {}
    all_ids = [
        doc_id
        for name in partition_names
        for doc_id in partitions[name]
    ]
    for doc_id in all_ids:
        data = json.loads((GOLD / f"{doc_id}.json").read_text(encoding="utf-8"))
        text = re.sub(r"\s+", " ", data.get("text", "")).strip().lower()
        texts[doc_id] = text
        fingerprints[doc_id] = shingles(text)

    exact_pairs = []
    near_pairs = []
    for left_index, left_name in enumerate(partition_names):
        for right_name in partition_names[left_index + 1:]:
            for left_id in partitions[left_name]:
                for right_id in partitions[right_name]:
                    if texts[left_id] == texts[right_id]:
                        exact_pairs.append(
                            (left_name, left_id, right_name, right_id)
                        )
                        continue
                    similarity = jaccard(
                        fingerprints[left_id],
                        fingerprints[right_id],
                    )
                    if similarity >= args.threshold:
                        near_pairs.append(
                            (
                                left_name,
                                left_id,
                                right_name,
                                right_id,
                                similarity,
                            )
                        )
    if exact_pairs:
        errors.append(f"完全重复文本跨集合: {exact_pairs}")
    if near_pairs:
        errors.append(
            "高相似文本跨集合: "
            + str([
                (left_name, left, right_name, right, round(score, 4))
                for left_name, left, right_name, right, score in near_pairs
            ])
        )

    summary = {
        "status": "passed" if not errors else "failed",
        "split_file": str(split_path),
        "threshold": args.threshold,
        "partitions": {name: len(partitions[name]) for name in partition_names},
        "exact_cross_pairs": len(exact_pairs),
        "near_cross_pairs": len(near_pairs),
        "errors": errors,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        " ".join(
            f"{name}={len(partitions[name])}" for name in partition_names
        )
        + " "
        + f"exact_cross_pairs={len(exact_pairs)} "
        f"near_cross_pairs={len(near_pairs)} "
        f"threshold={args.threshold}"
    )
    for error in errors:
        print("ERROR:", error)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

