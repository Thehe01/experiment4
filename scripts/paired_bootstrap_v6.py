"""Paired document bootstrap for the frozen v6 test predictions.

This script is read-only with respect to Gold and prediction files.  It samples
the same test documents for every method in each replicate, so uncertainty is
estimated at the document level without treating individual mentions as
independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from eval_metrics import (
    extract_entity_spans,
    extract_normalized_relation_facts,
    extract_relation_tuples,
)


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "paired_bootstrap_v6.json"
METHODS = ("rule", "multipass", "full", "protegi")
METRICS = ("entity", "strict_relation", "normalized_fact")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.name}\0{_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _items(annotation: dict, metric: str) -> set:
    entities = annotation.get("entities", [])
    relations = annotation.get("relations", [])
    if metric == "entity":
        return extract_entity_spans(entities)
    if metric == "strict_relation":
        return extract_relation_tuples(entities, relations)
    if metric == "normalized_fact":
        return extract_normalized_relation_facts(entities, relations)
    raise ValueError(f"unsupported metric: {metric}")


def _counts(prediction: dict, gold: dict, metric: str) -> tuple[int, int, int]:
    pred_items = _items(prediction, metric)
    gold_items = _items(gold, metric)
    return (
        len(pred_items & gold_items),
        len(pred_items - gold_items),
        len(gold_items - pred_items),
    )


def _f1(counts: tuple[int, int, int]) -> float:
    tp, fp, fn = counts
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _sum_counts(
    per_document: list[tuple[int, int, int]], indices: list[int] | None = None
) -> tuple[int, int, int]:
    selected = range(len(per_document)) if indices is None else indices
    tp = fp = fn = 0
    for index in selected:
        doc_tp, doc_fp, doc_fn = per_document[index]
        tp += doc_tp
        fp += doc_fp
        fn += doc_fn
    return tp, fp, fn


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        round(_percentile(ordered, 0.025), 6),
        round(_percentile(ordered, 0.975), 6),
    ]


def _paired_randomization_p_value(
    reference_counts: list[tuple[int, int, int]],
    method_counts: list[tuple[int, int, int]],
    *,
    observed_difference: float,
    resamples: int,
    seed: int,
) -> float:
    """文档内随机交换方法标签，计算双侧配对随机化 p 值。"""
    randomizer = random.Random(seed)
    extreme = 0
    threshold = abs(observed_difference)
    for _ in range(resamples):
        randomized_reference = []
        randomized_method = []
        for ref_count, method_count in zip(reference_counts, method_counts):
            if randomizer.getrandbits(1):
                randomized_reference.append(ref_count)
                randomized_method.append(method_count)
            else:
                randomized_reference.append(method_count)
                randomized_method.append(ref_count)
        difference = _f1(_sum_counts(randomized_reference)) - _f1(
            _sum_counts(randomized_method)
        )
        if abs(difference) >= threshold - 1e-15:
            extreme += 1
    return (extreme + 1) / (resamples + 1)


def analyze(
    *,
    methods: tuple[str, ...] = METHODS,
    reference: str = "protegi",
    resamples: int = 10000,
    seed: int = 20260913,
) -> dict:
    if resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if reference not in methods:
        raise ValueError("reference method must be included in methods")
    if reference not in methods:
        raise ValueError("reference method must be included in methods")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")

    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    doc_ids = list(split["test"])
    gold_paths = [GOLD_DIR / f"{doc_id}.json" for doc_id in doc_ids]
    annotations: dict[str, dict[str, dict]] = {"gold": {}}
    for doc_id, path in zip(doc_ids, gold_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        annotations["gold"][doc_id] = json.loads(path.read_text(encoding="utf-8"))

    input_provenance = {
        "split_file": str(SPLIT_FILE),
        "split_file_sha256": _sha256(SPLIT_FILE),
        "gold_aggregate_sha256": _aggregate_sha256(gold_paths),
        "prediction_aggregate_sha256": {},
    }
    for method in methods:
        prediction_dir = RESULTS_DIR / "raw_predictions" / f"v6_{method}"
        prediction_paths = [prediction_dir / f"{doc_id}.json" for doc_id in doc_ids]
        missing = [str(path) for path in prediction_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {method} predictions: {missing}")
        annotations[method] = {
            doc_id: json.loads(path.read_text(encoding="utf-8"))
            for doc_id, path in zip(doc_ids, prediction_paths)
        }
        input_provenance["prediction_aggregate_sha256"][method] = (
            _aggregate_sha256(prediction_paths)
        )

    per_document: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for method in methods:
        per_document[method] = {}
        for metric in METRICS:
            per_document[method][metric] = [
                _counts(
                    annotations[method][doc_id],
                    annotations["gold"][doc_id],
                    metric,
                )
                for doc_id in doc_ids
            ]

    randomizer = random.Random(seed)
    replicate_indices = [
        [randomizer.randrange(len(doc_ids)) for _ in doc_ids]
        for _ in range(resamples)
    ]
    bootstrap_values: dict[str, dict[str, list[float]]] = {
        method: {
            metric: [
                _f1(_sum_counts(per_document[method][metric], indices))
                for indices in replicate_indices
            ]
            for metric in METRICS
        }
        for method in methods
    }

    method_results = {}
    for method in methods:
        method_results[method] = {}
        for metric in METRICS:
            totals = _sum_counts(per_document[method][metric])
            method_results[method][metric] = {
                "point_f1": round(_f1(totals), 6),
                "bootstrap_95_ci": _interval(bootstrap_values[method][metric]),
                "tp": totals[0],
                "fp": totals[1],
                "fn": totals[2],
            }

    comparisons = {}
    for method in methods:
        if method == reference:
            continue
        comparisons[f"{reference}_minus_{method}"] = {}
        for metric in METRICS:
            differences = [
                reference_value - method_value
                for reference_value, method_value in zip(
                    bootstrap_values[reference][metric],
                    bootstrap_values[method][metric],
                )
            ]
            point_difference = (
                method_results[reference][metric]["point_f1"]
                - method_results[method][metric]["point_f1"]
            )
            strictly_greater = sum(diff > 0 for diff in differences)
            strictly_less = sum(diff < 0 for diff in differences)
            ties = resamples - strictly_greater - strictly_less
            randomization_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{reference}:{method}:{metric}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            p_value = _paired_randomization_p_value(
                per_document[reference][metric],
                per_document[method][metric],
                observed_difference=point_difference,
                resamples=resamples,
                seed=randomization_seed,
            )
            comparisons[f"{reference}_minus_{method}"][metric] = {
                "point_difference": round(point_difference, 6),
                "bootstrap_95_ci": _interval(differences),
                "p_value_two_sided": round(min(1.0, p_value), 6),
                "p_value_method": "paired_document_label_randomization_plus_one",
                "probability_reference_greater": round(
                    (strictly_greater + 0.5 * ties) / resamples, 6
                ),
                "bootstrap_tie_probability": round(ties / resamples, 6),
            }

    return {
        "analysis": "paired_document_bootstrap_v6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resamples": resamples,
        "seed": seed,
        "document_count": len(doc_ids),
        "reference_method": reference,
        "methods": list(methods),
        "input_provenance": input_provenance,
        "method_results": method_results,
        "comparisons_against_reference": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
    )
    parser.add_argument("--reference", default="protegi", choices=list(METHODS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    report = analyze(
        methods=tuple(args.methods),
        reference=args.reference,
        resamples=args.resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "methods": report["methods"],
                "reference_method": report["reference_method"],
                "resamples": report["resamples"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
