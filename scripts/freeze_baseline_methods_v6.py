"""Freeze and verify the completed v6 rule/multipass/full test artifacts.

The freeze is deliberately separate from the experiment runner.  Once the
manifest exists, run_v6_experiment.py refuses to overwrite any of the three
baseline methods.  Removing or replacing the manifest is therefore an
explicit unfreeze operation rather than an accidental consequence of
``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
RAW_ROOT = RESULTS / "raw_predictions"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
DATASET_MANIFEST = EXP_DIR / "data" / "dataset_freeze_manifest_v6.json"
STATUS_FILE = EXP_DIR / "data" / "review_status.json"
BASELINE_FREEZE_MANIFEST = (
    EXP_DIR / "data" / "baseline_methods_freeze_manifest_v1.json"
)
FROZEN_METHODS = ("rule", "multipass", "full")
DEPENDENCIES = (
    "config/runtime_profile.json",
    "scripts/run_v6_experiment.py",
    "scripts/freeze_baseline_methods_v6.py",
    "scripts/provider_config.py",
    "scripts/rule_baseline.py",
    "scripts/llm_methods.py",
    "scripts/eval_metrics.py",
    "scripts/schema.py",
    "scripts/prompts/multipass_prompts.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_hashes(directory: Path) -> tuple[dict[str, str], str]:
    hashes = {
        path.name: _sha256(path)
        for path in sorted(directory.glob("*.json"), key=lambda item: item.name)
    }
    aggregate = hashlib.sha256()
    for name, digest in hashes.items():
        aggregate.update(f"{name}\0{digest}\n".encode("utf-8"))
    return hashes, aggregate.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_completed_outputs(test_ids: list[str]) -> dict:
    """Recompute core metrics and prove Rule/Full provenance without API use."""
    os.environ.setdefault("V6_API_KEY", "v6-offline-freeze-audit")
    sys.path.insert(0, str(SCRIPT_DIR))

    import eval_metrics as EM
    import rule_baseline
    from llm_methods import apply_postprocess
    from schema import EXTRACTION_ENTITY_TYPES, EXTRACTION_RELATION_TYPES

    entity_types = set(EXTRACTION_ENTITY_TYPES)
    relation_types = set(EXTRACTION_RELATION_TYPES)
    gold_dir = EXP_DIR / "data" / "annotations" / "gold"

    def filt(record: dict) -> tuple[list[dict], list[dict]]:
        entities = [
            entity
            for entity in record.get("entities", [])
            if entity.get("type") in entity_types
        ]
        entity_ids = {entity["id"] for entity in entities}
        relations = [
            relation
            for relation in record.get("relations", [])
            if relation.get("type") in relation_types
            and relation.get("head") in entity_ids
            and relation.get("tail") in entity_ids
        ]
        return entities, relations

    validation: dict[str, dict] = {}
    for method in FROZEN_METHODS:
        prediction_dir = RAW_ROOT / f"v6_{method}"
        prediction_ids = sorted(path.stem for path in prediction_dir.glob("*.json"))
        if prediction_ids != sorted(test_ids):
            raise ValueError(
                f"{method} prediction set does not equal the frozen test split"
            )
        result_path = RESULTS / f"v6_{method}_test.json"
        result = _load_json(result_path)
        if result.get("method") != method or result.get("split") != "test":
            raise ValueError(f"invalid method/split metadata: {result_path}")
        if result.get("num_docs") != len(test_ids):
            raise ValueError(f"invalid document count: {result_path}")
        if result.get("schema_ready") is not True:
            raise ValueError(f"schema gate did not pass: {result_path}")

        entity_metrics = []
        relation_metrics = []
        normalized_relation_metrics = []
        for doc_id in test_ids:
            prediction = _load_json(prediction_dir / f"{doc_id}.json")
            gold = _load_json(gold_dir / f"{doc_id}.json")
            pred_entities, pred_relations = filt(prediction)
            gold_entities, gold_relations = filt(gold)
            entity_metrics.append(
                EM.calc_entity_metrics(pred_entities, gold_entities)
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
        recomputed = {
            "entity": EM.aggregate_metrics(entity_metrics),
            "relation": EM.aggregate_metrics(relation_metrics),
            "normalized_relation": EM.aggregate_metrics(
                normalized_relation_metrics
            ),
        }
        for key, value in recomputed.items():
            if value != result.get(key):
                raise ValueError(f"{method} {key} metrics do not reproduce")
        validation[method] = {
            "prediction_count": len(prediction_ids),
            "exact_test_document_set": True,
            "metrics_recomputed": True,
        }

    rule_mismatches = []
    full_mismatches = []
    multipass_error_stages = 0
    multipass_failed_stages = 0
    for doc_id in test_ids:
        gold = _load_json(gold_dir / f"{doc_id}.json")
        rule_prediction = _load_json(RAW_ROOT / "v6_rule" / f"{doc_id}.json")
        reproduced_rule = rule_baseline.extract(gold["text"])
        if (
            reproduced_rule.get("entities", [])
            != rule_prediction.get("entities", [])
            or reproduced_rule.get("relations", [])
            != rule_prediction.get("relations", [])
        ):
            rule_mismatches.append(doc_id)

        multipass = _load_json(
            RAW_ROOT / "v6_multipass" / f"{doc_id}.json"
        )
        for window in (multipass.get("trace") or {}).get("windows", []):
            for stage in ("stage1", "stage2"):
                trace = window.get(stage) or {}
                if trace.get("errors"):
                    multipass_error_stages += 1
                if trace.get("errors") and not trace.get("raw_response"):
                    multipass_failed_stages += 1

        full = _load_json(RAW_ROOT / "v6_full" / f"{doc_id}.json")
        reproduced_full = apply_postprocess(multipass)
        if (
            reproduced_full.get("entities", []) != full.get("entities", [])
            or reproduced_full.get("relations", []) != full.get("relations", [])
            or reproduced_full.get("_postprocess", {})
            != full.get("postprocess", {})
        ):
            full_mismatches.append(doc_id)

    if rule_mismatches:
        raise ValueError(f"Rule outputs are not reproducible: {rule_mismatches}")
    if multipass_failed_stages or multipass_error_stages:
        raise ValueError(
            "Multipass contains API error stages: "
            f"errors={multipass_error_stages}, failed={multipass_failed_stages}"
        )
    if full_mismatches:
        raise ValueError(
            "Full does not reproduce from frozen Multipass outputs: "
            f"{full_mismatches}"
        )
    validation["rule"]["deterministic_reproduction_mismatches"] = 0
    validation["multipass"].update(
        {"error_stages": 0, "failed_stage_calls": 0}
    )
    validation["full"].update(
        {
            "source_method": "multipass",
            "deterministic_postprocess_reproduction_mismatches": 0,
            "new_model_calls": 0,
        }
    )
    return validation


def create_freeze() -> dict:
    if BASELINE_FREEZE_MANIFEST.exists():
        ok, message = verify_freeze()
        if not ok:
            raise RuntimeError(
                "baseline freeze already exists but is invalid; explicit unfreeze "
                f"is required before replacement: {message}"
            )
        return _load_json(BASELINE_FREEZE_MANIFEST)

    split = _load_json(SPLIT_FILE)
    test_ids = list(split["test"])
    validation = _validate_completed_outputs(test_ids)
    methods = {}
    for method in FROZEN_METHODS:
        result_path = RESULTS / f"v6_{method}_test.json"
        result = _load_json(result_path)
        prediction_dir = RAW_ROOT / f"v6_{method}"
        prediction_hashes, aggregate_hash = _prediction_hashes(prediction_dir)
        methods[method] = {
            "result": {
                "path": result_path.relative_to(EXP_DIR).as_posix(),
                "sha256": _sha256(result_path),
            },
            "predictions": {
                "directory": prediction_dir.relative_to(EXP_DIR).as_posix(),
                "document_count": len(prediction_hashes),
                "aggregate_sha256": aggregate_hash,
                "file_sha256": prediction_hashes,
            },
            "metrics": {
                "entity": result["entity"],
                "relation": result["relation"],
                "normalized_relation": result["normalized_relation"],
            },
            "runtime_config": result["runtime_config"],
            "validation": validation[method],
        }

    dependencies = {}
    for relative in DEPENDENCIES:
        path = EXP_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"freeze dependency is missing: {path}")
        dependencies[relative] = _sha256(path)

    manifest = {
        "manifest_version": "v6-baseline-methods-freeze-v1",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "methods": list(FROZEN_METHODS),
            "split": "test",
            "document_count": len(test_ids),
            "policy": (
                "Rule, Multipass/P0 and Full test predictions and exported "
                "metrics are immutable. Full is deterministically derived from "
                "the frozen Multipass prediction batch without new model calls."
            ),
        },
        "dataset_freeze": {
            "path": DATASET_MANIFEST.relative_to(EXP_DIR).as_posix(),
            "sha256": _sha256(DATASET_MANIFEST),
        },
        "split": {
            "path": SPLIT_FILE.relative_to(EXP_DIR).as_posix(),
            "sha256": _sha256(SPLIT_FILE),
            "test_document_ids": test_ids,
        },
        "methods": methods,
        "dependencies": dependencies,
        "invariants": {
            "allow_force_overwrite": False,
            "allow_eval_only_result_rewrite": False,
            "explicit_unfreeze_required_for_rerun": True,
        },
    }
    BASELINE_FREEZE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status = _load_json(STATUS_FILE)
    status["baseline_methods_freeze"] = {
        "status": "frozen",
        "methods": list(FROZEN_METHODS),
        "split": "test",
        "document_count": len(test_ids),
        "manifest": BASELINE_FREEZE_MANIFEST.relative_to(EXP_DIR).as_posix(),
        "manifest_sha256": _sha256(BASELINE_FREEZE_MANIFEST),
        "full_reuses_multipass_predictions": True,
        "rerun_required": False,
    }
    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_freeze() -> tuple[bool, str]:
    if not BASELINE_FREEZE_MANIFEST.is_file():
        return False, "baseline freeze manifest is missing"
    try:
        manifest = _load_json(BASELINE_FREEZE_MANIFEST)
        if manifest.get("status") != "frozen":
            return False, "manifest status is not frozen"
        if tuple(manifest.get("scope", {}).get("methods", [])) != FROZEN_METHODS:
            return False, "frozen method set changed"
        for record_name in ("dataset_freeze", "split"):
            record = manifest[record_name]
            path = EXP_DIR / record["path"]
            if not path.is_file() or _sha256(path) != record["sha256"]:
                return False, f"{record_name} hash mismatch"
        for method in FROZEN_METHODS:
            record = manifest["methods"][method]
            result_path = EXP_DIR / record["result"]["path"]
            if not result_path.is_file():
                return False, f"{method} result is missing"
            if _sha256(result_path) != record["result"]["sha256"]:
                return False, f"{method} result hash mismatch"
            prediction_dir = EXP_DIR / record["predictions"]["directory"]
            hashes, aggregate = _prediction_hashes(prediction_dir)
            if hashes != record["predictions"]["file_sha256"]:
                return False, f"{method} prediction file hash mismatch"
            if aggregate != record["predictions"]["aggregate_sha256"]:
                return False, f"{method} prediction aggregate hash mismatch"
        for relative, expected in manifest.get("dependencies", {}).items():
            path = EXP_DIR / relative
            if not path.is_file() or _sha256(path) != expected:
                return False, f"dependency hash mismatch: {relative}"
        status = _load_json(STATUS_FILE).get("baseline_methods_freeze", {})
        if status.get("status") != "frozen":
            return False, "review status does not mark baselines frozen"
        if status.get("manifest_sha256") != _sha256(BASELINE_FREEZE_MANIFEST):
            return False, "review status manifest hash mismatch"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        return False, f"invalid baseline freeze manifest: {exc}"
    return True, "Rule, Multipass/P0 and Full test artifacts are frozen and intact"


def frozen_methods() -> set[str]:
    if not BASELINE_FREEZE_MANIFEST.is_file():
        return set()
    try:
        manifest = _load_json(BASELINE_FREEZE_MANIFEST)
        if manifest.get("status") != "frozen":
            return set()
        return set(manifest.get("scope", {}).get("methods", []))
    except (json.JSONDecodeError, OSError, TypeError):
        return set(FROZEN_METHODS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create",
        action="store_true",
        help="validate completed artifacts and create the one-way freeze manifest",
    )
    args = parser.parse_args()
    if args.create:
        manifest = create_freeze()
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "manifest": str(BASELINE_FREEZE_MANIFEST),
                    "methods": manifest["scope"]["methods"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    ok, message = verify_freeze()
    print(json.dumps({"ok": ok, "message": message}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
