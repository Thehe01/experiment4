"""Offline preflight for the Configuration-targeted ProTeGi pilot.

No model client is created and no network access is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protegi.document_context import extract_explicit_abbreviation_pairs
from protegi.optimizer import prepare_stage1_window_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "protegi_configuration_pilot_v1.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "protegi" / "configs" / "protegi_configuration_pilot_v1.yaml",
    )
    parser.add_argument(
        "--official-split",
        type=Path,
        default=ROOT / "data" / "train_dev_test_split_v7.json",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=ROOT / "data" / "annotations" / "gold",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _actual_counts(doc_ids: list[str], gold_dir: Path) -> dict:
    entity_types = [
        "Vulnerability",
        "Configuration",
        "Weakness",
        "AttackTechnique",
    ]
    document_entities = {entity_type: 0 for entity_type in entity_types}
    source_characters = 0
    abbreviation_pairs = 0
    for doc_id in doc_ids:
        document = json.loads(
            (gold_dir / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        text = document.get("text", "")
        source_characters += len(text)
        abbreviation_pairs += len(extract_explicit_abbreviation_pairs(text))
        for entity in document.get("entities", []):
            if entity.get("type") in document_entities:
                document_entities[entity["type"]] += 1

    samples = prepare_stage1_window_samples(
        doc_ids,
        gold_dir,
        include_document_abbreviations=True,
    )
    windows_by_document = {
        doc_id: sum(sample["doc_id"] == doc_id for sample in samples)
        for doc_id in doc_ids
    }
    window_entities = {entity_type: 0 for entity_type in entity_types}
    for sample in samples:
        for entity in sample.get("gold_entities", []):
            if entity.get("type") in window_entities:
                window_entities[entity["type"]] += 1

    return {
        "documents": len(doc_ids),
        "source_characters": source_characters,
        "runtime_windows": len(samples),
        "runtime_windows_by_document": windows_by_document,
        "document_entities": document_entities,
        "window_entities": window_entities,
        "explicit_abbreviation_pairs": abbreviation_pairs,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    official = json.loads(args.official_split.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    actual_hash = hashlib.sha256(args.official_split.read_bytes()).hexdigest()
    if manifest.get("parent_split_sha256") != actual_hash:
        raise ValueError("official split hash does not match pilot manifest")
    if manifest.get("test") != []:
        raise ValueError("pilot test list must be empty")
    if not set(manifest["train"]).issubset(set(official["train"])):
        raise ValueError("pilot Train contains a document outside official Train")
    if not set(manifest["dev"]).issubset(set(official["dev"])):
        raise ValueError("pilot Dev contains a document outside official Dev")
    if set(manifest["dev"]) & set(manifest["observed_diagnostic_dev_excluded"]):
        raise ValueError("fresh Dev overlaps the previously observed diagnostic Dev")
    if (set(manifest["train"]) | set(manifest["dev"])) & set(official["test"]):
        raise ValueError("pilot touches the official Test partition")

    expected_config = {
        "prompt_scope": "constrained",
        "document_abbreviation_context": True,
        "selection_entity_type": "Configuration",
        "error_focus_entity_type": "Configuration",
        "final_selection_entity_type": "Configuration",
        "include_p0_in_final_selection": True,
        "task_model": "hy3",
        "task_max_workers": 8,
        "optimizer_model": "muse-spark-1.3-contributor",
        "eval_batch_size": 8,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key} must equal {expected!r}")
    if config.get("guardrail_entity_types") != [
        "Vulnerability",
        "Weakness",
        "AttackTechnique",
    ]:
        raise ValueError("guardrail_entity_types do not match the protocol")
    if float(config.get("guardrail_max_f1_drop", -1)) != 0.02:
        raise ValueError("guardrail_max_f1_drop must equal preregistered 0.02")

    actual = {}
    for split_name in ("train", "dev"):
        actual[split_name] = _actual_counts(
            list(manifest[split_name]), args.gold_dir
        )
        if actual[split_name] != manifest["audit_counts"][split_name]:
            raise ValueError(
                f"{split_name} runtime counts differ from manifest: "
                f"{actual[split_name]!r}"
            )

    if args.output_dir and args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")

    print(json.dumps({
        "status": "preflight_passed",
        "model_calls": 0,
        "network_calls": 0,
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "counts": actual,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
