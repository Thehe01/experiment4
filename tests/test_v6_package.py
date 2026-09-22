"""v6 package boundary checks; no model or network calls are made."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE / "scripts") not in sys.path:
    sys.path.insert(0, str(PACKAGE / "scripts"))
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

GOLD = PACKAGE / "data" / "annotations" / "gold"
SPLIT = PACKAGE / "data" / "train_dev_test_split_v7.json"
MANIFEST = PACKAGE / "data" / "dataset_freeze_manifest_v6.json"
STATUS = PACKAGE / "data" / "review_status.json"
BASELINE_FREEZE = PACKAGE / "data" / "baseline_methods_freeze_manifest_v6.json"
PILOT_SPLIT = PACKAGE / "data" / "protegi_prompt_scope_pilot_v1.json"
PILOT_CONSTRAINED = PACKAGE / "protegi" / "configs" / "protegi_pilot_constrained.yaml"
PILOT_UNCONSTRAINED = PACKAGE / "protegi" / "configs" / "protegi_pilot_unconstrained.yaml"
CONFIGURATION_PILOT_SPLIT = PACKAGE / "data" / "protegi_configuration_pilot_v1.json"
CONFIGURATION_PILOT_CONFIG = (
    PACKAGE / "protegi" / "configs" / "protegi_configuration_pilot_v1.yaml"
)
MCPU_AUDIT = PACKAGE / "results" / "configuration_boundary_mcpu_v2_audit.json"
MCPU_RECEIPT = PACKAGE / "results" / "configuration_boundary_mcpu_v2_receipt.json"


class V6PackageTest(unittest.TestCase):
    def test_configuration_mcpu_v2_is_complete_and_regression_bound(self):
        audit = json.loads(MCPU_AUDIT.read_text(encoding="utf-8"))
        receipt = json.loads(MCPU_RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(audit["summary"]["configuration_mentions"], 406)
        self.assertEqual(audit["summary"]["blocking_errors"], 0)
        self.assertEqual(audit["summary"]["gate_status"], "gate_passed")
        self.assertFalse(receipt["prediction_outputs_used"])
        self.assertEqual(len(receipt["applied"]), 17)

        telerik = json.loads(
            (GOLD / "aa23-074a-telerik-cve-18935.json").read_text(
                encoding="utf-8"
            )
        )
        telerik_entity = next(
            entity for entity in telerik["entities"] if entity["id"] == "E81"
        )
        self.assertEqual(
            telerik_entity["text"],
            "Progress Telerik user interface (UI) for ASP.NET AJAX",
        )

        coldfusion = json.loads(
            (GOLD / "aa23-339a-coldfusion-cve-26360.json").read_text(
                encoding="utf-8"
            )
        )
        by_id = {entity["id"]: entity for entity in coldfusion["entities"]}
        self.assertEqual(by_id["E3"]["text"], "Adobe ColdFusion")
        for entity_id in ("E3", "E4", "E5", "E44", "E45"):
            self.assertEqual(
                by_id[entity_id]["normalized_id"],
                "cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*",
            )

    def test_configuration_pilot_is_fresh_targeted_and_test_free(self):
        sys.path.insert(0, str(PACKAGE))
        from protegi.document_context import extract_explicit_abbreviation_pairs
        from protegi.optimizer import prepare_stage1_window_samples

        official = json.loads(SPLIT.read_text(encoding="utf-8"))
        manifest = json.loads(
            CONFIGURATION_PILOT_SPLIT.read_text(encoding="utf-8")
        )
        config = yaml.safe_load(
            CONFIGURATION_PILOT_CONFIG.read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["test"], [])
        self.assertEqual(
            manifest["parent_split_sha256"],
            hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        )
        self.assertTrue(set(manifest["train"]).issubset(set(official["train"])))
        self.assertTrue(set(manifest["dev"]).issubset(set(official["dev"])))
        self.assertFalse(
            set(manifest["dev"])
            & set(manifest["observed_diagnostic_dev_excluded"])
        )
        self.assertFalse(
            (set(manifest["train"]) | set(manifest["dev"]))
            & set(official["test"])
        )

        for split_name in ("train", "dev"):
            samples = prepare_stage1_window_samples(
                manifest[split_name],
                GOLD,
                include_document_abbreviations=True,
            )
            actual_window_entities = {
                "Vulnerability": 0,
                "Configuration": 0,
                "Weakness": 0,
                "AttackTechnique": 0,
            }
            for sample in samples:
                for entity in sample["gold_entities"]:
                    actual_window_entities[entity["type"]] += 1
            expected = manifest["audit_counts"][split_name]
            self.assertEqual(len(samples), expected["runtime_windows"])
            self.assertEqual(
                actual_window_entities,
                expected["window_entities"],
            )
            actual_pairs = 0
            for doc_id in manifest[split_name]:
                document = json.loads(
                    (GOLD / f"{doc_id}.json").read_text(encoding="utf-8")
                )
                actual_pairs += len(
                    extract_explicit_abbreviation_pairs(document["text"])
                )
            self.assertEqual(
                actual_pairs,
                expected["explicit_abbreviation_pairs"],
            )

        self.assertEqual(config["prompt_scope"], "constrained")
        self.assertTrue(config["document_abbreviation_context"])
        self.assertFalse(config["vulnerability_anchored_backfill"])
        self.assertFalse(
            manifest["document_context"]["automatic_entity_backfill"]
        )
        self.assertEqual(config["selection_entity_type"], "Configuration")
        self.assertEqual(config["error_focus_entity_type"], "Configuration")
        self.assertEqual(config["final_selection_entity_type"], "Configuration")
        self.assertTrue(config["include_p0_in_final_selection"])
        self.assertEqual(config["task_model"], "hy3")
        self.assertEqual(config["task_max_workers"], 8)
        self.assertEqual(config["optimizer_model"], "muse-spark-1.3-contributor")

    def test_prompt_scope_pilot_is_matched_and_test_free(self):
        from protegi.optimizer import prepare_stage1_window_samples

        official = json.loads(SPLIT.read_text(encoding="utf-8"))
        pilot = json.loads(PILOT_SPLIT.read_text(encoding="utf-8"))
        self.assertEqual(pilot["manifest_version"], "protegi-prompt-scope-pilot-v1")
        self.assertEqual(pilot["test"], [])
        self.assertEqual(
            pilot["parent_split_sha256"],
            hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        )
        self.assertTrue(set(pilot["train"]).issubset(set(official["train"])))
        self.assertTrue(set(pilot["dev"]).issubset(set(official["dev"])))

        for split_name in ("train", "dev"):
            type_counts = {
                "Vulnerability": 0,
                "Configuration": 0,
                "Weakness": 0,
                "AttackTechnique": 0,
            }
            source_characters = 0
            for doc_id in pilot[split_name]:
                document = json.loads(
                    (GOLD / f"{doc_id}.json").read_text(encoding="utf-8")
                )
                text_length = len(document.get("text", ""))
                source_characters += text_length
                for entity in document.get("entities", []):
                    if entity.get("type") in type_counts:
                        type_counts[entity["type"]] += 1
            self.assertEqual(
                type_counts,
                pilot["audit_counts"][split_name]["entities"],
            )
            self.assertTrue(all(count > 0 for count in type_counts.values()))
            self.assertEqual(
                source_characters,
                pilot["audit_counts"][split_name]["source_characters"],
            )
            runtime_windows = len(
                prepare_stage1_window_samples(pilot[split_name], GOLD)
            )
            self.assertEqual(
                runtime_windows,
                pilot["audit_counts"][split_name]["runtime_windows"],
            )

        constrained = yaml.safe_load(PILOT_CONSTRAINED.read_text(encoding="utf-8"))
        unconstrained = yaml.safe_load(PILOT_UNCONSTRAINED.read_text(encoding="utf-8"))
        self.assertEqual(constrained.pop("prompt_scope"), "constrained")
        self.assertEqual(unconstrained.pop("prompt_scope"), "unconstrained")
        self.assertEqual(constrained, unconstrained)
        self.assertEqual(constrained["task_max_workers"], 8)
        self.assertEqual(constrained["eval_batch_size"], 8)

    def test_dataset_shape_and_partition(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        split = json.loads(SPLIT.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["dataset_version"],
            "v6-v5-gold-v9-mcpu-v2",
        )
        self.assertEqual(
            manifest["boundary_contract_version"],
            "chapter3-boundary-sync-v2",
        )
        self.assertEqual(manifest["gold_directory"], "data/annotations/gold")
        self.assertEqual(manifest["gold_document_count"], 105)
        self.assertEqual({key: len(split[key]) for key in ("train", "dev", "test")},
                         {"train": 63, "dev": 21, "test": 21})
        doc_ids = set().union(*(set(split[key]) for key in ("train", "dev", "test")))
        self.assertEqual(doc_ids,
                         set(manifest["gold_document_sha256"]))
        self.assertEqual(len(list(GOLD.glob("*.json"))), 105)

    def test_frozen_hashes_and_evidence(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        for doc_id in sorted(manifest["gold_document_sha256"]):
            path = GOLD / f"{doc_id}.json"
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(current, manifest["gold_document_sha256"][doc_id])
            digest.update(f"{doc_id}\0{current}\n".encode("utf-8"))
        self.assertEqual(digest.hexdigest(), manifest["gold_aggregate_sha256"])
        for name, record in manifest["supporting_evidence"].items():
            path = PACKAGE / record["path"]
            self.assertTrue(path.is_file(), path)
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(current, record["sha256"])

        accepted = []
        for path in GOLD.glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            for relation in document["relations"]:
                if relation["type"] != "exploited_by":
                    continue
                accepted.append((document["doc_id"], relation))
                self.assertIn(
                    "chapter3-boundary-sync-v2",
                    relation.get("adjudication_basis", ""),
                )
                self.assertNotIn(
                    "verified single-endpoint factual clause",
                    relation.get("adjudication_basis", ""),
                )
        self.assertEqual(len(accepted), 54)
        aa24_290a = json.loads(
            (GOLD / "aa24-290a.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("R6", {item["id"] for item in aa24_290a["relations"]})
        quarantine_record = manifest["supporting_evidence"]["boundary_quarantine"]
        quarantine = json.loads(
            (PACKAGE / quarantine_record["path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(quarantine["summary"]["rejected_ledger_items"], 62)
        self.assertEqual(quarantine["summary"]["cascade_removed_instantiates"], 47)

    def test_cpe_gold_audit_is_complete_and_bound(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        evidence = manifest["supporting_evidence"]
        audit_path = PACKAGE / evidence["cpe_audit"]["path"]
        receipt_path = PACKAGE / evidence["cpe_review_receipt"]["path"]
        quarantine_path = PACKAGE / evidence["cpe_quarantine"]["path"]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))

        self.assertFalse(audit["prediction_outputs_used"])
        self.assertEqual(audit["summary"]["gate_status"], "gate_passed")
        self.assertEqual(audit["summary"]["unresolved_items"], 0)
        self.assertEqual(audit["summary"]["configuration_mentions"], 406)
        # The v1 receipt is historical provenance for the applied CPE decisions.
        # The current audit is rebound independently by the v2 freeze manifest
        # after the MCPU family-level normalization pass.
        self.assertEqual(
            hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            evidence["cpe_audit"]["sha256"],
        )
        self.assertEqual(
            receipt["application"]["action_counts"],
            {"KEEP": 6, "REJECT": 2, "REVISE_CPE": 50},
        )
        self.assertEqual(quarantine["rejected_count"], 2)
        self.assertEqual(
            sum(len(item["relations"]) for item in quarantine["items"]), 3
        )

    def test_runner_gate_and_apo_block(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        sys.path.insert(0, str(PACKAGE / "scripts"))
        import run_v6_experiment as runner

        ready, message = runner._review_ready(SPLIT)
        self.assertFalse(ready)
        self.assertIn("未批准受控重跑", message)
        status = json.loads(STATUS.read_text(encoding="utf-8"))
        self.assertFalse(status["apo_prompt_status"]["ready"])
        self.assertEqual(
            status["apo_prompt_status"]["artifact"],
            "results/protegi_optimization/entity_protegi/final_entity_prompt.txt",
        )
        with self.assertRaises(RuntimeError):
            runner.run("apo", split="test", eval_only=True)

    def test_chapter3_boundary_rules_are_enforced(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        sys.path.insert(0, str(PACKAGE / "scripts"))
        from llm_methods import apply_postprocess, validate_apo_guidance

        text = "CVE-2024-1000 has an authentication bypass vulnerability."
        start = text.index("authentication bypass")
        raw = {
            "text": text,
            "entities": [{
                "id": "E1",
                "text": "authentication bypass",
                "type": "Weakness",
                "start": start,
                "end": start + len("authentication bypass"),
                "normalized_id": "CWE-287",
            }],
            "relations": [],
        }
        result = apply_postprocess(raw)
        self.assertEqual(result["entities"], [])
        self.assertEqual(
            result["_postprocess"]["rejected"][
                "entity_weakness_without_local_explicit_cwe"
            ],
            1,
        )

        text = "CVE-2024-1000 has server-side request forgery (CWE-918)."
        surface = "server-side request forgery"
        start = text.index(surface)
        raw = {
            "text": text,
            "entities": [{
                "id": "E1",
                "text": surface,
                "type": "Weakness",
                "start": start,
                "end": start + len(surface),
                "normalized_id": "CWE-918",
            }],
            "relations": [],
        }
        result = apply_postprocess(raw)
        self.assertEqual(result["entities"][0]["normalized_id"], "CWE-918")

        text = (
            "Application Layer Protocol T1071.001: attackers exploited "
            "CVE-2022-21587 and then ran a RAT for C2."
        )
        cve = "CVE-2022-21587"
        technique = "T1071.001"
        raw = {
            "text": text,
            "entities": [
                {
                    "id": "E1", "text": cve, "type": "Vulnerability",
                    "start": text.index(cve), "end": text.index(cve) + len(cve),
                    "normalized_id": cve,
                },
                {
                    "id": "E2", "text": technique, "type": "AttackTechnique",
                    "start": text.index(technique),
                    "end": text.index(technique) + len(technique),
                    "normalized_id": technique,
                },
            ],
            "relations": [{
                "id": "R1", "type": "exploited_by", "head": "E1", "tail": "E2",
                "evidence": text, "evidence_start": 0, "evidence_end": len(text),
            }],
        }
        result = apply_postprocess(raw)
        self.assertEqual(result["relations"], [])
        self.assertGreaterEqual(
            result["_postprocess"]["rejected"][
                "exploited_by_explicit_post_exploitation_sequence"
            ],
            1,
        )

        validate_apo_guidance(
            "Preserve an adjacent vendor only when it is in the same continuous product noun phrase.",
            "",
        )
        with self.assertRaisesRegex(ValueError, "boundary_conflicts"):
            validate_apo_guidance(
                "Always remove the vendor prefix from every Configuration.", ""
            )
        with self.assertRaisesRegex(ValueError, "boundary_conflicts"):
            validate_apo_guidance(
                "", "Use the maximal verbatim tail phrase for affects."
            )

    def test_evidence_ambiguity_counts_invalid_relations(self):
        sys.path.insert(0, str(PACKAGE / "scripts"))
        from eval_metrics import calc_evidence_ambiguity_metrics

        metrics = calc_evidence_ambiguity_metrics(
            [{"id": "E1", "type": "Vulnerability", "start": 0, "end": 3}],
            [{"id": "R1", "type": "affects", "head": "E1", "tail": "missing"}],
            "CVE",
        )
        self.assertEqual(metrics["total"], 1)
        self.assertEqual(metrics["ambiguous"], 1)
        self.assertEqual(metrics["rate"], 1.0)

    def test_conditional_relation_recall_separates_missing_endpoints(self):
        sys.path.insert(0, str(PACKAGE / "scripts"))
        from apo_metrics import conditional_relation_recall_by_type

        gold_entities = [
            {"id": "G1", "type": "Vulnerability", "start": 0, "end": 3},
            {"id": "G2", "type": "Configuration", "start": 4, "end": 7},
            {"id": "G3", "type": "Configuration", "start": 8, "end": 11},
        ]
        gold_relations = [
            {"id": "R1", "type": "affects", "head": "G1", "tail": "G2"},
            {"id": "R2", "type": "affects", "head": "G1", "tail": "G3"},
        ]
        pred_entities = [
            {"id": "P1", "type": "Vulnerability", "start": 0, "end": 3},
            {"id": "P2", "type": "Configuration", "start": 4, "end": 7},
        ]
        pred_relations = [
            {"id": "P_R1", "type": "affects", "head": "P1", "tail": "P2"}
        ]
        metrics = conditional_relation_recall_by_type(
            pred_entities,
            pred_relations,
            gold_entities,
            gold_relations,
            endpoint_entities=pred_entities,
        )["affects"]
        self.assertEqual(metrics["conditional_recall"], 1.0)
        self.assertEqual(metrics["eligible_gold"], 1)
        self.assertEqual(metrics["endpoint_missing"], 1)

    def test_stage1_all_entity_recall_preset_contract(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        sys.path.insert(0, str(PACKAGE / "scripts"))
        import apo_optimizer as optimizer
        self.assertEqual(optimizer.OPTIMIZER_REASONING_EFFORT, "xhigh")
        self.assertEqual(optimizer.EDITOR_REASONING_EFFORT, "xhigh")
        self.assertEqual(optimizer.CRITIC_FALLBACK_REASONING_EFFORT, "xhigh")

        preset = optimizer.APO_PRESETS["stage1_all_entity_recall_p0"]
        self.assertEqual(
            preset["phase_rounds"],
            {"entity": 4, "relation": 0, "joint": 0},
        )
        self.assertEqual(preset["objective_focus"], "all_entity_recall")
        self.assertFalse(preset["freeze_artifact"])
        self.assertEqual(
            [
                optimizer._atomic_target_for_round(
                    "entity", index, "all_entity_recall"
                )
                for index in range(1, 5)
            ],
            [
                ("Configuration",),
                ("Weakness",),
                ("Vulnerability",),
                ("AttackTechnique",),
            ],
        )

        by_type = {
            label: {"precision": 0.5, "recall": recall, "f1": 0.6}
            for label, recall in zip(
                optimizer.ENTITY_LABELS, (0.6, 0.7, 0.8, 0.9)
            )
        }
        expected_macro_f2 = optimizer._macro_fbeta(
            by_type, optimizer.ENTITY_LABELS, 2.0
        )
        record = {
            "stage1_guidance": "",
            "stage2_guidance": "",
            "metrics": {
                "candidate_entity_by_type": by_type,
                "candidate_entity_macro_f2": expected_macro_f2,
            },
        }
        self.assertEqual(
            optimizer._phase_score(record, "entity", "all_entity_recall"),
            expected_macro_f2,
        )
        value, metric_name = optimizer._target_value(
            record["metrics"], "Weakness", "all_entity_recall", "entity"
        )
        self.assertEqual(metric_name, "candidate_f2")
        self.assertEqual(value, optimizer._fbeta(by_type["Weakness"], 2.0))

        stage1_only_prediction = {
            "_trace": {
                "windows": [
                    {
                        "stage1_only": True,
                        "stage1": {"raw_response": '{"entities": []}'},
                        "stage2": {"skipped": True},
                    }
                ]
            }
        }
        self.assertTrue(
            optimizer._prediction_cacheable(stage1_only_prediction)
        )
        recall_guardrails = {
            "candidate_entity_recall_min": {
                label: metric["recall"] for label, metric in by_type.items()
            }
        }
        self.assertTrue(
            optimizer._passes_all_entity_recall_guardrails(
                {"candidate_entity_by_type": by_type}, recall_guardrails
            )
        )
        degraded = json.loads(json.dumps(by_type))
        degraded["Weakness"]["recall"] -= 0.01
        self.assertFalse(
            optimizer._passes_all_entity_recall_guardrails(
                {"candidate_entity_by_type": degraded}, recall_guardrails
            )
        )

    def test_configuration_recall_only_preset_contract(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        sys.path.insert(0, str(PACKAGE / "scripts"))
        import apo_optimizer as optimizer

        preset = optimizer.APO_PRESETS["configuration_recall_only_p0"]
        self.assertEqual(
            preset["phase_rounds"],
            {"entity": 2, "relation": 0, "joint": 0},
        )
        self.assertEqual(
            preset["objective_focus"], "configuration_recall_only"
        )
        self.assertFalse(preset["full_dev_recheck"])
        self.assertFalse(preset["freeze_artifact"])

        round1_policy = (
            "For Configuration, emit every later acronym or short form "
            "occurrence at its own source offset, even when the same product "
            "appeared earlier."
        )
        optimizer._validate_configuration_recall_candidate_policy(
            round1_policy, "repeated_alias_occurrence", 1
        )
        with self.assertRaisesRegex(ValueError, "precision_filter"):
            optimizer._validate_configuration_recall_candidate_policy(
                round1_policy + " Reject unrelated tools.",
                "repeated_alias_occurrence",
                1,
            )

    def test_test_gate_blocks_before_predictor(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        import run_v6_experiment as runner

        predictor_call_count = 0

        def fake_predictor(text, doc_id):
            nonlocal predictor_call_count
            predictor_call_count += 1
            return {"entities": [], "relations": []}

        original_predictor = runner.PREDICTORS.get("protegi")
        runner.PREDICTORS["protegi"] = fake_predictor
        try:
            with self.assertRaises(RuntimeError) as ctx:
                runner.run("protegi", split="test")
            self.assertIn("受控 final test 门禁已阻断运行", str(ctx.exception))
            self.assertEqual(predictor_call_count, 0)
        finally:
            if original_predictor is not None:
                runner.PREDICTORS["protegi"] = original_predictor

    def test_all_split_requires_test_gate(self):
        os.environ.setdefault("V6_API_KEY", "v6-offline-test-key")
        import run_v6_experiment as runner

        predictor_call_count = 0

        def fake_predictor(text, doc_id):
            nonlocal predictor_call_count
            predictor_call_count += 1
            return {"entities": [], "relations": []}

        original_predictor = runner.PREDICTORS.get("protegi")
        runner.PREDICTORS["protegi"] = fake_predictor
        try:
            with self.assertRaises(RuntimeError) as ctx:
                runner.run("protegi", split="all")
            self.assertIn("受控 final test 门禁已阻断运行", str(ctx.exception))
            self.assertEqual(predictor_call_count, 0)
        finally:
            if original_predictor is not None:
                runner.PREDICTORS["protegi"] = original_predictor

    def test_train_and_dev_do_not_require_final_test_gate(self):
        import run_v6_experiment as runner

        try:
            runner._assert_run_allowed("train", SPLIT)
            runner._assert_run_allowed("dev", SPLIT)
        except RuntimeError as exc:
            self.fail(f"_assert_run_allowed unexpectedly blocked train/dev split: {exc}")

    def test_independent_repo_freeze_manifest_path_is_valid(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        base_manifest_info = manifest.get("base_manifest", {})
        base_path_str = base_manifest_info.get("path")
        self.assertEqual(base_path_str, "data/dataset_freeze_manifest_v5.json")
        base_path = PACKAGE / base_path_str
        self.assertTrue(
            base_path.is_file(),
            f"Base manifest file does not exist at independent repo path: {base_path}",
        )
        self.assertEqual(
            hashlib.sha256(base_path.read_bytes()).hexdigest(),
            base_manifest_info.get("sha256"),
        )

    def test_run_rule_on_dev_allowed_and_not_blocked_by_baseline_freeze(self):
        import run_v6_experiment as runner

        try:
            runner._block_frozen_baseline_overwrite("rule", split="dev")
            runner._block_frozen_baseline_overwrite("rule", split="train")
            runner._block_frozen_baseline_overwrite("multipass", split="dev")
        except RuntimeError as exc:
            self.fail(f"_block_frozen_baseline_overwrite unexpectedly blocked dev/train: {exc}")

    def test_test_gate_blocks_even_if_review_status_tampered_when_freeze_manifest_disallows(self):
        import run_v6_experiment as runner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_status = Path(tmpdir) / "review_status.json"
            status_data = json.loads(STATUS.read_text(encoding="utf-8"))
            status_data["controlled_test_rerun_ready"] = True
            tmp_status.write_text(json.dumps(status_data), encoding="utf-8")

            old_status = runner.REVIEW_STATUS_FILE
            runner.REVIEW_STATUS_FILE = tmp_status
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    runner._assert_run_allowed("test", SPLIT)
                self.assertIn("未批准受控重跑", str(ctx.exception))
            finally:
                runner.REVIEW_STATUS_FILE = old_status

    def test_freeze_dataset_manifest_check_is_read_only(self):
        manifest_bytes_before = MANIFEST.read_bytes()
        hash_before = hashlib.sha256(manifest_bytes_before).hexdigest()

        import subprocess
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(PACKAGE / "scripts" / "freeze_dataset_manifest_v6.py"), "--check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PACKAGE),
        )
        self.assertEqual(result.returncode, 0, f"freeze check failed: {result.stderr}")
        self.assertIn('"check_status": "passed"', result.stdout)

        manifest_bytes_after = MANIFEST.read_bytes()
        hash_after = hashlib.sha256(manifest_bytes_after).hexdigest()
        self.assertEqual(hash_before, hash_after, "freeze manifest mutated during read-only --check!")


if __name__ == "__main__":
    unittest.main()
