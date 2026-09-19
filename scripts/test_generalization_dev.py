"""Generalization test for Stage 1 on fresh, unobserved Dev documents.

This script evaluates Stage 1 extraction (hy3 task model) on a fresh set of
Dev documents that were never seen or used in prompt-scope pilot or configuration pilot:
- aa24-249a (Ransomware / Atlassian Confluence / Windows Server / Sophos)
- aa23-187a (Snatch Ransomware / Netwrix Auditor)
- aa20-259a-iran-citrix-vpn-cve-19781 (Citrix NetScaler / VPN)
- accellion-fta-google-accellion-data-theft-extortion (Accellion FTA)

Compares:
1. Raw LLM Prediction (Baseline)
2. Vulnerability-Anchored Argument Backfilling + Refined Denoising (Window-level)
3. Document-level Global Coordinate Fusion (Document-level)

Zero access to Test documents.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

# Setup environment
os.environ.setdefault("V5_OPENCODE_SESSION_ID", f"bron-v6-gen-{uuid.uuid4().hex[:8]}")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from provider_config import apply_formal_apo_runtime_defaults
apply_formal_apo_runtime_defaults()

from llm_methods import make_extractor, merge_window_predictions
from protegi.prompts_p0 import ENTITY_PROMPT_P0
from protegi.optimizer import prepare_stage1_window_samples
from protegi.evaluator import TaskEvaluator
from protegi.entity_backfill import vulnerability_anchored_backfill
from protegi.metrics import calc_strict_entity_sample_counts, aggregate_micro_f1

FRESH_DEV_DOCS = [
    "aa24-249a",
    "aa23-187a",
    "aa20-259a-iran-citrix-vpn-cve-19781",
    "accellion-fta-google-accellion-data-theft-extortion",
]

def main() -> None:
    print("=" * 70)
    print("Stage 1 Generalization Test on Fresh Dev Documents")
    print("=" * 70)
    print(f"Target Documents ({len(FRESH_DEV_DOCS)}): {FRESH_DEV_DOCS}")

    gold_dir = ROOT / "data" / "annotations" / "gold"
    official_split = json.load(open(ROOT / "data" / "train_dev_test_split_v7.json", encoding="utf-8"))

    # Verify no test leakage
    test_set = set(official_split.get("test", []))
    for doc_id in FRESH_DEV_DOCS:
        assert doc_id in official_split["dev"], f"{doc_id} is not in Dev partition!"
        assert doc_id not in test_set, f"FATAL: {doc_id} is in Test partition!"
    print("[PASS] All target documents verified to belong strictly to official Dev.")

    # 1. Prepare samples with sentence-snapped windowing and abbreviation context
    print("\nPreparing window samples with snap_sentence_boundary=True ...")
    samples = prepare_stage1_window_samples(
        FRESH_DEV_DOCS,
        gold_dir,
        max_chars=3000,
        overlap=400,
        include_document_abbreviations=True,
        snap_sentence_boundary=True,
    )
    print(f"Total Windows: {len(samples)}")
    for doc_id in FRESH_DEV_DOCS:
        w_count = sum(s["doc_id"] == doc_id for s in samples)
        print(f"  {doc_id}: {w_count} windows")

    # Audit gold entities across windows
    gold_by_type = {}
    for s in samples:
        for e in s["gold_entities"]:
            t = e["type"]
            gold_by_type[t] = gold_by_type.get(t, 0) + 1
    print(f"Gold entities across windows: {gold_by_type}")

    # 2. Run Task Model (hy3) via TaskEvaluator
    print("\nRunning Task Model (hy3) inference on all windows ...")
    t0 = time.time()
    client = make_extractor()
    # First evaluate without backfill to get raw predictions
    evaluator_raw = TaskEvaluator(task_client=client, vulnerability_anchored_backfill=False)
    
    # We collect predictions
    abbrev_contexts = [s.get("document_abbreviations") for s in samples]
    raw_predictions = evaluator_raw.predict_stage1_texts(
        [s["text"] for s in samples],
        ENTITY_PROMPT_P0,
        document_abbreviations=abbrev_contexts,
    )
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s across {len(samples)} windows.")

    # 3. Evaluate Mode A: Raw Model (No Backfill)
    print("\n" + "=" * 70)
    print("Mode A: Raw LLM Predictions (P0 Baseline)")
    print("=" * 70)
    
    def score_predictions(preds_list, label=""):
        type_scores = {}
        total_tp, total_fp, total_fn = 0, 0, 0
        entity_types = ["Vulnerability", "Configuration", "Weakness", "AttackTechnique"]
        for etype in entity_types:
            t_tp, t_fp, t_fn = 0, 0, 0
            for s, preds in zip(samples, preds_list):
                gold = [e for e in s["gold_entities"] if e["type"] == etype]
                pred = [e for e in preds if e.get("type") == etype]
                tp, fp, fn = calc_strict_entity_sample_counts(pred, gold, allowed_types={etype})
                t_tp += tp
                t_fp += fp
                t_fn += fn
            res = aggregate_micro_f1(t_tp, t_fp, t_fn)
            type_scores[etype] = {
                "tp": t_tp, "fp": t_fp, "fn": t_fn,
                "precision": res.precision, "recall": res.recall, "f1": res.f1,
            }
            total_tp += t_tp
            total_fp += t_fp
            total_fn += t_fn
            print(f"{etype:18s} | TP: {t_tp:3d}, FP: {t_fp:3d}, FN: {t_fn:3d} | P: {res.precision*100:6.2f}%, R: {res.recall*100:6.2f}%, F1: {res.f1*100:6.2f}%")
        
        overall = aggregate_micro_f1(total_tp, total_fp, total_fn)
        print("-" * 70)
        print(f"OVERALL MICRO      | TP: {total_tp:3d}, FP: {total_fp:3d}, FN: {total_fn:3d} | P: {overall.precision*100:6.2f}%, R: {overall.recall*100:6.2f}%, F1: {overall.f1*100:6.2f}%")
        return {"by_type": type_scores, "overall": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": overall.precision, "recall": overall.recall, "f1": overall.f1}}

    res_raw = score_predictions(raw_predictions, "Raw Baseline")

    # 4. Evaluate Mode B: With Vulnerability-Anchored Backfill + Refined Denoising
    print("\n" + "=" * 70)
    print("Mode B: With Vulnerability-Anchored Backfill + Refined Denoising")
    print("=" * 70)
    enhanced_predictions = vulnerability_anchored_backfill(samples, raw_predictions, enabled=True)
    res_enhanced = score_predictions(enhanced_predictions, "Enhanced Backfill")

    # 5. Evaluate Mode C: Document-Level Global Coordinate Fusion
    print("\n" + "=" * 70)
    print("Mode C: Document-Level Global Coordinate Fusion (Global Graph View)")
    print("=" * 70)
    docs = {}
    for s, p_list in zip(samples, enhanced_predictions):
        docs.setdefault(s["doc_id"], []).append((s, p_list))

    doc_scores = {}
    entity_types = ["Vulnerability", "Configuration", "Weakness", "AttackTechnique"]
    doc_type_counts = {t: [0, 0, 0] for t in entity_types}
    
    for doc_id, items in docs.items():
        gold_data = json.load(open(gold_dir / f"{doc_id}.json", encoding="utf-8"))
        doc_preds = []
        for s, p_list in items:
            w_start = s["window_start"]
            win_ents = []
            for e in p_list:
                ge = dict(e)
                ge["start"] = ge["start"] + w_start
                ge["end"] = ge["end"] + w_start
                win_ents.append(ge)
            doc_preds.append({"entities": win_ents, "relations": []})

        merged = merge_window_predictions(doc_preds)
        
        for etype in entity_types:
            gold_etype = [e for e in gold_data["entities"] if e["type"] == etype]
            merged_etype = [e for e in merged["entities"] if e["type"] == etype]
            tp, fp, fn = calc_strict_entity_sample_counts(merged_etype, gold_etype, allowed_types={etype})
            doc_type_counts[etype][0] += tp
            doc_type_counts[etype][1] += fp
            doc_type_counts[etype][2] += fn

    doc_total_tp, doc_total_fp, doc_total_fn = 0, 0, 0
    for etype in entity_types:
        counts = doc_type_counts[etype]
        res = aggregate_micro_f1(*counts)
        doc_total_tp += counts[0]
        doc_total_fp += counts[1]
        doc_total_fn += counts[2]
        print(f"{etype:18s} | TP: {counts[0]:3d}, FP: {counts[1]:3d}, FN: {counts[2]:3d} | P: {res.precision*100:6.2f}%, R: {res.recall*100:6.2f}%, F1: {res.f1*100:6.2f}%")
    
    doc_overall = aggregate_micro_f1(doc_total_tp, doc_total_fp, doc_total_fn)
    print("-" * 70)
    print(f"DOC-LEVEL OVERALL  | TP: {doc_total_tp:3d}, FP: {doc_total_fp:3d}, FN: {doc_total_fn:3d} | P: {doc_overall.precision*100:6.2f}%, R: {doc_overall.recall*100:6.2f}%, F1: {doc_overall.f1*100:6.2f}%")

    # 6. Save results
    out_dir = ROOT / "results" / "generalization_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "stage1_fresh_dev_generalization_report.json"
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_documents": FRESH_DEV_DOCS,
        "window_count": len(samples),
        "inference_time_seconds": elapsed,
        "mode_a_raw": res_raw,
        "mode_b_enhanced": res_enhanced,
        "mode_c_doc_level": {
            "by_type": {
                t: aggregate_micro_f1(*doc_type_counts[t]).to_dict()
                for t in entity_types
            },
            "overall": doc_overall.to_dict(),
        },
    }
    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Generalization test report saved to {report_file}")

if __name__ == "__main__":
    main()
