"""Full Dev partition (21 documents) evaluation for Stage 1 Entity Extraction.

Evaluates Stage 1 performance across the entire official Dev partition (319 windows):
1. Mode A: Raw Model (P0 Zero-shot baseline)
2. Mode B: Enhanced (Vulnerability-Anchored Backfilling + Protected Server Whitelist)
3. Mode C: Document-Level Coordinate Fusion
4. In-depth Root Cause Diagnosis specifically for Configuration entities (all FNs and FPs).

Includes disk checkpointing for progressive resumption and zero access to Test set.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Setup environment
os.environ.setdefault("V5_OPENCODE_SESSION_ID", f"bron-v6-devfull-{uuid.uuid4().hex[:8]}")
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Full Dev Evaluation")
    parser.add_argument("--cache-name", type=str, default="dev_full_predictions_mcpu_v2.json", help="Cache filename in results/dev_full_evaluation")
    parser.add_argument("--report-name", type=str, default="dev_full_stage1_report_mcpu_v2.json", help="Report filename in results/dev_full_evaluation")
    parser.add_argument("--workers", type=int, default=8, help="Max concurrency workers")
    args = parser.parse_args()

    print("=" * 75)
    print(f"Stage 1 Full Dev Evaluation (All 21 Official Dev Documents)")
    print(f"Contract: chapter3-boundary-sync-v2 (MCPU v2)")
    print(f"Cache: {args.cache_name} | Workers: {args.workers}")
    print("=" * 75)

    gold_dir = ROOT / "data" / "annotations" / "gold"
    official_split = json.load(open(ROOT / "data" / "train_dev_test_split_v7.json", encoding="utf-8"))

    dev_docs = official_split["dev"]
    test_set = set(official_split.get("test", []))

    # Guardrail check
    for doc_id in dev_docs:
        assert doc_id not in test_set, f"FATAL LEAKAGE: {doc_id} is in Test set!"
    print(f"[PASS] All {len(dev_docs)} documents verified to belong strictly to official Dev.")

    out_dir = ROOT / "results" / "dev_full_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / args.cache_name

    # 1. Prepare all 319 windows
    print("\nPreparing window samples with snap_sentence_boundary=True ...")
    samples = prepare_stage1_window_samples(
        dev_docs,
        gold_dir,
        max_chars=3000,
        overlap=400,
        include_document_abbreviations=True,
        snap_sentence_boundary=True,
    )
    print(f"Total Dev Windows: {len(samples)}")

    gold_type_counts = Counter()
    config_by_doc = Counter()
    for s in samples:
        for e in s["gold_entities"]:
            t = e["type"]
            gold_type_counts[t] += 1
            if t == "Configuration":
                config_by_doc[s["doc_id"]] += 1
    print(f"Gold entities across all Dev windows: {dict(gold_type_counts)}")
    print(f"Configuration distribution across {len(config_by_doc)} docs: {dict(config_by_doc)}")

    # 2. Load existing cache or pre-populate from fresh dev cache
    cached_predictions: Dict[str, List[Dict[str, Any]]] = {}
    if cache_file.is_file():
        try:
            cached_data = json.load(open(cache_file, encoding="utf-8"))
            if isinstance(cached_data, dict):
                cached_predictions = cached_data
            elif isinstance(cached_data, list):
                for item in cached_data:
                    cached_predictions[item["sample_id"]] = item.get("raw_predictions", [])
            print(f"[CACHE] Loaded {len(cached_predictions)} cached predictions from {cache_file.name}")
        except Exception as e:
            print(f"[CACHE] Could not load cache: {e}")

    # Also check if scratch fresh_dev_predictions has any matching samples (legacy cache only)
    if args.cache_name == "dev_full_predictions_cache.json":
        fresh_cache_path = Path("C:/Users/18870/.gemini/antigravity/brain/635170ab-70be-47db-9d36-4e4a27510213/scratch/fresh_dev_predictions.json")
        if fresh_cache_path.is_file():
            try:
                fresh_items = json.load(open(fresh_cache_path, encoding="utf-8"))
                imported = 0
                for it in fresh_items:
                    sid = it.get("sample_id")
                    if sid and sid not in cached_predictions and "raw_predictions" in it:
                        cached_predictions[sid] = it["raw_predictions"]
                        imported += 1
                if imported > 0:
                    print(f"[CACHE] Imported {imported} predictions from fresh dev scratch cache.")
            except Exception:
                pass

    # 3. Determine missing windows to run
    missing_samples = [s for s in samples if s["sample_id"] not in cached_predictions]
    print(f"Windows to run: {len(missing_samples)} / {len(samples)} ({len(cached_predictions)} already cached)")

    if missing_samples:
        print(f"\nRunning Task Model inference on {len(missing_samples)} windows (max_workers={args.workers}) ...")
        t0 = time.time()
        client = make_extractor()
        evaluator_raw = TaskEvaluator(task_client=client, max_workers=args.workers, vulnerability_anchored_backfill=False)

        # Batch in chunks of 32 to save progress progressively
        chunk_size = 32
        for i in range(0, len(missing_samples), chunk_size):
            chunk = missing_samples[i : i + chunk_size]
            print(f"  Processing chunk {i//chunk_size + 1}/{(len(missing_samples)-1)//chunk_size + 1} ({len(chunk)} windows)...")
            abbrev_ctxs = [s.get("document_abbreviations") for s in chunk]
            chunk_preds = evaluator_raw.predict_stage1_texts(
                [s["text"] for s in chunk],
                ENTITY_PROMPT_P0,
                document_abbreviations=abbrev_ctxs,
            )
            for s, preds in zip(chunk, chunk_preds):
                cached_predictions[s["sample_id"]] = preds
            
            # Save intermediate cache
            json.dump(cached_predictions, open(cache_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"Inference completed in {elapsed:.1f}s.")

    # Assemble raw predictions aligned with samples list
    raw_predictions = [cached_predictions[s["sample_id"]] for s in samples]

    # 4. Mode A: Raw Model Evaluation
    print("\n" + "=" * 75)
    print("Mode A: Raw LLM Predictions (P0 Baseline)")
    print("=" * 75)

    def score_by_type(preds_list: List[List[Dict[str, Any]]], label: str = "") -> Dict[str, Any]:
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
        print("-" * 75)
        print(f"OVERALL MICRO      | TP: {total_tp:3d}, FP: {total_fp:3d}, FN: {total_fn:3d} | P: {overall.precision*100:6.2f}%, R: {overall.recall*100:6.2f}%, F1: {overall.f1*100:6.2f}%")
        return {"by_type": type_scores, "overall": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": overall.precision, "recall": overall.recall, "f1": overall.f1}}

    res_raw = score_by_type(raw_predictions, "Raw Baseline")

    # 5. Mode B: Enhanced Model (Backfilling + Server Whitelist)
    print("\n" + "=" * 75)
    print("Mode B: Vulnerability-Anchored Backfill + Protected Server Whitelist")
    print("=" * 75)
    enhanced_predictions = vulnerability_anchored_backfill(samples, raw_predictions, enabled=True)
    res_enhanced = score_by_type(enhanced_predictions, "Enhanced")

    # 6. Per-Document Configuration Breakdown
    print("\n" + "=" * 75)
    print("Configuration Strict Performance Per Document (Mode B)")
    print("=" * 75)
    print(f"{'Document ID':50s} | Gold | {'TP':3s} | {'FP':3s} | {'FN':3s} | {'Recall':7s} | {'F1':7s}")
    print("-" * 88)

    doc_config_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gold_count": 0})
    for s, preds in zip(samples, enhanced_predictions):
        doc_id = s["doc_id"]
        gold_configs = [e for e in s["gold_entities"] if e["type"] == "Configuration"]
        pred_configs = [e for e in preds if e.get("type") == "Configuration"]
        tp, fp, fn = calc_strict_entity_sample_counts(pred_configs, gold_configs, allowed_types={"Configuration"})
        doc_config_stats[doc_id]["tp"] += tp
        doc_config_stats[doc_id]["fp"] += fp
        doc_config_stats[doc_id]["fn"] += fn
        doc_config_stats[doc_id]["gold_count"] += len(gold_configs)

    for doc_id in sorted(dev_docs):
        st = doc_config_stats[doc_id]
        res = aggregate_micro_f1(st["tp"], st["fp"], st["fn"])
        rec_str = f"{res.recall*100:5.1f}%" if st["gold_count"] > 0 else "N/A"
        f1_str = f"{res.f1*100:5.1f}%" if st["gold_count"] > 0 else "N/A"
        print(f"{doc_id:50s} | {st['gold_count']:4d} | {st['tp']:3d} | {st['fp']:3d} | {st['fn']:3d} | {rec_str:7s} | {f1_str:7s}")

    # 7. Deep Diagnosis of Configuration FNs (False Negatives)
    print("\n" + "=" * 75)
    print("DEEP DIAGNOSIS: Configuration False Negatives (FNs)")
    print("=" * 75)

    fn_details = []
    for s, preds in zip(samples, enhanced_predictions):
        gold_configs = [e for e in s["gold_entities"] if e["type"] == "Configuration"]
        pred_configs = [e for e in preds if e.get("type") == "Configuration"]
        
        for g in gold_configs:
            # Check if strictly matched
            matched = False
            overlap_pred = None
            for p in pred_configs:
                if p["start"] == g["start"] and p["end"] == g["end"]:
                    matched = True
                    break
                # Check overlap
                if max(p["start"], g["start"]) < min(p["end"], g["end"]):
                    overlap_pred = p
            
            if not matched:
                w_text = s["text"]
                g_start, g_end = g["start"], g["end"]
                ctx = w_text[max(0, g_start - 35) : min(len(w_text), g_end + 35)].replace("\n", " ")
                
                # Determine cause category
                if overlap_pred:
                    cause = f"Boundary Mismatch (Predicted: '{overlap_pred.get('text')}' [{overlap_pred['start']}:{overlap_pred['end']}])"
                else:
                    # Check if text appears anywhere in the window
                    pred_texts = [p.get("text", "") for p in pred_configs]
                    cause = f"Completely Missing (Window preds: {pred_texts})"

                fn_details.append({
                    "doc_id": s["doc_id"],
                    "sample_id": s["sample_id"],
                    "gold_text": g["text"],
                    "gold_span": [g["start"], g["end"]],
                    "context": ctx,
                    "cause": cause,
                    "overlap_pred": overlap_pred,
                })

    print(f"Total Configuration FNs across Dev: {len(fn_details)}")
    for idx, fn in enumerate(fn_details, 1):
        print(f"\n[FN #{idx:02d}] {fn['doc_id']} ({fn['sample_id']})")
        print(f"  Gold Entity : '{fn['gold_text']}' {fn['gold_span']}")
        print(f"  Context     : ...{fn['context']}...")
        print(f"  Root Cause  : {fn['cause']}")

    # 8. Diagnosis of Configuration FPs (False Positives)
    print("\n" + "=" * 75)
    print("DEEP DIAGNOSIS: Configuration False Positives (FPs)")
    print("=" * 75)
    
    fp_counter = Counter()
    for s, preds in zip(samples, enhanced_predictions):
        gold_configs = [e for e in s["gold_entities"] if e["type"] == "Configuration"]
        pred_configs = [e for e in preds if e.get("type") == "Configuration"]
        for p in pred_configs:
            if not any(p["start"] == g["start"] and p["end"] == g["end"] for g in gold_configs):
                fp_counter[p.get("text", "")] += 1

    print(f"Total Configuration FPs: {sum(fp_counter.values())} across {len(fp_counter)} unique strings")
    print("Top 20 False Positive Strings:")
    for txt, cnt in fp_counter.most_common(20):
        print(f"  {cnt:2d}x: '{txt}'")

    # 9. Save full report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_documents": dev_docs,
        "window_count": len(samples),
        "mode_a_raw": res_raw,
        "mode_b_enhanced": res_enhanced,
        "configuration_per_doc": {d: dict(doc_config_stats[d]) for d in dev_docs},
        "configuration_fns": fn_details,
        "configuration_fps_top": fp_counter.most_common(30),
    }

    report_file = out_dir / args.report_name
    json.dump(report_data, open(report_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] Full report successfully saved to: {report_file}")


if __name__ == "__main__":
    main()
