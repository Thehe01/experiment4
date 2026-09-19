"""v6 威胁情报文本知识图谱端到端实验编排运行器.

基于 chapter3-no-capec-v1 抽取模式与 4.6-mcpu-mention-fact-dual-layer-v1 标注协议。
抽取层目标为 4 类实体和 3 类文本关系；
上层 AttackTechnique -> AttackTactic -> KillChainPhase 的补全由 complete_bron_layer.py 单独评估。

用法：
    python run_v6_experiment.py --method rule
    python run_v6_experiment.py --method llm_manual
    python run_v6_experiment.py --method multipass
    python run_v6_experiment.py --method full
    python run_v6_experiment.py --method protegi
    python run_v6_experiment.py --method apo       # (legacy / historical only)
    python run_v6_experiment.py --method apo_full  # (legacy / historical only)
    python run_v6_experiment.py --method full --eval-only
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXP_DIR))

from provider_config import apply_formal_apo_runtime_defaults

# Keep APO development selection and its final benchmark evaluation on the
# same task runtime. Explicit environment variables still override the profile.
apply_formal_apo_runtime_defaults()

import eval_metrics as EM
import rule_baseline
from llm_methods import (
    APO_ARTIFACT,
    PROTEGI_FINAL_ARTIFACT,
    load_apo_prompt_artifact,
    load_protegi_final_artifact,
    predict_full,
    predict_llm_apo,
    predict_llm_apo_full,
    predict_llm_manual,
    predict_llm_multipass,
    predict_llm_protegi,
    runtime_config,
)
from schema import (
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_TYPES,
    SCHEMA_VERSION,
)
from freeze_baseline_methods_v6 import (
    BASELINE_FREEZE_MANIFEST,
    frozen_methods,
    verify_freeze as verify_baseline_freeze,
)

GOLD = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
REVIEW_STATUS_FILE = EXP_DIR / "data" / "review_status.json"
FREEZE_MANIFEST_FILE = EXP_DIR / "data" / "dataset_freeze_manifest_v6.json"
RESULTS = EXP_DIR / "results"
GUIDELINE_FILE = EXP_DIR / "ANNOTATION_GUIDELINE.md"
FRESH_GOLD_AUDIT_FILE = RESULTS / "gold_strategy_audit_v6_v7.json"
FRESH_LEAKAGE_AUDIT_FILE = RESULTS / "split_leakage_audit_v6_v7.json"
EXT_ENT = set(EXTRACTION_ENTITY_TYPES)
EXT_REL = set(EXTRACTION_RELATION_TYPES)


def _block_frozen_baseline_overwrite(method: str, split: str = "test") -> None:
    """Protect completed baseline artifacts even when --force is supplied."""
    if split not in {"test", "all"}:
        return
    if method not in frozen_methods():
        return
    ok, message = verify_baseline_freeze()
    if not ok:
        raise RuntimeError(
            "前三种基线方法已冻结，但冻结完整性检查失败；为防止进一步覆盖，"
            f"当前运行被阻断：{message}"
        )
    raise RuntimeError(
        f"方法 {method} 的实验与结果已由 {BASELINE_FREEZE_MANIFEST} 冻结；"
        "--force 和 --eval-only 均不能覆盖。若确需重跑，必须先执行显式解冻。"
    )



def filt(ann):
    ents = [e for e in ann.get("entities", []) if e.get("type") in EXT_ENT]
    keep = {e["id"] for e in ents}
    rels = [
        r for r in ann.get("relations", [])
        if r.get("type") in EXT_REL
        and r.get("head") in keep
        and r.get("tail") in keep
    ]
    return ents, rels


def predict_rule(text, doc_id):
    """规则基线不读取 doc_id；词典仅由冻结 train/dev 标注构建。"""
    return rule_baseline.extract(text)


PREDICTORS = {
    "rule": predict_rule,
    "llm_manual": predict_llm_manual,
    "multipass": predict_llm_multipass,
    "full": predict_full,
    "protegi": predict_llm_protegi,
    # Legacy / historical methods - not valid as current ProTeGi formal artifact:
    "apo": predict_llm_apo,
    "apo_full": predict_llm_apo_full,
}


def _method_runtime_config(method: str) -> dict:
    config = runtime_config()
    if method == "protegi":
        artifact = load_protegi_final_artifact()
        config["protegi"] = {
            "artifact": str(PROTEGI_FINAL_ARTIFACT),
            "artifact_sha256": hashlib.sha256(
                PROTEGI_FINAL_ARTIFACT.read_bytes()
            ).hexdigest(),
            "artifact_version": artifact.get("artifact_version"),
            "prompt_scope": artifact.get("prompt_scope"),
            "task_model": artifact.get("task_model"),
            "optimizer_model": artifact.get("optimizer_model"),
            "entity_prompt_sha256": artifact.get("entity_prompt_sha256"),
            "relation_prompt_sha256": artifact.get("relation_prompt_sha256"),
        }
    elif method in {"apo", "apo_full"}:
        artifact = load_apo_prompt_artifact()
        config["apo"] = {
            "artifact": str(APO_ARTIFACT),
            "artifact_sha256": hashlib.sha256(
                APO_ARTIFACT.read_bytes()
            ).hexdigest(),
            "algorithm": artifact.get("algorithm"),
            "score": artifact.get("score"),
            "selected_round": artifact.get("selected_round"),
            "selected_candidate": artifact.get("selected_candidate"),
            "target_f1_gain": (artifact.get("target_performance") or {}).get(
                "target_f1_gain"
            ),
            "formal_validity": "historical only; not valid as current ProTeGi formal artifact",
        }
    return config


def _aggregate_by_type(per_doc, metric_name):
    pool = Counter()
    tp_pool = Counter()
    fp_pool = Counter()
    fn_pool = Counter()
    for d in per_doc:
        for t, m in d[metric_name].items():
            pool[t] += 1
            tp_pool[t] += m.get("tp", 0)
            fp_pool[t] += m.get("fp", 0)
            fn_pool[t] += m.get("fn", 0)
    out = {}
    for t in sorted(pool):
        tp = tp_pool[t]
        fp = fp_pool[t]
        fn = fn_pool[t]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[t] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def load_docs(split="test", split_file=SPLIT_FILE):
    split_file = Path(split_file)
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    if split == "all":
        ids = []
        for part in ("train", "dev", "test"):
            ids.extend(split_data.get(part, []))
    else:
        ids = split_data.get(split, [])
    docs = []
    for doc_id in ids:
        p = GOLD / f"{doc_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"缺少 Gold 文件：{p}")
        docs.append(p)
    return docs


def _review_ready(requested_split_file=SPLIT_FILE):
    if not REVIEW_STATUS_FILE.is_file():
        return False, "缺少 data/review_status.json"
    status = json.loads(REVIEW_STATUS_FILE.read_text(encoding="utf-8"))
    if not status.get("human_reannotation_complete"):
        return False, "人工复标未完成"
    if not status.get("adjudication_complete"):
        return False, "分歧裁决未完成"
    if not status.get("dataset_frozen"):
        return False, "数据集未冻结"
    if not status.get("controlled_test_rerun_ready"):
        return False, "未批准受控重跑"
    boundary = status.get("boundary_sync", {})
    if (
        boundary.get("contract_version") != "chapter3-boundary-sync-v2"
        or boundary.get("complete") is not True
    ):
        return False, "未通过 chapter3-boundary-sync-v2 MCPU 边界对齐门禁"
    if not FREEZE_MANIFEST_FILE.is_file():
        return False, "缺少 data/dataset_freeze_manifest_v6.json"
    manifest = json.loads(FREEZE_MANIFEST_FILE.read_text(encoding="utf-8"))
    base_record = manifest.get("base_manifest", {})
    base_path = EXP_DIR / str(base_record.get("path", ""))
    if not base_path.is_file():
        return False, "v6 冻结清单绑定的 v5 基础清单缺失"
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_record.get("sha256"):
        return False, "v6 冻结清单绑定的 v5 基础清单已发生变化"
    split_path = EXP_DIR / manifest["split_file"]
    if Path(requested_split_file).resolve() != split_path.resolve():
        return False, "当前运行未使用冻结清单指定的正式划分文件"
    current_split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if current_split_hash != manifest.get("split_sha256"):
        return False, "冻结后的划分文件已发生变化"
    aggregate = hashlib.sha256()
    for doc_id in sorted(manifest.get("gold_document_sha256", {})):
        path = GOLD / f"{doc_id}.json"
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_hash != manifest["gold_document_sha256"][doc_id]:
            return False, f"v6 冻结后的 Gold 已发生变化：{doc_id}"
        aggregate.update(f"{doc_id}\0{current_hash}\n".encode("utf-8"))
    if aggregate.hexdigest() != manifest.get("gold_aggregate_sha256"):
        return False, "Gold 聚合哈希与冻结清单不一致"
    current_gold_ids = {
        path.stem
        for path in GOLD.glob("*.json")
        if path.name != "_manifest.json"
    }
    frozen_gold_ids = set(manifest.get("gold_document_sha256", {}))
    if current_gold_ids != frozen_gold_ids:
        return False, "当前 Gold 文档集合与冻结清单不一致"
    evidence = manifest.get("supporting_evidence", {})
    for evidence_name in (
        "review_receipt",
        "audit_report",
        "boundary_audit_report",
        "leakage_report",
        "guideline",
        "boundary_review_csv",
        "boundary_review_md",
        "boundary_contract",
        "boundary_quarantine",
        "configuration_mcpu_policy",
        "configuration_mcpu_audit",
        "configuration_mcpu_receipt",
        "configuration_mcpu_review_csv",
        "configuration_mcpu_review_md",
    ):
        record = evidence.get(evidence_name, {})
        evidence_path = EXP_DIR / str(record.get("path", ""))
        if not evidence_path.is_file():
            return False, f"冻结证据缺失：{evidence_name}"
        current_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if current_hash != record.get("sha256"):
            return False, f"冻结证据哈希不一致：{evidence_name}"
    audit_record = evidence.get("audit_report", {})
    audit_path = EXP_DIR / str(audit_record.get("path", ""))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("gate_status") != "gate_passed":
        return False, "冻结审计报告未通过语义门禁"
    leakage_record = evidence.get("leakage_report", {})
    leakage_path = EXP_DIR / str(leakage_record.get("path", ""))
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    if leakage.get("status") != "passed":
        return False, "冻结泄漏审计报告未通过"
    if not FRESH_GOLD_AUDIT_FILE.is_file():
        return False, "缺少当前 results/gold_strategy_audit_v6_v7.json；请重新运行审计"
    fresh_audit = json.loads(FRESH_GOLD_AUDIT_FILE.read_text(encoding="utf-8"))
    if fresh_audit.get("gate_status") != "gate_passed":
        return False, "当前 Gold 审计报告未通过语义门禁"
    if fresh_audit.get("split_sha256") != current_split_hash:
        return False, "当前 Gold 审计报告未绑定当前 v7 划分"
    guideline_hash = hashlib.sha256(GUIDELINE_FILE.read_bytes()).hexdigest()
    if fresh_audit.get("guideline", {}).get("sha256") != guideline_hash:
        return False, "当前 Gold 审计报告未绑定当前标注指南"
    if not FRESH_LEAKAGE_AUDIT_FILE.is_file():
        return False, "缺少当前 results/split_leakage_audit_v6_v7.json；请重新运行审计"
    fresh_leakage = json.loads(FRESH_LEAKAGE_AUDIT_FILE.read_text(encoding="utf-8"))
    if fresh_leakage.get("status") != "passed":
        return False, "当前划分防泄漏审计未通过"
    return True, "v6 Gold 已通过 chapter3-boundary-sync-v2 MCPU 门禁，可进行受控重跑（非正式人工 IAA）"


def _assert_run_allowed(split: str, split_file: Path) -> None:
    """受控 final test 强门禁。

    在任何 load_docs、predictor 调用、生成预测或读取 test gold 前执行。
    对 dev/train split 允许正常开发实验。
    对 test/all split，当且仅当 _review_ready() 通过且 controlled_test_rerun_ready 为 True 时才允许。
    若未达到准入条件，必须在此处 hard fail 阻断，保证 predictor 调用次数严格为 0。
    """
    if split in {"test", "all"}:
        status = (
            json.loads(REVIEW_STATUS_FILE.read_text(encoding="utf-8"))
            if REVIEW_STATUS_FILE.is_file()
            else {}
        )
        manifest = (
            json.loads(FREEZE_MANIFEST_FILE.read_text(encoding="utf-8"))
            if FREEZE_MANIFEST_FILE.is_file()
            else {}
        )
        if (
            status.get("controlled_test_rerun_ready") is not True
            or manifest.get("controlled_test_rerun_ready") is not True
        ):
            raise RuntimeError(
                "受控 final test 门禁已阻断运行：未批准受控重跑。"
                "当前仓库在 controlled_test_rerun_ready 开启且全量门禁就绪前，"
                "禁止对 test/all split 执行任何加载、预测或评测。"
            )
        ready, message = _review_ready(split_file)
        if not ready:
            raise RuntimeError(
                f"受控 final test 门禁已阻断运行：{message}。"
                "当前仓库在 controlled_test_rerun_ready 开启且全量门禁就绪前，"
                "禁止对 test/all split 执行任何加载、预测或评测。"
            )


def run(method, split="test", force=False, eval_only=False, split_file=SPLIT_FILE):
    split_file = Path(split_file)
    _assert_run_allowed(split, split_file)
    _block_frozen_baseline_overwrite(method, split)
    if method == "protegi":
        load_protegi_final_artifact()
    elif method in {"apo", "apo_full"}:
        status = json.loads(REVIEW_STATUS_FILE.read_text(encoding="utf-8"))
        apo_status = status.get("apo_prompt_status", {})
        if apo_status.get("ready") is not True:
            raise RuntimeError(
                "当前 v6 尚未绑定可运行的非 P0 APO 提示产物；P0 只作为基线，"
                "请先在 v7 固定 dev 集重新运行 apo_optimizer.py 并执行 promote_apo_v6.py "
                "(注：APO 为历史遗留方法，正式实验请使用 protegi)"
            )
        load_apo_prompt_artifact()
    predict = PREDICTORS[method]
    docs = load_docs(split, split_file=split_file)
    pred_dir = RESULTS / "raw_predictions" / f"v6_{method}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def process_doc(args):
        i, path = args
        out_path = pred_dir / path.name
        if eval_only:
            if not out_path.exists():
                raise FileNotFoundError(f"缺少预测文件：{out_path}")
            return f"  [{i}/{len(docs)}] {path.stem}: 读取已有预测"
        if out_path.exists() and not force:
            return f"  [{i}/{len(docs)}] {path.stem}: 已存在，跳过"
        gold = json.loads(path.read_text(encoding="utf-8"))
        start = time.time()
        prediction = predict(gold["text"], path.stem)
        record = {
            "doc_id": path.stem,
            "text": gold["text"],
            "schema_version": SCHEMA_VERSION,
            "entities": prediction.get("entities", []),
            "relations": prediction.get("relations", []),
            "trace": prediction.get("_trace"),
            "resource": prediction.get("_resource"),
            "postprocess": prediction.get("_postprocess"),
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return (
            f"  [{i}/{len(docs)}] {path.stem}: "
            f"{len(record['entities'])}E/{len(record['relations'])}R "
            f"({time.time() - start:.0f}s)"
        )

    work = list(enumerate(docs, 1))
    if method in {"llm_manual", "multipass", "apo", "protegi"} and not eval_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for message in executor.map(process_doc, work):
                print(message, flush=True)
    else:
        for item in work:
            print(process_doc(item), flush=True)

    entity_metrics = []
    relation_metrics = []
    normalized_relation_metrics = []
    evidence_ambiguity_metrics = []
    gold_evidence_ambiguity_metrics = []
    scr_metrics = []
    na_metrics = []
    per_doc = []
    gold_relation_counts = Counter()
    gold_relation_docs = Counter()
    output_audit = Counter()
    postprocess_rejections = Counter()
    for path in docs:
        gold = json.loads(path.read_text(encoding="utf-8"))
        pred_path = pred_dir / path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少预测文件：{pred_path}")
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        pred_entities, pred_relations = filt(pred)
        gold_entities, gold_relations = filt(gold)
        output_audit["raw_entities"] += len(pred.get("entities", []))
        output_audit["accepted_entities"] += len(pred_entities)
        output_audit["raw_relations"] += len(pred.get("relations", []))
        output_audit["accepted_relations"] += len(pred_relations)
        if not gold_relations:
            output_audit["negative_relation_documents"] += 1
            output_audit["relations_on_negative_documents"] += len(pred_relations)
            if pred_relations:
                output_audit["negative_documents_with_prediction"] += 1
        postprocess_rejections.update(
            (pred.get("postprocess") or {}).get("rejected", {})
        )
        gold_relation_counts.update(r["type"] for r in gold_relations)
        gold_relation_docs.update({r["type"] for r in gold_relations})

        em = EM.calc_entity_metrics(pred_entities, gold_entities)
        rm = EM.calc_relation_metrics(
            pred_entities, pred_relations, gold_entities, gold_relations
        )
        entity_metrics.append(em)
        relation_metrics.append(rm)
        normalized_relation_metrics.append(
            EM.calc_normalized_relation_metrics(
                pred_entities, pred_relations, gold_entities, gold_relations
            )
        )
        evidence_ambiguity_metrics.append(
            EM.calc_evidence_ambiguity_metrics(
                pred.get("entities", []), pred.get("relations", []),
                pred.get("text", gold.get("text", "")),
            )
        )
        gold_evidence_ambiguity_metrics.append(
            EM.calc_evidence_ambiguity_metrics(
                gold.get("entities", []), gold.get("relations", []),
                gold.get("text", ""),
            )
        )
        scr_metrics.append(EM.calc_scr(pred.get("entities", []), pred.get("relations", [])))
        na_metrics.append(EM.calc_na(pred_entities, gold_entities))
        per_doc.append({
            "entity_by_type": EM.calc_entity_metrics_by_type(pred_entities, gold_entities),
            "relation_by_type": EM.calc_relation_metrics_by_type(
                pred_entities, pred_relations, gold_entities, gold_relations
            ),
            "normalized_relation_by_type": EM.calc_normalized_relation_metrics_by_type(
                pred_entities, pred_relations, gold_entities, gold_relations
            ),
        })

    entity = EM.aggregate_metrics(entity_metrics)
    relation = EM.aggregate_metrics(relation_metrics)
    normalized_relation = EM.aggregate_metrics(normalized_relation_metrics)
    review_ready, review_message = _review_ready(split_file)
    schema_ready = (
        gold_relation_counts["exploited_by"] > 0
        and gold_relation_docs["exploited_by"] > 0
        and review_ready
    )
    status_payload = json.loads(REVIEW_STATUS_FILE.read_text(encoding="utf-8"))
    output = {
        "method": method,
        "schema_version": SCHEMA_VERSION,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_config": (
            _method_runtime_config(method)
            if method in {"llm_manual", "multipass", "apo", "apo_full", "full", "protegi"}
            else {
                "extractor": "deterministic-rule-baseline",
                "lexicon_source": "frozen train+dev annotations only (development-fitted rule baseline)",
                "split_file": split_file.name,
                "uses_document_topic": False,
            }
        ),
        "split": split,
        "split_file": split_file.name,
        "num_docs": len(docs),
        "entity": entity,
        "relation": relation,
        "entity_by_type": _aggregate_by_type(per_doc, "entity_by_type"),
        "relation_by_type": _aggregate_by_type(per_doc, "relation_by_type"),
        "normalized_relation": normalized_relation,
        "normalized_relation_by_type": _aggregate_by_type(
            per_doc, "normalized_relation_by_type"
        ),
        "evidence_ambiguity": {
            "prediction": EM.aggregate_evidence_ambiguity(
                evidence_ambiguity_metrics
            ),
            "gold": EM.aggregate_evidence_ambiguity(
                gold_evidence_ambiguity_metrics
            ),
        },
        "scr": EM.aggregate_scr(scr_metrics),
        "na": EM.aggregate_na(na_metrics),
        "gold_relation_counts": dict(sorted(gold_relation_counts.items())),
        "gold_relation_doc_counts": dict(sorted(gold_relation_docs.items())),
        "output_audit": {
            **dict(sorted(output_audit.items())),
            "entity_rejection_count": (
                output_audit["raw_entities"] - output_audit["accepted_entities"]
            ),
            "relation_rejection_count": (
                output_audit["raw_relations"] - output_audit["accepted_relations"]
            ),
            "mean_relations_per_negative_document": (
                output_audit["relations_on_negative_documents"]
                / output_audit["negative_relation_documents"]
                if output_audit["negative_relation_documents"]
                else 0.0
            ),
            "negative_document_prediction_rate": (
                output_audit["negative_documents_with_prediction"]
                / output_audit["negative_relation_documents"]
                if output_audit["negative_relation_documents"]
                else 0.0
            ),
        },
        "postprocess_rejections": dict(sorted(postprocess_rejections.items())),
        "schema_ready": schema_ready,
        "controlled_test_rerun_ready": schema_ready,
        "formal_experiment_ready": bool(
            status_payload.get("formal_experiment_ready", False)
        ),
        "freeze_manifest": str(FREEZE_MANIFEST_FILE.relative_to(EXP_DIR)),
        "review_status": review_message,
    }
    result_path = RESULTS / f"v6_{method}_{split}.json"
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== v6 威胁情报图谱抽取层 | 方法={method} | {split}={len(docs)} 篇 ===")
    print(
        f"  实体 P/R/F1 = {entity['precision']:.3f} / "
        f"{entity['recall']:.3f} / {entity['f1']:.3f}"
    )
    print(
        f"  关系 P/R/F1 = {relation['precision']:.3f} / "
        f"{relation['recall']:.3f} / {relation['f1']:.3f}"
    )
    print(f"  Gold 关系分布 = {output['gold_relation_counts']}")
    print(f"  SCR = {output['scr']}")
    print(f"  NA  = {output['na']}")
    print(f"  输出审计 = {output['output_audit']}")
    if output["postprocess_rejections"]:
        print(f"  质量控制拒绝原因 = {output['postprocess_rejections']}")
    if not output["schema_ready"]:
        print(f"  警告：正式实验尚未就绪（{review_message}）。")
        if gold_relation_counts["exploited_by"] == 0:
            print("        当前 Gold 中尚无 exploited_by。")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="rule", choices=list(PREDICTORS))
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "dev", "test", "all"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖未冻结的已有预测；不能覆盖已冻结的 rule/multipass/full",
    )
    parser.add_argument("--eval-only", action="store_true", help="只评估已有预测")
    parser.add_argument(
        "--split-file",
        default=str(SPLIT_FILE),
        help="训练/开发/测试切分 JSON；默认使用 v7 划分 (63/21/21)",
    )
    args = parser.parse_args()
    run(
        args.method,
        split=args.split,
        force=args.force,
        eval_only=args.eval_only,
        split_file=args.split_file,
    )
