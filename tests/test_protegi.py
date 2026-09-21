"""Unit tests for the ProTeGi framework in experiments/v6/protegi.

Tests all pure-logic components without external network or LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

V6_ROOT = Path(__file__).resolve().parents[1]
if str(V6_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(V6_ROOT / "scripts"))
if str(V6_ROOT) not in sys.path:
    sys.path.insert(0, str(V6_ROOT))

from protegi.models import (
    CallStats,
    ErrorExample,
    EvaluationResult,
    PromptCandidate,
    PromptGradient,
)
from protegi.metrics import (
    aggregate_micro_f1,
    calc_same_type_jaccard_overlap_counts,
    calc_strict_entity_sample_counts,
    calc_strict_relation_sample_counts,
)
from protegi.selectors import UCBPromptSelector, UniformPromptSelector
from protegi.mutators import (
    PromptEditor,
    _extract_blocks,
    format_error_examples_for_prompt,
    deduplicate_and_sample_successors,
)
from protegi.lineage import PromptLineageTracker
from protegi.entity_cache import EntityCacheManager, compute_prompt_hash
from protegi.prompts_p0 import (
    ENTITY_PROMPT_P0,
    RELATION_PROMPT_P0,
    extract_immutable_contract,
    extract_optimizable_guidance,
    replace_optimizable_guidance,
)
from protegi.contract_validator import PromptContractValidator
from protegi.evaluator import TaskEvaluator, focus_entity_error_examples
from protegi.document_context import (
    extract_explicit_abbreviation_pairs,
    format_abbreviation_context,
)
from protegi.optimizer import select_final_candidate
from schema import (
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_TYPES,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
)


class TestProTeGiCore(unittest.TestCase):

    def test_task_evaluator_uses_eight_way_window_concurrency(self):
        class ConcurrentFakeClient:
            config = {}

            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def call_fn(self, **kwargs):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return '{"entities": []}'

        client = ConcurrentFakeClient()
        evaluator = TaskEvaluator(task_client=client, max_workers=8)
        samples = [
            {"sample_id": f"s{i}", "text": f"sample {i}", "gold_entities": []}
            for i in range(8)
        ]
        result, errors = evaluator.evaluate_stage1_batch(
            samples,
            ENTITY_PROMPT_P0,
        )
        self.assertEqual(evaluator.max_workers, 8)
        self.assertEqual(evaluator.call_count, 8)
        self.assertGreater(client.max_active, 1)
        self.assertLessEqual(client.max_active, 8)
        self.assertEqual((result.tp, result.fp, result.fn), (0, 0, 0))
        self.assertEqual(errors, [])

    def test_metrics_strict_entity_calculation(self):
        gold = [
            {"id": "E1", "type": "Vulnerability", "start": 10, "end": 20},
            {"id": "E2", "type": "Configuration", "start": 30, "end": 40},
        ]
        pred = [
            {"id": "p1", "type": "Vulnerability", "start": 10, "end": 20},  # TP
            {"id": "p2", "type": "Configuration", "start": 30, "end": 45},  # FP (boundary mismatch)
        ]
        tp, fp, fn = calc_strict_entity_sample_counts(pred, gold)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

        result = aggregate_micro_f1(tp, fp, fn)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 0.5)
        self.assertEqual(result.f1, 0.5)

    def test_metrics_strict_relation_calculation(self):
        gold_ents = [
            {"id": "E1", "type": "Vulnerability", "start": 0, "end": 10},
            {"id": "E2", "type": "Configuration", "start": 20, "end": 30},
        ]
        gold_rels = [
            {"id": "R1", "type": "affects", "head": "E1", "tail": "E2"}
        ]
        pred_ents = [
            {"id": "E1", "type": "Vulnerability", "start": 0, "end": 10},
            {"id": "E2", "type": "Configuration", "start": 20, "end": 30},
        ]
        pred_rels = [
            {"id": "r1", "type": "affects", "head": "E1", "tail": "E2"}
        ]
        tp, fp, fn = calc_strict_relation_sample_counts(
            pred_ents, pred_rels, gold_ents, gold_rels
        )
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_ucb_selector(self):
        selector = UCBPromptSelector(c=2.0, total_pull_budget_per_round=4, batch_size=2)
        c1 = PromptCandidate(candidate_id="c1", prompt_text="prompt 1")
        c2 = PromptCandidate(candidate_id="c2", prompt_text="prompt 2")

        # Initial state
        self.assertEqual(c1.num_evaluations, 0)
        selector.compute_ucb_scores([c1, c2], total_t=1)
        self.assertEqual(c1.ucb_score, float("inf"))

        # Update c1 with batch (1 TP, 0 FP, 0 FN -> F1 = 1.0)
        res1 = EvaluationResult(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        selector.update_candidate_with_batch(c1, res1, samples_count=2)
        self.assertEqual(c1.num_evaluations, 1)
        self.assertEqual(c1.estimated_reward, 1.0)

        # Update c2 with batch (0 TP, 1 FP, 1 FN -> F1 = 0.0)
        res2 = EvaluationResult(tp=0, fp=1, fn=1, precision=0.0, recall=0.0, f1=0.0)
        selector.update_candidate_with_batch(c2, res2, samples_count=2)
        self.assertEqual(c2.estimated_reward, 0.0)

        # Rank Top-1
        top = selector.rank_and_select_top_k([c1, c2], top_k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].candidate_id, "c1")
        self.assertEqual(c1.selection_status, "selected")
        self.assertEqual(c2.selection_status, "dropped")

    def test_uniform_selector(self):
        selector = UniformPromptSelector(total_pull_budget_per_round=4, batch_size=2)
        c1 = PromptCandidate(candidate_id="c1", prompt_text="short")
        c2 = PromptCandidate(candidate_id="c2", prompt_text="longer prompt")

        # Give both equal evaluation batches
        res1 = EvaluationResult(tp=2, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        selector.update_candidate_with_batch(c1, res1, 2)
        selector.update_candidate_with_batch(c2, res1, 2)

        # Tie-breaker should prefer shorter prompt
        top = selector.rank_and_select_top_k([c1, c2], top_k=1)
        self.assertEqual(top[0].candidate_id, "c1")

    def test_mutators_block_extraction(self):
        raw = "Here is feedback:\n<START>\nDirection 1: fix weakness span\n<END>\n<START>\nDirection 2: fix cpe\n<END>"
        blocks = _extract_blocks(raw, "<START>", "<END>")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], "Direction 1: fix weakness span")
        self.assertEqual(blocks[1], "Direction 2: fix cpe")

    def test_deduplication_and_sampling(self):
        parent = PromptCandidate(candidate_id="p0", prompt_text="original prompt")
        s1 = PromptCandidate(candidate_id="s1", prompt_text="new prompt A")
        s2 = PromptCandidate(candidate_id="s2", prompt_text="new prompt A")  # Duplicate of s1
        s3 = PromptCandidate(candidate_id="s3", prompt_text="original prompt")  # Duplicate of parent
        s4 = PromptCandidate(candidate_id="s4", prompt_text="new prompt B")

        sampled, stats = deduplicate_and_sample_successors(parent, [s1, s2, s3, s4], max_successors=2, seed=42)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(stats["duplicate_count"], 2)
        self.assertEqual(stats["unique_count"], 2)

    def test_lineage_tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PromptLineageTracker()
            c0 = PromptCandidate(candidate_id="P0", prompt_text="root", round_idx=0)
            c1 = PromptCandidate(candidate_id="C1", prompt_text="child", parent_id="P0", round_idx=1)
            tracker.register_candidate(c0)
            tracker.register_candidate(c1)
            tracker.register_candidate(c1)

            json_out = Path(tmpdir) / "lineage.json"
            dot_out = Path(tmpdir) / "lineage.dot"
            tracker.export_json(json_out, final_candidate_id="C1")
            tracker.export_dot(dot_out, final_candidate_id="C1")

            self.assertTrue(json_out.is_file())
            self.assertTrue(dot_out.is_file())
            data = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(data["total_nodes"], 2)
            self.assertEqual(data["final_candidate_id"], "C1")
            self.assertEqual(data["trace_to_root"], ["P0", "C1"])
            self.assertEqual(data["total_edges"], 1)
            self.assertEqual(data["nodes"]["C1"]["prompt_text"], "child")

    def test_entity_cache_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            manager = EntityCacheManager(cache_dir)

            class MockEvaluator:
                max_workers = 8

                def predict_stage1_window(self, text, prompt):
                    return [{"id": "E1", "type": "Vulnerability", "start": 0, "end": 4}]

            samples = [{
                "sample_id": "doc1_w0",
                "text": "text sample",
                "gold_entities": [],
                "gold_relations": [{"type": "affects", "head": "E1", "tail": "E2"}],
            }]
            prompt = "Sample prompt"
            cache_file = manager.build_and_save_cache(MockEvaluator(), prompt, samples, "train")
            self.assertTrue(cache_file.is_file())

            # Load matching hash
            loaded = manager.load_cache(
                "train",
                expected_prompt_hash=compute_prompt_hash(prompt),
                require_gold_relations=True,
                expected_task_max_workers=8,
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["fixed_entities"][0]["id"], "E1")

            # Mismatched hash raises ValueError
            with self.assertRaises(ValueError):
                manager.load_cache("train", expected_prompt_hash="wrong_hash")

            self.assertEqual(compute_prompt_hash("a\r\nb\n"), compute_prompt_hash("a\nb"))

    def test_entity_p0_schema_consistency(self):
        """验证 ENTITY_PROMPT_P0 与 schema.py 中定义的实体类型完全一致。"""
        val_res = PromptContractValidator.validate_entity_prompt(ENTITY_PROMPT_P0)
        self.assertTrue(val_res.is_valid, f"ENTITY_PROMPT_P0 结构校验失败: {val_res.error_message}")
        for ent_type in EXTRACTION_ENTITY_TYPES:
            self.assertIn(ent_type.lower(), ENTITY_PROMPT_P0.lower())

    def test_relation_p0_schema_consistency(self):
        """验证 RELATION_PROMPT_P0 与 schema.py 中定义的关系类型完全一致。"""
        val_res = PromptContractValidator.validate_relation_prompt(RELATION_PROMPT_P0)
        self.assertTrue(val_res.is_valid, f"RELATION_PROMPT_P0 结构校验失败: {val_res.error_message}")
        for rel_type in EXTRACTION_RELATION_TYPES:
            self.assertIn(rel_type.lower(), RELATION_PROMPT_P0.lower())

    def test_relation_direction_consistency(self):
        """验证 RELATION_PROMPT_P0 中的关系方向严格匹配 schema.py 定义：
        affects: Vulnerability -> Configuration
        instantiates: Vulnerability -> Weakness
        exploited_by: Vulnerability -> AttackTechnique
        """
        for rel, (head, tail) in EXTRACTION_RELATION_ARGUMENT_TYPES.items():
            expected_pattern = f"{head} -> {tail}"
            self.assertIn(
                expected_pattern.lower(),
                RELATION_PROMPT_P0.lower(),
                f"关系 {rel} 未在 RELATION_PROMPT_P0 中找到标准方向定义 {expected_pattern}"
            )

    def test_required_placeholders(self):
        """验证输入占位符在 P0 提示词中正确保留。"""
        self.assertIn("{text}", ENTITY_PROMPT_P0)
        self.assertIn("{text}", RELATION_PROMPT_P0)
        self.assertIn("{entities}", RELATION_PROMPT_P0)

    def test_candidate_contract_validator(self):
        """测试 PromptContractValidator 能准确拦截结构缺陷或方向倒置的非法 Prompt。"""
        # 1. 正常合法 Prompt 应通过
        self.assertTrue(PromptContractValidator.validate_entity_prompt(ENTITY_PROMPT_P0).is_valid)
        self.assertTrue(PromptContractValidator.validate_relation_prompt(RELATION_PROMPT_P0).is_valid)

        # 2. 缺少 {text} 占位符应被拦截
        bad_entity = ENTITY_PROMPT_P0.replace("{text}", "NO_PLACEHOLDER")
        val_missing_text = PromptContractValidator.validate_entity_prompt(bad_entity)
        self.assertFalse(val_missing_text.is_valid)
        self.assertTrue(any("冻结契约" in r for r in val_missing_text.reasons))

        # 3. 故意反转关系方向：affects: Configuration -> Vulnerability 必须被严密拦截
        inverted_relation = RELATION_PROMPT_P0.replace(
            "affects: Vulnerability -> Configuration",
            "affects: Configuration -> Vulnerability"
        )
        val_inverted = PromptContractValidator.validate_relation_prompt(inverted_relation)
        self.assertFalse(val_inverted.is_valid, "未能拦截反转关系方向 Configuration -> Vulnerability！")
        self.assertTrue(any("冻结契约" in r for r in val_inverted.reasons))

        # 4. Stage 1 guidance 不得引入关系输出或重定义标签。
        leaked_relation = replace_optimizable_guidance(
            ENTITY_PROMPT_P0,
            "affects: Vulnerability -> Configuration; output source and target.",
        )
        self.assertFalse(
            PromptContractValidator.validate_entity_prompt(leaked_relation).is_valid
        )

    def test_unconstrained_scope_keeps_only_runtime_interface(self):
        """无约束臂允许语义契约变化，但必须保留固定评价接口。"""
        rewritten = ENTITY_PROMPT_P0.replace(
            "A description-only phrase without a locally explicit CWE is out of scope.",
            "A description-only phrase may be inferred from context.",
        )
        self.assertFalse(
            PromptContractValidator.validate_candidate(
                "entity", rewritten, prompt_scope="constrained"
            ).is_valid
        )
        self.assertTrue(
            PromptContractValidator.validate_candidate(
                "entity", rewritten, prompt_scope="unconstrained"
            ).is_valid
        )

        missing_placeholder = rewritten.replace("{text}", "TEXT_HERE")
        invalid = PromptContractValidator.validate_candidate(
            "entity", missing_placeholder, prompt_scope="unconstrained"
        )
        self.assertFalse(invalid.is_valid)
        self.assertTrue(any("{text}" in reason for reason in invalid.reasons))

    def test_prompt_editor_only_replaces_guidance(self):
        class FakeClient:
            config = {}

            @staticmethod
            def call_fn(**kwargs):
                return "<START>Check all spans, then apply the frozen rules conservatively.</END>"

        parent = PromptCandidate(candidate_id="P_E0", prompt_text=ENTITY_PROMPT_P0)
        gradient = PromptGradient(
            gradient_id="g1",
            parent_prompt_id="P_E0",
            error_group_id="eg1",
            gradient_text="The guidance lacks an explicit verification pass.",
            round_idx=0,
        )
        edited = PromptEditor(FakeClient()).edit_prompt(parent, gradient, [])
        self.assertIsNotNone(edited)
        self.assertEqual(
            extract_immutable_contract(edited.prompt_text),
            extract_immutable_contract(ENTITY_PROMPT_P0),
        )
        self.assertNotEqual(
            extract_optimizable_guidance(edited.prompt_text),
            extract_optimizable_guidance(ENTITY_PROMPT_P0),
        )

    def test_unconstrained_prompt_editor_rewrites_complete_prompt(self):
        rewritten_prompt = ENTITY_PROMPT_P0.replace(
            "Apply the frozen definitions conservatively",
            "Apply the task definitions with a two-pass verification procedure",
        )

        class FakeClient:
            config = {}

            @staticmethod
            def call_fn(**kwargs):
                return f"<START>{rewritten_prompt}<END>"

        parent = PromptCandidate(candidate_id="P_E0", prompt_text=ENTITY_PROMPT_P0)
        gradient = PromptGradient(
            gradient_id="g_full",
            parent_prompt_id="P_E0",
            error_group_id="eg_full",
            gradient_text="The complete prompt lacks a verification procedure.",
            round_idx=0,
        )
        edited = PromptEditor(
            FakeClient(), prompt_scope="unconstrained"
        ).edit_prompt(parent, gradient, [])
        self.assertIsNotNone(edited)
        self.assertEqual(edited.prompt_text, rewritten_prompt)
        self.assertEqual(edited.generation_type, "gradient_edit_full")
        self.assertTrue(
            PromptContractValidator.validate_candidate(
                "entity", edited.prompt_text, prompt_scope="unconstrained"
            ).is_valid
        )

    def test_relation_evaluator_uses_gold_entity_mapping(self):
        class FakeClient:
            config = {}

        evaluator = TaskEvaluator(task_client=FakeClient())
        evaluator.predict_stage2_window = lambda text, entities, prompt, sample_id=None: [
            {"type": "affects", "head": "P1", "tail": "P2"}
        ]
        sample = {
            "sample_id": "s1",
            "text": "CVE product",
            "fixed_entities": [
                {"id": "P1", "type": "Vulnerability", "start": 0, "end": 3},
                {"id": "P2", "type": "Configuration", "start": 4, "end": 11},
            ],
            "gold_entities": [
                {"id": "G1", "type": "Vulnerability", "start": 0, "end": 3},
                {"id": "G2", "type": "Configuration", "start": 4, "end": 11},
            ],
            "gold_relations": [{"type": "affects", "head": "G1", "tail": "G2"}],
        }
        result, _ = evaluator.evaluate_stage2_batch([sample], "prompt", collect_errors=False)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))

    def test_ucb_total_budget(self):
        """确认 UCB 评估拉动总数严格等于 total_pull_budget_per_round。"""
        selector = UCBPromptSelector(
            c=2.0,
            total_pull_budget_per_round=64,
            batch_size=8,
            min_pulls_per_candidate=2,
        )
        cands = [PromptCandidate(candidate_id=f"c_{i}", prompt_text=f"prompt {i}") for i in range(10)]
        def mock_eval(c, pull_idx):
            return EvaluationResult(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        history = selector.execute_evaluation_budget(cands, mock_eval)
        self.assertEqual(len(history), 64)
        self.assertEqual(sum(c.num_evaluations for c in cands), 64)
        for cand in cands:
            first_indices = [
                item["candidate_pull_index"]
                for item in history
                if item["candidate_id"] == cand.candidate_id
            ][:2]
            self.assertEqual(first_indices, [0, 1])

    def test_ucb_rejects_insufficient_minimum_budget(self):
        selector = UCBPromptSelector(
            total_pull_budget_per_round=3,
            min_pulls_per_candidate=2,
        )
        candidates = [
            PromptCandidate(candidate_id="c1", prompt_text="p1"),
            PromptCandidate(candidate_id="c2", prompt_text="p2"),
        ]
        with self.assertRaises(ValueError):
            selector.execute_evaluation_budget(candidates, lambda c, i: None)

    def test_uniform_total_budget(self):
        """确认 Uniform 评估拉动总数严格等于 total_pull_budget_per_round。"""
        selector = UniformPromptSelector(total_pull_budget_per_round=64, batch_size=8)
        cands = [PromptCandidate(candidate_id=f"c_{i}", prompt_text=f"prompt {i}") for i in range(10)]
        def mock_eval(c, pull_idx):
            return EvaluationResult(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        history = selector.execute_evaluation_budget(cands, mock_eval)
        self.assertEqual(len(history), 64)
        self.assertEqual(sum(c.num_evaluations for c in cands), 64)

    def test_uniform_equal_allocation(self):
        """确认 Uniform 在候选之间拉动次数最大差距不超过 1。"""
        selector = UniformPromptSelector(total_pull_budget_per_round=64, batch_size=8)
        cands = [PromptCandidate(candidate_id=f"c_{i}", prompt_text=f"prompt {i}") for i in range(10)]
        def mock_eval(c, pull_idx):
            return EvaluationResult(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        selector.execute_evaluation_budget(cands, mock_eval)
        pulls = [c.num_evaluations for c in cands]
        # 64 pulls among 10 candidates -> 4 candidates get 7, 6 candidates get 6 -> diff == 1
        self.assertLessEqual(max(pulls) - min(pulls), 1)
        self.assertEqual(sum(pulls), 64)

    def test_ucb_uniform_same_total_budget(self):
        """确认在相同候选集与配置下，UCB 与 Uniform 的总 Pull 数完全相等。"""
        total_budget = 64
        ucb_selector = UCBPromptSelector(c=2.0, total_pull_budget_per_round=total_budget, batch_size=8)
        uni_selector = UniformPromptSelector(total_pull_budget_per_round=total_budget, batch_size=8)
        ucb_cands = [PromptCandidate(candidate_id=f"ucb_{i}", prompt_text=f"p {i}") for i in range(16)]
        uni_cands = [PromptCandidate(candidate_id=f"uni_{i}", prompt_text=f"p {i}") for i in range(16)]
        def mock_eval(c, pull_idx):
            return EvaluationResult(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        ucb_hist = ucb_selector.execute_evaluation_budget(ucb_cands, mock_eval)
        uni_hist = uni_selector.execute_evaluation_budget(uni_cands, mock_eval)
        self.assertEqual(len(ucb_hist), total_budget)
        self.assertEqual(len(uni_hist), total_budget)
        self.assertEqual(sum(c.num_evaluations for c in ucb_cands), sum(c.num_evaluations for c in uni_cands))
        self.assertEqual(sum(c.num_evaluations for c in ucb_cands), total_budget)

    def test_p0_not_in_final_protegi_selection(self):
        """确保 Initial P0 baseline 绝不进入 Full ProTeGi 的最终决选（杜绝 APO-v2 回退 P0 逻辑）。"""
        p0 = PromptCandidate(
            candidate_id="P_E0",
            prompt_text="p0 text",
            generation_type="initial",
            metrics={"dev_f1": 0.99},
        )
        evolved_1 = PromptCandidate(
            candidate_id="c_edit_1",
            prompt_text="evolved 1",
            generation_type="gradient_edit",
            metrics={"dev_f1": 0.85},
        )
        evolved_2 = PromptCandidate(
            candidate_id="c_para_2",
            prompt_text="evolved 2",
            generation_type="paraphrase",
            metrics={"dev_f1": 0.88},
        )
        beam = [p0, evolved_1, evolved_2]

        finalists = [c for c in beam if c.generation_type != "initial"]
        def final_sort_key(c: PromptCandidate):
            dev_f1 = c.metrics.get("dev_f1", 0.0)
            dev_prec = c.metrics.get("dev_precision", 0.0)
            return (dev_f1, dev_prec, -len(c.prompt_text), c.candidate_id)

        winner = max(finalists, key=final_sort_key)
        self.assertNotEqual(winner.candidate_id, "P_E0")
        self.assertEqual(winner.candidate_id, "c_para_2")

    def test_explicit_abbreviation_context_is_lexical_only(self):
        text = (
            "Acme Secure Gateway (ASG) was assessed. ASG remained online. "
            "NCE (Network Control Engine) was mentioned separately."
        )
        pairs = extract_explicit_abbreviation_pairs(text)
        self.assertEqual(
            [(item["short_form"], item["long_form"]) for item in pairs],
            [
                ("ASG", "Acme Secure Gateway"),
                ("NCE", "Network Control Engine"),
            ],
        )
        self.assertEqual(extract_explicit_abbreviation_pairs("ASG remained online."), [])
        rendered = format_abbreviation_context(pairs)
        self.assertIn("ASG = Acme Secure Gateway", rendered)
        self.assertNotIn("Configuration", rendered)

    def test_document_context_is_fixed_runtime_input(self):
        class CapturingClient:
            config = {}

            def __init__(self):
                self.prompt = None

            def call_fn(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return '{"entities": []}'

        client = CapturingClient()
        evaluator = TaskEvaluator(task_client=client)
        evaluator.predict_stage1_window(
            "ASG is affected.",
            ENTITY_PROMPT_P0,
            "ASG = Acme Secure Gateway",
        )
        self.assertIn("explicit-abbreviation-context-v1", client.prompt)
        self.assertIn("not automatically a Configuration", client.prompt)
        self.assertIn("ASG is affected.", client.prompt)

    def test_configuration_overlap_metric_is_diagnostic(self):
        gold = [{"type": "Configuration", "start": 10, "end": 20}]
        pred = [{"type": "Configuration", "start": 10, "end": 18}]
        self.assertEqual(
            calc_strict_entity_sample_counts(
                pred, gold, allowed_types={"Configuration"}
            ),
            (0, 1, 1),
        )
        self.assertEqual(
            calc_same_type_jaccard_overlap_counts(
                pred,
                gold,
                allowed_types={"Configuration"},
                threshold=0.5,
            ),
            (1, 0, 0),
        )

    def test_configuration_error_focus_projects_other_types_away(self):
        error = ErrorExample(
            sample_id="s1",
            input_text="CVE-1 affects Acme Gateway.",
            gold_output={"entities": [
                {"type": "Vulnerability", "start": 0, "end": 5},
                {"type": "Configuration", "start": 14, "end": 26},
            ]},
            predicted_output={"entities": [
                {"type": "Weakness", "start": 0, "end": 5},
                {"type": "Configuration", "start": 14, "end": 21},
            ]},
        )
        focused = focus_entity_error_examples([error], "Configuration")
        self.assertEqual(len(focused), 1)
        self.assertEqual(
            {entity["type"] for entity in focused[0].gold_output["entities"]},
            {"Configuration"},
        )
        self.assertEqual(
            focused[0].error_details["boundary_overlap_pairs"],
            1,
        )
        self.assertEqual(focused[0].error_details["source_split"], "train")

    def test_selector_can_use_configuration_strict_f1(self):
        selector = UCBPromptSelector(
            total_pull_budget_per_round=2,
            batch_size=1,
            objective_entity_type="Configuration",
        )
        candidate = PromptCandidate(candidate_id="c1", prompt_text="prompt")
        result = EvaluationResult(
            tp=10,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            details={"by_type": {
                "Configuration": {
                    "tp": 1,
                    "fp": 1,
                    "fn": 1,
                    "f1": 0.5,
                }
            }},
        )
        selector.update_candidate_with_batch(candidate, result, 1)
        self.assertEqual((candidate.tp, candidate.fp, candidate.fn), (1, 1, 1))
        self.assertEqual(candidate.estimated_reward, 0.5)

    def test_final_selection_uses_strict_configuration_and_guardrails(self):
        p0 = PromptCandidate("P_E0", "p0", generation_type="initial")
        unsafe = PromptCandidate("unsafe", "unsafe prompt")
        safe = PromptCandidate("safe", "safe prompt")

        def result(config_f1, vulnerability_f1, overall_f1):
            return EvaluationResult(
                tp=8,
                fp=2,
                fn=2,
                precision=0.8,
                recall=0.8,
                f1=overall_f1,
                details={"by_type": {
                    "Configuration": {"f1": config_f1, "applicable": True},
                    "Vulnerability": {
                        "f1": vulnerability_f1,
                        "applicable": True,
                    },
                    "Weakness": {"f1": 0.7, "applicable": True},
                    "AttackTechnique": {"f1": 0.8, "applicable": True},
                }},
            )

        winner, audits = select_final_candidate(
            [p0, unsafe, safe],
            {
                "P_E0": result(0.30, 0.90, 0.80),
                "unsafe": result(0.60, 0.80, 0.86),
                "safe": result(0.45, 0.89, 0.83),
            },
            p0_candidate_id="P_E0",
            objective_entity_type="Configuration",
            guardrail_entity_types=[
                "Vulnerability",
                "Weakness",
                "AttackTechnique",
            ],
            guardrail_max_f1_drop=0.02,
        )
        self.assertEqual(winner.candidate_id, "safe")
        self.assertFalse(audits["unsafe"]["eligible"])
        self.assertFalse(audits["safe"]["overlap_metrics_used_for_selection"])

    def test_retry_api_call_success(self):
        from protegi.retry_utils import retry_api_call

        attempts = 0

        def flaky_call():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionResetError("Remote disconnected")
            return "success"

        res = retry_api_call(flaky_call, max_retries=4, initial_delay=0.01, backoff_factor=1.5)
        self.assertEqual(res, "success")
        self.assertEqual(attempts, 3)

    def test_retry_api_call_fatal_error(self):
        from protegi.retry_utils import retry_api_call

        def fatal_call():
            raise ValueError("Invalid format")

        with self.assertRaises(ValueError):
            retry_api_call(fatal_call, max_retries=3, initial_delay=0.01)

    def test_normalize_configuration_entity(self):
        from protegi.entity_backfill import normalize_configuration_entity

        # Trailing noise should be trimmed
        raw_ent = {
            "id": "E1",
            "text": "Microsoft Exchange application",
            "type": "Configuration",
            "start": 10,
            "end": 40,
        }
        cleaned = normalize_configuration_entity(raw_ent)
        self.assertEqual(cleaned["text"], "Microsoft Exchange")
        self.assertEqual(cleaned["end"], 28)

        # Protected official Server product should be preserved intact
        protected_win = {
            "id": "E2",
            "text": "Windows Server",
            "type": "Configuration",
            "start": 50,
            "end": 64,
        }
        self.assertEqual(normalize_configuration_entity(protected_win), protected_win)

        protected_ex = {
            "id": "E3",
            "text": "Exchange Server",
            "type": "Configuration",
            "start": 0,
            "end": 15,
        }
        self.assertEqual(normalize_configuration_entity(protected_ex), protected_ex)

        # Non-configuration should not be touched
        vuln_ent = {
            "id": "E4",
            "text": "CVE-2021-1234 server",
            "type": "Vulnerability",
            "start": 0,
            "end": 20,
        }
        self.assertEqual(normalize_configuration_entity(vuln_ent), vuln_ent)

    def test_vulnerability_anchored_backfill_logic(self):
        from protegi.entity_backfill import vulnerability_anchored_backfill

        samples = [
            {
                "sample_id": "doc1_w0",
                "doc_id": "doc1",
                "text": "Attackers targeted Microsoft Exchange application in 2021.\n\nCVE-2021-26855 affects Exchange severely.",
            },
            {
                "sample_id": "doc1_w1",
                "doc_id": "doc1",
                "text": "Administrators should review Exchange logs. Moreover, CVE-2021-26858 in Exchange allows file writes.",
            },
            {
                "sample_id": "doc2_w0",
                "doc_id": "doc2",
                "text": "CVE-2023-1234 was found in unknown product.",
            },
        ]

        # Model predicted "Microsoft Exchange application" in doc1_w0
        # and nothing in doc1_w1 or doc2_w0
        predictions = [
            [
                {
                    "id": "E1",
                    "text": "Microsoft Exchange application",
                    "type": "Configuration",
                    "start": 19,
                    "end": 49,
                    "normalized_id": "cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*",
                }
            ],
            [],
            [],
        ]

        updated = vulnerability_anchored_backfill(samples, predictions, enabled=True)

        # In doc1_w0: "Microsoft Exchange application" trimmed to "Microsoft Exchange" (19:37).
        # Also in doc1_w0: second sentence has CVE-2021-26855 and "Exchange".
        self.assertEqual(updated[0][0]["text"], "Microsoft Exchange")
        self.assertEqual(updated[0][0]["end"], 37)
        self.assertTrue(any(e["text"] == "Exchange" and e.get("_source") == "cve_anchored_backfill" for e in updated[0]))

        # In doc1_w1:
        # Sentence 1: "Administrators should review Exchange logs." (NO CVE -> NO backfill!)
        # Sentence 2: "Moreover, CVE-2021-26858 in Exchange allows file writes." (HAS CVE -> BACKFILLED!)
        self.assertEqual(len(updated[1]), 1)
        self.assertEqual(updated[1][0]["text"], "Exchange")
        self.assertEqual(updated[1][0]["_source"], "cve_anchored_backfill")
        self.assertEqual(updated[1][0]["normalized_id"], "cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*")

        # In doc2_w0:
        # doc2 has NO seed -> no backfill, len is 0 (no cross-doc leakage!)
        self.assertEqual(len(updated[2]), 0)

    def test_task_evaluator_backfill_integration(self):
        class FakeClient:
            config = {}

        evaluator = TaskEvaluator(
            task_client=FakeClient(),
            vulnerability_anchored_backfill=True,
        )
        evaluator.predict_stage1_texts = lambda texts, prompt, document_abbreviations=None, sample_ids=None: [
            [{"id": "E1", "text": "Microsoft Exchange application", "type": "Configuration", "start": 0, "end": 30}],
            [],
        ]
        samples = [
            {"sample_id": "docA_w0", "doc_id": "docA", "text": "Microsoft Exchange application in action."},
            {"sample_id": "docA_w1", "doc_id": "docA", "text": "CVE-2021-1234 affects Exchange today."},
        ]
        res, _ = evaluator.evaluate_stage1_batch(samples, "prompt", capture_predictions=True)
        preds = res.details["predictions"]
        # w0 entity trimmed to Microsoft Exchange
        self.assertEqual(preds[0]["pred_entities"][0]["text"], "Microsoft Exchange")
        # w1 entity backfilled with Exchange
        self.assertEqual(len(preds[1]["pred_entities"]), 1)
        self.assertEqual(preds[1]["pred_entities"][0]["text"], "Exchange")
        self.assertEqual(preds[1]["pred_entities"][0]["_source"], "cve_anchored_backfill")

    def test_snap_to_sentence_boundary_prefers_cve_and_paragraphs(self):
        from llm_methods import snap_to_sentence_boundary

        sample_text = (
            "Prefix filler text that occupies some length.\n"
            "CVE-2023-9999\n"
            " (CWE-100)\n"
            "An arbitrary code execution in Acme System allows remote attackers."
        )
        # raw_start points into middle of "Acme System allows..."
        raw_idx = sample_text.find("allows")
        snapped = snap_to_sentence_boundary(
            sample_text,
            raw_start=raw_idx,
            min_start=0,
            max_start=len(sample_text),
            radius=200,
        )
        # Should snap backward to the start of "CVE-2023-9999"
        expected_cve_start = sample_text.find("CVE-2023-9999")
        self.assertEqual(snapped, expected_cve_start)

    def test_build_text_windows_sentence_snapping_avoids_headless_fragments(self):
        from llm_methods import build_text_windows

        # Construct text where raw jump would cut a sentence in half
        block1 = "P" * 2800 + ".\n"
        cve_sentence = "CVE-2023-1234 is a severe vulnerability in Acme Gateway and below.\n"
        block2 = "S" * 2000 + ".\n"
        full_text = block1 + cve_sentence + block2

        # 1. Legacy mode (snap_sentence_boundary=False)
        legacy_windows = build_text_windows(full_text, max_chars=3000, overlap=400, snap_sentence_boundary=False)
        # 2. Snapped mode (snap_sentence_boundary=True)
        snapped_windows = build_text_windows(full_text, max_chars=3000, overlap=400, snap_sentence_boundary=True)

        self.assertGreaterEqual(len(snapped_windows), 2)
        # Snapped window 1 should start cleanly at a sentence or CVE boundary
        win1_text = snapped_windows[1]["text"]
        self.assertTrue(win1_text.startswith("CVE-2023-1234") or win1_text.startswith("P") or win1_text.startswith("S"))

    def test_dense_split_off_is_byte_identical_to_legacy(self):
        from llm_methods import build_text_windows

        doc = (
            V6_ROOT / "data" / "annotations" / "gold" / "aa24-207a.json"
        ).read_text(encoding="utf-8")
        import json as _json

        text = _json.loads(doc)["text"]
        legacy = build_text_windows(text, max_chars=3000, overlap=400)
        off = build_text_windows(
            text, max_chars=3000, overlap=400, dense_run_split=False,
            dense_min_ids=25, dense_min_span=1000, dense_gap=120,
            dense_max_ids=20,
        )
        self.assertEqual(off, legacy)
        # None（环境默认关闭）同样一致。
        none = build_text_windows(text, max_chars=3000, overlap=400)
        self.assertEqual(none, legacy)

    def test_find_flat_dense_runs_needs_no_gold(self):
        from llm_methods import find_flat_dense_runs

        flat = " ".join(f"CVE-2023-{10000 + i:05d} Product{i}" for i in range(45))
        self.assertGreater(len(flat), 1000)
        runs = find_flat_dense_runs(
            flat, min_ids=25, min_span=1000, gap=120
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]["hits"]), 45)
        # 同样内容一旦保留换行（表格形态），即不定为拍扁 run。
        tabled = "\n".join(
            f"CVE-2023-{10000 + i:05d} Product{i}" for i in range(45)
        )
        self.assertEqual(
            find_flat_dense_runs(
                tabled, min_ids=25, min_span=1000, gap=120
            ),
            [],
        )

    def test_dense_split_killer_window_into_bounded_chunks(self):
        import json as _json
        import re as _re
        from llm_methods import (
            build_text_windows,
            find_flat_dense_runs,
        )

        doc = _json.loads(
            (V6_ROOT / "data" / "annotations" / "gold" / "aa24-207a.json")
            .read_text(encoding="utf-8")
        )
        legacy = build_text_windows(doc["text"], max_chars=3000, overlap=400)
        base = legacy[5]
        self.assertEqual((base["start"], base["end"]), (11830, 14711))
        params = dict(
            dense_run_split=True, dense_min_ids=25, dense_min_span=1000,
            dense_gap=120, dense_max_ids=20, dense_seam=100,
        )
        subs = build_text_windows(
            doc["text"], max_chars=3000, overlap=400, **params
        )
        # 基窗 w5 被替换为覆盖它的子窗；其余窗逐字节不变。
        self.assertGreater(len(subs), len(legacy))
        killers = [w for w in subs if w["start"] >= 11830 and w["end"] <= 14711]
        self.assertGreaterEqual(len(killers), 2)
        id_re = _re.compile(r"CVE-\d{4}-\d+|CWE-\d+|\bT\d{4}(?:\.\d{3})?\b")
        for sub in killers:
            self.assertLessEqual(len(sub["text"]), 3000)
            # 不动点性质：子窗内不再含 qualifying run（同参数复测）。
            self.assertEqual(
                find_flat_dense_runs(
                    sub["text"], min_ids=25, min_span=1000, gap=120
                ),
                [],
            )
            # 经验安全界内（可正常窗 317a_w4 含 36 个 ID）。
            self.assertLessEqual(len(id_re.findall(sub["text"])), 32)
        # 全覆盖：基窗字符每点至少被一子窗覆盖。
        covered = bytearray(len(base["text"]))
        for sub in killers:
            for pos in range(sub["start"] - 11830, sub["end"] - 11830):
                covered[pos] = 1
        self.assertTrue(all(covered))
        # 非 killer 窗不受影响。
        untouched_legacy = [w for w in legacy if w["end"] <= 11830]
        untouched_new = [w for w in subs if w["end"] <= 11830]
        self.assertEqual(untouched_new, untouched_legacy)

    def test_window_split_version_descriptors(self):
        from llm_methods import window_split_version

        self.assertEqual(window_split_version(False), "window-split-v1")
        self.assertEqual(
            window_split_version(True, 25, 1000, 120, 20, 100),
            "window-split-v2:dense25-1000-120-20-s100",
        )

    def test_backfill_3char_acronym_and_abbreviation_pairing(self):
        from protegi.entity_backfill import (
            harvest_document_configuration_seeds,
            vulnerability_anchored_backfill,
            is_valid_configuration_seed,
        )

        # 1. Validation test
        self.assertTrue(is_valid_configuration_seed("ZCS"))
        self.assertFalse(is_valid_configuration_seed("rdp"))
        self.assertFalse(is_valid_configuration_seed("ssh"))
        self.assertFalse(is_valid_configuration_seed("cve"))

        # 2. Abbreviation pairing test: seed long form -> short form derived
        items = [
            (
                {
                    "sample_id": "docZ_w0",
                    "doc_id": "docZ",
                    "text": "Zimbra Collaboration Suite (ZCS) is vulnerable.",
                    "document_abbreviations": "- ZCS = Zimbra Collaboration Suite",
                },
                [
                    {
                        "id": "E1",
                        "text": "Zimbra Collaboration Suite",
                        "type": "Configuration",
                        "normalized_id": "cpe:2.3:a:zimbra:collaboration:*:*:*:*:*:*:*:*",
                    }
                ],
            )
        ]
        seeds = harvest_document_configuration_seeds(items)
        self.assertIn("Zimbra Collaboration Suite", seeds)
        self.assertIn("ZCS", seeds)
        self.assertEqual(seeds["ZCS"], "cpe:2.3:a:zimbra:collaboration:*:*:*:*:*:*:*:*")

        # 3. CVE backfill with 3-character acronym and optional CVE whitespace
        samples = [
            items[0][0],
            {
                "sample_id": "docZ_w1",
                "doc_id": "docZ",
                "text": "Threat actors targeted CVE - 2022-27925 in unpatched ZCS deployments.",
            },
        ]
        preds = [items[0][1], []]
        updated = vulnerability_anchored_backfill(samples, preds, enabled=True)
        self.assertEqual(len(updated[1]), 1)
        self.assertEqual(updated[1][0]["text"], "ZCS")
        self.assertEqual(updated[1][0]["_source"], "cve_anchored_backfill")
        self.assertEqual(updated[1][0]["normalized_id"], "cpe:2.3:a:zimbra:collaboration:*:*:*:*:*:*:*:*")

    def test_backfill_server_product_dual_derivation(self):
        from protegi.entity_backfill import (
            harvest_document_configuration_seeds,
            vulnerability_anchored_backfill,
        )

        items = [
            (
                {
                    "sample_id": "docEx_w0",
                    "doc_id": "docEx",
                    "text": "Microsoft Exchange Server has multiple zero-day vulnerabilities.",
                },
                [
                    {
                        "id": "E1",
                        "text": "Microsoft Exchange Server",
                        "type": "Configuration",
                        "normalized_id": "cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*",
                    }
                ],
            )
        ]
        seeds = harvest_document_configuration_seeds(items)
        # Should derive Exchange Server, Microsoft Exchange, and Exchange
        self.assertIn("Microsoft Exchange Server", seeds)
        self.assertIn("Microsoft Exchange", seeds)
        self.assertIn("Exchange Server", seeds)
        self.assertIn("Exchange", seeds)

        # Backfill window with only "Exchange" and CVE
        samples = [
            items[0][0],
            {
                "sample_id": "docEx_w1",
                "doc_id": "docEx",
                "text": "A remote code execution CVE-2021-26855 in Exchange was observed in the wild.",
            },
        ]
        preds = [items[0][1], []]
        updated = vulnerability_anchored_backfill(samples, preds, enabled=True)
        self.assertEqual(len(updated[1]), 1)
        self.assertEqual(updated[1][0]["text"], "Exchange")
        self.assertEqual(updated[1][0]["_source"], "cve_anchored_backfill")


class TestProTeGiFormalArtifactValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        cls.freeze_manifest = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        cls.split_sha = hashlib.sha256(cls.split_file.read_bytes()).hexdigest()
        cls.freeze_sha = hashlib.sha256(cls.freeze_manifest.read_bytes()).hexdigest()
        fm_data = json.loads(cls.freeze_manifest.read_text(encoding="utf-8"))
        cls.gold_agg_sha = fm_data["gold_aggregate_sha256"]
        cls.window_construction = fm_data["window_construction"]["version"]

    def _create_valid_artifact_fixture(self, tmp_dir: Path) -> tuple[Path, dict]:
        ent_prompt = tmp_dir / "valid_entity_prompt.txt"
        ent_prompt.write_text("Valid entity prompt text", encoding="utf-8")
        rel_prompt = tmp_dir / "valid_relation_prompt.txt"
        rel_prompt.write_text("Valid relation prompt text", encoding="utf-8")

        artifact = {
            "artifact_version": "protegi-final-v1",
            "schema_version": "chapter3-no-capec-v1",
            "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
            "boundary_contract_version": "chapter3-boundary-sync-v2",
            "prompt_scope": "constrained",
            "split_file": "data/train_dev_test_split_v7.json",
            "split_sha256": self.split_sha,
            "dataset_freeze_manifest_path": "data/dataset_freeze_manifest_v6.json",
            "dataset_freeze_manifest_sha256": self.freeze_sha,
            "gold_aggregate_sha256": self.gold_agg_sha,
            "entity_prompt_path": str(ent_prompt),
            "entity_prompt_sha256": hashlib.sha256(ent_prompt.read_bytes()).hexdigest(),
            "relation_prompt_path": str(rel_prompt),
            "relation_prompt_sha256": hashlib.sha256(rel_prompt.read_bytes()).hexdigest(),
            "window_construction": self.window_construction,
            "task_runtime": {
                "model": "hy3",
                "max_workers": 8,
                "temperature": 0.0,
                "thinking": "disabled",
                "reasoning_effort": "none",
                "top_p": 0.95,
                "max_tokens": 4096,
                "window_chars": 3000,
                "window_overlap": 400,
                "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            },
            "frozen_for_test": True,
            "formal_eligible": True,
            "test_gold_loaded": False,
            "test_predictions_generated": False,
        }
        art_path = tmp_dir / "protegi_final_artifact.json"
        art_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        return art_path, artifact

    def test_protegi_artifact_requires_current_boundary_contract(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["boundary_contract_version"] = "tampered-boundary-v0"
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("boundary_contract_version", str(ctx.exception))

    def test_protegi_artifact_requires_current_annotation_protocol(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["annotation_protocol_version"] = "tampered-protocol-v0"
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("annotation_protocol_version", str(ctx.exception))

    def test_protegi_artifact_requires_current_split_hash(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["split_sha256"] = "0" * 64
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("划分哈希", str(ctx.exception))

    def test_protegi_artifact_requires_current_gold_hash(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["gold_aggregate_sha256"] = "0" * 64
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("Gold 聚合哈希", str(ctx.exception))

    def test_protegi_artifact_requires_entity_prompt_hash(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["entity_prompt_sha256"] = "0" * 64
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("实体提示词内容哈希", str(ctx.exception))

    def test_protegi_artifact_requires_relation_prompt_hash(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["relation_prompt_sha256"] = "0" * 64
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("关系提示词内容哈希", str(ctx.exception))

    def test_legacy_apo_cannot_be_promoted_as_protegi(self):
        from promote_protegi_v6 import promote_protegi

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            apo_prompt = tmp / "final_prompt.json"
            apo_prompt.write_text(
                json.dumps({
                    "stage1_guidance": "legacy guidance",
                    "stage2_guidance": "legacy guidance",
                    "apo_candidate_selected": True,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                promote_protegi(
                    entity_prompt_path=apo_prompt,
                    relation_prompt_path=tmp / "rel.txt",
                    entity_summary_path=tmp / "ent_summary.json",
                    relation_summary_path=tmp / "rel_summary.json",
                    entity_cache_train_manifest_path=tmp / "train_manifest.json",
                    entity_cache_dev_manifest_path=tmp / "dev_manifest.json",
                )
            self.assertIn("APO", str(ctx.exception))

    def test_old_entity_cache_rejected_after_gold_hash_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_mgr = EntityCacheManager(tmp)
            (tmp / "entity_cache_dev.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "entity_prompt_sha256": "abc",
                "schema_version": "chapter3-no-capec-v1",
                "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                "boundary_contract_version": "chapter3-boundary-sync-v2",
                "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                "split_sha256": "valid_split",
                "gold_aggregate_sha256": "old_tampered_gold_hash",
                "dataset_freeze_manifest_sha256": "valid_freeze",
            }
            (tmp / "entity_cache_dev_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                cache_mgr.load_cache(
                    "dev",
                    expected_gold_aggregate_sha256="current_gold_hash",
                    expected_split_sha256="valid_split",
                )
            self.assertIn("gold_aggregate_sha256", str(ctx.exception))
            self.assertIn("Gold 已变更，缓存自动失效", str(ctx.exception))

    def test_entity_cache_rejected_on_window_construction_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_mgr = EntityCacheManager(tmp)
            (tmp / "entity_cache_dev.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "entity_prompt_sha256": "abc",
                "schema_version": "chapter3-no-capec-v1",
                "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                "boundary_contract_version": "chapter3-boundary-sync-v2",
                "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                "split_sha256": "valid_split",
                "gold_aggregate_sha256": "valid_gold",
                "dataset_freeze_manifest_sha256": "valid_freeze",
                "window_construction": "window-split-v1",
                "cache_file_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                "num_samples": 1,
                "sample_ids_sha256": hashlib.sha256(b"unknown\0").hexdigest(),
                "gold_relation_count": 0,
            }
            (tmp / "entity_cache_dev_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                cache_mgr.load_cache(
                    "dev",
                    validate_freeze_binding=False,
                    expected_window_construction=(
                        "window-split-v2:dense25-1000-120-20-s100"
                    ),
                )
            self.assertIn("window_construction", str(ctx.exception))

    def test_old_entity_cache_rejected_after_split_hash_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_mgr = EntityCacheManager(tmp)
            (tmp / "entity_cache_dev.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "entity_prompt_sha256": "abc",
                "schema_version": "chapter3-no-capec-v1",
                "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                "boundary_contract_version": "chapter3-boundary-sync-v2",
                "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                "split_sha256": "old_tampered_split_hash",
                "gold_aggregate_sha256": "valid_gold",
                "dataset_freeze_manifest_sha256": "valid_freeze",
            }
            (tmp / "entity_cache_dev_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                cache_mgr.load_cache(
                    "dev",
                    expected_split_sha256="current_split_hash",
                    expected_gold_aggregate_sha256="valid_gold",
                )
            self.assertIn("split_sha256", str(ctx.exception))
            self.assertIn("划分已变更，缓存自动失效", str(ctx.exception))

    def test_stage2_cache_never_falls_back_to_gold_entities(self):
        evaluator = TaskEvaluator(task_model="mock-model", task_client=None)
        samples_with_gold_only = [
            {
                "sample_id": "test_sample_1",
                "text": "Apache Log4j has CVE-2021-44228.",
                "entities": [{"id": "E1", "type": "Configuration", "start": 0, "end": 12}],
                "relations": [],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            evaluator.evaluate_stage2_batch(
                samples=samples_with_gold_only,
                full_relation_prompt="dummy_prompt",
            )
        self.assertIn("缺少 fixed_entities 字段", str(ctx.exception))
        self.assertIn("严禁回退到 Gold entities", str(ctx.exception))

    def test_custom_split_cannot_be_promoted_as_formal_protegi(self):
        from promote_protegi_v6 import promote_protegi

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ent_p = tmp / "entity_prompt.txt"
            ent_p.write_text("Prompt", encoding="utf-8")
            rel_p = tmp / "relation_prompt.txt"
            rel_p.write_text("Prompt", encoding="utf-8")
            ent_s = tmp / "entity_summary.json"
            ent_s.write_text(json.dumps({"formal_eligible": False}), encoding="utf-8")
            rel_s = tmp / "relation_summary.json"
            rel_s.write_text(json.dumps({"formal_eligible": True}), encoding="utf-8")
            train_m = tmp / "train_manifest.json"
            train_m.write_text("{}", encoding="utf-8")
            dev_m = tmp / "dev_manifest.json"
            dev_m.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                promote_protegi(
                    entity_prompt_path=ent_p,
                    relation_prompt_path=rel_p,
                    entity_summary_path=ent_s,
                    relation_summary_path=rel_s,
                    entity_cache_train_manifest_path=train_m,
                    entity_cache_dev_manifest_path=dev_m,
                )
            self.assertIn("formal_eligible", str(ctx.exception))

    def test_protegi_artifact_requires_formal_eligible(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path, artifact = self._create_valid_artifact_fixture(Path(tmpdir))
            artifact["formal_eligible"] = False
            art_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(art_path)
            self.assertIn("formal_eligible=true", str(ctx.exception))

    def test_cache_fails_closed_when_freeze_manifest_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_mgr = EntityCacheManager(tmp)
            (tmp / "entity_cache_dev.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = {
                "entity_prompt_sha256": "abc",
                "cache_file_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                "num_samples": 1,
                "sample_ids_sha256": hashlib.sha256(b"unknown\0").hexdigest(),
                "gold_relation_count": 0,
            }
            (tmp / "entity_cache_dev_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                cache_mgr.load_cache(
                    "dev",
                    validate_freeze_binding=True,
                    freeze_manifest_path=tmp / "nonexistent_freeze_manifest.json",
                )
            self.assertIn("冻结清单文件不存在", str(ctx.exception))

    def test_dry_run_summary_cannot_be_promoted_even_if_files_exist(self):
        from promote_protegi_v6 import promote_protegi

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ent_p = tmp / "entity_prompt.txt"
            ent_p.write_text("Valid Prompt", encoding="utf-8")
            rel_p = tmp / "relation_prompt.txt"
            rel_p.write_text("Valid Prompt", encoding="utf-8")
            ent_s = tmp / "entity_summary.json"
            ent_s.write_text(json.dumps({"formal_eligible": True, "dry_run": True}), encoding="utf-8")
            rel_s = tmp / "relation_summary.json"
            rel_s.write_text(json.dumps({"formal_eligible": True, "dry_run": False}), encoding="utf-8")
            train_m = tmp / "train_manifest.json"
            train_m.write_text("{}", encoding="utf-8")
            dev_m = tmp / "dev_manifest.json"
            dev_m.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                promote_protegi(
                    entity_prompt_path=ent_p,
                    relation_prompt_path=rel_p,
                    entity_summary_path=ent_s,
                    relation_summary_path=rel_s,
                    entity_cache_train_manifest_path=train_m,
                    entity_cache_dev_manifest_path=dev_m,
                )
            self.assertIn("dry-run", str(ctx.exception))

    def test_canonical_test_leakage_rejected_with_custom_split(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            split_data = json.loads((V6_ROOT / "data" / "train_dev_test_split_v7.json").read_text(encoding="utf-8"))
            canonical_test_doc = split_data["test"][0]
            custom_split = {
                "train": [canonical_test_doc],
                "dev": [],
                "test": [],
            }
            custom_split_path = tmp / "leaky_split.json"
            custom_split_path.write_text(json.dumps(custom_split), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(V6_ROOT / "scripts" / "run_protegi.py"),
                    "--split-file",
                    str(custom_split_path),
                    "--stage",
                    "entity",
                    "--allow-custom-split",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(V6_ROOT),
            )
            self.assertNotEqual(result.returncode, 0)
            combined_output = (result.stderr or "") + (result.stdout or "")
            self.assertIn("canonical test", combined_output)

    def test_promote_protegi_cli_supports_individual_file_args(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(V6_ROOT / "scripts" / "promote_protegi_v6.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(V6_ROOT),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--entity-prompt", result.stdout or "")
        self.assertIn("--relation-prompt", result.stdout or "")
        self.assertIn("--entity-summary", result.stdout or "")
        self.assertIn("--relation-summary", result.stdout or "")
        self.assertIn("--entity-cache-train-manifest", result.stdout or "")
        self.assertIn("--entity-cache-dev-manifest", result.stdout or "")


class TestFormalGoldIntegrity(unittest.TestCase):
    """Formal 模式必须验证实际 Gold 内容（重算 SHA），修改后在模型调用前 fail。"""

    def _make_mini_freeze(self, tmp: Path, docs: dict[str, str]) -> tuple[Path, Path]:
        gold_dir = tmp / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        doc_hashes: dict[str, str] = {}
        for doc_id, content in docs.items():
            payload = json.dumps(
                {"doc_id": doc_id, "text": content, "entities": [], "relations": []},
                ensure_ascii=False,
                sort_keys=True,
            )
            path = gold_dir / f"{doc_id}.json"
            path.write_text(payload, encoding="utf-8")
            doc_hashes[doc_id] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        aggregate = hashlib.sha256()
        for doc_id in sorted(doc_hashes):
            aggregate.update(
                f"{doc_id}\0{doc_hashes[doc_id]}\n".encode("utf-8")
            )
        manifest = {
            "gold_document_count": len(doc_hashes),
            "gold_aggregate_sha256": aggregate.hexdigest(),
            "gold_document_sha256": dict(sorted(doc_hashes.items())),
        }
        manifest_path = tmp / "freeze.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return gold_dir, manifest_path

    def test_formal_protegi_rejects_modified_gold_file(self):
        from protegi.gold_integrity import verify_frozen_gold_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_dir, manifest = self._make_mini_freeze(
                tmp, {"docA": "hello world", "docB": "second doc"}
            )
            # 篡改其中一篇 Gold 内容（不修改文件名与数量）
            target = gold_dir / "docA.json"
            target.write_text(
                json.dumps(
                    {"doc_id": "docA", "text": "TAMPERED", "entities": [], "relations": []}
                ),
                encoding="utf-8",
            )
            ok, msg = verify_frozen_gold_integrity(gold_dir, manifest)
            self.assertFalse(ok)
            self.assertIn("docA", msg)

    def test_formal_protegi_rejects_missing_gold_file(self):
        from protegi.gold_integrity import verify_frozen_gold_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_dir, manifest = self._make_mini_freeze(
                tmp, {"docA": "hello", "docB": "world"}
            )
            (gold_dir / "docB.json").unlink()
            ok, msg = verify_frozen_gold_integrity(gold_dir, manifest)
            self.assertFalse(ok)
            self.assertIn("缺失", msg)

    def test_formal_protegi_rejects_extra_gold_file(self):
        from protegi.gold_integrity import verify_frozen_gold_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_dir, manifest = self._make_mini_freeze(
                tmp, {"docA": "hello", "docB": "world"}
            )
            (gold_dir / "docEXTRA.json").write_text(
                json.dumps({"doc_id": "docEXTRA"}), encoding="utf-8"
            )
            ok, msg = verify_frozen_gold_integrity(gold_dir, manifest)
            self.assertFalse(ok)
            self.assertIn("额外", msg)

    def test_formal_protegi_rejects_gold_aggregate_mismatch(self):
        from protegi.gold_integrity import verify_frozen_gold_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_dir, manifest = self._make_mini_freeze(
                tmp, {"docA": "hello", "docB": "world"}
            )
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["gold_aggregate_sha256"] = "0" * 64
            manifest.write_text(json.dumps(data), encoding="utf-8")
            ok, msg = verify_frozen_gold_integrity(gold_dir, manifest)
            self.assertFalse(ok)
            self.assertIn("聚合哈希", msg)

    def test_formal_cache_build_fails_before_model_call_on_tampered_gold(self):
        from protegi.entity_cache import EntityCacheManager
        from protegi.gold_integrity import verify_frozen_gold_integrity

        class CountingEvaluator:
            max_workers = 8
            client = type("C", (), {"config": {"model": "hy3"}})()
            call_count = 0

            def predict_stage1_texts(self, texts, prompt, document_abbreviations=None):
                type(self).call_count += len(texts)
                return [[] for _ in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_dir, manifest = self._make_mini_freeze(
                tmp, {"docA": "hello world", "docB": "second"}
            )
            split_file = tmp / "split.json"
            split_file.write_text(json.dumps({"train": [], "dev": []}), encoding="utf-8")
            # 篡改 Gold 后，preflight 应先失败，模型调用次数保持 0
            (gold_dir / "docA.json").write_text(
                json.dumps({"doc_id": "docA", "text": "TAMPERED"}), encoding="utf-8"
            )
            ok, _ = verify_frozen_gold_integrity(gold_dir, manifest)
            self.assertFalse(ok)
            evaluator = CountingEvaluator()
            CountingEvaluator.call_count = 0
            cache_mgr = EntityCacheManager(tmp / "cache")
            samples = [
                {
                    "sample_id": "docA_w0",
                    "text": "hello",
                    "gold_entities": [],
                    "gold_relations": [],
                }
            ]
            with self.assertRaises(RuntimeError):
                cache_mgr.build_and_save_cache(
                    evaluator,
                    "prompt",
                    samples,
                    "train",
                    gold_dir=gold_dir,
                    freeze_manifest_path=manifest,
                    split_file_path=split_file,
                    validate_formal_gold=True,
                )
            self.assertEqual(CountingEvaluator.call_count, 0)


class TestProtegiFinalRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        cls.freeze_manifest = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        cls.split_sha = hashlib.sha256(cls.split_file.read_bytes()).hexdigest()
        cls.freeze_sha = hashlib.sha256(cls.freeze_manifest.read_bytes()).hexdigest()
        fm_data = json.loads(cls.freeze_manifest.read_text(encoding="utf-8"))
        cls.gold_agg_sha = fm_data["gold_aggregate_sha256"]
        cls.window_construction = fm_data["window_construction"]["version"]

    def _valid_artifact(self, tmp: Path) -> tuple[Path, dict]:
        ent = tmp / "e.txt"
        ent.write_text("e", encoding="utf-8")
        rel = tmp / "r.txt"
        rel.write_text("r", encoding="utf-8")
        artifact = {
            "artifact_version": "protegi-final-v1",
            "schema_version": "chapter3-no-capec-v1",
            "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
            "boundary_contract_version": "chapter3-boundary-sync-v2",
            "prompt_scope": "constrained",
            "split_sha256": self.split_sha,
            "dataset_freeze_manifest_sha256": self.freeze_sha,
            "gold_aggregate_sha256": self.gold_agg_sha,
            "entity_prompt_path": str(ent),
            "entity_prompt_sha256": hashlib.sha256(ent.read_bytes()).hexdigest(),
            "relation_prompt_path": str(rel),
            "relation_prompt_sha256": hashlib.sha256(rel.read_bytes()).hexdigest(),
            "window_construction": self.window_construction,
            "task_runtime": {
                "model": "hy3",
                "max_workers": 8,
                "temperature": 0.0,
                "thinking": "disabled",
                "reasoning_effort": "none",
                "top_p": 0.95,
                "max_tokens": 4096,
                "window_chars": 3000,
                "window_overlap": 400,
                "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            },
            "frozen_for_test": True,
            "formal_eligible": True,
            "test_gold_loaded": False,
            "test_predictions_generated": False,
        }
        path = tmp / "art.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path, artifact

    def test_protegi_artifact_requires_task_runtime(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            path, artifact = self._valid_artifact(Path(tmpdir))
            del artifact["task_runtime"]
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(path)
            self.assertIn("task_runtime", str(ctx.exception))

    def test_protegi_artifact_rejects_incomplete_task_runtime(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            path, artifact = self._valid_artifact(Path(tmpdir))
            del artifact["task_runtime"]["top_p"]
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(path)
            self.assertIn("top_p", str(ctx.exception))

    def test_protegi_artifact_rejects_invalid_window_runtime(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            path, artifact = self._valid_artifact(Path(tmpdir))
            artifact["task_runtime"]["window_overlap"] = 3000
            artifact["task_runtime"]["window_chars"] = 3000
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(path)
            self.assertIn("window", str(ctx.exception).lower())

    def test_protegi_artifact_rejects_missing_window_construction(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            path, artifact = self._valid_artifact(Path(tmpdir))
            del artifact["window_construction"]
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(path)
            self.assertIn("window_construction", str(ctx.exception))

    def test_protegi_artifact_rejects_window_construction_mismatch(self):
        from llm_methods import load_protegi_final_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            path, artifact = self._valid_artifact(Path(tmpdir))
            artifact["window_construction"] = "window-split-v999-nonexistent"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_protegi_final_artifact(path)
            self.assertIn("window_construction", str(ctx.exception))

    def test_predict_llm_protegi_uses_frozen_task_runtime(self):
        import llm_methods
        from protegi import evaluator as evaluator_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path, artifact = self._valid_artifact(tmp)
            artifact["task_runtime"].update({
                "temperature": 0.0,
                "thinking": "disabled",
                "reasoning_effort": "none",
                "top_p": 0.95,
                "max_tokens": 4096,
                "window_chars": 100,
                "window_overlap": 10,
            })
            path.write_text(json.dumps(artifact), encoding="utf-8")

            captured: dict = {}
            orig_windows = llm_methods.build_text_windows

            def spy_windows(text, max_chars=3000, overlap=400, **kwargs):
                captured["window_chars"] = max_chars
                captured["window_overlap"] = overlap
                return orig_windows(text, max_chars=max_chars, overlap=overlap, **kwargs)

            class SpyEvaluator:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.max_workers = kwargs.get("max_workers", 8)
                    self.vulnerability_anchored_backfill = kwargs.get(
                        "vulnerability_anchored_backfill", False
                    )

                def predict_stage1_window(self, text, prompt, document_abbreviations=None):
                    return []

                def predict_stage2_window(self, text, entities, prompt):
                    return []

            orig_evaluator = evaluator_module.TaskEvaluator
            orig_windows_fn = llm_methods.build_text_windows
            llm_methods.build_text_windows = spy_windows
            evaluator_module.TaskEvaluator = SpyEvaluator
            try:
                llm_methods.predict_llm_protegi("short text", "docX", artifact_path=path)
            finally:
                llm_methods.build_text_windows = orig_windows_fn
                evaluator_module.TaskEvaluator = orig_evaluator
            self.assertEqual(captured.get("task_model"), "hy3")
            self.assertEqual(captured.get("task_temperature"), 0.0)
            self.assertEqual(captured.get("task_thinking"), "disabled")
            self.assertEqual(captured.get("task_reasoning_effort"), "none")
            self.assertEqual(captured.get("window_chars"), 100)
            self.assertEqual(captured.get("window_overlap"), 10)

    def test_predict_llm_protegi_does_not_use_environment_runtime(self):
        import llm_methods
        from protegi import evaluator as evaluator_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path, artifact = self._valid_artifact(tmp)
            artifact["task_runtime"].update({
                "temperature": 0.0,
                "thinking": "disabled",
                "reasoning_effort": "none",
            })
            path.write_text(json.dumps(artifact), encoding="utf-8")

            captured: dict = {}
            orig_temp = llm_methods.API_TEMPERATURE
            orig_thinking = llm_methods.API_THINKING
            orig_effort = llm_methods.API_REASONING_EFFORT
            llm_methods.API_TEMPERATURE = 0.99
            llm_methods.API_THINKING = "enabled"
            llm_methods.API_REASONING_EFFORT = "high"

            class SpyEvaluator:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.max_workers = kwargs.get("max_workers", 8)
                    self.vulnerability_anchored_backfill = kwargs.get(
                        "vulnerability_anchored_backfill", False
                    )

                def predict_stage1_window(self, text, prompt, document_abbreviations=None):
                    return []

                def predict_stage2_window(self, text, entities, prompt):
                    return []

            orig_evaluator = evaluator_module.TaskEvaluator
            evaluator_module.TaskEvaluator = SpyEvaluator
            try:
                llm_methods.predict_llm_protegi("text", "docY", artifact_path=path)
            finally:
                evaluator_module.TaskEvaluator = orig_evaluator
                llm_methods.API_TEMPERATURE = orig_temp
                llm_methods.API_THINKING = orig_thinking
                llm_methods.API_REASONING_EFFORT = orig_effort
            self.assertEqual(captured.get("task_temperature"), 0.0)
            self.assertEqual(captured.get("task_thinking"), "disabled")
            self.assertEqual(captured.get("task_reasoning_effort"), "none")
            self.assertNotEqual(captured.get("task_temperature"), 0.99)


class TestPromoteProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        cls.freeze_manifest = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        cls.split_sha = hashlib.sha256(cls.split_file.read_bytes()).hexdigest()
        cls.freeze_sha = hashlib.sha256(cls.freeze_manifest.read_bytes()).hexdigest()
        fm_data = json.loads(cls.freeze_manifest.read_text(encoding="utf-8"))
        cls.gold_agg_sha = fm_data["gold_aggregate_sha256"]
        cls.window_construction = fm_data["window_construction"]["version"]

    def _write_prompts(self, tmp: Path):
        from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0

        ent = tmp / "final_entity_prompt.txt"
        rel = tmp / "final_relation_prompt.txt"
        ent.write_text(ENTITY_PROMPT_P0, encoding="utf-8")
        rel.write_text(RELATION_PROMPT_P0, encoding="utf-8")
        return ent, rel

    def _task_config(self, **overrides):
        cfg = {
            "task_model": "hy3",
            "task_max_workers": 8,
            "task_temperature": 0.0,
            "task_thinking": "disabled",
            "task_reasoning_effort": "none",
            "task_top_p": 0.95,
            "task_max_tokens": 4096,
            "optimizer_model": "muse-spark-1.3-contributor",
            "window_chars": 3000,
            "window_overlap": 400,
            "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        cfg.update(overrides)
        cfg["effective_task_runtime"] = {
            "model": cfg["task_model"],
            "max_workers": cfg["task_max_workers"],
            "temperature": cfg["task_temperature"],
            "thinking": cfg["task_thinking"],
            "reasoning_effort": cfg["task_reasoning_effort"],
            "top_p": cfg["task_top_p"],
            "max_tokens": cfg["task_max_tokens"],
            "window_chars": cfg["window_chars"],
            "window_overlap": cfg["window_overlap"],
            "document_abbreviation_context": cfg["document_abbreviation_context"],
            "vulnerability_anchored_backfill": cfg["vulnerability_anchored_backfill"],
        }
        return cfg

    def _make_valid_promote_fixture(self, tmp: Path):
        from protegi.entity_cache import compute_prompt_hash

        ent_path, rel_path = self._write_prompts(tmp)
        ent_text = ent_path.read_text(encoding="utf-8")
        rel_text = rel_path.read_text(encoding="utf-8")
        ent_raw = hashlib.sha256(ent_path.read_bytes()).hexdigest()
        rel_raw = hashlib.sha256(rel_path.read_bytes()).hexdigest()
        config_hash = hashlib.sha256(b"formal-config").hexdigest()
        base_bind = {
            "split_file_sha256_raw_bytes": self.split_sha,
            "dataset_freeze_manifest_sha256_raw_bytes": self.freeze_sha,
            "config_file_sha256_raw_bytes": config_hash,
        }
        ent_summary = {
            "stage": "entity",
            "method": "protegi",
            "prompt_scope": "constrained",
            "experiment_pair_id": "protegi-prompt-scope-v1",
            "formal_eligible": True,
            "dry_run": False,
            "winner_prompt_sha256": compute_prompt_hash(ent_text),
            "winner_prompt_sha256_raw_bytes": ent_raw,
            "window_construction": self.window_construction,
            "config": self._task_config(),
            "input_bindings": dict(base_bind),
        }
        rel_summary = {
            "stage": "relation",
            "method": "protegi",
            "prompt_scope": "constrained",
            "experiment_pair_id": "protegi-prompt-scope-v1",
            "formal_eligible": True,
            "dry_run": False,
            "winner_prompt_sha256": compute_prompt_hash(rel_text),
            "winner_prompt_sha256_raw_bytes": rel_raw,
            "window_construction": self.window_construction,
            "config": self._task_config(),
            "input_bindings": dict(base_bind),
        }
        ent_summary_path = tmp / "entity_summary.json"
        rel_summary_path = tmp / "relation_summary.json"
        ent_summary_path.write_text(
            json.dumps(ent_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rel_summary_path.write_text(
            json.dumps(rel_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        task_runtime = {
            "model": "hy3",
            "max_workers": 8,
            "temperature": 0.0,
            "thinking": "disabled",
            "reasoning_effort": "none",
            "top_p": 0.95,
            "max_tokens": 4096,
            "window_chars": 3000,
            "window_overlap": 400,
            "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        for name in ("train", "dev"):
            manifest = {
                "split_name": name,
                "entity_prompt_sha256": ent_raw,
                "prompt_scope": "constrained",
                "task_max_workers": 8,
                "window_chars": 3000,
                "window_overlap": 400,
                "window_construction": self.window_construction,
                "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
                "task_runtime": dict(task_runtime),
                "schema_version": "chapter3-no-capec-v1",
                "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                "boundary_contract_version": "chapter3-boundary-sync-v2",
                "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                "split_sha256": self.split_sha,
                "gold_aggregate_sha256": self.gold_agg_sha,
                "dataset_freeze_manifest_sha256": self.freeze_sha,
            }
            (tmp / f"{name}_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return {
            "entity_prompt": ent_path,
            "relation_prompt": rel_path,
            "entity_summary": ent_summary_path,
            "relation_summary": rel_summary_path,
            "train_manifest": tmp / "train_manifest.json",
            "dev_manifest": tmp / "dev_manifest.json",
            "output": tmp / "artifact.json",
        }

    def _promote(self, paths: dict):
        from promote_protegi_v6 import promote_protegi

        return promote_protegi(
            entity_prompt_path=paths["entity_prompt"],
            relation_prompt_path=paths["relation_prompt"],
            entity_summary_path=paths["entity_summary"],
            relation_summary_path=paths["relation_summary"],
            entity_cache_train_manifest_path=paths["train_manifest"],
            entity_cache_dev_manifest_path=paths["dev_manifest"],
            prompt_scope="constrained",
            output_artifact_path=paths["output"],
        )

    def _tweak_guidance_valid(self, prompt_text: str) -> str:
        from protegi.prompts_p0 import replace_optimizable_guidance

        current = prompt_text
        # 追加一句合法的执行策略，不触碰冻结契约与禁用模式。
        from protegi.prompts_p0 import extract_optimizable_guidance

        guidance = extract_optimizable_guidance(prompt_text) or ""
        return replace_optimizable_guidance(
            current, guidance + " Verify each span against the source text."
        )

    def test_promote_rejects_entity_prompt_not_matching_summary_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            tampered = self._tweak_guidance_valid(
                paths["entity_prompt"].read_text(encoding="utf-8")
            )
            paths["entity_prompt"].write_text(tampered, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn(
                "final prompt does not match summary winner", str(ctx.exception)
            )

    def test_promote_rejects_relation_prompt_not_matching_summary_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            tampered = self._tweak_guidance_valid(
                paths["relation_prompt"].read_text(encoding="utf-8")
            )
            # relation guidance 同样追加合法策略，保持契约通过但哈希失配。
            paths["relation_prompt"].write_text(tampered, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn(
                "final prompt does not match summary winner", str(ctx.exception)
            )

    def test_promote_rejects_stage_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["entity_summary"].read_text(encoding="utf-8"))
            data["stage"] = "relation"
            paths["entity_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("stage", str(ctx.exception).lower())

    def test_promote_rejects_prompt_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            data["prompt_scope"] = "unconstrained"
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("prompt_scope", str(ctx.exception))

    def test_promote_rejects_experiment_pair_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            data["experiment_pair_id"] = "different-pair"
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("experiment_pair_id", str(ctx.exception))

    def test_promote_rejects_window_construction_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            data["window_construction"] = "window-split-v999-nonexistent"
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("window_construction", str(ctx.exception))

    def test_promote_rejects_missing_window_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["entity_summary"].read_text(encoding="utf-8"))
            del data["window_construction"]
            paths["entity_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("window_construction", str(ctx.exception))

    def test_promote_rejects_cache_window_construction_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            for name in ("train_manifest", "dev_manifest"):
                data = json.loads(paths[name].read_text(encoding="utf-8"))
                data["window_construction"] = "window-split-v999-nonexistent"
                paths[name].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("window_construction", str(ctx.exception))

    def test_promote_rejects_summary_split_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["entity_summary"].read_text(encoding="utf-8"))
            data["input_bindings"]["split_file_sha256_raw_bytes"] = "0" * 64
            paths["entity_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("split_file_sha256", str(ctx.exception))

    def test_promote_rejects_summary_freeze_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            data["input_bindings"][
                "dataset_freeze_manifest_sha256_raw_bytes"
            ] = "0" * 64
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("dataset_freeze_manifest_sha256", str(ctx.exception))

    def test_promote_rejects_task_runtime_mismatch_between_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_valid_promote_fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            data["config"]["task_temperature"] = 0.7
            data["config"]["effective_task_runtime"]["temperature"] = 0.7
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("Task Runtime", str(ctx.exception))


class TestAllowCustomSplitNeverFormal(unittest.TestCase):
    def test_allow_custom_split_is_never_formal_eligible(self):
        from run_protegi import compute_formal_eligibility

        # canonical 路径 + flag：仍必须 non-formal（无模型调用的纯函数断言）
        self.assertFalse(
            compute_formal_eligibility(dry_run=False, allow_custom_split=True)
        )
        self.assertFalse(
            compute_formal_eligibility(dry_run=True, allow_custom_split=False)
        )
        self.assertFalse(
            compute_formal_eligibility(dry_run=True, allow_custom_split=True)
        )
        self.assertTrue(
            compute_formal_eligibility(dry_run=False, allow_custom_split=False)
        )

    def test_allow_custom_split_skips_formal_promotion_eligibility_even_when_paths_match(self):
        from promote_protegi_v6 import promote_protegi
        from protegi.entity_cache import compute_prompt_hash
        from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0

        split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        freeze_file = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        split_sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
        freeze_sha = hashlib.sha256(freeze_file.read_bytes()).hexdigest()
        fm = json.loads(freeze_file.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ent = tmp / "final_entity_prompt.txt"
            rel = tmp / "final_relation_prompt.txt"
            ent.write_text(ENTITY_PROMPT_P0, encoding="utf-8")
            rel.write_text(RELATION_PROMPT_P0, encoding="utf-8")
            ent_text = ent.read_text(encoding="utf-8")
            rel_text = rel.read_text(encoding="utf-8")
            bind = {
                "split_file_sha256_raw_bytes": split_sha,
                "dataset_freeze_manifest_sha256_raw_bytes": freeze_sha,
                "config_file_sha256_raw_bytes": hashlib.sha256(b"c").hexdigest(),
            }
            eff = {
                "model": "hy3", "max_workers": 8, "temperature": 0.0,
                "thinking": "disabled", "reasoning_effort": "none",
                "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            }
            # 即使路径全是 canonical，custom run 的 summary 仍 formal_eligible=false
            ent_sum = {
                "stage": "entity", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": False, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(ent_text),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    ent.read_bytes()
                ).hexdigest(),
                "config": {
                    "task_model": "hy3", "task_max_workers": 8,
                    "task_temperature": 0.0, "task_thinking": "disabled",
                    "task_reasoning_effort": "none", "task_top_p": 0.95,
                    "task_max_tokens": 4096, "effective_task_runtime": dict(eff),
                },
                "input_bindings": dict(bind),
            }
            rel_sum = {
                "stage": "relation", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": False, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(rel_text),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    rel.read_bytes()
                ).hexdigest(),
                "config": {
                    "task_model": "hy3", "task_max_workers": 8,
                    "task_temperature": 0.0, "task_thinking": "disabled",
                    "task_reasoning_effort": "none", "task_top_p": 0.95,
                    "task_max_tokens": 4096, "effective_task_runtime": dict(eff),
                },
                "input_bindings": dict(bind),
            }
            ent_sum_p = tmp / "ent_sum.json"
            rel_sum_p = tmp / "rel_sum.json"
            ent_sum_p.write_text(json.dumps(ent_sum), encoding="utf-8")
            rel_sum_p.write_text(json.dumps(rel_sum), encoding="utf-8")
            cache_rt = dict(eff)
            for name in ("train", "dev"):
                (tmp / f"{name}_m.json").write_text(
                    json.dumps({
                        "split_name": name,
                        "entity_prompt_sha256": hashlib.sha256(
                            ent.read_bytes()
                        ).hexdigest(),
                        "prompt_scope": "constrained", "task_max_workers": 8,
                        "window_chars": 3000, "window_overlap": 400,
                        "document_abbreviation_context": False,
                        "vulnerability_anchored_backfill": False,
                        "task_runtime": cache_rt,
                        "schema_version": "chapter3-no-capec-v1",
                        "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                        "boundary_contract_version": "chapter3-boundary-sync-v2",
                        "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                        "split_sha256": split_sha,
                        "gold_aggregate_sha256": fm["gold_aggregate_sha256"],
                        "dataset_freeze_manifest_sha256": freeze_sha,
                    }),
                    encoding="utf-8",
                )
            with self.assertRaises(ValueError) as ctx:
                promote_protegi(
                    entity_prompt_path=ent, relation_prompt_path=rel,
                    entity_summary_path=ent_sum_p,
                    relation_summary_path=rel_sum_p,
                    entity_cache_train_manifest_path=tmp / "train_m.json",
                    entity_cache_dev_manifest_path=tmp / "dev_m.json",
                    prompt_scope="constrained",
                    output_artifact_path=tmp / "art.json",
                )
            self.assertIn("formal_eligible", str(ctx.exception))


class TestEntityCacheBackfillExecution(unittest.TestCase):
    def _samples(self):
        return [
            {
                "sample_id": "docA_w0", "doc_id": "docA",
                "text": "CVE-2021-1234 affects Exchange today.",
                "gold_entities": [], "gold_relations": [],
            },
            {
                "sample_id": "docA_w1", "doc_id": "docA",
                "text": "CVE-2021-1234 affects Exchange again.",
                "gold_entities": [], "gold_relations": [],
            },
        ]

    def _evaluator(self):
        class FakeEvaluator:
            max_workers = 8
            vulnerability_anchored_backfill = False
            client = type("C", (), {"config": {
                "model": "hy3", "temperature": 0.0, "thinking": "disabled",
                "reasoning_effort": "none", "top_p": 0.95, "max_tokens": 4096,
            }})()

            def predict_stage1_texts(self, texts, prompt, document_abbreviations=None):
                return [[] for _ in texts]

        return FakeEvaluator()

    def test_entity_cache_executes_backfill_when_runtime_true(self):
        import protegi.entity_backfill as bf_module
        from protegi.entity_cache import EntityCacheManager

        calls = {"n": 0}
        orig = bf_module.vulnerability_anchored_backfill

        def spy(samples, predictions, enabled=True):
            calls["n"] += 1
            return orig(samples, predictions, enabled=enabled)

        bf_module.vulnerability_anchored_backfill = spy
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mgr = EntityCacheManager(Path(tmpdir) / "c1")
                mgr.build_and_save_cache(
                    self._evaluator(), "p", self._samples(), "train",
                    prompt_scope="constrained",
                    freeze_manifest_path=Path(tmpdir) / "no_freeze.json",
                    split_file_path=Path(tmpdir) / "no_split.json",
                    schema_version="chapter3-no-capec-v1",
                    annotation_protocol_version="4.6-mcpu-mention-fact-dual-layer-v1",
                    boundary_contract_version="chapter3-boundary-sync-v2",
                    dataset_version="v6-v5-gold-v9-mcpu-v2",
                    gold_aggregate_sha256="x", split_sha256="y",
                    dataset_freeze_manifest_sha256="z",
                    window_chars=3000, window_overlap=400,
                    document_abbreviation_context=False,
                    vulnerability_anchored_backfill=True,
                )
        finally:
            bf_module.vulnerability_anchored_backfill = orig
        self.assertEqual(calls["n"], 1)

    def test_entity_cache_does_not_execute_backfill_when_runtime_false(self):
        import protegi.entity_backfill as bf_module
        from protegi.entity_cache import EntityCacheManager

        calls = {"n": 0}
        orig = bf_module.vulnerability_anchored_backfill

        def spy(samples, predictions, enabled=True):
            calls["n"] += 1
            return orig(samples, predictions, enabled=enabled)

        bf_module.vulnerability_anchored_backfill = spy
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mgr = EntityCacheManager(Path(tmpdir) / "c2")
                mgr.build_and_save_cache(
                    self._evaluator(), "p", self._samples(), "train",
                    prompt_scope="constrained",
                    freeze_manifest_path=Path(tmpdir) / "no_freeze.json",
                    split_file_path=Path(tmpdir) / "no_split.json",
                    schema_version="chapter3-no-capec-v1",
                    annotation_protocol_version="4.6-mcpu-mention-fact-dual-layer-v1",
                    boundary_contract_version="chapter3-boundary-sync-v2",
                    dataset_version="v6-v5-gold-v9-mcpu-v2",
                    gold_aggregate_sha256="x", split_sha256="y",
                    dataset_freeze_manifest_sha256="z",
                    window_chars=3000, window_overlap=400,
                    document_abbreviation_context=False,
                    vulnerability_anchored_backfill=False,
                )
        finally:
            bf_module.vulnerability_anchored_backfill = orig
        self.assertEqual(calls["n"], 0)

    def test_entity_cache_manifest_backfill_matches_actual_execution(self):
        import protegi.entity_backfill as bf_module
        from protegi.entity_cache import EntityCacheManager

        for flag, expect_calls in ((True, 1), (False, 0)):
            calls = {"n": 0}
            orig = bf_module.vulnerability_anchored_backfill

            def spy(samples, predictions, enabled=True):
                calls["n"] += 1
                return orig(samples, predictions, enabled=enabled)

            bf_module.vulnerability_anchored_backfill = spy
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    mgr = EntityCacheManager(tmp / "c")
                    mgr.build_and_save_cache(
                        self._evaluator(), "p", self._samples(), "train",
                        prompt_scope="constrained",
                        freeze_manifest_path=tmp / "no_freeze.json",
                        split_file_path=tmp / "no_split.json",
                        schema_version="chapter3-no-capec-v1",
                        annotation_protocol_version="4.6-mcpu-mention-fact-dual-layer-v1",
                        boundary_contract_version="chapter3-boundary-sync-v2",
                        dataset_version="v6-v5-gold-v9-mcpu-v2",
                        gold_aggregate_sha256="x", split_sha256="y",
                        dataset_freeze_manifest_sha256="z",
                        window_chars=3000, window_overlap=400,
                        document_abbreviation_context=False,
                        vulnerability_anchored_backfill=flag,
                    )
                    manifest = json.loads(
                        (tmp / "c" / "entity_cache_train_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
            finally:
                bf_module.vulnerability_anchored_backfill = orig
            self.assertEqual(calls["n"], expect_calls)
            self.assertEqual(
                manifest["vulnerability_anchored_backfill"], flag
            )
            self.assertEqual(
                manifest["task_runtime"]["vulnerability_anchored_backfill"], flag
            )


class TestPromoteCanonicalHashRequired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        cls.freeze_manifest = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        cls.split_sha = hashlib.sha256(cls.split_file.read_bytes()).hexdigest()
        cls.freeze_sha = hashlib.sha256(cls.freeze_manifest.read_bytes()).hexdigest()
        cls.gold_agg = json.loads(
            cls.freeze_manifest.read_text(encoding="utf-8")
        )["gold_aggregate_sha256"]

    def _fixture(self, tmp: Path):
        from protegi.entity_cache import compute_prompt_hash
        from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0

        ent = tmp / "final_entity_prompt.txt"
        rel = tmp / "final_relation_prompt.txt"
        ent.write_text(ENTITY_PROMPT_P0, encoding="utf-8")
        rel.write_text(RELATION_PROMPT_P0, encoding="utf-8")
        eff = {
            "model": "hy3", "max_workers": 8, "temperature": 0.0,
            "thinking": "disabled", "reasoning_effort": "none",
            "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
            "window_overlap": 400, "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        bind = {
            "split_file_sha256_raw_bytes": self.split_sha,
            "dataset_freeze_manifest_sha256_raw_bytes": self.freeze_sha,
            "config_file_sha256_raw_bytes": hashlib.sha256(b"c").hexdigest(),
        }
        base_cfg = {
            "task_model": "hy3", "task_max_workers": 8, "task_temperature": 0.0,
            "task_thinking": "disabled", "task_reasoning_effort": "none",
            "task_top_p": 0.95, "task_max_tokens": 4096,
            "window_chars": 3000, "window_overlap": 400,
            "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
            "effective_task_runtime": dict(eff),
        }
        ent_sum = {
            "stage": "entity", "prompt_scope": "constrained",
            "experiment_pair_id": "protegi-prompt-scope-v1",
            "formal_eligible": True, "dry_run": False,
            "winner_prompt_sha256": compute_prompt_hash(
                ent.read_text(encoding="utf-8")
            ),
            "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                ent.read_bytes()
            ).hexdigest(),
            "config": dict(base_cfg), "input_bindings": dict(bind),
        }
        rel_sum = {
            "stage": "relation", "prompt_scope": "constrained",
            "experiment_pair_id": "protegi-prompt-scope-v1",
            "formal_eligible": True, "dry_run": False,
            "winner_prompt_sha256": compute_prompt_hash(
                rel.read_text(encoding="utf-8")
            ),
            "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                rel.read_bytes()
            ).hexdigest(),
            "config": dict(base_cfg), "input_bindings": dict(bind),
        }
        ent_p = tmp / "ent_sum.json"
        rel_p = tmp / "rel_sum.json"
        ent_p.write_text(json.dumps(ent_sum), encoding="utf-8")
        rel_p.write_text(json.dumps(rel_sum), encoding="utf-8")
        for name in ("train", "dev"):
            (tmp / f"{name}_m.json").write_text(
                json.dumps({
                    "split_name": name,
                    "entity_prompt_sha256": hashlib.sha256(
                        ent.read_bytes()
                    ).hexdigest(),
                    "prompt_scope": "constrained", "task_max_workers": 8,
                    "window_chars": 3000, "window_overlap": 400,
                    "document_abbreviation_context": False,
                    "vulnerability_anchored_backfill": False,
                    "task_runtime": dict(eff),
                    "schema_version": "chapter3-no-capec-v1",
                    "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                    "boundary_contract_version": "chapter3-boundary-sync-v2",
                    "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                    "split_sha256": self.split_sha,
                    "gold_aggregate_sha256": self.gold_agg,
                    "dataset_freeze_manifest_sha256": self.freeze_sha,
                }),
                encoding="utf-8",
            )
        return {
            "entity_prompt": ent, "relation_prompt": rel,
            "entity_summary": ent_p, "relation_summary": rel_p,
            "train_manifest": tmp / "train_m.json",
            "dev_manifest": tmp / "dev_m.json",
            "output": tmp / "art.json",
        }

    def _promote(self, paths: dict):
        from promote_protegi_v6 import promote_protegi

        return promote_protegi(
            entity_prompt_path=paths["entity_prompt"],
            relation_prompt_path=paths["relation_prompt"],
            entity_summary_path=paths["entity_summary"],
            relation_summary_path=paths["relation_summary"],
            entity_cache_train_manifest_path=paths["train_manifest"],
            entity_cache_dev_manifest_path=paths["dev_manifest"],
            prompt_scope="constrained",
            output_artifact_path=paths["output"],
        )

    def test_promote_rejects_missing_entity_winner_canonical_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            data = json.loads(paths["entity_summary"].read_text(encoding="utf-8"))
            del data["winner_prompt_sha256"]
            paths["entity_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("winner_prompt_sha256", str(ctx.exception))

    def test_promote_rejects_missing_relation_winner_canonical_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._fixture(Path(tmpdir))
            data = json.loads(paths["relation_summary"].read_text(encoding="utf-8"))
            del data["winner_prompt_sha256"]
            paths["relation_summary"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self._promote(paths)
            self.assertIn("winner_prompt_sha256", str(ctx.exception))


class TestEffectiveRuntime(unittest.TestCase):
    def test_run_protegi_records_effective_task_runtime(self):
        import yaml
        from run_protegi import normalize_config_with_effective_runtime

        cfg_path = (
            V6_ROOT / "protegi" / "configs"
            / "protegi_formal_constrained_muse.yaml"
        )
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        out = normalize_config_with_effective_runtime(dict(config))
        self.assertIn("effective_task_runtime", out)
        eff = out["effective_task_runtime"]
        self.assertEqual(eff["model"], "hy3")
        self.assertEqual(eff["max_workers"], 8)
        self.assertEqual(eff["window_chars"], 3000)
        self.assertEqual(eff["window_overlap"], 400)

    def test_effective_runtime_contains_all_required_fields(self):
        import yaml
        from protegi.runtime_contract import TASK_RUNTIME_FIELDS
        from run_protegi import normalize_config_with_effective_runtime

        cfg_path = (
            V6_ROOT / "protegi" / "configs"
            / "protegi_formal_constrained_muse.yaml"
        )
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        out = normalize_config_with_effective_runtime(dict(config))
        self.assertEqual(
            set(out["effective_task_runtime"].keys()), set(TASK_RUNTIME_FIELDS)
        )

    def test_formal_arms_share_frozen_window_construction(self):
        import yaml
        from run_protegi import normalize_config_with_effective_runtime

        names = [
            "protegi_formal_muse.yaml",
            "protegi_formal_constrained_muse.yaml",
            "protegi_formal_unconstrained_muse.yaml",
        ]
        parsed = []
        for name in names:
            cfg = yaml.safe_load(
                (V6_ROOT / "protegi" / "configs" / name).read_text(
                    encoding="utf-8"
                )
            )
            parsed.append(cfg)
        # matched-budget：除 prompt_scope 外三臂完全一致。
        stripped = [
            {k: v for k, v in cfg.items() if k != "prompt_scope"}
            for cfg in parsed
        ]
        self.assertEqual(stripped[0], stripped[1])
        self.assertEqual(stripped[0], stripped[2])
        # 冻结的窗口构造描述符。
        out = normalize_config_with_effective_runtime(dict(parsed[0]))
        self.assertTrue(out["dense_run_split"])
        self.assertEqual(
            out["window_construction"],
            "window-split-v2:dense25-1000-120-20-s100",
        )
        self.assertEqual(out["window_chars"], 3000)
        self.assertEqual(out["window_overlap"], 400)

    def test_formal_preflight_accepts_frozen_construction(self):
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        version = manifest["window_construction"]["version"]
        # 一致即通过（无返回值；正式路径经 normalize 后恒带尺寸字段）。
        validate_formal_window_construction(
            {
                "window_construction": version,
                "window_chars": 3000,
                "window_overlap": 400,
            },
            manifest,
        )

    def test_formal_preflight_rejects_provisional_construction(self):
        import copy
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        provisional = copy.deepcopy(manifest)
        provisional["window_construction"]["params_provisional"] = True
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {"window_construction": "window-split-v2:dense25-1000-120-20-s100"},
                provisional,
            )
        self.assertIn("params_provisional", str(ctx.exception))

    def test_formal_preflight_rejects_construction_mismatch(self):
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {"window_construction": "window-split-v1"}, manifest
            )
        self.assertIn("window_construction", str(ctx.exception).lower())

    def test_formal_preflight_accepts_matching_window_size(self):
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        version = manifest["window_construction"]["version"]
        validate_formal_window_construction(
            {
                "window_construction": version,
                "window_chars": 3000,
                "window_overlap": 400,
            },
            manifest,
        )

    def test_formal_preflight_rejects_window_chars_mismatch(self):
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        version = manifest["window_construction"]["version"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {
                    "window_construction": version,
                    "window_chars": 2500,
                    "window_overlap": 400,
                },
                manifest,
            )
        self.assertIn("formal window_chars mismatch", str(ctx.exception))

    def test_formal_preflight_rejects_window_overlap_mismatch(self):
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        version = manifest["window_construction"]["version"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {
                    "window_construction": version,
                    "window_chars": 3000,
                    "window_overlap": 300,
                },
                manifest,
            )
        self.assertIn("formal window_overlap mismatch", str(ctx.exception))

    def test_formal_preflight_rejects_manifest_missing_max_chars(self):
        import copy
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        tampered = copy.deepcopy(manifest)
        del tampered["window_construction"]["max_chars"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {
                    "window_construction": manifest["window_construction"][
                        "version"
                    ],
                    "window_chars": 3000,
                    "window_overlap": 400,
                },
                tampered,
            )
        self.assertIn("formal window_chars mismatch", str(ctx.exception))

    def test_formal_preflight_rejects_manifest_missing_overlap(self):
        import copy
        from run_protegi import validate_formal_window_construction

        manifest = json.loads(
            (V6_ROOT / "data" / "dataset_freeze_manifest_v6.json").read_text(
                encoding="utf-8"
            )
        )
        tampered = copy.deepcopy(manifest)
        del tampered["window_construction"]["overlap"]
        with self.assertRaises(RuntimeError) as ctx:
            validate_formal_window_construction(
                {
                    "window_construction": manifest["window_construction"][
                        "version"
                    ],
                    "window_chars": 3000,
                    "window_overlap": 400,
                },
                tampered,
            )
        self.assertIn("formal window_overlap mismatch", str(ctx.exception))

    def test_promote_uses_recorded_effective_runtime(self):
        from protegi.entity_cache import compute_prompt_hash
        from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0
        from promote_protegi_v6 import promote_protegi

        split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        freeze_file = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        split_sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
        freeze_sha = hashlib.sha256(freeze_file.read_bytes()).hexdigest()
        gold_agg = json.loads(freeze_file.read_text(encoding="utf-8"))[
            "gold_aggregate_sha256"
        ]
        live_construction = json.loads(freeze_file.read_text(encoding="utf-8"))[
            "window_construction"
        ]["version"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ent = tmp / "final_entity_prompt.txt"
            rel = tmp / "final_relation_prompt.txt"
            ent.write_text(ENTITY_PROMPT_P0, encoding="utf-8")
            rel.write_text(RELATION_PROMPT_P0, encoding="utf-8")
            # recorded effective 一致（0.0），顶层 task_temperature 故意分歧：
            # 若 promotion 用重推导会误判，用 recorded 则应通过。
            eff = {
                "model": "hy3", "max_workers": 8, "temperature": 0.0,
                "thinking": "disabled", "reasoning_effort": "none",
                "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            }
            bind = {
                "split_file_sha256_raw_bytes": split_sha,
                "dataset_freeze_manifest_sha256_raw_bytes": freeze_sha,
                "config_file_sha256_raw_bytes": hashlib.sha256(b"c").hexdigest(),
            }
            ent_cfg = {
                "task_model": "hy3", "task_max_workers": 8,
                "task_temperature": 0.99, "task_thinking": "disabled",
                "task_reasoning_effort": "none", "task_top_p": 0.95,
                "task_max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
                "effective_task_runtime": dict(eff),
            }
            rel_cfg = {
                "task_model": "hy3", "task_max_workers": 8,
                "task_temperature": 0.0, "task_thinking": "disabled",
                "task_reasoning_effort": "none", "task_top_p": 0.95,
                "task_max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
                "effective_task_runtime": dict(eff),
            }
            ent_sum = {
                "stage": "entity", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": True, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(
                    ent.read_text(encoding="utf-8")
                ),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    ent.read_bytes()
                ).hexdigest(),
                "window_construction": live_construction,
                "config": ent_cfg, "input_bindings": dict(bind),
            }
            rel_sum = {
                "stage": "relation", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": True, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(
                    rel.read_text(encoding="utf-8")
                ),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    rel.read_bytes()
                ).hexdigest(),
                "window_construction": live_construction,
                "config": rel_cfg, "input_bindings": dict(bind),
            }
            ent_p = tmp / "ent_sum.json"
            rel_p = tmp / "rel_sum.json"
            ent_p.write_text(json.dumps(ent_sum), encoding="utf-8")
            rel_p.write_text(json.dumps(rel_sum), encoding="utf-8")
            for name in ("train", "dev"):
                (tmp / f"{name}_m.json").write_text(
                    json.dumps({
                        "split_name": name,
                        "entity_prompt_sha256": hashlib.sha256(
                            ent.read_bytes()
                        ).hexdigest(),
                        "prompt_scope": "constrained", "task_max_workers": 8,
                        "window_chars": 3000, "window_overlap": 400,
                        "window_construction": live_construction,
                        "document_abbreviation_context": False,
                        "vulnerability_anchored_backfill": False,
                        "task_runtime": dict(eff),
                        "schema_version": "chapter3-no-capec-v1",
                        "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                        "boundary_contract_version": "chapter3-boundary-sync-v2",
                        "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                        "split_sha256": split_sha,
                        "gold_aggregate_sha256": gold_agg,
                        "dataset_freeze_manifest_sha256": freeze_sha,
                    }),
                    encoding="utf-8",
                )
            artifact = promote_protegi(
                entity_prompt_path=ent, relation_prompt_path=rel,
                entity_summary_path=ent_p, relation_summary_path=rel_p,
                entity_cache_train_manifest_path=tmp / "train_m.json",
                entity_cache_dev_manifest_path=tmp / "dev_m.json",
                prompt_scope="constrained",
                output_artifact_path=tmp / "art.json",
            )
            self.assertEqual(artifact["task_runtime"]["temperature"], 0.0)

    def test_promote_rejects_missing_effective_runtime_for_formal_summary(self):
        from protegi.entity_cache import compute_prompt_hash
        from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0
        from promote_protegi_v6 import promote_protegi

        split_file = V6_ROOT / "data" / "train_dev_test_split_v7.json"
        freeze_file = V6_ROOT / "data" / "dataset_freeze_manifest_v6.json"
        split_sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
        freeze_sha = hashlib.sha256(freeze_file.read_bytes()).hexdigest()
        gold_agg = json.loads(freeze_file.read_text(encoding="utf-8"))[
            "gold_aggregate_sha256"
        ]
        live_construction = json.loads(freeze_file.read_text(encoding="utf-8"))[
            "window_construction"
        ]["version"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ent = tmp / "final_entity_prompt.txt"
            rel = tmp / "final_relation_prompt.txt"
            ent.write_text(ENTITY_PROMPT_P0, encoding="utf-8")
            rel.write_text(RELATION_PROMPT_P0, encoding="utf-8")
            bind = {
                "split_file_sha256_raw_bytes": split_sha,
                "dataset_freeze_manifest_sha256_raw_bytes": freeze_sha,
                "config_file_sha256_raw_bytes": hashlib.sha256(b"c").hexdigest(),
            }
            base_cfg = {
                "task_model": "hy3", "task_max_workers": 8,
                "task_temperature": 0.0, "task_thinking": "disabled",
                "task_reasoning_effort": "none", "task_top_p": 0.95,
                "task_max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            }
            ent_sum = {
                "stage": "entity", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": True, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(
                    ent.read_text(encoding="utf-8")
                ),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    ent.read_bytes()
                ).hexdigest(),
                "config": dict(base_cfg), "input_bindings": dict(bind),
            }
            rel_sum = {
                "stage": "relation", "prompt_scope": "constrained",
                "experiment_pair_id": "protegi-prompt-scope-v1",
                "formal_eligible": True, "dry_run": False,
                "winner_prompt_sha256": compute_prompt_hash(
                    rel.read_text(encoding="utf-8")
                ),
                "winner_prompt_sha256_raw_bytes": hashlib.sha256(
                    rel.read_bytes()
                ).hexdigest(),
                "config": dict(base_cfg), "input_bindings": dict(bind),
            }
            ent_p = tmp / "ent_sum.json"
            rel_p = tmp / "rel_sum.json"
            ent_p.write_text(json.dumps(ent_sum), encoding="utf-8")
            rel_p.write_text(json.dumps(rel_sum), encoding="utf-8")
            eff = {
                "model": "hy3", "max_workers": 8, "temperature": 0.0,
                "thinking": "disabled", "reasoning_effort": "none",
                "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
                "window_overlap": 400, "document_abbreviation_context": False,
                "vulnerability_anchored_backfill": False,
            }
            for name in ("train", "dev"):
                (tmp / f"{name}_m.json").write_text(
                    json.dumps({
                        "split_name": name,
                        "entity_prompt_sha256": hashlib.sha256(
                            ent.read_bytes()
                        ).hexdigest(),
                        "prompt_scope": "constrained", "task_max_workers": 8,
                        "window_chars": 3000, "window_overlap": 400,
                        "window_construction": live_construction,
                        "document_abbreviation_context": False,
                        "vulnerability_anchored_backfill": False,
                        "task_runtime": dict(eff),
                        "schema_version": "chapter3-no-capec-v1",
                        "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
                        "boundary_contract_version": "chapter3-boundary-sync-v2",
                        "dataset_version": "v6-v5-gold-v9-mcpu-v2",
                        "split_sha256": split_sha,
                        "gold_aggregate_sha256": gold_agg,
                        "dataset_freeze_manifest_sha256": freeze_sha,
                    }),
                    encoding="utf-8",
                )
            with self.assertRaises(ValueError) as ctx:
                promote_protegi(
                    entity_prompt_path=ent, relation_prompt_path=rel,
                    entity_summary_path=ent_p, relation_summary_path=rel_p,
                    entity_cache_train_manifest_path=tmp / "train_m.json",
                    entity_cache_dev_manifest_path=tmp / "dev_m.json",
                    prompt_scope="constrained",
                    output_artifact_path=tmp / "art.json",
                )
            self.assertIn("effective_task_runtime", str(ctx.exception))

    def test_stage1_stage2_effective_runtime_must_match(self):
        from promote_protegi_v6 import _extract_task_runtime

        eff_a = {
            "model": "hy3", "max_workers": 8, "temperature": 0.0,
            "thinking": "disabled", "reasoning_effort": "none",
            "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
            "window_overlap": 400, "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        eff_b = dict(eff_a)
        eff_b["temperature"] = 0.7
        ra = _extract_task_runtime(
            {"config": {"effective_task_runtime": eff_a}}, "Stage 1"
        )
        rb = _extract_task_runtime(
            {"config": {"effective_task_runtime": eff_b}}, "Stage 2"
        )
        self.assertNotEqual(ra["temperature"], rb["temperature"])


    def test_stage1_stage2_effective_runtime_must_match(self):
        from promote_protegi_v6 import _extract_task_runtime

        eff_a = {
            "model": "hy3", "max_workers": 8, "temperature": 0.0,
            "thinking": "disabled", "reasoning_effort": "none",
            "top_p": 0.95, "max_tokens": 4096, "window_chars": 3000,
            "window_overlap": 400, "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        eff_b = dict(eff_a)
        eff_b["temperature"] = 0.7
        ra = _extract_task_runtime(
            {"config": {"effective_task_runtime": eff_a}}, "Stage 1"
        )
        rb = _extract_task_runtime(
            {"config": {"effective_task_runtime": eff_b}}, "Stage 2"
        )
        self.assertNotEqual(ra["temperature"], rb["temperature"])


class TestSearchStabilityOffline(unittest.TestCase):
    """Stage 1 运行稳定性：纯离线测试，不调用真实模型。"""

    def _runtime(self, **overrides):
        rt = {
            "model": "mock-task", "max_workers": 1, "temperature": 0.0,
            "thinking": "disabled", "reasoning_effort": "none",
            "top_p": 0.95, "max_tokens": 16, "window_chars": 3000,
            "window_overlap": 400, "document_abbreviation_context": False,
            "vulnerability_anchored_backfill": False,
        }
        rt.update(overrides)
        return rt

    def _config(self, **overrides):
        cfg = {
            "prompt_scope": "constrained",
            "experiment_pair_id": "test-stability-v1",
            "_config_file_sha256": "test-config-sha",
            "seed": 42, "beam_width": 1, "optimization_steps": 1,
            "minibatch_size": 1, "eval_batch_size": 1,
            "total_pull_budget_per_round": 4, "min_pulls_per_candidate": 1,
            "errors_per_group": 1, "gradients_per_error_group": 1,
            "max_error_groups": 1, "edits_per_gradient": 1,
            "paraphrases_per_edit": 0, "successors_per_parent": 1,
            "ucb_c": 2.0,
            "task_model": "mock-task", "task_max_workers": 1,
            "task_temperature": 0.0, "task_thinking": "disabled",
            "task_reasoning_effort": "none", "task_top_p": 0.95,
            "task_max_tokens": 16,
            "optimizer_model": "mock-opt",
            "effective_task_runtime": self._runtime(),
        }
        cfg.update(overrides)
        return cfg

    def _bindings(self):
        return {
            "split_sha256": "test-split-sha",
            "gold_aggregate_sha256": "test-gold-agg",
            "freeze_manifest_sha256": "test-freeze-sha",
            "config_file_sha256": "test-config-sha",
        }

    def _samples(self, n=3):
        return [
            {
                "sample_id": f"d{i}_w0", "doc_id": f"d{i}",
                "text": f"CVE-2021-100{i} affects Acme Widget {i}.",
                "gold_entities": [
                    {"id": "E1", "type": "Vulnerability",
                     "start": 0, "end": 14}
                ],
                "gold_relations": [],
            }
            for i in range(n)
        ]

    class _TaskClient:
        """可编程 task client：ok / budget / transient 三种行为。"""

        def __init__(self, mode="ok", fail_on=0):
            self.config = {
                "model": "mock-task", "temperature": 0.0,
                "thinking": "disabled", "reasoning_effort": "none",
                "top_p": 0.95, "max_tokens": 16,
            }
            self.mode = mode
            self.fail_on = fail_on
            self.calls = 0
            self.call_prompts = []

        def call_fn(self, prompt="", system_prompt="", config=None):
            from llm_extractor import ModelOutputBudgetExhaustedError

            self.calls += 1
            self.call_prompts.append(prompt)
            if self.mode == "budget_always":
                raise ModelOutputBudgetExhaustedError(16, 16)
            if self.mode == "budget_on_bad" and "BAD" in prompt:
                raise ModelOutputBudgetExhaustedError(16, 16)
            if self.mode == "budget_from" and self.calls >= self.fail_on:
                raise ModelOutputBudgetExhaustedError(16, 16)
            if self.mode == "transient_once" and self.calls >= self.fail_on:
                raise RuntimeError("simulated 500 internal error")
            if self.mode == "transient_always":
                raise RuntimeError("simulated 500 internal error")
            return '{"entities": []}'

    class _OptClient:
        config = {}

        def __init__(self, bad_edit=False):
            self.bad_edit = bad_edit

        def call_fn(self, prompt="", system_prompt="", config=None):
            if "critic" in system_prompt:
                return "<START>The guidance lacks explicit span verification.<END>"
            if "engineer" in system_prompt:
                if self.bad_edit:
                    return "<START>BAD marker guidance, verify spans carefully.<END>"
                return "<START>Verify each emitted span against the source text.<END>"
            return "<START>Verify each emitted span against the source text.<END>"

    def _optimizer(self, tmp: Path, task_client, opt_client=None, **cfg_over):
        from protegi.optimizer import ProTeGiOptimizer

        return ProTeGiOptimizer(
            stage="entity", method="protegi",
            config=self._config(**cfg_over),
            output_dir=tmp / "out",
            gold_dir=tmp / "gold",
            split_file=tmp / "split.json",
            entity_cache_dir=tmp / "ec",
            task_client=task_client,
            optimizer_client=opt_client or self._OptClient(),
            freeze_bindings=self._bindings(),
        )

    def test_budget_exhaustion_retry_once_then_candidate_invalid(self):
        from protegi.evaluator import TaskEvaluator
        from protegi.search_stability import WindowBudgetExhaustedError

        client = self._TaskClient(mode="budget_always")
        evaluator = TaskEvaluator(task_client=client, max_workers=1)
        samples = self._samples(n=2)
        with self.assertRaises(WindowBudgetExhaustedError):
            evaluator.evaluate_stage1_batch(samples, "prompt text")
        # 同一窗口在完全相同 runtime 下最多重试 1 次（首调 + 1 次重试）。
        self.assertEqual(client.calls, 2)

    def test_invalid_candidate_does_not_receive_fake_f1(self):
        from protegi.models import PromptCandidate
        from protegi.search_stability import (
            CandidateBudgetExhaustedError,
            INVALID_BUDGET_EXHAUSTED,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = self._optimizer(
                Path(tmpdir), self._TaskClient(), self._OptClient()
            )
            cand = PromptCandidate(candidate_id="c_bad", prompt_text="BAD prompt")
            err = CandidateBudgetExhaustedError(
                candidate_id="c_bad", stage="entity",
                prompt_hash="abc", sample_ids=["s1"],
                failure_reason="budget exhausted",
            )
            optimizer._mark_candidate_invalid(cand, err, round_idx=1)
            self.assertEqual(cand.selection_status, INVALID_BUDGET_EXHAUSTED)
            self.assertEqual(cand.estimated_reward, -1.0)
            self.assertNotIn("train_f1", cand.metrics)
            self.assertNotIn("dev_f1", cand.metrics)
            self.assertEqual(
                cand.metrics["evaluation_status"], INVALID_BUDGET_EXHAUSTED
            )

    def test_invalid_candidate_never_enters_beam_winner(self):
        from protegi.optimizer import ProTeGiOptimizer

        saved = []
        orig_save = ProTeGiOptimizer._save_checkpoint

        def spy(self, **kwargs):
            saved.append((kwargs.get("phase"), kwargs.get("next_round")))
            return orig_save(self, **kwargs)

        ProTeGiOptimizer._save_checkpoint = spy
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                optimizer = self._optimizer(
                    tmp, self._TaskClient(mode="budget_on_bad"),
                    self._OptClient(bad_edit=True),
                )
                winner = optimizer.run_optimization(self._samples(3), self._samples(1))
                self.assertEqual(winner.candidate_id, "P_E0")
                bad_id = "c_r1_p0_g0_edit"
                node = optimizer.lineage_tracker.nodes.get(bad_id)
                self.assertIsNotNone(node)
                self.assertEqual(
                    node["selection_status"], "invalid_budget_exhausted"
                )
                self.assertNotIn("train_f1", node["metrics"])
                summary = json.loads(
                    (tmp / "out" / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    summary["search_stability"]["candidates_invalid_budget_exhausted"], 1
                )
                self.assertEqual(summary["winner_candidate_id"], "P_E0")
                # 中止分支同样落检查点（含 round-0 与 abort 轮）。
                self.assertIn(("search", 1), saved)
        finally:
            ProTeGiOptimizer._save_checkpoint = orig_save

    def test_p0_budget_exhaustion_hard_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = self._optimizer(
                Path(tmpdir), self._TaskClient(mode="budget_always"),
                self._OptClient(),
            )
            with self.assertRaises(RuntimeError) as ctx:
                optimizer.run_optimization(self._samples(2), self._samples(1))
            self.assertIn("P0", str(ctx.exception))

    def test_p0_round_parent_eval_budget_hard_fails(self):
        """P0 在 round parent 评估（梯度 minibatch）持续超限 → 整 run 硬失败。

        调用时序（已标定）：init 占 1 次调用，parent minibatch 恰为第 2 次。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            client = self._TaskClient(mode="budget_from", fail_on=2)
            optimizer = self._optimizer(tmp, client, self._OptClient())
            with self.assertRaises(RuntimeError) as ctx:
                optimizer.run_optimization(self._samples(4), self._samples(1))
            self.assertIn("P0", str(ctx.exception))
            self.assertIn("round=1", str(ctx.exception))
            # init(1) + parent 首调/重试(2,3) 后即硬失败，未进入选择。
            self.assertEqual(client.calls, 3)
            where = [e["where"] for e in optimizer.stability["failure_log"]]
            self.assertNotIn("round_selection_aborted", where)
            self.assertFalse((tmp / "out" / "summary.json").exists())

    def test_p0_ucb_eval_budget_hard_fails(self):
        """P0 在 UCB 选择评估中持续超限 → 整 run 硬失败（非 abort 继续）。

        调用时序（已标定）：init(1) + parent(2) 成功，第 4 次调用
        （P0 的选择 pull）首调/重试(4,5)均超限。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            client = self._TaskClient(mode="budget_from", fail_on=4)
            optimizer = self._optimizer(tmp, client, self._OptClient())
            with self.assertRaises(RuntimeError) as ctx:
                optimizer.run_optimization(self._samples(4), self._samples(1))
            self.assertIn("P0", str(ctx.exception))
            self.assertIn("round=1", str(ctx.exception))
            self.assertEqual(client.calls, 5)
            where = [e["where"] for e in optimizer.stability["failure_log"]]
            self.assertNotIn("round_selection_aborted", where)
            self.assertFalse((tmp / "out" / "summary.json").exists())

    def test_http_retry_exhausted_writes_checkpoint(self):
        import protegi.evaluator as evaluator_module
        import protegi.retry_utils as retry_utils

        orig = retry_utils.retry_api_call

        def fast_retry(fn, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["max_retries"] = 2
            kwargs["initial_delay"] = 0
            kwargs["max_delay"] = 0
            return orig(fn, *args, **kwargs)

        evaluator_module.retry_api_call = fast_retry
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                optimizer = self._optimizer(
                    tmp, self._TaskClient(mode="transient_always"),
                    self._OptClient(),
                )
                with self.assertRaises(RuntimeError) as ctx:
                    optimizer.run_optimization(self._samples(2), self._samples(1))
                self.assertIn("500", str(ctx.exception))
                ckpt_path = tmp / "out" / "search_checkpoint.json"
                self.assertTrue(ckpt_path.is_file())
                ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
                self.assertEqual(ckpt["phase"], "search")
                self.assertEqual(ckpt["next_round"], 0)
                self.assertEqual(ckpt["stage"], "entity")
                self.assertIn("effective_task_runtime", ckpt)
        finally:
            evaluator_module.retry_api_call = orig

    def test_resume_restores_exact_search_state(self):
        import random
        import protegi.evaluator as evaluator_module
        import protegi.retry_utils as retry_utils
        from protegi.optimizer import ProTeGiOptimizer

        orig = retry_utils.retry_api_call

        def fast_retry(fn, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["max_retries"] = 2
            kwargs["initial_delay"] = 0
            kwargs["max_delay"] = 0
            return orig(fn, *args, **kwargs)

        evaluator_module.retry_api_call = fast_retry
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                train = self._samples(3)
                dev = self._samples(1)
                fail_client = self._TaskClient(mode="transient_once", fail_on=4)
                opt1 = self._optimizer(tmp, fail_client, self._OptClient())
                with self.assertRaises(RuntimeError):
                    opt1.run_optimization(train, dev)
                self.assertTrue((tmp / "out" / "search_checkpoint.json").is_file())

                ok_client = self._TaskClient(mode="ok")
                bindings = self._bindings()
                from protegi.search_stability import (
                    implementation_hashes, sample_ids_hash,
                )

                resume_bindings = {
                    **bindings, "stage": "entity", "method": "protegi",
                    "prompt_scope": "constrained",
                    "experiment_pair_id": "test-stability-v1",
                    "effective_task_runtime": self._runtime(),
                    "implementation": implementation_hashes(Path.cwd()),
                    "train_sample_ids_sha256": sample_ids_hash(
                        [s["sample_id"] for s in train]
                    ),
                    "dev_sample_ids_sha256": sample_ids_hash(
                        [s["sample_id"] for s in dev]
                    ),
                }
                opt2 = ProTeGiOptimizer(
                    stage="entity", method="protegi",
                    config=self._config(), output_dir=tmp / "out",
                    gold_dir=tmp / "gold", split_file=tmp / "split.json",
                    entity_cache_dir=tmp / "ec", task_client=ok_client,
                    optimizer_client=self._OptClient(),
                    freeze_bindings=bindings, resume=True,
                    resume_bindings=resume_bindings,
                )
                winner = opt2.run_optimization(train, dev)
                self.assertIsNotNone(winner.candidate_id)
                summary = json.loads(
                    (tmp / "out" / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["search_stability"]["resume_count"], 1)
                self.assertGreaterEqual(
                    summary["search_stability"]["evaluation_cache_hits"], 2
                )
                # RNG 精确恢复：redo 的 round-1 minibatch 与首轮首次抽取一致。
                expected_batch = random.Random(42).sample(train, 1)
                expected_ids = [s["sample_id"] for s in expected_batch]
                grad_batches = json.loads(
                    (tmp / "out" / "round_1" / "gradient_minibatches.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(grad_batches[0]["sample_ids"], expected_ids)
        finally:
            evaluator_module.retry_api_call = orig

    def test_resume_rejects_hash_mismatch(self):
        from protegi.search_stability import (
            save_search_checkpoint, validate_checkpoint_bindings,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bindings = {
                "stage": "entity", "method": "protegi",
                "prompt_scope": "constrained",
                "experiment_pair_id": "test-stability-v1",
                "config_file_sha256": "test-config-sha",
                "split_sha256": "test-split-sha",
                "gold_aggregate_sha256": "test-gold-agg",
                "freeze_manifest_sha256": "test-freeze-sha",
                "effective_task_runtime": self._runtime(),
                "implementation": {"a.py": "abc"},
                "train_sample_ids_sha256": "t", "dev_sample_ids_sha256": "d",
            }
            save_search_checkpoint(tmp, {**bindings, "phase": "search", "next_round": 1})
            ckpt = json.loads(
                (tmp / "search_checkpoint.json").read_text(encoding="utf-8")
            )
            bad_split = dict(bindings, split_sha256="tampered")
            with self.assertRaises(ValueError) as ctx:
                validate_checkpoint_bindings(ckpt, bad_split)
            self.assertIn("split_sha256", str(ctx.exception))
            bad_runtime = dict(
                bindings, effective_task_runtime=self._runtime(temperature=0.7)
            )
            with self.assertRaises(ValueError) as ctx:
                validate_checkpoint_bindings(ckpt, bad_runtime)
            self.assertIn("effective_task_runtime", str(ctx.exception))

    def test_evaluation_cache_avoids_duplicate_task_calls(self):
        from protegi.models import PromptCandidate

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            client = self._TaskClient(mode="ok")
            optimizer = self._optimizer(tmp, client, self._OptClient())
            samples = self._samples(2)
            cand = PromptCandidate(candidate_id="c1", prompt_text="prompt one")
            res1, _ = optimizer._evaluate_candidate_batch(cand, samples)
            calls_after_first = client.calls
            self.assertGreater(calls_after_first, 0)
            res2, _ = optimizer._evaluate_candidate_batch(cand, samples)
            self.assertEqual(client.calls, calls_after_first)
            self.assertEqual((res2.tp, res2.fp, res2.fn), (res1.tp, res1.fp, res1.fn))
            other = PromptCandidate(candidate_id="c2", prompt_text="prompt two")
            optimizer._evaluate_candidate_batch(other, samples)
            self.assertGreater(client.calls, calls_after_first)

    def test_counter_sync_never_goes_negative_on_stale_watermark(self):
        """resume 旧水位 + 新 evaluator(0)：同步不得产生负增量。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer = self._optimizer(tmp, self._TaskClient(mode="ok"))
            optimizer.stability["transient_api_retries"] = 5
            optimizer.stability["budget_exhaustion_retries"] = 3
            optimizer.call_stats.transient_api_retries = 5
            optimizer.call_stats.budget_exhaustion_retries = 3
            # 模拟陈旧水位（旧进程累计值）+ 全新 evaluator（0）。
            optimizer._last_synced_eval_transient = 7
            optimizer._last_synced_eval_budget = 4
            optimizer.evaluator.transient_api_retries = 0
            optimizer.evaluator.budget_exhaustion_retries = 0
            optimizer._sync_evaluator_counters()
            self.assertEqual(optimizer.stability["transient_api_retries"], 5)
            self.assertEqual(optimizer.stability["budget_exhaustion_retries"], 3)
            self.assertEqual(optimizer.call_stats.transient_api_retries, 5)
            self.assertEqual(optimizer.call_stats.budget_exhaustion_retries, 3)
            self.assertEqual(optimizer._last_synced_eval_transient, 0)
            self.assertEqual(optimizer._last_synced_eval_budget, 0)
            where = [e["where"] for e in optimizer.stability["failure_log"]]
            self.assertIn("counter_sync_negative_transient", where)
            self.assertIn("counter_sync_negative_budget", where)
            # 水位重置后，新增量仍可正常同步。
            optimizer.evaluator.transient_api_retries = 2
            optimizer.evaluator.budget_exhaustion_retries = 1
            optimizer._sync_evaluator_counters()
            self.assertEqual(optimizer.stability["transient_api_retries"], 7)
            self.assertEqual(optimizer.stability["budget_exhaustion_retries"], 4)

    def test_checkpoint_never_persists_invalid_beam_members(self):
        """_save_checkpoint 必须清掉 INVALID beam 并持久化其身份。"""
        import json as _json
        from protegi.models import PromptCandidate
        from protegi.search_stability import (
            INVALID_BUDGET_EXHAUSTED,
            INVALID_OUTPUT_AMPLIFICATION,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer = self._optimizer(tmp, self._TaskClient(mode="ok"))
            valid = PromptCandidate(
                candidate_id="c_valid", prompt_text="prompt valid",
            )
            valid.selection_status = "selected"
            bad_budget = PromptCandidate(
                candidate_id="c_bad_budget", prompt_text="prompt bad",
            )
            bad_budget.selection_status = INVALID_BUDGET_EXHAUSTED
            bad_amp = PromptCandidate(
                candidate_id="c_bad_amp", prompt_text="prompt amp",
            )
            bad_amp.selection_status = INVALID_OUTPUT_AMPLIFICATION
            optimizer._invalid_candidate_ids = {"c_bad_budget", "c_bad_amp"}
            samples = self._samples(2)
            optimizer._save_checkpoint(
                phase="search", next_round=2,
                beam=[valid, bad_budget, bad_amp],
                p0_candidate=valid,
                train_samples=samples, dev_samples=samples[:1],
                reason="unit test",
            )
            ckpt = _json.loads(
                (tmp / "out" / "search_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            beam_ids = [c["candidate_id"] for c in ckpt["beam"]]
            self.assertEqual(beam_ids, ["c_valid"])
            self.assertEqual(
                sorted(ckpt["invalid_candidate_ids"]),
                ["c_bad_amp", "c_bad_budget"],
            )

    def test_resume_restores_invalid_ids_and_cleans_legacy_dirty_beam(self):
        """旧检查点（beam 残留 INVALID、无 stored 键）恢复时身份不丢、beam 清掉。"""
        from protegi.models import PromptCandidate
        from protegi.optimizer import ProTeGiOptimizer
        from protegi.search_stability import (
            INVALID_BUDGET_EXHAUSTED,
            implementation_hashes,
            sample_ids_hash,
            save_search_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train = self._samples(3)
            dev = self._samples(1)
            valid = PromptCandidate(
                candidate_id="c_valid", prompt_text="prompt valid",
            )
            valid.selection_status = "selected"
            legacy_bad = PromptCandidate(
                candidate_id="c_legacy_bad", prompt_text="prompt bad",
            )
            legacy_bad.selection_status = INVALID_BUDGET_EXHAUSTED
            bindings = self._bindings()
            payload = {
                **bindings,
                "stage": "entity", "method": "protegi",
                "prompt_scope": "constrained",
                "experiment_pair_id": "test-stability-v1",
                "config_file_sha256": "test-config-sha",
                "effective_task_runtime": self._runtime(),
                "implementation": implementation_hashes(Path.cwd()),
                "phase": "search", "next_round": 2,
                "beam": [valid.to_dict(), legacy_bad.to_dict()],
                "p0_candidate": valid.to_dict(),
                "lineage": {"nodes": {}, "edges": []},
                "call_stats": {},
                "evaluated_candidate_ids": [],
                # 故意不写 invalid_candidate_ids：模拟旧检查点。
                "stability": None,
                "train_sample_ids": [s["sample_id"] for s in train],
                "dev_sample_ids": [s["sample_id"] for s in dev],
                "train_sample_ids_sha256": sample_ids_hash(
                    [s["sample_id"] for s in train]
                ),
                "dev_sample_ids_sha256": sample_ids_hash(
                    [s["sample_id"] for s in dev]
                ),
                "reason": "legacy dirty beam",
            }
            (tmp / "out").mkdir(parents=True, exist_ok=True)
            save_search_checkpoint(tmp / "out", payload)
            resume_bindings = {
                **bindings, "stage": "entity", "method": "protegi",
                "prompt_scope": "constrained",
                "experiment_pair_id": "test-stability-v1",
                "config_file_sha256": "test-config-sha",
                "effective_task_runtime": self._runtime(),
                "implementation": implementation_hashes(Path.cwd()),
                "train_sample_ids_sha256": sample_ids_hash(
                    [s["sample_id"] for s in train]
                ),
                "dev_sample_ids_sha256": sample_ids_hash(
                    [s["sample_id"] for s in dev]
                ),
            }
            optimizer = ProTeGiOptimizer(
                stage="entity", method="protegi",
                config=self._config(), output_dir=tmp / "out",
                gold_dir=tmp / "gold", split_file=tmp / "split.json",
                entity_cache_dir=tmp / "ec",
                task_client=self._TaskClient(mode="ok"),
                optimizer_client=self._OptClient(),
                freeze_bindings=bindings, resume=True,
                resume_bindings=resume_bindings,
            )
            # _init 恢复阶段：INVALID 身份从 beam 派生，不丢。
            self.assertIn(
                "c_legacy_bad", optimizer._invalid_candidate_ids
            )
            # 加载侧清理：遗留脏 beam 成员不得进入 active beam。
            cleaned = optimizer._drop_invalid_from_beam(
                [PromptCandidate.from_dict(c) for c in
                 optimizer._resumed_checkpoint.get("beam", [])]
            )
            self.assertEqual(
                [c.candidate_id for c in cleaned], ["c_valid"]
            )


class TestOutputExpansionGuard(unittest.TestCase):
    """Output Expansion Guard：纯离线，不调用真实模型。"""

    SMOKING_GUN_GUIDANCE = (
        "Proceed linearly across the full content covering all areas such as "
        "prose, tables, lists, and captions, inspecting every token position "
        "for a directly stated match.\n"
        "Create an individual entry for each offset-distinct occurrence, "
        "counting repeated identical strings as well as nested or overlapping "
        "matches at the same location separately; do not collapse duplicates "
        "or merge several occurrences into a single excerpt."
    )

    # run7 中预算耗尽、但属于精确判断语义、必须放行的两条 guidance。
    RUN7_PRECISE_EDIT_GUIDANCE = (
        "Scan the entire input, including prose, tables, lists, captions, and "
        "line-broken identifiers.\n"
        "Apply a strict local-block gate to any affected-product candidate:\n"
        "1. Bound its local block to one sentence, one table row, one list item, "
        "or one caption line; do not cross boundaries to find support.\n"
        "2. Require that the same block contains an explicit vulnerability identifier.\n"
        "3. Require that the same block explicitly links that identifier to that "
        "candidate as affected.\n"
        "4. If either requirement fails, suppress the candidate and do not infer "
        "support from elsewhere in the document.\n"
        "Verify each retained span against the source before returning."
    )
    RUN7_PRECISE_PARA_GUIDANCE = (
        "Examine the full input across prose, tables, lists, captions, and "
        "identifiers broken over lines, and enforce the fixed definitions with "
        "conservative strictness. For any code token appearing alongside name "
        "wording, confine the span exclusively to the code characters: begin at "
        "the initial code character and terminate directly after the terminal "
        "code character. Omit all adjacent name wording, parentheses, commas, "
        "punctuation, and whitespace from the span. When a single grouping holds "
        "several codes, decompose it and create one distinct minimal "
        "code-characters-only span for each code, then confirm each span through "
        "exact character-by-character matching to the source before emitting."
    )

    def test_output_expansion_guard_rejects_run7_smoking_gun(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        result = OutputExpansionGuard.validate_guidance(self.SMOKING_GUN_GUIDANCE)
        self.assertFalse(result)
        self.assertTrue(
            any(
                r.startswith("output_amplification_contract_violation")
                for r in result.reasons
            )
        )

    def test_output_expansion_guard_rejects_every_token_position(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Inspect every token position for matches.",
            "Check each token position carefully.",
            "Cover all token positions in document order.",
        ):
            with self.subTest(text=text):
                self.assertFalse(OutputExpansionGuard.validate_guidance(text))

    def test_output_expansion_guard_rejects_duplicate_preservation(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Do not collapse duplicates across windows.",
            "Preserve duplicates for recall.",
            "Keep duplicates emitted.",
            "Retain duplicates in the output.",
            "Do not deduplicate spans.",
        ):
            with self.subTest(text=text):
                self.assertFalse(OutputExpansionGuard.validate_guidance(text))

    def test_output_expansion_guard_rejects_unconditional_nested_overlap_enumeration(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Emit all nested spans found in the window.",
            "Output every overlapping span without filtering.",
            "List nested or overlapping matches separately.",
        ):
            with self.subTest(text=text):
                self.assertFalse(OutputExpansionGuard.validate_guidance(text))

    def test_output_expansion_guard_rejects_all_possible_spans(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Enumerate all possible spans in the text.",
            "Check every possible span for entity evidence.",
            "List all candidate spans before deciding.",
            "Extract every substring as a candidate.",
            "Enumerate tokens and spans exhaustively.",
        ):
            with self.subTest(text=text):
                self.assertFalse(OutputExpansionGuard.validate_guidance(text))

    def test_output_expansion_guard_allows_normal_span_verification(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Check every predicted span against the source text.",
            "Deduplicate exact repeated (type, start, end) entities.",
            "Inspect each candidate entity for compliance with the MCPU boundary rule.",
            "Do not enumerate every token; emit each verified entity once.",
        ):
            with self.subTest(text=text):
                result = OutputExpansionGuard.validate_guidance(text)
                self.assertTrue(result, result.reasons)

    def test_output_expansion_guard_allows_conditional_legitimate_overlap(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        result = OutputExpansionGuard.validate_guidance(
            "Overlapping mentions may be emitted only when each independently "
            "satisfies the frozen entity boundary contract."
        )
        self.assertTrue(result, result.reasons)

    def test_output_expansion_guard_negation_exempts_affirmative_rules(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            "Never inspect every token position; emit only verified entities.",
            "Do not create an individual entry for each occurrence separately.",
            "Never keep duplicates; deduplicate strictly before returning.",
            "Do not output all nested spans unconditionally.",
            "Do not list all candidate spans; decide each verified entity once.",
            "Avoid emitting every substring as a candidate.",
        ):
            with self.subTest(text=text):
                result = OutputExpansionGuard.validate_guidance(text)
                self.assertTrue(result, result.reasons)

    def test_output_expansion_guard_rule_error_fails_closed(self):
        import protegi.output_expansion_guard as guard_module
        from protegi.output_expansion_guard import OutputExpansionGuard

        original_rules = guard_module._RULES

        def _boom(text):
            raise RuntimeError("simulated rule bug")

        guard_module._RULES = (("token_position_enumeration", _boom),)
        try:
            result = OutputExpansionGuard.validate_guidance(
                "Inspect every token position for matches."
            )
        finally:
            guard_module._RULES = original_rules
        self.assertFalse(result)
        self.assertTrue(
            any(r.startswith("guard_rule_error:") for r in result.reasons)
        )

    def test_output_expansion_guard_allows_run7_precision_guidance(self):
        from protegi.output_expansion_guard import OutputExpansionGuard

        for text in (
            self.RUN7_PRECISE_EDIT_GUIDANCE,
            self.RUN7_PRECISE_PARA_GUIDANCE,
        ):
            result = OutputExpansionGuard.validate_guidance(text)
            self.assertTrue(result, result.reasons)

    def _expansion_mini_optimizer(self, tmp: Path):
        base = TestSearchStabilityOffline()
        task_client = base._TaskClient(mode="ok")

        class GuardTripOptClient(base._OptClient):
            def call_fn(self, prompt="", system_prompt="", config=None):
                if "engineer" in system_prompt:
                    return (
                        "<START>Inspect every token position and emit "
                        "each occurrence separately.<END>"
                    )
                return super().call_fn(
                    prompt=prompt, system_prompt=system_prompt, config=config
                )

        optimizer = base._optimizer(tmp, task_client, GuardTripOptClient())
        return optimizer, task_client

    def test_output_expansion_candidate_never_calls_task_model(self):
        base = TestSearchStabilityOffline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer, task_client = self._expansion_mini_optimizer(tmp)
            winner = optimizer.run_optimization(base._samples(3), base._samples(1))
            self.assertEqual(winner.candidate_id, "P_E0")
            # 被 guard 拒绝的 edit 其 guidance marker 从未进入任何 task 调用。
            for prompt in task_client.call_prompts:
                self.assertNotIn("every token position", prompt)

    def test_output_expansion_candidate_never_enters_beam_or_winner(self):
        from protegi.search_stability import INVALID_OUTPUT_AMPLIFICATION

        base = TestSearchStabilityOffline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer, _ = self._expansion_mini_optimizer(tmp)
            winner = optimizer.run_optimization(base._samples(3), base._samples(1))
            self.assertEqual(winner.candidate_id, "P_E0")
            bad_id = "c_r1_p0_g0_edit"
            node = optimizer.lineage_tracker.nodes.get(bad_id)
            self.assertIsNotNone(node)
            self.assertEqual(node["selection_status"], INVALID_OUTPUT_AMPLIFICATION)
            self.assertEqual(
                node["metrics"].get("failure_reason"),
                "output_amplification_contract_violation",
            )
            self.assertNotIn("train_f1", node["metrics"])
            beam_ids = [
                c["candidate_id"]
                for c in json.loads(
                    (tmp / "out" / "round_1" / "beam.json").read_text(encoding="utf-8")
                )
            ]
            self.assertNotIn(bad_id, beam_ids)
            summary = json.loads(
                (tmp / "out" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["search_stability"]["candidates_invalid_output_amplification"],
                1,
            )
            self.assertEqual(
                summary["search_stability"]["candidates_invalid_budget_exhausted"], 0
            )

    def test_frozen_p0_output_expansion_violation_hard_fails(self):
        import protegi.optimizer as optimizer_module
        from protegi.prompts_p0 import (
            ENTITY_PROMPT_P0,
            replace_optimizable_guidance,
        )

        base = TestSearchStabilityOffline()
        bad_p0 = replace_optimizable_guidance(
            ENTITY_PROMPT_P0, self.SMOKING_GUN_GUIDANCE
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer = base._optimizer(
                tmp, base._TaskClient(mode="ok"), base._OptClient()
            )
            original = optimizer_module.ENTITY_PROMPT_P0
            optimizer_module.ENTITY_PROMPT_P0 = bad_p0
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    optimizer.run_optimization(base._samples(2), base._samples(1))
            finally:
                optimizer_module.ENTITY_PROMPT_P0 = original
            self.assertIn("hard fail", str(ctx.exception))

    def test_checkpoint_records_startup_snapshot_not_save_time(self):
        import protegi.optimizer as optimizer_module

        base = TestSearchStabilityOffline()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            optimizer = base._optimizer(
                tmp, base._TaskClient(mode="ok"), base._OptClient()
            )
            sentinel = {"protegi/optimizer.py": "startup-sentinel"}
            optimizer._startup_implementation = dict(sentinel)
            original = optimizer_module.implementation_hashes
            optimizer_module.implementation_hashes = lambda root: {
                "protegi/optimizer.py": "save-time-drifted"
            }
            try:
                path = optimizer._save_checkpoint(
                    phase="search", next_round=1, beam=[],
                    p0_candidate=None, train_samples=[], dev_samples=[],
                    reason="probe",
                )
            finally:
                optimizer_module.implementation_hashes = original
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["implementation"], sentinel)

    def test_repeated_identical_sentence_negation(self):
        from protegi.output_expansion_guard import OutputExpansionGuard as G

        self.assertFalse(
            G.validate_guidance("Count repeated identical strings in each window.")
        )
        self.assertTrue(
            G.validate_guidance(
                "Do not count repeated identical strings; "
                "emit each verified entity once."
            )
        )
        # 否定域止于句界：第二句无否定仍拒绝。
        self.assertFalse(
            G.validate_guidance(
                "Emit verified entities. "
                "Count repeated identical strings in each window."
            )
        )

    def test_enumerate_combo_sentence_negation(self):
        from protegi.output_expansion_guard import OutputExpansionGuard as G

        self.assertFalse(
            G.validate_guidance("Enumerate all spans in the window.")
        )
        self.assertTrue(
            G.validate_guidance("Never enumerate tokens; verify each span instead.")
        )
        # 否定域止于句界：第二句无否定仍拒绝。
        self.assertFalse(
            G.validate_guidance(
                "Verify spans first. Enumerate tokens and spans exhaustively."
            )
        )


class TestOutputContractV2Diagnostic(unittest.TestCase):
    """输出范式 v2：纯离线诊断测试，不调用真实模型，不碰冻结契约。"""

    def test_v2_contract_keeps_frozen_definitions_byte_identical(self):
        from protegi.output_contract_v2 import build_v2_immutable_contract
        from protegi.prompts_p0 import ENTITY_IMMUTABLE_CONTRACT

        v2 = build_v2_immutable_contract()
        # 定义段落逐字节保留：四类型标题行必须原样存在。
        for anchor in (
            "- Vulnerability: every explicit CVE identifier mention.",
            "- Weakness: an explicit CWE identifier,",
            "- Configuration: the Minimum Canonical Product Unit (MCPU)",
            "- AttackTechnique: an explicit ATT&CK T-code",
        ):
            self.assertIn(anchor, v2)
            self.assertEqual(
                v2.count(anchor),
                ENTITY_IMMUTABLE_CONTRACT.count(anchor),
            )
        # 规则与示例已替换为最小字段版本。
        self.assertNotIn('"id": "E1"', v2)
        self.assertNotIn('"text": "exact substring"', v2)
        self.assertIn("OUTPUT_PARADIGM_VERSION: output-contract-v2-diagnostic-v1", v2)
        self.assertIn(
            "Return exactly these entity fields: type, start, end, normalized_id.",
            v2,
        )
        # 7 条 few-shot 记录全部转为最小字段。
        self.assertEqual(v2.count('{"type": "'), 7 + 1)  # 7 示例 + 1 返回示例

    def test_frozen_contract_untouched_by_v2(self):
        from protegi.prompts_p0 import ENTITY_IMMUTABLE_CONTRACT

        self.assertIn(
            "6. Return exactly these entity fields: "
            "id, text, type, start, end, normalized_id.",
            ENTITY_IMMUTABLE_CONTRACT,
        )
        self.assertNotIn("output-contract-v2", ENTITY_IMMUTABLE_CONTRACT)

    def test_v2_prompt_passes_output_expansion_guard(self):
        from protegi.output_contract_v2 import build_v2_entity_prompt
        from protegi.output_expansion_guard import OutputExpansionGuard

        result = OutputExpansionGuard.validate(
            build_v2_entity_prompt(), stage="entity"
        )
        self.assertTrue(result, result.reasons)

    def test_v2_parser_recovers_id_text_and_collapses_duplicates(self):
        from protegi.output_contract_v2 import parse_minimal_entity_mentions

        text = "CVE-2021-44228 in Log4j, CVE-2021-44228 again."
        first = text.find("CVE-2021-44228")
        second = text.find("CVE-2021-44228", first + 1)
        raw = [
            {"type": "Vulnerability", "start": first, "end": first + 14,
             "normalized_id": "CVE-2021-44228"},
            # 同一 span 重复：必须坍缩。
            {"type": "Vulnerability", "start": first, "end": first + 14,
             "normalized_id": "CVE-2021-44228"},
            # 同一表面不同 mention：合法，保留。
            {"type": "Vulnerability", "start": second, "end": second + 14,
             "normalized_id": "CVE-2021-44228"},
            # 无偏移记录：v2 不做表面扩展，直接丢弃。
            {"type": "Vulnerability", "normalized_id": "CVE-2021-44228"},
            # 非法类型与越界偏移：丢弃。
            {"type": "CAPEC", "start": 0, "end": 3},
            {"type": "Vulnerability", "start": -1, "end": 99999},
        ]
        entities, diag = parse_minimal_entity_mentions(text, raw)
        self.assertEqual(diag["raw_count"], 6)
        self.assertEqual(diag["duplicates_collapsed"], 1)
        self.assertEqual(diag["invalid_dropped"], 3)
        self.assertEqual(diag["kept_count"], 2)
        self.assertEqual(diag["nested_overlap_pairs"], 0)
        self.assertEqual([e["id"] for e in entities], ["E1", "E2"])
        self.assertEqual(entities[0]["text"], "CVE-2021-44228")
        self.assertEqual(
            (entities[0]["start"], entities[0]["end"]), (first, first + 14)
        )

    def test_v2_parser_counts_nested_overlap_without_dropping(self):
        from protegi.output_contract_v2 import parse_minimal_entity_mentions

        text = "Microsoft Exchange Server is affected."
        raw = [
            {"type": "Configuration", "start": 0, "end": 25},
            {"type": "Configuration", "start": 10, "end": 25},
        ]
        entities, diag = parse_minimal_entity_mentions(text, raw)
        self.assertEqual(diag["kept_count"], 2)
        self.assertEqual(diag["nested_overlap_pairs"], 1)

    def test_v2_parser_empty_list_roundtrip(self):
        from protegi.output_contract_v2 import parse_minimal_entity_mentions

        entities, diag = parse_minimal_entity_mentions("plain text", [])
        self.assertEqual(entities, [])
        self.assertEqual(diag["kept_count"], 0)
        self.assertEqual(diag["duplicates_collapsed"], 0)

    def test_v2_metric_parity_with_v1_spans(self):
        from protegi.output_contract_v2 import parse_minimal_entity_mentions

        text = "CVE-2021-44228 in Log4j (T1190)."
        v1_style = [
            {"id": "E9", "text": "CVE-2021-44228", "type": "Vulnerability",
             "start": 0, "end": 14, "normalized_id": "CVE-2021-44228"},
            {"id": "E3", "text": "Log4j", "type": "Configuration",
             "start": 18, "end": 23, "normalized_id": None},
        ]
        v2_style = [
            {"type": "Vulnerability", "start": 0, "end": 14,
             "normalized_id": "CVE-2021-44228"},
            {"type": "Configuration", "start": 18, "end": 23},
        ]
        gold = [
            {"id": "E1", "text": "CVE-2021-44228", "type": "Vulnerability",
             "start": 0, "end": 14, "normalized_id": "CVE-2021-44228"},
            {"id": "E2", "text": "Log4j", "type": "Configuration",
             "start": 18, "end": 23, "normalized_id": None},
        ]
        v2_entities, _ = parse_minimal_entity_mentions(text, v2_style)
        self.assertEqual(
            calc_strict_entity_sample_counts(v1_style, gold),
            calc_strict_entity_sample_counts(v2_entities, gold),
        )

    def test_v2_canonical_output_smaller_than_v1(self):
        import sys as _sys

        from protegi.output_contract_v2 import estimate_canonical_output_chars

        gold_dir = V6_ROOT / "data" / "annotations" / "gold"
        _sys.path.insert(0, str(V6_ROOT / "scripts"))
        from llm_methods import build_text_windows

        doc = json.loads((gold_dir / "aa24-207a.json").read_text(encoding="utf-8"))
        wins = build_text_windows(doc["text"], max_chars=3000, overlap=400)
        win = wins[5]
        local = [
            e for e in doc["entities"]
            if e.get("start") is not None and e.get("end") is not None
            and e["start"] >= win["start"] and e["end"] <= win["end"]
        ]
        local = [
            {**e, "start": e["start"] - win["start"],
             "end": e["end"] - win["start"]}
            for e in local
        ]
        self.assertGreaterEqual(len(local), 40)
        v1 = estimate_canonical_output_chars(
            win["text"], local, paradigm="v1"
        )
        v2 = estimate_canonical_output_chars(
            win["text"], local, paradigm="v2"
        )
        self.assertLess(v2, v1)
        # CVE 枚举窗上，normalized_id（CPE 长串）占主导，去掉 id/text
        # 约省 25-30%；阈值取 0.75 留余量。
        self.assertLess(v2, v1 * 0.75)

    def test_v2p_contract_keeps_text_drops_id_only(self):
        from protegi.output_contract_v2 import build_v2p_immutable_contract
        from protegi.prompts_p0 import ENTITY_IMMUTABLE_CONTRACT

        v2p = build_v2p_immutable_contract()
        self.assertIn("text-grounded-v1", v2p)
        # text 字段保留（接地证据），id 移除。
        self.assertNotIn('"id": "E1"', v2p)
        self.assertIn('"text": "CVE-2021-44228"', v2p)
        self.assertIn(
            "Return exactly these entity fields: "
            "text, type, start, end, normalized_id.",
            v2p,
        )
        # 定义段落逐字节保留。
        for anchor in (
            "- Vulnerability: every explicit CVE identifier mention.",
            "- Configuration: the Minimum Canonical Product Unit (MCPU)",
        ):
            self.assertIn(anchor, v2p)

    def test_v2p_prompt_passes_output_expansion_guard(self):
        from protegi.output_contract_v2 import build_v2p_entity_prompt
        from protegi.output_expansion_guard import OutputExpansionGuard

        result = OutputExpansionGuard.validate(
            build_v2p_entity_prompt(), stage="entity"
        )
        self.assertTrue(result, result.reasons)

    def test_v2p_records_parse_through_frozen_parser(self):
        import sys as _sys

        _sys.path.insert(0, str(V6_ROOT / "scripts"))
        from llm_methods import parse_entity_mentions

        text = "CVE-2021-44228 in Log4j (T1190)."
        # v2' 形状：无 id、有 text；偏移故意给错一条，验证程序端修复。
        raw = [
            {"text": "CVE-2021-44228", "type": "Vulnerability",
             "start": 0, "end": 14, "normalized_id": "CVE-2021-44228"},
            {"text": "Log4j", "type": "Configuration",
             "start": 99, "end": 104},
            {"text": "Log4j", "type": "Configuration",
             "start": 99, "end": 104},
        ]
        entities, _ = parse_entity_mentions(text, raw)
        spans = {(e["type"], e["start"], e["end"]) for e in entities}
        # 正确偏移 + 表面搜索修复各一条，去重后恰好两条。
        self.assertIn(("Vulnerability", 0, 14), spans)
        self.assertIn(("Configuration", 18, 23), spans)
        self.assertEqual(len(entities), 2)
        self.assertEqual([e["id"] for e in entities], ["E1", "E2"])


if __name__ == "__main__":
    unittest.main()


