"""ProTeGi 提示词突变与演化生成器。

实现 constrained / unconstrained 两个实验臂共用的三级生成机制：
1. 文本梯度生成器 (GradientGenerator): 根据错误样本群生成自然语言改进梯度；
2. 编辑器 (PromptEditor): 按实验臂重写 guidance 或完整语义提示；
3. 蒙特卡洛释义器 (MonteCarloParaphraser): 按同一实验臂扩展变体。
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from protegi.models import CallStats, ErrorExample, PromptCandidate, PromptGradient
from protegi.prompts_p0 import (
    extract_immutable_contract,
    extract_optimizable_guidance,
    replace_optimizable_guidance,
)
from protegi.retry_utils import retry_api_call
from protegi.templates import (
    EDIT_TEMPLATE,
    GRADIENT_TEMPLATE,
    PARAPHRASE_TEMPLATE,
    UNCONSTRAINED_EDIT_TEMPLATE,
    UNCONSTRAINED_GRADIENT_TEMPLATE,
    UNCONSTRAINED_PARAPHRASE_TEMPLATE,
)


PROMPT_SCOPES = {"constrained", "unconstrained"}


def _validate_prompt_scope(prompt_scope: str) -> str:
    scope = str(prompt_scope).strip().lower()
    if scope not in PROMPT_SCOPES:
        raise ValueError(
            f"未知 prompt_scope: {prompt_scope!r}；必须为 {sorted(PROMPT_SCOPES)}"
        )
    return scope


def _runtime_interface_description(prompt_text: str) -> Tuple[str, str]:
    """根据阶段占位符返回无约束编辑器必须保留的运行接口。"""
    if "{entities}" in prompt_text:
        return (
            "{text}, {entities}",
            '"relations"; each relation item: "source", "target", "type", '
            '"evidence_start", "evidence_end"',
        )
    return (
        "{text}",
        '"entities"; each entity item: "id", "text", "type", "start", '
        '"end", "normalized_id"',
    )


def _extract_blocks(
    text: str,
    tag_start: str = "<START>",
    tag_end: str = "<END>",
    *,
    allow_untagged: bool = False,
) -> List[str]:
    """提取标签块；候选生成默认对未按格式返回的响应 fail-closed。支持 <END> 与 </END> 闭合标签。"""
    end_core = tag_end.strip("<>/")
    end_pattern = rf"(?:{re.escape(tag_end)}|</{re.escape(end_core)}>)"
    pattern = re.compile(rf"{re.escape(tag_start)}\s*(.*?)\s*{end_pattern}", re.DOTALL)
    matches = pattern.findall(text)
    blocks = [m.strip() for m in matches if m.strip()]
    if blocks:
        return blocks

    if allow_untagged:
        cleaned = text.strip()
        if cleaned:
            return [cleaned]
    return []


def format_error_examples_for_prompt(errors: List[ErrorExample], max_errors: int = 4) -> str:
    """将错误样本格式化为适合送入批评与编辑模板的紧凑字符串。"""
    formatted_items = []
    for i, err in enumerate(errors[:max_errors], 1):
        item_str = (
            f"--- Error Example {i} (Sample ID: {err.sample_id}) ---\n"
            f"[Precomputed Error Profile]:\n{json.dumps(err.error_details, ensure_ascii=False, indent=2)}\n"
            f"[Input Text Snippet]:\n{err.input_text[:800]}\n"
            f"[Gold Output (Ground Truth)]:\n{json.dumps(err.gold_output, ensure_ascii=False, indent=2)}\n"
            f"[Model Predicted Output (Incorrect)]:\n{json.dumps(err.predicted_output, ensure_ascii=False, indent=2)}"
        )
        formatted_items.append(item_str)
    return "\n\n".join(formatted_items)


class GradientGenerator:
    """自然语言文本梯度生成器。"""

    def __init__(
        self,
        optimizer_client,
        call_stats: Optional[CallStats] = None,
        *,
        prompt_scope: str = "constrained",
    ):
        self.client = optimizer_client
        self.stats = call_stats
        self.prompt_scope = _validate_prompt_scope(prompt_scope)

    def generate_gradients(
        self,
        parent_candidate: PromptCandidate,
        errors: List[ErrorExample],
        errors_per_group: int = 4,
        gradients_per_error_group: int = 4,
        max_error_groups: int = 1,
    ) -> List[PromptGradient]:
        """将错误样本分群，并为每个群调用批评模型生成自然语言文本梯度。"""
        if not errors:
            return []

        gradients: List[PromptGradient] = []
        # 按 errors_per_group 切分错误样本并限制最多处理 max_error_groups 个群
        group_count = 0
        for group_idx, offset in enumerate(range(0, len(errors), errors_per_group)):
            if group_count >= max_error_groups:
                break
            group_count += 1
            group_errors = errors[offset : offset + errors_per_group]
            error_group_id = f"G{parent_candidate.round_idx}_{group_idx}"
            formatted_errors = format_error_examples_for_prompt(group_errors, max_errors=errors_per_group)

            template = (
                GRADIENT_TEMPLATE
                if self.prompt_scope == "constrained"
                else UNCONSTRAINED_GRADIENT_TEMPLATE
            )
            prompt_content = (
                template.replace("{prompt}", parent_candidate.prompt_text)
                .replace("{error_examples}", formatted_errors)
                .replace("{num_feedbacks}", str(gradients_per_error_group))
            )

            if self.stats:
                self.stats.optimizer_model_calls += 1
                self.stats.optimizer_input_tokens += len(prompt_content) // 4

            raw_response = retry_api_call(
                self.client.call_fn,
                prompt=prompt_content,
                system_prompt="You are an expert cybersecurity prompt critic. Identify specific, actionable directions to fix prompt deficiencies.",
                config=self.client.config,
            )

            if self.stats:
                self.stats.optimizer_output_tokens += len(raw_response) // 4

            extracted_reasons = _extract_blocks(raw_response, "<START>", "<END>")
            for reason_idx, reason_text in enumerate(extracted_reasons[:gradients_per_error_group]):
                grad_id = f"grad_{parent_candidate.candidate_id}_{group_idx}_{reason_idx}"
                gradients.append(
                    PromptGradient(
                        gradient_id=grad_id,
                        parent_prompt_id=parent_candidate.candidate_id,
                        error_group_id=error_group_id,
                        gradient_text=reason_text,
                        round_idx=parent_candidate.round_idx,
                    )
                )

        return gradients


class PromptEditor:
    """按实验臂重写 guidance 或完整语义提示的编辑生成器。"""

    def __init__(
        self,
        optimizer_client,
        call_stats: Optional[CallStats] = None,
        *,
        prompt_scope: str = "constrained",
    ):
        self.client = optimizer_client
        self.stats = call_stats
        self.prompt_scope = _validate_prompt_scope(prompt_scope)

    def edit_prompt(
        self,
        parent_candidate: PromptCandidate,
        gradient: PromptGradient,
        errors: List[ErrorExample],
        errors_per_group: int = 4,
        next_candidate_id: str = "c_edit",
    ) -> Optional[PromptCandidate]:
        """根据单条梯度生成 1 个 guidance 或完整 Prompt 变体。"""
        formatted_errors = format_error_examples_for_prompt(errors, max_errors=errors_per_group)

        if self.prompt_scope == "constrained":
            immutable_contract = extract_immutable_contract(parent_candidate.prompt_text)
            guidance = extract_optimizable_guidance(parent_candidate.prompt_text)
            if immutable_contract is None or guidance is None:
                return None
            prompt_content = (
                EDIT_TEMPLATE.replace("{immutable_contract}", immutable_contract)
                .replace("{guidance}", guidance)
                .replace("{error_examples}", formatted_errors)
                .replace("{gradient}", gradient.gradient_text)
            )
            system_prompt = (
                "You are an expert prompt engineer. Return only revised operational "
                "guidance wrapped in <START> and <END>."
            )
        else:
            required_placeholders, required_output_fields = _runtime_interface_description(
                parent_candidate.prompt_text
            )
            prompt_content = (
                UNCONSTRAINED_EDIT_TEMPLATE.replace(
                    "{prompt}", parent_candidate.prompt_text
                )
                .replace("{error_examples}", formatted_errors)
                .replace("{gradient}", gradient.gradient_text)
                .replace("{required_placeholders}", required_placeholders)
                .replace("{required_output_fields}", required_output_fields)
            )
            system_prompt = (
                "You are an expert prompt engineer. Return only the complete rewritten "
                "prompt wrapped in <START> and <END>."
            )

        if self.stats:
            self.stats.optimizer_model_calls += 1
            self.stats.optimizer_input_tokens += len(prompt_content) // 4

        raw_response = retry_api_call(
            self.client.call_fn,
            prompt=prompt_content,
            system_prompt=system_prompt,
            config=self.client.config,
        )

        if self.stats:
            self.stats.optimizer_output_tokens += len(raw_response) // 4

        blocks = _extract_blocks(raw_response, "<START>", "<END>")
        if not blocks:
            return None

        rewritten_block = blocks[0].strip()
        if not rewritten_block:
            return None

        if self.prompt_scope == "constrained":
            try:
                rewritten_prompt = replace_optimizable_guidance(
                    parent_candidate.prompt_text,
                    rewritten_block,
                )
            except ValueError:
                return None
            generation_type = "gradient_edit"
        else:
            rewritten_prompt = rewritten_block
            generation_type = "gradient_edit_full"

        return PromptCandidate(
            candidate_id=next_candidate_id,
            prompt_text=rewritten_prompt,
            parent_id=parent_candidate.candidate_id,
            generation_type=generation_type,
            gradient_id=gradient.gradient_id,
            round_idx=parent_candidate.round_idx + 1,
        )


class MonteCarloParaphraser:
    """按实验臂对 guidance 或完整提示做蒙特卡洛释义。"""

    def __init__(
        self,
        optimizer_client,
        call_stats: Optional[CallStats] = None,
        *,
        prompt_scope: str = "constrained",
    ):
        self.client = optimizer_client
        self.stats = call_stats
        self.prompt_scope = _validate_prompt_scope(prompt_scope)

    def paraphrase_prompt(
        self,
        base_candidate: PromptCandidate,
        num_paraphrases: int = 2,
        id_prefix: str = "c_para",
    ) -> List[PromptCandidate]:
        """为已编辑 Prompt 生成与实验臂匹配的释义后继。"""
        paraphrases: List[PromptCandidate] = []

        if self.prompt_scope == "constrained":
            guidance = extract_optimizable_guidance(base_candidate.prompt_text)
            if guidance is None:
                return paraphrases
            prompt_content = PARAPHRASE_TEMPLATE.replace("{guidance}", guidance)
            system_prompt = (
                "You are an expert prompt variation generator. Return only paraphrased "
                "operational guidance wrapped in <START> and <END>."
            )
        else:
            required_placeholders, required_output_fields = _runtime_interface_description(
                base_candidate.prompt_text
            )
            prompt_content = (
                UNCONSTRAINED_PARAPHRASE_TEMPLATE.replace(
                    "{prompt}", base_candidate.prompt_text
                )
                .replace("{required_placeholders}", required_placeholders)
                .replace("{required_output_fields}", required_output_fields)
            )
            system_prompt = (
                "You are an expert prompt variation generator. Return only the complete "
                "paraphrased prompt wrapped in <START> and <END>."
            )

        for p_idx in range(num_paraphrases):
            if self.stats:
                self.stats.optimizer_model_calls += 1
                self.stats.optimizer_input_tokens += len(prompt_content) // 4

            raw_response = retry_api_call(
                self.client.call_fn,
                prompt=prompt_content,
                system_prompt=system_prompt,
                config=self.client.config,
            )

            if self.stats:
                self.stats.optimizer_output_tokens += len(raw_response) // 4

            blocks = _extract_blocks(raw_response, "<START>", "<END>")
            if not blocks:
                continue

            paraphrased_block = blocks[0].strip()
            if not paraphrased_block:
                continue

            if self.prompt_scope == "constrained":
                try:
                    paraphrased_text = replace_optimizable_guidance(
                        base_candidate.prompt_text,
                        paraphrased_block,
                    )
                except ValueError:
                    continue
                generation_type = "paraphrase"
            else:
                paraphrased_text = paraphrased_block
                generation_type = "paraphrase_full"

            paraphrases.append(
                PromptCandidate(
                    candidate_id=f"{id_prefix}_{p_idx}",
                    prompt_text=paraphrased_text,
                    parent_id=base_candidate.candidate_id,
                    generation_type=generation_type,
                    gradient_id=base_candidate.gradient_id,
                    round_idx=base_candidate.round_idx,
                )
            )

        return paraphrases


def deduplicate_and_sample_successors(
    parent_candidate: PromptCandidate,
    successors: List[PromptCandidate],
    max_successors: int = 8,
    seed: int = 42,
    call_stats: Optional[CallStats] = None,
) -> Tuple[List[PromptCandidate], Dict[str, int]]:
    """去除完全重复的 Prompt 并固定随机抽样出指定数量的候选后继。"""
    raw_count = len(successors)
    parent_norm = "".join(parent_candidate.prompt_text.split())

    seen_signatures = {parent_norm}
    unique_successors: List[PromptCandidate] = []

    for s in successors:
        sig = "".join(s.prompt_text.split())
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_successors.append(s)

    duplicate_count = raw_count - len(unique_successors)

    # 随机抽样
    rng = random.Random(seed)
    if len(unique_successors) > max_successors:
        sampled = rng.sample(unique_successors, max_successors)
    else:
        sampled = list(unique_successors)

    if call_stats:
        call_stats.num_generated_candidates += raw_count
        call_stats.num_duplicate_candidates += duplicate_count

    stats = {
        "generated_count": raw_count,
        "duplicate_count": duplicate_count,
        "unique_count": len(unique_successors),
        "sampled_count": len(sampled),
    }
    return sampled, stats
