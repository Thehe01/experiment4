"""ProTeGi Candidate Output Expansion Guard（确定性规则）。

冻结语义：每个独立满足 frozen entity boundary contract 的 entity
mention 最多输出一次。Candidate 可以优化“如何判断实体”，但不得
通过逐 token / 逐 substring / 重复 / 嵌套-重叠穷举等方式扩大任务
输出基数，否则 Task Model 在输出侧组合爆炸（见 run7 冒烟枪）。

实现为确定性 regex/pattern 规则，不调用 LLM。注意：
- 不机械禁止一切 overlap：附条件独立成立的 overlap 允许；
- 针对“扩大候选集合/重复排放”，不针对普通逐预测验证
  （"every predicted span" 与 "every possible span" 不是一回事）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

REASON = "output_amplification_contract_violation"

_NORM_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _NORM_WS.sub(" ", text or "").strip()


@dataclass
class ExpansionGuardResult:
    is_valid: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(self.reasons)

    def __bool__(self) -> bool:
        return self.is_valid


# 否定词：出现在放大动词前 25 字符内时，该条动词驱动规则跳过
# （如 "do not enumerate every token" 是在禁枚举，必须放行）。
_NEGATION_RE = re.compile(
    r"\b(do\s+not|never|avoid|don't|does\s+not|did\s+not|"
    r"prohibit\w*|forbid\w*|without|suppress\w*)\b",
    re.IGNORECASE,
)

# 条件状语：同句出现则 unconditional-nested/overlap 规则跳过
# （独立成立的合法 overlap 仍然允许）。
_CONDITIONAL_RE = re.compile(
    r"\bonly\s+(when|if)\b|\bindependently\s+satisf\w*\b"
    r"|\bthat\s+satisf\w*\b|\bwhich\s+satisf\w*\b",
    re.IGNORECASE,
)


def _negated(text: str, match_start: int, window: int = 25) -> bool:
    return bool(_NEGATION_RE.search(text[max(0, match_start - window):match_start]))


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"[.!?\n]+", text) if s.strip()]


def _rule_token_position(text: str) -> Optional[str]:
    """every/each/all + token position(s)：逐 token 穷举。"""
    pattern = re.compile(
        r"\b(every|each|all)\s+token\s+positions?\b", re.IGNORECASE
    )
    match = pattern.search(text)
    return match.group(0) if match else None


def _rule_occurrence_separate(text: str) -> Optional[str]:
    """逐 occurrence 分立输出 / offset-distinct / individual entry。"""
    patterns = (
        re.compile(r"\b(each|every)\s+occurrences?\s+separately\b", re.IGNORECASE),
        re.compile(r"\boffset-distinct\s+occurrences?\b", re.IGNORECASE),
        re.compile(r"\b(each|every)\s+offset-distinct\b", re.IGNORECASE),
        re.compile(r"\bindividual\s+entr\w*\s+for\s+each\b", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


_AMPLIFY_VERB_RE = re.compile(
    r"\b(count\w*|emit\w*|creat\w+|output\w*|list\w*|retain\w*|keep\w*|preserve\w*)\b",
    re.IGNORECASE,
)
_REPEATED_IDENTICAL_RE = re.compile(
    r"\brepeated\s+identical\s+strings?\b", re.IGNORECASE
)
_SEPARATELY_RE = re.compile(
    r"\b(separately|each|individual\w*)\b", re.IGNORECASE
)


def _rule_repeated_identical(text: str) -> Optional[str]:
    """count/emit/... repeated identical strings（要求放大动词或分立副词）。"""
    for rep in _REPEATED_IDENTICAL_RE.finditer(text):
        before = text[max(0, rep.start() - 80):rep.start()]
        verb = None
        for candidate in _AMPLIFY_VERB_RE.finditer(before):
            verb = candidate
        if verb is not None and not _negated(
            text, max(0, rep.start() - 80) + verb.start()
        ):
            return text[verb.start() + max(0, rep.start() - 80):rep.end()]
        after = text[rep.end():rep.end() + 80]
        sep = _SEPARATELY_RE.search(after)
        if sep:
            return rep.group(0) + " ... " + sep.group(0)
    return None


_ANTI_DEDUP_RE = re.compile(
    r"\bdo\s+not\s+(collapse|deduplicate|dedup\w*|merge)\b", re.IGNORECASE
)
_DEDUP_OBJECT_RE = re.compile(
    r"\bduplicat\w*|repeats?|repetitions?|occurrences?\b", re.IGNORECASE
)


def _rule_anti_dedup(text: str) -> Optional[str]:
    """do not collapse/deduplicate/...（禁止去重）与 preserve/keep/retain duplicates。"""
    if re.search(r"\bdo\s+not\s+deduplicate\b", text, re.IGNORECASE):
        return "do not deduplicate"
    keep = re.search(
        r"\b(preserve|keep|retain)\s+duplicates?\b", text, re.IGNORECASE
    )
    if keep:
        return keep.group(0)
    for match in _ANTI_DEDUP_RE.finditer(text):
        after = text[match.end():match.end() + 60]
        if _DEDUP_OBJECT_RE.search(after):
            return match.group(0)
    return None


_UNCONDITIONAL_SPAN_RE = re.compile(
    r"\b(all|every)\s+(nested|overlapping)\s+(spans?|mentions?|matches?|entities?)\b",
    re.IGNORECASE,
)
_NESTED_OVERLAP_SEP_RE = re.compile(
    r"\bnested\s+or\s+overlapping\b[\s\S]{0,60}?\bseparately\b",
    re.IGNORECASE,
)


def _rule_nested_overlap(text: str) -> Optional[str]:
    """无条件输出全部 nested/overlapping（附条件独立成立则放行）。"""
    for sentence in _sentences(text):
        if _CONDITIONAL_RE.search(sentence):
            continue
        match = _UNCONDITIONAL_SPAN_RE.search(sentence)
        if match:
            return match.group(0)
        match = _NESTED_OVERLAP_SEP_RE.search(sentence)
        if match:
            snippet = _normalize(match.group(0))
            return snippet[:80]
    return None


def _rule_possible_candidate_spans(text: str) -> Optional[str]:
    """all/every + possible/candidate + spans；all/every + substrings。"""
    patterns = (
        re.compile(
            r"\b(all|every)\s+(possible|candidate)\s+spans?\b", re.IGNORECASE
        ),
        re.compile(r"\b(all|every)\s+substr\w*\b", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


_ENUMERATE_RE = re.compile(r"\benumerat\w*\b", re.IGNORECASE)
_ENUMERATE_OBJECT_RE = re.compile(
    r"\b(spans?|tokens?|substrings?|matches?|occurrences?|entities?)\b",
    re.IGNORECASE,
)


def _rule_enumerate_combo(text: str) -> Optional[str]:
    """enumerate + span/token/substring/match/occurrence/entity（否定豁免）。"""
    for match in _ENUMERATE_RE.finditer(text):
        if _negated(text, match.start()):
            continue
        after = text[match.end():match.end() + 40]
        obj = _ENUMERATE_OBJECT_RE.search(after)
        if obj:
            return (match.group(0) + " ... " + obj.group(0))[:80]
    return None


_RULES = (
    ("token_position_enumeration", _rule_token_position),
    ("occurrence_separate_enumeration", _rule_occurrence_separate),
    ("repeated_identical_strings", _rule_repeated_identical),
    ("anti_dedup", _rule_anti_dedup),
    ("unconditional_nested_overlap", _rule_nested_overlap),
    ("possible_candidate_spans", _rule_possible_candidate_spans),
    ("enumerate_combination", _rule_enumerate_combo),
)


class OutputExpansionGuard:
    """确定性输出膨胀守卫：只看语义模式，不看长度，不调模型。"""

    @classmethod
    def validate_guidance(cls, guidance_text: str) -> ExpansionGuardResult:
        text = _normalize(guidance_text)
        if not text:
            return ExpansionGuardResult(True, [])
        for rule_id, rule_fn in _RULES:
            try:
                hit = rule_fn(text)
            except Exception:
                continue
            if hit:
                snippet = _normalize(str(hit))[:80]
                return ExpansionGuardResult(
                    False,
                    [f"{REASON}:{rule_id}:{snippet}"],
                )
        return ExpansionGuardResult(True, [])

    @classmethod
    def validate(
        cls, prompt_text: str, *, stage: str = "entity"
    ) -> ExpansionGuardResult:
        """校验完整候选 prompt：有可优化块则只查该块，否则查全文。"""
        _ = stage
        target: Optional[str] = None
        try:
            from protegi.prompts_p0 import extract_optimizable_guidance

            target = extract_optimizable_guidance(prompt_text or "")
        except Exception:
            target = None
        return cls.validate_guidance(
            target if target is not None else (prompt_text or "")
        )
