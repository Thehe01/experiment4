"""第四章 LLM 抽取方法。

文本抽取层只处理 Configuration、Vulnerability、Weakness 和
AttackTechnique，以及 affects、instantiates 和 exploited_by。上层的
AttackTactic、KillChainPhase 及其关系由 complete_bron_layer.py 补全。
"""
import sys, os, json, re, time, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from llm_extractor import (  # noqa: E402
    LLMExtractor,
    ModelOutputBudgetExhaustedError,
)
from schema import (  # noqa: E402
    BOUNDARY_CONTRACT_VERSION,
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
    SCHEMA_VERSION,
)
from provider_config import (  # noqa: E402
    apply_formal_apo_runtime_defaults,
    public_provider_config,
    resolve_provider,
)

apply_formal_apo_runtime_defaults()
_PROVIDER = resolve_provider()
API_BASE_URL = str(_PROVIDER["base_url"])
API_MODEL = str(_PROVIDER["model"])
API_TEMPERATURE = float(os.environ.get("V3_LLM_TEMPERATURE", "0.0"))
API_THINKING = os.environ.get("V3_LLM_THINKING", "disabled").strip().lower()
API_REASONING_EFFORT = os.environ.get(
    "V3_LLM_REASONING_EFFORT", ""
).strip().lower()
API_MAX_TOKENS = int(os.environ.get("V3_LLM_MAX_TOKENS", "4096"))
API_MAX_ESCALATED_TOKENS = max(
    API_MAX_TOKENS,
    int(
        os.environ.get(
            "V3_LLM_MAX_ESCALATED_TOKENS",
            str(API_MAX_TOKENS),
        )
    ),
)
API_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("V3_API_TIMEOUT_SECONDS", "180")
)
API_TRANSPORT_MAX_RETRIES = int(
    os.environ.get("V3_API_TRANSPORT_MAX_RETRIES", "0")
)
STAGE1_PROMPT_VERSION = "chapter3-no-capec-v1-mcpu-stage1-v2"
STAGE2_PROMPT_VERSION = "chapter3-no-capec-v1-boundary-sync-stage2-v2"
PROMPT_VERSION = f"{STAGE1_PROMPT_VERSION}|{STAGE2_PROMPT_VERSION}"
POSTPROCESS_VERSION = "chapter3-no-capec-v1-mcpu-postprocess-v2"
WINDOW_CHARS = int(os.environ.get("V3_LLM_WINDOW_CHARS", "3000"))
WINDOW_OVERLAP = int(os.environ.get("V3_LLM_WINDOW_OVERLAP", "400"))
LLM_MAX_WORKERS = max(1, int(os.environ.get("V3_LLM_MAX_WORKERS", "8")))
MAX_EVIDENCE_OFFSET_REPAIR_CHARS = 1200
APO_ARTIFACT = EXP_DIR / "results" / "apo_optimization_v6" / "final_prompt.json"
CURRENT_SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"

_APO_FORBIDDEN_TERMS = (
    "attackpattern",
    "capec",
    "attacktactic",
    "killchainphase",
    "leverages",
    "realizes",
    "implies",
    "belongs_to_phase",
)
_APO_FORBIDDEN_ID_PATTERNS = (
    re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    re.compile(r"\bCWE-\d+\b", re.IGNORECASE),
    re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE),
    re.compile(r"\bTA\d{4}\b", re.IGNORECASE),
    re.compile(r"\bKC-[A-Z0-9_-]+\b", re.IGNORECASE),
)
_APO_DIRECTION_TYPE = (
    r"Configuration|Vulnerability|Weakness|Attack[\s_-]*Technique"
)
_APO_FROM_TO_PATTERN = re.compile(
    rf"\bfrom\s+({_APO_DIRECTION_TYPE})\s+to\s+({_APO_DIRECTION_TYPE})\b",
    re.IGNORECASE,
)
_APO_ARROW_PATTERN = re.compile(
    rf"\b({_APO_DIRECTION_TYPE})\s*(?:->|→)\s*({_APO_DIRECTION_TYPE})\b",
    re.IGNORECASE,
)
_APO_LINK_TO_PATTERN = re.compile(
    rf"\blink\s+({_APO_DIRECTION_TYPE})\s+to\s+({_APO_DIRECTION_TYPE})\b",
    re.IGNORECASE,
)
_APO_STAGE2_ENTITY_TYPE_PATTERN = re.compile(
    rf"\b(?:{_APO_DIRECTION_TYPE})\b",
    re.IGNORECASE,
)
_APO_STAGE2_DIRECTIONAL_PATTERNS = (
    re.compile(r"\bdirection\b", re.IGNORECASE),
    re.compile(r"\bgrammatical\s+(?:subject|object)\b", re.IGNORECASE),
    re.compile(r"\bsubject\s*(?:/|and)\s*object\b", re.IGNORECASE),
    re.compile(
        r"\bfrom\s+(?:the\s+)?(?:source|target)\s+to\s+"
        r"(?:the\s+)?(?:source|target)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsource\s+(?:is|as)\b", re.IGNORECASE),
    re.compile(r"\btarget\s+(?:is|as)\b", re.IGNORECASE),
)
_APO_STAGE2_FACT_BLOCK_CONFLICT_PATTERNS = (
    # Frozen affects policy also permits parenthetical titles, labelled rows,
    # and a CVE-scoped CPE heading/list; candidate guidance cannot narrow all
    # direct facts to prose with a particular grammatical form.
    re.compile(r"\b(?:same|one)\s+(?:sentence|clause)\b", re.IGNORECASE),
    re.compile(r"\b(?:direct|syntactic)\s+object\b", re.IGNORECASE),
    re.compile(r"\bobject\s+of\s+(?:the\s+)?relation\s+verb\b", re.IGNORECASE),
    re.compile(
        r"\b(?:require|must|only)\b.{0,100}\b(?:explicit\s+)?verb\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:do\s+not|never)\s+rely\b.{0,100}"
        r"\b(?:list|listing|table|punctuation)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:no|without|absent)\b.{0,100}"
        r"\b(?:explicit\s+)?relation\s+wording\b.{0,120}"
        r"\b(?:abstain|output\s+no\s+relation)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_APO_STAGE1_FACT_BLOCK_CONFLICT_PATTERNS = (
    re.compile(
        r"\bconfiguration\b.{0,240}\b(?:same|one)\s+(?:sentence|clause)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bconfiguration\b.{0,240}\b(?:direct|syntactic)\s+object\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_APO_STAGE1_SCHEMA_REDEFINITION_PATTERNS = (
    re.compile(
        r"\bconfiguration\b.{0,220}\b(?:setting|parameter|threshold|"
        r"flag|option|parameter\s+value|port\s+number)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bconfiguration\b.{0,220}\b(?:full|complete)\s+(?:product\s+)?"
        r"name\b.{0,80}\b(?:include|including|with)\b.{0,50}\bversion\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_APO_BOUNDARY_CONFLICT_PATTERNS = (
    re.compile(
        r"\bmaximal\s+(?:verbatim\s+)?(?:tail|product|configuration)\s+phrase\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:always|must)\b.{0,80}\b(?:drop|remove|strip)\b.{0,80}"
        r"\bvendor\b|\b(?:drop|remove|strip)\b.{0,80}\bvendor\b.{0,80}"
        r"\b(?:always|must)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:copy|attach|include|preserve)\b.{0,100}\bvendor\b.{0,120}"
        r"\b(?:another|previous|nearby|distant)\b.{0,60}"
        r"\b(?:column|row|sentence|clause|conjunct|mention)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bweakness\b.{0,160}\b(?:infer|guess|map|normalize)\b.{0,100}"
        r"\bwithout\b.{0,50}\b(?:explicit|literal|local)\s+cwe\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _normalize_guidance_type(value: str) -> str:
    compact = re.sub(r"[\s_-]+", "", value).casefold()
    return {
        "configuration": "Configuration",
        "vulnerability": "Vulnerability",
        "weakness": "Weakness",
        "attacktechnique": "AttackTechnique",
    }[compact]


def _invalid_guidance_directions(guidance: str) -> list[str]:
    invalid = []
    statements = re.split(r"[.!?;\n]+", guidance)
    for statement in statements:
        lowered = statement.casefold()
        for relation_type, expected in EXTRACTION_RELATION_ARGUMENT_TYPES.items():
            if relation_type.casefold() not in lowered:
                continue
            for pattern in (
                _APO_FROM_TO_PATTERN,
                _APO_ARROW_PATTERN,
                _APO_LINK_TO_PATTERN,
            ):
                for match in pattern.finditer(statement):
                    observed = tuple(
                        _normalize_guidance_type(value)
                        for value in match.groups()
                    )
                    if observed != expected:
                        invalid.append(
                            f"{relation_type}:{observed[0]}->{observed[1]}"
                        )
    return sorted(set(invalid))


def _api_key():
    configured = _PROVIDER.get("key")
    if configured:
        return str(configured)
    is_deepseek = "api.deepseek.com" in API_BASE_URL.casefold()
    if is_deepseek:
        env = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("V3_API_KEY")
            or os.environ.get("V2_API_KEY")
        )
        key_files = (
            EXP_DIR / ".deepseek_api_key",
            ROOT / "experiments" / "v2" / ".api_key",
        )
    else:
        env = (
            os.environ.get("V3_API_KEY")
            or os.environ.get("V2_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        key_files = (
            EXP_DIR / ".api_key",
            ROOT / "experiments" / "v2" / ".api_key",
        )
    if env:
        return env.strip()
    for kf in key_files:
        if kf.exists():
            k = kf.read_text(encoding="utf-8").strip()
            if k:
                return k
    return None


def make_extractor(
    temperature: float | None = None,
    *,
    model: str | None = None,
    thinking: str | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    reasoning_effort: str | None = None,
) -> LLMExtractor:
    """Create an extraction or optimizer client with an explicit runtime.

    APO may use a stronger critic/editor model while every candidate is still
    evaluated by the frozen task model.  Keeping the override here avoids
    mutating module-level task settings or leaking optimizer choices into task
    prediction cache keys.
    """
    key = _api_key()
    if not key:
        raise RuntimeError(
            "无有效 API key：请设置 V5_API_KEY/V3_API_KEY，或将 provider "
            "对应的 key 写入 experiments/v5 下的密钥文件；不要复用其他项目密钥"
        )
    if temperature is None:
        temperature = API_TEMPERATURE
    if model is None:
        model = API_MODEL
    if thinking is None:
        thinking = API_THINKING
    if max_tokens is None:
        max_tokens = API_MAX_TOKENS
    if top_p is None:
        top_p = 0.95
    if reasoning_effort is None:
        reasoning_effort = API_REASONING_EFFORT
        if (
            "muse-spark" in model.casefold()
            and "contributor" in model.casefold()
            and not reasoning_effort
        ):
            reasoning_effort = "xhigh"
    endpoint = (
        "/v1/responses"
        if "muse-spark" in model.casefold()
        and "contributor" in model.casefold()
        else "/v1/chat/completions"
    )
    return LLMExtractor(backend="openai_compatible", config={
        "model": model, "temperature": temperature,
        "max_tokens": max_tokens,
        "max_escalated_tokens": max(max_tokens, API_MAX_ESCALATED_TOKENS),
        "top_p": float(top_p), "api_key": key, "base_url": API_BASE_URL,
        "endpoint": endpoint,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "timeout": API_REQUEST_TIMEOUT_SECONDS,
        "max_retries": API_TRANSPORT_MAX_RETRIES})


def runtime_config() -> dict:
    """返回可公开写入实验记录的模型配置，不包含 API key。"""
    return {
        "model": API_MODEL,
        "base_url": API_BASE_URL,
        "provider": public_provider_config()["name"],
        "endpoint": (
            "/v1/responses"
            if "muse-spark" in API_MODEL.casefold()
            and "contributor" in API_MODEL.casefold()
            else "/v1/chat/completions"
        ),
        "temperature": API_TEMPERATURE,
        "thinking": API_THINKING,
        "reasoning_effort": API_REASONING_EFFORT,
        "max_tokens": API_MAX_TOKENS,
        "max_escalated_tokens": API_MAX_ESCALATED_TOKENS,
        "top_p": 0.95,
        "max_retries": 3,
        "request_timeout_seconds": API_REQUEST_TIMEOUT_SECONDS,
        "transport_max_retries": API_TRANSPORT_MAX_RETRIES,
        "prompt_version": PROMPT_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "window_chars": WINDOW_CHARS,
        "window_overlap": WINDOW_OVERLAP,
        "max_workers": LLM_MAX_WORKERS,
    }


def stage1_runtime_config() -> dict:
    """Stage 1 缓存独立于 Stage 2 提示版本，保证关系 APO 严格配对。"""
    config = runtime_config()
    config["prompt_version"] = STAGE1_PROMPT_VERSION
    return config


def validate_apo_guidance(stage1_guidance: str, stage2_guidance: str) -> None:
    """验证 APO 只能改写下层任务的通用判定指导，不能改动本体边界。

    显式编号也被禁止写入优化增量，避免候选提示记忆开发集中的具体
    CVE/CWE/ATT&CK 标识并形成数据泄漏。
    """
    for stage_name, guidance in (
        ("stage1_guidance", stage1_guidance),
        ("stage2_guidance", stage2_guidance),
    ):
        if not isinstance(guidance, str):
            raise TypeError(f"{stage_name} 必须为字符串")
        lowered = guidance.casefold()
        compact = re.sub(r"[\s_-]+", "", lowered)
        illegal_terms = [
            term
            for term in _APO_FORBIDDEN_TERMS
            if term in lowered
            or re.sub(r"[\s_-]+", "", term) in compact
        ]
        illegal_ids = [
            pattern.pattern
            for pattern in _APO_FORBIDDEN_ID_PATTERNS
            if pattern.search(guidance)
        ]
        invalid_directions = _invalid_guidance_directions(guidance)
        # Stage 2 may reference the four frozen extraction entity types when
        # stating endpoint decision rules.  It cannot create entities, and
        # wrong argument directions are still rejected below.  Treating every
        # type-name reference as a schema redefinition made valid relation APO
        # candidates impossible to express.
        stage2_entity_types = []
        stage2_directional_language = (
            [
                pattern.pattern
                for pattern in _APO_STAGE2_DIRECTIONAL_PATTERNS
                if pattern.search(guidance)
            ]
            if stage_name == "stage2_guidance"
            else []
        )
        stage2_fact_block_conflicts = (
            [
                pattern.pattern
                for pattern in _APO_STAGE2_FACT_BLOCK_CONFLICT_PATTERNS
                if pattern.search(guidance)
            ]
            if stage_name == "stage2_guidance"
            else []
        )
        stage1_fact_block_conflicts = (
            [
                pattern.pattern
                for pattern in _APO_STAGE1_FACT_BLOCK_CONFLICT_PATTERNS
                if pattern.search(guidance)
            ]
            if stage_name == "stage1_guidance"
            else []
        )
        stage1_schema_redefinitions = (
            [
                pattern.pattern
                for pattern in _APO_STAGE1_SCHEMA_REDEFINITION_PATTERNS
                if pattern.search(guidance)
            ]
            if stage_name == "stage1_guidance"
            else []
        )
        boundary_conflicts = [
            pattern.pattern
            for pattern in _APO_BOUNDARY_CONFLICT_PATTERNS
            if pattern.search(guidance)
        ]
        if (
            illegal_terms
            or illegal_ids
            or invalid_directions
            or stage2_entity_types
            or stage2_directional_language
            or stage2_fact_block_conflicts
            or stage1_fact_block_conflicts
            or stage1_schema_redefinitions
            or boundary_conflicts
        ):
            raise ValueError(
                f"{stage_name} 含越界模式或开发集特定编号："
                f"terms={illegal_terms}, id_patterns={illegal_ids}, "
                f"invalid_directions={invalid_directions}, "
                f"stage2_entity_types={stage2_entity_types}, "
                f"stage2_directional_language={stage2_directional_language}, "
                f"stage2_fact_block_conflicts={stage2_fact_block_conflicts}, "
                f"stage1_fact_block_conflicts={stage1_fact_block_conflicts}, "
                f"stage1_schema_redefinitions={stage1_schema_redefinitions}, "
                f"boundary_conflicts={boundary_conflicts}"
            )


def validate_apo_textual_gradient(gradient: str, phase: str) -> None:
    """Validate a private optimizer diagnosis, not an injected guidance block.

    A textual gradient may name frozen labels and discuss directional errors;
    unlike final guidance it is never appended to the task prompt.  It must
    still avoid development-set identifiers, upper-layer schema terms, wrong
    argument directions, and Stage-1 entity-definition changes.
    """
    if not isinstance(gradient, str) or not gradient.strip():
        raise TypeError("textual_gradient 必须为非空字符串")
    if phase not in {"entity", "relation", "joint"}:
        raise ValueError(f"未知 APO 阶段：{phase}")
    lowered = gradient.casefold()
    compact = re.sub(r"[\s_-]+", "", lowered)
    illegal_terms = [
        term
        for term in _APO_FORBIDDEN_TERMS
        if term in lowered or re.sub(r"[\s_-]+", "", term) in compact
    ]
    illegal_ids = [
        pattern.pattern
        for pattern in _APO_FORBIDDEN_ID_PATTERNS
        if pattern.search(gradient)
    ]
    invalid_directions = _invalid_guidance_directions(gradient)
    stage1_schema_redefinitions = []
    if phase in {"entity", "joint"}:
        stage1_schema_redefinitions = [
            pattern.pattern
            for pattern in _APO_STAGE1_SCHEMA_REDEFINITION_PATTERNS
            if pattern.search(gradient)
        ]
    if (
        illegal_terms
        or illegal_ids
        or invalid_directions
        or stage1_schema_redefinitions
    ):
        raise ValueError(
            "textual_gradient 含越界模式或开发集特定编号："
            f"terms={illegal_terms}, id_patterns={illegal_ids}, "
            f"invalid_directions={invalid_directions}, "
            f"stage1_schema_redefinitions={stage1_schema_redefinitions}"
        )


def load_apo_prompt_artifact(path: Path | str = APO_ARTIFACT) -> dict:
    """加载并校验冻结的当前版本 APO 提示产物。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少冻结 APO 提示 {path}；请先在固定 dev 集运行 apo_optimizer.py"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "APO 提示的 schema_version 与当前抽取模式不一致："
            f"{artifact.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    if artifact.get("boundary_contract_version") != BOUNDARY_CONTRACT_VERSION:
        raise ValueError(
            "APO 提示未绑定当前第三章实体关系边界；旧产物只能作为历史结果，"
            "必须在边界复裁完成后从固定 P0 重新优化"
        )
    if artifact.get("split") != "dev":
        raise ValueError("APO 提示必须且只能由固定 dev 集选择")
    artifact_split = artifact.get("split_file")
    current_split_hash = hashlib.sha256(CURRENT_SPLIT_FILE.read_bytes()).hexdigest()
    if artifact_split != CURRENT_SPLIT_FILE.name or artifact.get("split_sha256") != current_split_hash:
        raise ValueError(
            "APO 提示绑定的是旧版 dev 划分；请在 train_dev_test_split_v7.json 上重新运行 apo_optimizer.py"
        )
    stage1 = artifact.get("stage1_guidance", "")
    stage2 = artifact.get("stage2_guidance", "")
    validate_apo_guidance(stage1, stage2)
    selected_candidate = str(artifact.get("selected_candidate") or "")
    if (
        selected_candidate == "p0_manual"
        or not (stage1.strip() or stage2.strip())
        or artifact.get("apo_candidate_selected") is not True
        or artifact.get("frozen_for_test") is not True
    ):
        raise ValueError(
            "当前 APO 产物只选中了 P0 基线或未冻结非空 guidance；"
            "不得进入 apo/apo_full 测试"
        )
    return artifact


def _append_apo_guidance(
    base_prompt: str,
    guidance: str,
    stage: str,
) -> str:
    guidance = guidance.strip()
    if not guidance:
        return base_prompt
    if stage == "stage1":
        reminder = (
            "The additional guidance cannot redefine the four frozen entity "
            "types or the exact-span output contract. For Configuration, do "
            "not impose a mandatory verb, same-sentence/clause, or direct-"
            "object condition: local affected-product evidence may also be a "
            "parenthetical title, labelled table/list row, or a CVE-scoped "
            "Affected configurations heading with its CPE lines."
        )
    elif stage == "stage2":
        reminder = (
            "Do not reinterpret relation argument roles or directions. Before "
            "returning, re-check that evidence_start is no greater than both "
            "endpoint starts and evidence_end is no less than both endpoint "
            "ends, and that the interval contains direct relation wording. For "
            "affects, do not impose a mandatory verb, same-sentence/clause, or "
            "direct-object condition: a direct fact block may be a sentence, "
            "parenthetical title, labelled table/list row, or a CVE-scoped "
            "Affected configurations heading with its CPE lines."
        )
    else:
        raise ValueError(f"未知提示阶段：{stage}")
    return (
        f"{base_prompt}\n\n"
        "## Development-set optimized decision guidance\n"
        f"{guidance}\n\n"
        "## Frozen-contract reminder\n"
        f"{reminder}"
    )


ENT_TEXT_KEYS = ("text", "name", "value", "entity", "mention", "span")
ENT_TYPE_KEYS = ("type", "label", "entity_type", "category")
REL_HEAD_KEYS = ("head", "source", "subject", "from", "h")
REL_TAIL_KEYS = ("tail", "target", "object", "to", "t")


def _pick(d, keys):
    for k in keys:
        if d.get(k) is not None:
            return str(d.get(k))
    return None


def _coerce_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _all_text_spans(text: str, surface: str) -> list[tuple[int, int, str]]:
    """返回表面文本在原文中的全部出现位置，优先使用大小写严格匹配。"""
    if not surface:
        return []
    spans = []
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            break
        spans.append((index, index + len(surface), text[index:index + len(surface)]))
        start = index + max(1, len(surface))
    if spans:
        return spans

    lower_text = text.casefold()
    lower_surface = surface.casefold()
    start = 0
    while True:
        index = lower_text.find(lower_surface, start)
        if index < 0:
            break
        spans.append((index, index + len(surface), text[index:index + len(surface)]))
        start = index + max(1, len(surface))
    return spans


def _entity_spans(text: str, entity: dict, surface: str):
    """验证模型偏移；缺失偏移时扩展同一表面文本的全部显式提及。

    对 Configuration 而言，模型若明确给出错误偏移，不能把这一个错误
    预测扩展成文中所有同名产品/厂商提及。这样会将一条位置错误的预测放大
    为多条 affects 候选。真正缺失偏移时仍保留全部显式提及，以支持同名产品
    关联不同 CVE 的合法情形。
    """
    start = _coerce_int(entity.get("start"))
    end = _coerce_int(entity.get("end"))
    if (
        start is not None
        and end is not None
        and 0 <= start < end <= len(text)
        and text[start:end].casefold() == surface.casefold()
    ):
        return [(start, end, text[start:end])]
    offsets_supplied = start is not None or end is not None
    fallback_spans = _all_text_spans(text, surface)
    if entity.get("type") == "Configuration" and offsets_supplied:
        # A uniquely occurring surface form can be repaired mechanically
        # without inventing which mention the model intended.  Keep rejecting
        # ambiguous repeated names (for example a bare vendor emitted with
        # 0,0 offsets), because expanding those would recreate the previous
        # cross-fact Configuration false positives.
        return fallback_spans if len(fallback_spans) == 1 else []
    return fallback_spans


_CONFIGURATION_DESCRIPTOR_RE = re.compile(
    r"\b(?:"
    r"privilege\s+escalation|"
    r"heap\s+buffer\s+overflow(?:\s+flaw)?|"
    r"buffer\s+overflow(?:\s+flaw)?|"
    r"remote\s+code\s+execution|"
    r"arbitrary\s+(?:code\s+execution|file\s+write)|"
    r"sql\s+injection|"
    r"cross[-\s]site\s+scripting|"
    r"authentication\s+bypass|"
    r"command\s+line\s+argument\s+parsing"
    r")\b",
    re.IGNORECASE,
)
_CONFIGURATION_TRAILING_CONTEXT_RE = re.compile(
    r"\s+(?:"
    r"software|"
    r"web\s+management\s+user\s+interface|"
    r"smart\s+install|"
    r"server\s+and\s+data\s+center"
    r")\s*$",
    re.IGNORECASE,
)
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:References|Resources|Further\s+Reading|Related\s+Links)\s*:?\s*$"
)
_INLINE_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:References|Resources)\s*:?\s*(?=\[\d+\])"
)
_FLATTENED_REFERENCE_HEADING_RE = re.compile(
    r"\b(?:References|Resources)\s*:?\s*(?=(?:\[\d+\]|\d+[.)])\s+)",
    re.IGNORECASE,
)
_REFERENCE_ITEM_RE = re.compile(r"\[\d+\]")
_REFERENCE_SECTION_STOP_RE = re.compile(
    r"(?im)^\s*(?:Contact\s+Information|Appendix(?:\s+[A-Z])?|Mitigations|"
    r"Indicators\s+of\s+Compromise|Acknowledg(?:e)?ments?|Conclusion)\s*:?\s*$"
)
_INLINE_REFERENCE_SECTION_STOP_RE = re.compile(
    r"\b(?:Contact\s+Information|Appendix\s+[A-Z]|Mitigations|"
    r"Indicators\s+of\s+Compromise)\b",
    re.IGNORECASE,
)
_TRAILING_CONFIGURATION_ACRONYM_RE = re.compile(
    r"\s+\([A-Z][A-Z0-9-]{1,9}\)\s*$"
)
_GENERIC_SINGLE_PRODUCT_TERMS = {
    "appliance", "firewall", "gateway", "security", "server",
}


def _reference_section_ranges(text: str) -> list[tuple[int, int]]:
    """Return explicit reference/resource sections without using model output.

    Only standalone headings start an excluded zone.  A known same-level
    section heading ends it; otherwise it conservatively extends to EOF.  This
    deliberately does not treat an in-sentence word such as ``references`` as
    a section marker.
    """
    ranges = []
    for heading in _REFERENCE_HEADING_RE.finditer(text):
        stop = _REFERENCE_SECTION_STOP_RE.search(text, heading.end())
        ranges.append((heading.start(), stop.start() if stop else len(text)))
    for heading in _INLINE_REFERENCE_HEADING_RE.finditer(text):
        stop = _INLINE_REFERENCE_SECTION_STOP_RE.search(text, heading.end())
        ranges.append((heading.start(), stop.start() if stop else len(text)))
    for heading in _FLATTENED_REFERENCE_HEADING_RE.finditer(text):
        stop = _INLINE_REFERENCE_SECTION_STOP_RE.search(text, heading.end())
        ranges.append((heading.start(), stop.start() if stop else len(text)))
    return ranges


def _position_in_ranges(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _configuration_is_numbered_reference(text: str, start: int) -> bool:
    """Return True for any mention inside an explicit reference/resource zone."""
    return _position_in_ranges(start, _reference_section_ranges(text))


def _cpe_product_match(surface: str, normalized_id) -> tuple[int, int, str] | None:
    """Locate the CPE product component as explicit tokens in ``surface``.

    The model-provided CPE is used only as a boundary consistency check.  No
    absent product token is inferred or inserted into the source text.
    """
    value = str(normalized_id or "").strip()
    parts = value.split(":")
    if len(parts) < 5 or parts[0].casefold() != "cpe" or parts[1] != "2.3":
        return None
    product = parts[4].strip().casefold()
    if not product or product in {"*", "-"}:
        return None
    product_key = re.sub(r"[^a-z0-9]", "", product)
    if not product_key:
        return None
    tokens = list(re.finditer(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*", surface))
    for left in range(len(tokens)):
        joined = ""
        for right in range(left, len(tokens)):
            joined += re.sub(
                r"[^a-z0-9]", "", tokens[right].group(0).casefold()
            )
            if joined == product_key:
                return tokens[left].start(), tokens[right].end(), product
            if len(joined) >= len(product_key):
                break
    return None


def _configuration_is_bare_cpe_vendor(surface: str, normalized_id) -> bool:
    """Reject a vendor name that does not explicitly name a concrete product."""
    value = str(normalized_id or "").strip()
    parts = value.split(":")
    if len(parts) < 5 or parts[0].casefold() != "cpe" or parts[1] != "2.3":
        return False
    surface_key = re.sub(r"[^a-z0-9]", "", surface.casefold())
    vendor_key = re.sub(r"[^a-z0-9]", "", parts[3].casefold())
    product_key = re.sub(r"[^a-z0-9]", "", parts[4].casefold())
    return bool(
        surface_key
        and surface_key == vendor_key
        and product_key != vendor_key
    )


def _incomplete_repeated_vendor_conjunction_end(
    surface: str,
    product_match: tuple[int, int, str] | None,
) -> int | None:
    """Return the end of a complete first product before a truncated peer."""
    if product_match is None:
        return None
    conjunctions = list(re.finditer(r"\s+and\s+", surface, re.IGNORECASE))
    if len(conjunctions) != 1:
        return None
    conjunction = conjunctions[0]
    left_tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", surface[:conjunction.start()]
    )
    right_tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", surface[conjunction.end():]
    )
    if not (
        len(left_tokens) >= 3
        and 1 <= len(right_tokens) < 3
        and left_tokens[0].casefold() == right_tokens[0].casefold()
        and product_match[1] <= conjunction.start()
    ):
        return None
    return product_match[1]


def _split_configuration_conjunction(
    entity: dict,
) -> tuple[list[dict], bool]:
    """Split only clearly repeated product names joined by ``and``.

    Safe forms are a shared all-uppercase product token (``Cisco IOS and IOS
    XE``) or an explicitly repeated vendor token on both sides.  Edition names
    such as ``Server and Data Center`` are handled as a suffix and are not
    treated as two products.
    """
    surface = entity["text"]
    conjunctions = list(re.finditer(r"\s+and\s+", surface, re.IGNORECASE))
    if len(conjunctions) != 1:
        return [entity], False
    conjunction = conjunctions[0]
    left = surface[:conjunction.start()].strip()
    right = surface[conjunction.end():].strip()
    if not left or not right:
        return [entity], False
    left_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", left)
    right_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", right)
    if not left_tokens or not right_tokens:
        return [entity], False
    shared_acronym = (
        left_tokens[-1] == right_tokens[0]
        and left_tokens[-1].isupper()
        and len(left_tokens[-1]) >= 2
    )
    repeated_vendor = (
        len(left_tokens) >= 3
        and len(right_tokens) >= 3
        and left_tokens[0].casefold() == right_tokens[0].casefold()
    )
    if not (shared_acronym or repeated_vendor):
        return [entity], False

    left_start = entity["start"] + len(surface[:conjunction.start()]) - len(
        surface[:conjunction.start()].lstrip()
    )
    left_end = entity["start"] + conjunction.start()
    while left_end > left_start and surface[left_end - entity["start"] - 1].isspace():
        left_end -= 1
    right_local_start = conjunction.end()
    while right_local_start < len(surface) and surface[right_local_start].isspace():
        right_local_start += 1
    right_start = entity["start"] + right_local_start
    right_end = entity["end"]
    group = f"{entity['start']}:{entity['end']}"
    variants = []
    for start, end in ((left_start, left_end), (right_start, right_end)):
        variant = dict(entity)
        variant.update({
            "start": start,
            "end": end,
            "text": surface[start - entity["start"]:end - entity["start"]],
            "_configuration_split_group": group,
        })
        variants.append(variant)
    return variants, True


def _configuration_entity_variants(text: str, entity: dict):
    """Return conservative, source-grounded Configuration span variants."""
    actions = []
    start = int(entity["start"])
    end = int(entity["end"])
    if not (0 <= start < end <= len(text)):
        return [entity], actions
    if _configuration_is_numbered_reference(text, start):
        return [], ["entity_configuration_reference_context"]

    candidate = dict(entity)
    candidate["text"] = text[start:end]
    surface = candidate["text"]
    if surface.strip().casefold().startswith("cpe:2.3:"):
        return [candidate], actions

    trailing_acronym = _TRAILING_CONFIGURATION_ACRONYM_RE.search(surface)
    if trailing_acronym and trailing_acronym.start() > 0:
        repaired_end = start + trailing_acronym.start()
        while repaired_end > start and text[repaired_end - 1].isspace():
            repaired_end -= 1
        if repaired_end > start:
            candidate.update({
                "end": repaired_end,
                "text": text[start:repaired_end],
            })
            surface = candidate["text"]
            actions.append("entity_configuration_span_repaired")
    if _configuration_is_bare_cpe_vendor(
        surface,
        candidate.get("normalized_id") or candidate.get("normalized"),
    ):
        return [], ["entity_configuration_bare_vendor"]
    suffix = _CONFIGURATION_TRAILING_CONTEXT_RE.search(surface)
    if suffix and suffix.start() > 0:
        end = start + suffix.start()
        while end > start and text[end - 1].isspace():
            end -= 1
        candidate.update({"end": end, "text": text[start:end]})
        surface = candidate["text"]
        actions.append("entity_configuration_span_repaired")

    descriptor = _CONFIGURATION_DESCRIPTOR_RE.search(surface)
    if (
        descriptor
        and not surface[:descriptor.start()].strip()
        and not surface[descriptor.end():].strip()
    ):
        return [], ["entity_configuration_descriptor_without_product"]
    product_match = _cpe_product_match(
        surface,
        candidate.get("normalized_id") or candidate.get("normalized"),
    )
    colon = surface.find(":")
    if descriptor:
        if product_match is None:
            return [], ["entity_configuration_descriptor_without_product"]
        product_start, product_end, product = product_match
        if colon >= 0 and product not in _GENERIC_SINGLE_PRODUCT_TERMS:
            start = candidate["start"] + product_start
        end = candidate["start"] + product_end
        candidate.update({"start": start, "end": end, "text": text[start:end]})
        actions.append("entity_configuration_span_repaired")
    elif colon >= 0 and product_match is not None:
        product_start, product_end, product = product_match
        if product not in _GENERIC_SINGLE_PRODUCT_TERMS:
            start = candidate["start"] + product_start
            end = candidate["start"] + product_end
            candidate.update({"start": start, "end": end, "text": text[start:end]})
            actions.append("entity_configuration_span_repaired")

    variants, was_split = _split_configuration_conjunction(candidate)
    if was_split:
        actions.append("entity_configuration_conjunction_split")
        return variants, list(dict.fromkeys(actions))
    incomplete_end = _incomplete_repeated_vendor_conjunction_end(
        candidate["text"],
        _cpe_product_match(
            candidate["text"],
            candidate.get("normalized_id") or candidate.get("normalized"),
        ),
    )
    if incomplete_end is not None:
        end = candidate["start"] + incomplete_end
        candidate.update({"end": end, "text": text[candidate["start"]:end]})
        actions.append("entity_configuration_span_repaired")
    return [candidate], list(dict.fromkeys(actions))


def parse_entity_mentions(text: str, raw_entities) -> tuple[list[dict], dict[str, list[str]]]:
    """用统一规则解析实体，并为相同表面文本保留每次显式出现。

    返回的 alias 映射允许关系解析器把模型原始实体 ID 映射到一个或多个
    mention ID；具体关系端点再由证据所在位置消歧。
    """
    records = {}
    alias_keys: dict[str, set[tuple[int, int, str]]] = {}
    for raw_entity in raw_entities or []:
        if not isinstance(raw_entity, dict):
            continue
        surface = _pick(raw_entity, ENT_TEXT_KEYS)
        entity_type = _pick(raw_entity, ENT_TYPE_KEYS) or ""
        if not surface or not surface.strip():
            continue
        surface = surface.strip()
        spans = _entity_spans(text, raw_entity, surface)
        if not spans and entity_type == "Weakness":
            cwe = re.search(r"CWE-\d+", surface, re.IGNORECASE)
            if cwe:
                spans = _all_text_spans(text, cwe.group(0))
        keys = []
        for start, end, matched in spans:
            base_entity = {
                "text": matched,
                "type": entity_type,
                "start": start,
                "end": end,
                "normalized_id": (
                    raw_entity.get("normalized_id")
                    or raw_entity.get("normalized")
                ),
            }
            variants = [base_entity]
            if entity_type == "Configuration":
                variants, _ = _configuration_entity_variants(text, base_entity)
            for variant in variants:
                key = (variant["start"], variant["end"], entity_type)
                records.setdefault(key, variant)
                keys.append(key)
        aliases = {
            str(value)
            for value in (
                raw_entity.get("id"),
                raw_entity.get("name"),
                surface,
            )
            if value is not None
        }
        for alias in aliases:
            alias_keys.setdefault(alias, set()).update(keys)

    ordered_keys = sorted(records, key=lambda item: (item[0], item[1], item[2]))
    key_to_id = {}
    entities = []
    for key in ordered_keys:
        entity = dict(records[key])
        entity["id"] = f"E{len(entities) + 1}"
        key_to_id[key] = entity["id"]
        entities.append(entity)

    alias_to_ids = {
        alias: [key_to_id[key] for key in ordered_keys if key in keys]
        for alias, keys in alias_keys.items()
    }
    for entity in entities:
        alias_to_ids.setdefault(entity["id"], [entity["id"]])
        alias_to_ids.setdefault(entity["text"], []).append(entity["id"])
    for alias, ids in alias_to_ids.items():
        alias_to_ids[alias] = list(dict.fromkeys(ids))
    return entities, alias_to_ids


def _span_distance(left: dict, right: dict) -> int:
    if left["end"] <= right["start"]:
        return right["start"] - left["end"]
    if right["end"] <= left["start"]:
        return left["start"] - right["end"]
    return 0


def _evidence_spans(text: str, evidence: str) -> list[tuple[int, int, str]]:
    return _all_text_spans(text, evidence.strip()) if evidence.strip() else []


def _relation_evidence(
    text: str,
    raw_relation: dict,
) -> tuple[str, list[tuple[int, int, str]], int | None, int | None]:
    """优先使用模型选择的字符区间，并由原文切片构造证据。"""
    start = raw_relation.get("evidence_start")
    end = raw_relation.get("evidence_end")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(text)
    ):
        evidence = text[start:end]
        return evidence, [(start, end, evidence)], start, end

    evidence = (
        raw_relation.get("evidence")
        if isinstance(raw_relation.get("evidence"), str)
        else ""
    )
    spans = _evidence_spans(text, evidence)
    if len(spans) == 1:
        evidence_start, evidence_end, _ = spans[0]
        return evidence, spans, evidence_start, evidence_end
    return evidence, spans, None, None


def _relation_pairs(
    head_ids: list[str],
    tail_ids: list[str],
    entity_by_id: dict[str, dict],
    evidence_spans: list[tuple[int, int, str]],
    relation_type: str = "",
) -> list[tuple[str, str]]:
    """依据证据位置将原始实体别名解析为具体 mention 端点。"""
    pairs = []
    for evidence_start, evidence_end, _ in evidence_spans:
        heads = [
            entity_by_id[entity_id]
            for entity_id in head_ids
            if entity_id in entity_by_id
            and evidence_start <= entity_by_id[entity_id]["start"]
            and entity_by_id[entity_id]["end"] <= evidence_end
        ]
        tails = [
            entity_by_id[entity_id]
            for entity_id in tail_ids
            if entity_id in entity_by_id
            and evidence_start <= entity_by_id[entity_id]["start"]
            and entity_by_id[entity_id]["end"] <= evidence_end
        ]
        candidates = [
            (_span_distance(head, tail), head["start"], tail["start"], head["id"], tail["id"])
            for head in heads
            for tail in tails
            if head["id"] != tail["id"]
        ]
        if relation_type == "affects":
            split_groups = {}
            for tail in tails:
                group = tail.get("_configuration_split_group")
                if group:
                    split_groups.setdefault(group, []).append(tail)
            expandable = [
                group_tails
                for group_tails in split_groups.values()
                if len(group_tails) >= 2
            ]
            if len(expandable) == 1 and heads:
                for tail in sorted(
                    expandable[0], key=lambda item: (item["start"], item["end"])
                ):
                    head = min(
                        heads,
                        key=lambda item: (
                            _span_distance(item, tail), item["start"], item["end"]
                        ),
                    )
                    if head["id"] != tail["id"]:
                        pairs.append((head["id"], tail["id"]))
                continue
        if candidates:
            _, _, _, head_id, tail_id = min(candidates)
            pairs.append((head_id, tail_id))
    if pairs:
        return list(dict.fromkeys(pairs))

    candidates = [
        (
            _span_distance(entity_by_id[head_id], entity_by_id[tail_id]),
            entity_by_id[head_id]["start"],
            entity_by_id[tail_id]["start"],
            head_id,
            tail_id,
        )
        for head_id in head_ids
        for tail_id in tail_ids
        if head_id in entity_by_id
        and tail_id in entity_by_id
        and head_id != tail_id
    ]
    if not candidates:
        return []
    _, _, _, head_id, tail_id = min(candidates)
    return [(head_id, tail_id)]


def _evidence_aligned_equivalent_ids(
    entity_ids: list[str],
    entity_by_id: dict[str, dict],
    evidence_spans: list[tuple[int, int, str]],
) -> list[str]:
    """Repair a wrong repeated-mention ID only when evidence is unambiguous.

    Models sometimes select an earlier entity ID even though their evidence
    interval contains a later occurrence of the same normalized entity.  This
    helper changes the endpoint only when none of the selected mentions is in
    the evidence and exactly one equivalent mention is.  It never substitutes
    a different normalized entity or guesses when the evidence is ambiguous.
    """
    if not entity_ids or not evidence_spans:
        return entity_ids

    def inside_evidence(entity: dict) -> bool:
        return any(
            evidence_start <= entity["start"]
            and entity["end"] <= evidence_end
            for evidence_start, evidence_end, _ in evidence_spans
        )

    selected = [
        entity_by_id[entity_id]
        for entity_id in entity_ids
        if entity_id in entity_by_id
    ]
    if any(inside_evidence(entity) for entity in selected):
        return entity_ids

    def equivalent(candidate: dict, reference: dict) -> bool:
        if candidate.get("type") != reference.get("type"):
            return False
        candidate_normalized = candidate.get("normalized_id")
        reference_normalized = reference.get("normalized_id")
        if candidate_normalized and reference_normalized:
            return candidate_normalized == reference_normalized
        return candidate.get("text") == reference.get("text")

    candidates = {
        candidate["id"]
        for candidate in entity_by_id.values()
        if inside_evidence(candidate)
        and any(equivalent(candidate, reference) for reference in selected)
    }
    if len(candidates) == 1:
        return list(candidates)
    return entity_ids


def _repair_one_sided_evidence_interval(
    text: str,
    evidence_start: int | None,
    evidence_end: int | None,
    head: dict,
    tail: dict,
) -> tuple[int, int] | None:
    """Expand a valid one-sided model interval to both selected endpoints.

    The repair is purely mechanical: the original interval must exactly cover
    at least one selected mention, and the continuous expanded interval is
    capped so a distant co-occurrence cannot be silently converted into local
    evidence.
    """
    if not (
        isinstance(evidence_start, int)
        and not isinstance(evidence_start, bool)
        and isinstance(evidence_end, int)
        and not isinstance(evidence_end, bool)
        and 0 <= evidence_start < evidence_end <= len(text)
    ):
        return None
    contains_head = (
        evidence_start <= head["start"] < head["end"] <= evidence_end
    )
    contains_tail = (
        evidence_start <= tail["start"] < tail["end"] <= evidence_end
    )
    if not (contains_head or contains_tail):
        return None
    repaired_start = min(evidence_start, head["start"], tail["start"])
    repaired_end = max(evidence_end, head["end"], tail["end"])
    if repaired_end - repaired_start > MAX_EVIDENCE_OFFSET_REPAIR_CHARS:
        return None
    return repaired_start, repaired_end


def parse_relations(
    text: str,
    raw_relations,
    entities: list[dict],
    alias_to_ids: dict[str, list[str]],
) -> list[dict]:
    """单阶段与多阶段共享的关系解析和证据端点消歧。"""
    entity_by_id = {entity["id"]: entity for entity in entities}
    relations = []
    seen = set()
    for raw_relation in raw_relations or []:
        if not isinstance(raw_relation, dict):
            continue
        head_alias = _pick(raw_relation, REL_HEAD_KEYS)
        tail_alias = _pick(raw_relation, REL_TAIL_KEYS)
        relation_type = (
            raw_relation.get("type")
            or raw_relation.get("relation")
            or raw_relation.get("label")
            or ""
        )
        head_ids = alias_to_ids.get(str(head_alias), [])
        tail_ids = alias_to_ids.get(str(tail_alias), [])
        evidence, evidence_spans, evidence_start, evidence_end = (
            _relation_evidence(text, raw_relation)
        )
        original_head_ids = head_ids
        original_tail_ids = tail_ids
        head_ids = _evidence_aligned_equivalent_ids(
            head_ids,
            entity_by_id,
            evidence_spans,
        )
        tail_ids = _evidence_aligned_equivalent_ids(
            tail_ids,
            entity_by_id,
            evidence_spans,
        )
        for head_id, tail_id in _relation_pairs(
            head_ids,
            tail_ids,
            entity_by_id,
            evidence_spans,
            relation_type if isinstance(relation_type, str) else "",
        ):
            key = (relation_type, head_id, tail_id)
            if key in seen:
                continue
            seen.add(key)
            relation = {
                "id": f"R{len(relations) + 1}",
                "type": relation_type if isinstance(relation_type, str) else "",
                "head": head_id,
                "tail": tail_id,
            }
            if head_ids != original_head_ids or tail_ids != original_tail_ids:
                relation["endpoint_mention_repaired"] = True
            head = entity_by_id[head_id]
            tail = entity_by_id[tail_id]
            repaired_interval = _repair_one_sided_evidence_interval(
                text,
                evidence_start,
                evidence_end,
                head,
                tail,
            )
            if repaired_interval is not None:
                repaired_start, repaired_end = repaired_interval
                relation["evidence"] = text[repaired_start:repaired_end]
                relation["evidence_start"] = repaired_start
                relation["evidence_end"] = repaired_end
                if (repaired_start, repaired_end) != (
                    evidence_start,
                    evidence_end,
                ):
                    relation["evidence_offset_repaired"] = True
                    relation["original_evidence_start"] = evidence_start
                    relation["original_evidence_end"] = evidence_end
            else:
                if evidence:
                    relation["evidence"] = evidence
                if evidence_start is not None and evidence_end is not None:
                    relation["evidence_start"] = evidence_start
                    relation["evidence_end"] = evidence_end
            relations.append(relation)
    return relations


def snap_to_sentence_boundary(
    text: str,
    raw_start: int,
    min_start: int,
    max_start: int,
    radius: int = 250,
) -> int:
    """在 raw_start 附近寻找最自然的语义起点（CVE 标题首、段落首、句首）。"""
    search_start = max(min_start, raw_start - radius)
    search_end = min(max_start, raw_start + radius // 2)
    if search_start >= search_end:
        return raw_start

    search_chunk = text[search_start:search_end]

    # 优先 1：是否有 CVE 标题（威胁情报最强语义单元）
    cve_matches = list(
        re.finditer(r"(?:\n|^)(CVE-\d{4}-\d{4,})", search_chunk, re.IGNORECASE)
    )
    if cve_matches:
        best_cve = min(
            cve_matches, key=lambda m: abs((search_start + m.start(1)) - raw_start)
        )
        cve_abs_pos = search_start + best_cve.start(1)
        if cve_abs_pos > min_start:
            return cve_abs_pos

    # 优先 2：段落分隔符 \n\n
    p_pos = text.rfind("\n\n", search_start, search_end)
    if p_pos >= 0 and p_pos + 2 > min_start:
        return p_pos + 2

    # 优先 3：句号+空格/换行
    for delimiter in (". \n", ".\n", ". ", "。\n", "。"):
        pos = text.rfind(delimiter, search_start, search_end)
        if pos >= 0 and pos + len(delimiter) > min_start:
            return pos + len(delimiter)

    # 优先 4：普通换行符 \n
    n_pos = text.rfind("\n", search_start, search_end)
    if n_pos >= 0 and n_pos + 1 > min_start:
        return n_pos + 1

    return raw_start


def build_text_windows(
    text: str,
    max_chars: int = WINDOW_CHARS,
    overlap: int = WINDOW_OVERLAP,
    snap_sentence_boundary: bool = False,
) -> list[dict]:
    """按段落/句子边界构造重叠窗口，并保留全局字符偏移。

    Args:
        text: 待切分长文本。
        max_chars: 窗口最大字符数。
        overlap: 窗口间重叠目标字符数。
        snap_sentence_boundary: 若为 True，窗口左边界向最近的 CVE 标题、段落首或句首吸附，
            彻底消除无主语的断头句。默认 False 保留历史预注册审计兼容性。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [{"start": 0, "end": len(text), "text": text}]
    if not 0 <= overlap < max_chars:
        raise ValueError("window overlap 必须满足 0 <= overlap < max_chars")

    windows = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            floor = start + max_chars // 2
            candidates = []
            for delimiter in ("\n\n", "\n", "。", ". "):
                position = text.rfind(delimiter, floor, hard_end)
                if position >= 0:
                    candidates.append(position + len(delimiter))
            if candidates:
                end = max(candidates)
        if end <= start:
            end = hard_end
        windows.append({"start": start, "end": end, "text": text[start:end]})
        if end >= len(text):
            break
        raw_start = max(start + 1, end - overlap)
        if snap_sentence_boundary and raw_start < end:
            raw_start = snap_to_sentence_boundary(
                text,
                raw_start,
                min_start=start + 1,
                max_start=min(end - 50, raw_start + 150),
                radius=250,
            )
        start = raw_start
    return windows


def _offset_prediction(prediction: dict, offset: int) -> dict:
    entities = []
    for entity in prediction.get("entities", []):
        shifted = dict(entity)
        shifted["start"] = int(shifted["start"]) + offset
        shifted["end"] = int(shifted["end"]) + offset
        entities.append(shifted)
    relations = []
    for relation in prediction.get("relations", []):
        shifted = dict(relation)
        if isinstance(shifted.get("evidence_start"), int):
            shifted["evidence_start"] += offset
        if isinstance(shifted.get("evidence_end"), int):
            shifted["evidence_end"] += offset
        relations.append(shifted)
    return {
        "entities": entities,
        "relations": relations,
        "_trace": prediction.get("_trace"),
    }


def merge_window_predictions(predictions: list[dict]) -> dict:
    """合并重叠窗口结果，以全局字符跨度去重并重写关系端点。"""
    entity_records = {}
    local_to_key = {}
    traces = []
    for window_index, prediction in enumerate(predictions):
        traces.append(prediction.get("_trace"))
        for entity in prediction.get("entities", []):
            key = (entity["start"], entity["end"], entity.get("type", ""))
            entity_records.setdefault(key, dict(entity))
            local_to_key[(window_index, entity["id"])] = key

    ordered_keys = sorted(entity_records, key=lambda item: (item[0], item[1], item[2]))
    key_to_id = {}
    entities = []
    for key in ordered_keys:
        entity = dict(entity_records[key])
        entity["id"] = f"E{len(entities) + 1}"
        key_to_id[key] = entity["id"]
        entities.append(entity)

    relations = []
    seen = set()
    for window_index, prediction in enumerate(predictions):
        for relation in prediction.get("relations", []):
            head_key = local_to_key.get((window_index, relation.get("head")))
            tail_key = local_to_key.get((window_index, relation.get("tail")))
            if head_key not in key_to_id or tail_key not in key_to_id:
                continue
            head_id = key_to_id[head_key]
            tail_id = key_to_id[tail_key]
            key = (relation.get("type", ""), head_id, tail_id)
            if head_id == tail_id or key in seen:
                continue
            seen.add(key)
            merged = {
                "id": f"R{len(relations) + 1}",
                "type": relation.get("type", ""),
                "head": head_id,
                "tail": tail_id,
            }
            if relation.get("evidence"):
                merged["evidence"] = relation["evidence"]
            if isinstance(relation.get("evidence_start"), int):
                merged["evidence_start"] = relation["evidence_start"]
            if isinstance(relation.get("evidence_end"), int):
                merged["evidence_end"] = relation["evidence_end"]
            relations.append(merged)
    return {
        "entities": entities,
        "relations": relations,
        "_trace": {"windows": traces, "window_count": len(predictions)},
    }


def _loose_json(output: str) -> dict:
    cands = [output]
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
    if m:
        cands.append(m.group(1))
    a, b = output.find("{"), output.rfind("}")
    if a != -1 and b != -1 and b > a:
        cands.append(output[a:b + 1])
    for c in cands:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return {"entities": [], "relations": []}


def _call_with_retries(extractor, prompt, system_prompt, max_retries=3):
    """执行模型调用并保存非敏感运行轨迹。"""
    errors = []
    call_config = dict(extractor.config)
    initial_max_tokens = int(call_config.get("max_tokens") or 0)
    escalation_limit = max(
        initial_max_tokens,
        int(call_config.get("max_escalated_tokens") or initial_max_tokens),
    )
    escalated = False
    for attempt in range(1, max_retries + 1):
        try:
            output = extractor.call_fn(prompt, system_prompt, call_config)
            if not isinstance(output, str) or not output.strip():
                raise RuntimeError("模型返回空内容")
            trace = {
                "attempts": attempt,
                "errors": errors,
                "called_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if escalated:
                trace["budget_escalation"] = {
                    "initial_max_tokens": initial_max_tokens,
                    "escalated_max_tokens": int(call_config["max_tokens"]),
                    "attempts_after_escalation": 1,
                }
            return output, trace
        except ModelOutputBudgetExhaustedError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            current_max_tokens = int(call_config.get("max_tokens") or 0)
            if not escalated and escalation_limit > current_max_tokens:
                call_config = {
                    **extractor.config,
                    "max_tokens": escalation_limit,
                }
                escalated = True
                continue
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            status_code = getattr(exc, "status_code", None)
            lowered = str(exc).casefold()
            terminal = status_code in {401, 402, 403} or any(
                marker in lowered
                for marker in (
                    "insufficient_balance",
                    "insufficient account balance",
                    "invalid api key",
                    "authentication",
                )
            )
            if terminal:
                raise RuntimeError(
                    "模型API返回终止性鉴权或余额错误，已停止重试："
                    f"status_code={status_code}, error={errors[-1]}"
                ) from exc
            # A budget-escalated call is deliberately attempted only once.
            # Repeating the same larger request would recreate the previous
            # 3 x max-token cost spike without adding a new recovery path.
            if escalated:
                break
            if attempt < max_retries:
                time.sleep(2.0)
    trace = {
        "attempts": attempt,
        "errors": errors,
        "called_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if escalated:
        trace["budget_escalation"] = {
            "initial_max_tokens": initial_max_tokens,
            "escalated_max_tokens": int(call_config["max_tokens"]),
            "attempts_after_escalation": 1,
        }
    return "", trace


def llm_extract_raw(extractor, text, prompt, system_prompt, max_retries=3) -> dict:
    """调用 LLM，并用共享解析器恢复全部实体提及及关系端点。"""
    data = {"entities": [], "relations": []}
    output, trace = _call_with_retries(
        extractor, prompt, system_prompt, max_retries=max_retries
    )
    parsed = _loose_json(output)
    if parsed.get("entities") or parsed.get("relations"):
        data = parsed
    elif trace["errors"]:
        print(f"    LLM 调用失败: {trace['errors'][-1]}")
    ents, aliases = parse_entity_mentions(text, data.get("entities"))
    rels = parse_relations(
        text,
        data.get("relations"),
        ents,
        aliases,
    )
    return {
        "entities": ents,
        "relations": rels,
        "_trace": {
            **trace,
            "raw_response": output,
            "parsed_nonempty": bool(
                parsed.get("entities") or parsed.get("relations")
            ),
        },
    }


_EXTRACTOR = None


def _ext():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = make_extractor()
    return _EXTRACTOR


# ===== 下层单轮 prompt：4 类实体 + 3 类文本关系 =====
LOWER_SYSTEM = ("You are a cybersecurity knowledge extraction expert. Extract ONLY the requested "
                "lower-layer entities and relations. Output valid JSON only, no explanation.")

LOWER_FEWSHOT = '''### Input:
Attackers exploited CVE-2021-44228 in the Log4j component to gain initial access by exploiting a public-facing application (T1190).
### Output:
{"entities":[{"id":"E1","text":"CVE-2021-44228","type":"Vulnerability","start":20,"end":34,"normalized_id":"CVE-2021-44228"},{"id":"E2","text":"Log4j","type":"Configuration","start":42,"end":47,"normalized_id":"cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"},{"id":"E3","text":"T1190","type":"AttackTechnique","start":124,"end":129,"normalized_id":"T1190"}],"relations":[{"id":"R1","type":"affects","head":"E1","tail":"E2","evidence_start":20,"evidence_end":57},{"id":"R2","type":"exploited_by","head":"E1","tail":"E3","evidence_start":0,"evidence_end":131}]}'''

LOWER_TEMPLATE = '''## Task
Extract lower-layer cybersecurity entities and relations from the CTI text.

## Entity Types (ONLY these 4)
1. Vulnerability — CVE identifiers (e.g., CVE-2021-44228). normalized_id = the CVE id.
2. Weakness — an explicit CWE mention, or a core flaw phrase only when the same local fact block explicitly supplies the matching CWE identifier. A description-only phrase without a local explicit CWE is a review candidate and MUST NOT be emitted as a formal entity. normalized_id = that locally explicit CWE id.
3. Configuration — the Minimum Canonical Product Unit (MCPU): the shortest continuous source span that preserves the canonical affected product/software/component/device identity and can support a direct affects fact, or a full literal CPE URI. Retain an explicitly adjacent vendor when it belongs to the same continuous product noun phrase; never import a vendor across a table column, conjunction, list item, sentence, or non-contiguous text. Retain Server, Gateway, Manager, Appliance, Controller, Service, Driver and similar designators only when they are part of the official product or directly affected component identity; remove purely descriptive class nouns. Exclude ordinary release versions, updates, patches, builds and deployment qualifiers. Preserve a full literal CPE URI and a lexicalized generation or device-model token required for identity, such as SMBv1. normalized_id must be a canonical CPE 2.3 URI: copy a literal CPE, or infer only supported part/vendor/product fields and use * for unspecified release fields; use null if vendor/product is uncertain, never a free-form product name.
4. AttackTechnique — annotate every explicit ATT&CK ID occurrence (for example T1190 or T1505.003). Annotate a clearly named technique only when no adjacent T ID is present and it maps unambiguously to one ATT&CK ID. If the name and its T ID are adjacent, return only the T-ID span. normalized_id = the ATT&CK id.

## Relation Types (ONLY these 3)
1. affects — Vulnerability -> Configuration
2. instantiates — Vulnerability -> Weakness
3. exploited_by — Vulnerability -> AttackTechnique

## Rules
1. CRITICAL: Entities MUST be exact substrings copied from the text, character by character. Do NOT hallucinate names or summarize.
2. CRITICAL: Do NOT output IDs (like T1505.003 or CWE-89) in the "text" field unless it is literally written in the text. Put IDs ONLY in the "normalized_id" field.
3. Return 0-based start and end offsets for every entity; end is exclusive and text[start:end] MUST equal the entity text.
4. Return every explicit CVE, CWE and ATT&CK-ID occurrence. For Configuration, keep every local mention that anchors a directly stated CVE-product fact in a self-contained fact block, including repeated statements of the same normalized fact; omit unrelated products and repeats that do not restate the direct relation locally.
5. Extract ONLY the 4 entity types and 3 relation types above. Do NOT extract AttackPattern, AttackTactic, KillChainPhase or any upper-layer relation.
6. MCPU boundary: keep an adjacent vendor only inside the same continuous product noun phrase; do not cross structural boundaries to add one. Keep official product/component designators, but strip generic descriptive class nouns, "vulnerability"/"flaw"/"attack" wording, and ordinary release/update/patch/build tokens.
7. Consequence phrases (remote code execution, privilege escalation, denial of service, information disclosure) are impacts, NOT weaknesses — do not extract them.
8. normalized_id: Vulnerability→CVE id, Weakness→a CWE id explicitly present in the entity span or same local fact block, AttackTechnique→ATT&CK id, Configuration→canonical CPE 2.3 URI (never a free-form product name). Never infer a CWE from a general phrase, a CVE database lookup, or model background knowledge.
9. Generic placeholders such as "multiple products", "various products", "affected systems" and "software" are NOT Configuration entities.
10. Do not extract Configuration from navigation, tags, reference URLs, tool or mitigation lists, or general attack context unless that local text directly identifies the product as affected by an explicit CVE.
11. A labeled CVE-to-Vendor/Product table or list row can directly support affects without a prose verb when the same row contains the exact CVE and an identifiable product/component. Do not use a vendor-only or flaw-description-only cell. Explicit unrelated/not affected/does not affect wording blocks affects; an actor's not-exploited statement does not erase a separately explicit CVE-product assignment.
12. Every relation must return evidence_start and evidence_end for one exact continuous source interval. The interval must contain the exact character spans of the selected head and tail mentions plus the relation wording; the same surface text at another position is not sufficient. Co-occurrence alone is insufficient.
13. Create exploited_by only when the text directly states that the specified CVE is exploited through the ATT&CK technique. Do not link techniques that merely describe reconnaissance, persistence, credential theft, command and control, or other pre-/post-exploitation activity. "Exploited the CVE and then ran a RAT for C2" does not support an exploited_by edge to the command-and-control technique.

## Example
__FEWSHOT__

## Input Text
__TEXT__

## Output
Output valid JSON only.'''


def lower_prompt(text):
    return LOWER_TEMPLATE.replace("__FEWSHOT__", LOWER_FEWSHOT).replace("__TEXT__", text)


def predict_llm_manual(text, doc_id):
    """单轮下层抽取；按共享窗口和共享解析器处理，不执行模式后处理。"""
    predictions = []
    for window in build_text_windows(text):
        local = llm_extract_raw(
            _ext(),
            window["text"],
            lower_prompt(window["text"]),
            LOWER_SYSTEM,
        )
        predictions.append(_offset_prediction(local, window["start"]))
    return merge_window_predictions(predictions)


# ============ full 方法：复用 APO 原始预测并执行 schema 约束后处理 ============
LOWER_ENT_TYPES = set(EXTRACTION_ENTITY_TYPES)
CONSEQUENCE_WEAKNESS_TERMS = {
    "remote code execution",
    "rce",
    "arbitrary code execution",
    "code execution",
    "privilege escalation",
    "elevation of privilege",
    "denial of service",
    "information disclosure",
    "information exposure",
}
EXPLOIT_TERMS = re.compile(
    r"\b(exploit(?:ed|ing|s|ation)?|leverag(?:e|ed|ing)|trigger(?:ed|ing|s)?)\b|"
    r"利用|漏洞利用|触发",
    re.IGNORECASE,
)
POST_EXPLOIT_SEQUENCE = re.compile(
    r"\bafter\s+(?:successfully\s+)?exploit(?:ing|ation)?\b|"
    r"\bfollowing\s+(?:the\s+)?exploit(?:ation)?\b|"
    r"\bexploit(?:ed|ing)\b.{0,120}\b(?:and\s+then|then|subsequently|and\s+ran)\b",
    re.IGNORECASE | re.DOTALL,
)
POST_EXPLOIT_ACTIVITY = re.compile(
    r"\b(?:RAT|C2|command[- ]and[- ]control|persistence|credential(?:s)?|"
    r"lateral\s+movement|arbitrary\s+code\s+execution|indirect\s+command\s+execution)\b|"
    r"持久化|凭据访问|横向移动|命令与控制|任意代码执行",
    re.IGNORECASE,
)


def _is_explicit_post_exploitation_sequence(evidence: str) -> bool:
    """Reject only explicit temporal transitions to a later attack activity."""
    return bool(
        POST_EXPLOIT_SEQUENCE.search(evidence)
        and POST_EXPLOIT_ACTIVITY.search(evidence)
    )


def _canonical_cve(text):
    """Normalize a CVE mention while tolerating PDF-introduced spaces."""
    compact = re.sub(r"\s+", "", str(text or "")).upper()
    match = re.fullmatch(r"CVE-?(\d{4})-(\d{4,7})", compact)
    if not match:
        return None
    return f"CVE-{match.group(1)}-{match.group(2)}"


_PDF_CVE_MENTION_RE = re.compile(
    r"\bCVE-\s*\d{4}-\s*\d{4,7}\b", re.IGNORECASE
)


def _canonical_cpe(value):
    """Accept only complete CPE 2.3 URIs for Configuration normalization."""
    candidate = str(value or "").strip()
    fields = candidate.split(":")
    if (
        len(fields) != 13
        or fields[0].casefold() != "cpe"
        or fields[1] != "2.3"
        or fields[2].casefold() not in {"a", "o", "h"}
        or fields[3] in {"", "*"}
        or fields[4] in {"", "*"}
    ):
        return None
    return candidate.casefold()


def _canonical_cwe(value):
    match = re.fullmatch(r"CWE-?(\d+)", str(value or "").strip(), re.IGNORECASE)
    return f"CWE-{match.group(1)}" if match else None


def _local_fact_block(text: str, start: int, end: int) -> str:
    """Return a conservative sentence/line block around one mention."""
    left_boundaries = [text.rfind(mark, 0, start) for mark in ("\n", ".", "!", "?", "。", "！", "？")]
    left = max(left_boundaries) + 1
    right_candidates = [
        pos
        for mark in ("\n", ".", "!", "?", "。", "！", "？")
        if (pos := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left:right]


def _locally_explicit_weakness_id(entity: dict, source_text: str) -> str | None:
    surface = str(entity.get("text") or "")
    explicit = re.search(r"CWE-\s*\d+", surface, re.IGNORECASE)
    if explicit:
        return _canonical_cwe(re.sub(r"\s+", "", explicit.group(0)))
    current = _canonical_cwe(entity.get("normalized_id"))
    start, end = entity.get("start"), entity.get("end")
    if not current or not source_text or not isinstance(start, int) or not isinstance(end, int):
        return None
    block = _local_fact_block(source_text, start, end)
    explicit_ids = {
        _canonical_cwe(re.sub(r"\s+", "", match.group(0)))
        for match in re.finditer(r"CWE-\s*\d+", block, re.IGNORECASE)
    }
    return current if current in explicit_ids else None


def _normalize_norm_id(e, source_text: str = ""):
    typ = e.get("type"); text = (e.get("text") or "").strip(); low = text.lower(); cur = e.get("normalized_id")
    if typ == "Vulnerability":
        return _canonical_cve(text) or cur
    if typ == "Weakness":
        return _locally_explicit_weakness_id(e, source_text)
    if typ == "AttackTechnique":
        m = re.search(r"T\d{4}(?:\.\d{3})?", text, re.IGNORECASE)
        return m.group(0).upper() if m else cur
    if typ == "Configuration":
        return _canonical_cpe(cur)
    return cur


def _equivalent_mention_identity(entity):
    """Return a conservative identity key for repeated local mentions."""
    normalized = _normalize_norm_id(entity)
    if normalized:
        return entity.get("type"), str(normalized).casefold()
    text = str(entity.get("text") or "").strip()
    if not text:
        return None
    return entity.get("type"), text.casefold()


def _locally_reanchor_equivalent_pair(
    head: dict,
    tail: dict,
    evidence_interval,
    entity_by_id: dict[str, dict],
):
    """Choose a uniquely closer repeated-mention pair inside the evidence.

    A broad evidence interval can contain both an earlier heading mention and
    the later mention that actually participates in the local fact.  Reanchor
    only among mentions with the same conservative identity, and only when a
    single pair is strictly closer than the model-selected pair.  Ties remain
    untouched instead of silently choosing the first occurrence.
    """
    if evidence_interval is None:
        return head, tail, False
    evidence_start, evidence_end = evidence_interval
    head_identity = _equivalent_mention_identity(head)
    tail_identity = _equivalent_mention_identity(tail)
    if head_identity is None or tail_identity is None:
        return head, tail, False

    def candidates(reference_identity):
        return [
            entity
            for entity in entity_by_id.values()
            if _equivalent_mention_identity(entity) == reference_identity
            and evidence_start <= entity["start"]
            and entity["end"] <= evidence_end
        ]

    scored_pairs = [
        (_span_distance(candidate_head, candidate_tail), candidate_head, candidate_tail)
        for candidate_head in candidates(head_identity)
        for candidate_tail in candidates(tail_identity)
        if candidate_head["id"] != candidate_tail["id"]
    ]
    if not scored_pairs:
        return head, tail, False
    best_distance = min(score for score, _, _ in scored_pairs)
    best_pairs = [
        (candidate_head, candidate_tail)
        for score, candidate_head, candidate_tail in scored_pairs
        if score == best_distance
    ]
    selected_distance = _span_distance(head, tail)
    if len(best_pairs) != 1 or best_distance >= selected_distance:
        return head, tail, False
    best_head, best_tail = best_pairs[0]
    return best_head, best_tail, True


def _recover_local_exploited_cve_mentions(raw: dict, source_text: str):
    """Recover only PDF-spaced CVE variants inside exact exploit evidence.

    This is deliberately relation-local, rather than a document-wide CVE
    regex fallback: a selected Vulnerability endpoint must already establish
    the canonical CVE identity, and the relation must provide exact evidence
    offsets.  Reference-list occurrences outside that evidence are untouched.
    """
    raw_entities = list(raw.get("entities", []))
    by_id = {
        entity.get("id"): entity
        for entity in raw_entities
        if isinstance(entity, dict) and entity.get("id") is not None
    }
    occupied = {
        (entity.get("start"), entity.get("end"), entity.get("type"))
        for entity in raw_entities
        if isinstance(entity, dict)
    }
    recovered = []
    for relation in raw.get("relations", []):
        if not isinstance(relation, dict) or relation.get("type") != "exploited_by":
            continue
        selected_head = by_id.get(relation.get("head"))
        if not selected_head or selected_head.get("type") != "Vulnerability":
            continue
        canonical = _canonical_cve(selected_head.get("text"))
        evidence = relation.get("evidence")
        start = relation.get("evidence_start")
        end = relation.get("evidence_end")
        if not (
            canonical
            and isinstance(evidence, str)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(source_text)
            and source_text[start:end] == evidence
        ):
            continue
        for match in _PDF_CVE_MENTION_RE.finditer(evidence):
            mention_start = start + match.start()
            mention_end = start + match.end()
            key = (mention_start, mention_end, "Vulnerability")
            if key in occupied or _canonical_cve(match.group(0)) != canonical:
                continue
            occupied.add(key)
            recovered.append({
                "id": f"__relation_local_cve_{len(recovered) + 1}",
                "text": source_text[mention_start:mention_end],
                "type": "Vulnerability",
                "start": mention_start,
                "end": mention_end,
                "normalized_id": canonical,
            })
    known_cves = {
        _canonical_cve(entity.get("text"))
        for entity in raw_entities
        if isinstance(entity, dict) and entity.get("type") == "Vulnerability"
    }
    for fact in _explicit_exploited_by_facts(source_text):
        key = (fact["cve_start"], fact["cve_end"], "Vulnerability")
        canonical = fact["cve_id"]
        if key in occupied or canonical not in known_cves:
            continue
        occupied.add(key)
        recovered.append({
            "id": f"__relation_local_cve_{len(recovered) + 1}",
            "text": source_text[fact["cve_start"]:fact["cve_end"]],
            "type": "Vulnerability",
            "start": fact["cve_start"],
            "end": fact["cve_end"],
            "normalized_id": canonical,
        })
    return raw_entities + recovered, len(recovered)


def _relation_evidence_interval(source_text, relation, head, tail):
    """Return the exact evidence interval covering these specific mentions."""
    evidence = relation.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        return None
    for entity in (head, tail):
        if not isinstance(entity, dict):
            return None
        if not isinstance(entity.get("start"), int) or not isinstance(
            entity.get("end"), int
        ):
            return None

    evidence_start = relation.get("evidence_start")
    evidence_end = relation.get("evidence_end")
    offsets_present = evidence_start is not None or evidence_end is not None
    intervals = []
    if offsets_present:
        if not (
            isinstance(evidence_start, int)
            and not isinstance(evidence_start, bool)
            and isinstance(evidence_end, int)
            and not isinstance(evidence_end, bool)
            and 0 <= evidence_start < evidence_end <= len(source_text)
            and source_text[evidence_start:evidence_end] == evidence
        ):
            return None
        intervals.append((evidence_start, evidence_end))
    else:
        cursor = source_text.find(evidence)
        while cursor >= 0:
            intervals.append((cursor, cursor + len(evidence)))
            cursor = source_text.find(evidence, cursor + 1)

    for start, end in intervals:
        if (
            start <= head["start"] < head["end"] <= end
            and start <= tail["start"] < tail["end"] <= end
        ):
            return start, end
    return None


_EXPLICIT_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*\u2022\u2023\u25e6\u25aa\u25cf\uf077\uf0a7\uf0b7]|\d+[.)])\s+"
)
_INLINE_UNICODE_BULLET_RE = re.compile(
    r"(?<!\S)[\u2022\u2023\u25e6\u25aa\u25cf\uf077\uf0a7\uf0b7](?=\s)"
)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_CVE_TOKEN_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
MAX_TABLE_ROW_PARSE_CHARS = 2000
_AFFECTED_CONFIGURATION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:(?:affected\s+)?configurations?|affected\s+(?:products?|versions?))\s*:.*$"
)
_CVE_RECORD_LINE_RE = re.compile(r"(?im)^\s*CVE-\d{4}-\d{4,7}\b")


@dataclass(frozen=True)
class _FactBlock:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _FactBlockIndex:
    structural: tuple[_FactBlock, ...]
    excluded: tuple[_FactBlock, ...]


def _is_explicit_table_row(line: str) -> bool:
    """Accept only bounded, visibly delimited physical table rows."""
    if not line or len(line) > MAX_TABLE_ROW_PARSE_CHARS:
        return False
    if line.count("|") >= 2:
        return sum(bool(cell.strip()) for cell in line.split("|")) >= 2
    if "\t" in line:
        return sum(bool(cell.strip()) for cell in line.split("\t")) >= 2
    return False


def _build_fact_block_index(source_text: str) -> _FactBlockIndex:
    """Parse explicit structure once without treating a flattened PDF as one row."""
    structural = []
    line_start = 0
    while line_start <= len(source_text):
        line_end = source_text.find("\n", line_start)
        if line_end < 0:
            line_end = len(source_text)
        line = source_text[line_start:line_end]
        markers = []
        line_marker = _EXPLICIT_LIST_ITEM_RE.match(line)
        if line_marker:
            markers.append((line_marker.start(), line_marker.end()))
        markers.extend(
            (match.start(), match.end())
            for match in _INLINE_UNICODE_BULLET_RE.finditer(line)
        )
        markers = sorted(set(markers))
        for marker_index, (marker_start, _marker_end) in enumerate(markers):
            item_end = (
                markers[marker_index + 1][0]
                if marker_index + 1 < len(markers)
                else len(line)
            )
            structural.append(_FactBlock(
                "list",
                line_start + marker_start,
                line_start + item_end,
            ))
        if not markers and _is_explicit_table_row(line):
            structural.append(_FactBlock("table", line_start, line_end))
        if line_end >= len(source_text):
            break
        line_start = line_end + 1

    excluded = tuple(
        _FactBlock("reference", start, end)
        for start, end in _reference_section_ranges(source_text)
    )
    return _FactBlockIndex(tuple(structural), excluded)


def _fact_block_at(
    fact_block_index: _FactBlockIndex,
    position: int,
    *,
    excluded: bool = False,
):
    blocks = fact_block_index.excluded if excluded else fact_block_index.structural
    containing = [block for block in blocks if block.start <= position < block.end]
    if not containing:
        return None
    return min(containing, key=lambda block: block.end - block.start)


def _explicit_fact_item_at(
    source_text: str,
    position: int,
    fact_block_index: _FactBlockIndex | None = None,
):
    """Return a clear list/table item containing ``position``, if any.

    Plain wrapped prose and whitespace-aligned PDF tables are deliberately not
    inferred as rows.  Explicit bullet/numbered list lines, unambiguous inline
    Unicode bullets emitted by PDF extraction, and pipe/tab-delimited table
    rows are structural fact blocks here.  ASCII ``-`` and ``*`` remain
    line-start-only markers so ordinary prose is not split accidentally.
    """
    if not 0 <= position < len(source_text):
        return None
    index = fact_block_index or _build_fact_block_index(source_text)
    block = _fact_block_at(index, position)
    return (block.kind, block.start, block.end) if block else None


def _crosses_explicit_fact_items(source_text: str, head: dict, tail: dict) -> bool:
    """Return True only when endpoints lie in different explicit items.

    A CVE heading followed by a scoped CPE bullet stays valid: an endpoint
    outside a list/table item does not trigger the guard.  This blocks only
    obvious bullet-to-bullet or row-to-row fan-out.
    """
    head_item = _explicit_fact_item_at(source_text, int(head["start"]))
    tail_item = _explicit_fact_item_at(source_text, int(tail["start"]))
    return bool(head_item and tail_item and head_item != tail_item)


def _sentence_bounds_at(source_text: str, position: int):
    """Return the conservative prose sentence containing ``position``."""
    if not 0 <= position < len(source_text):
        return None
    start = 0
    for match in _SENTENCE_END_RE.finditer(source_text):
        if match.end() <= position:
            start = match.end()
            continue
        return start, match.end()
    return start, len(source_text)


def _item_evidence_is_local(item, evidence_interval, source_text: str) -> bool:
    """Allow an item itself, plus the immediately preceding table header."""
    kind, start, end = item
    evidence_start, evidence_end = evidence_interval
    if start <= evidence_start and evidence_end <= end:
        return True
    if kind != "table":
        return False
    header_start = source_text.rfind("\n", 0, max(0, start - 1)) + 1
    header_end = start - 1
    if header_end <= header_start:
        return False
    header = source_text[header_start:header_end]
    return (
        header.count("|") >= 2
        and header_start <= evidence_start
        and evidence_end <= end
    )


def _affected_configuration_scope(source_text: str, head: dict, tail: dict):
    """Return a uniquely CVE-scoped ``Affected configurations`` block.

    A CPE/product line may be linked across a labelled ``Configurations:`` or
    ``Affected configurations:`` section only when the local record names
    exactly one CVE and that CVE is the relation head.  This is the structural
    exception to ordinary same-sentence matching.
    """
    head_cve = _canonical_cve(head.get("text"))
    if not head_cve:
        return None
    tail_start = int(tail["start"])
    for heading in _AFFECTED_CONFIGURATION_HEADING_RE.finditer(source_text):
        if tail_start < heading.end():
            continue
        section_end = len(source_text)
        blank = re.search(r"\n\s*\n", source_text[heading.end():])
        if blank:
            section_end = heading.end() + blank.start()
        next_heading = _AFFECTED_CONFIGURATION_HEADING_RE.search(
            source_text, heading.end()
        )
        if next_heading and next_heading.start() < section_end:
            section_end = next_heading.start()
        next_cve_line = re.search(
            r"(?im)^\s*CVE-\d{4}-\d{4,7}\b", source_text[heading.end():section_end]
        )
        if next_cve_line:
            section_end = heading.end() + next_cve_line.start()
        if not heading.end() <= tail_start < section_end:
            continue
        record_lines = list(
            _CVE_RECORD_LINE_RE.finditer(source_text, 0, heading.end())
        )
        if record_lines:
            # Structured NVD-like records may contain blank metadata fields
            # between the CVE heading and a later ``Configurations:`` field.
            scope_start = record_lines[-1].start()
        else:
            previous_blank = source_text.rfind("\n\n", 0, heading.start())
            scope_start = previous_blank + 2 if previous_blank >= 0 else 0
        cve_ids = {
            _canonical_cve(match.group(0))
            for match in _CVE_TOKEN_RE.finditer(source_text[scope_start:heading.end()])
        }
        if (
            cve_ids == {head_cve}
            and scope_start <= int(head["start"]) < heading.end()
        ):
            return scope_start, section_end
    return None


def _affects_fact_block_violation(
    source_text: str,
    evidence_interval,
    head: dict,
    tail: dict,
    fact_block_index: _FactBlockIndex | None = None,
):
    """Explain why an ``affects`` pair lacks one local, direct fact block.

    Explicit list/table rows are fact blocks.  In ordinary prose both endpoints
    and the model-provided evidence must stay in one sentence.  A uniquely
    CVE-scoped ``Affected configurations`` section is the sole cross-line
    exception, so a heading cannot fan one product list out to several CVEs.
    """
    fact_block_index = fact_block_index or _build_fact_block_index(source_text)
    if (
        _fact_block_at(fact_block_index, int(head["start"]), excluded=True)
        or _fact_block_at(fact_block_index, int(tail["start"]), excluded=True)
    ):
        return "reference_or_navigation_context"
    head_item = _explicit_fact_item_at(
        source_text, int(head["start"]), fact_block_index
    )
    tail_item = _explicit_fact_item_at(
        source_text, int(tail["start"]), fact_block_index
    )
    if head_item and tail_item:
        if head_item != tail_item:
            return "cross_fact_block"
        if not _item_evidence_is_local(head_item, evidence_interval, source_text):
            return "evidence_cross_fact_block"
        return None

    scope = _affected_configuration_scope(source_text, head, tail)
    if scope is not None:
        scope_start, scope_end = scope
        evidence_start, evidence_end = evidence_interval
        if scope_start <= evidence_start and evidence_end <= scope_end:
            return None
        return "evidence_cross_scoped_section"

    if head_item or tail_item:
        return "unscoped_fact_item"

    head_sentence = _sentence_bounds_at(source_text, int(head["start"]))
    tail_sentence = _sentence_bounds_at(source_text, int(tail["start"]))
    if head_sentence != tail_sentence:
        return "cross_sentence"
    evidence_start, evidence_end = evidence_interval
    if not (
        head_sentence[0] <= evidence_start
        and evidence_end <= head_sentence[1]
    ):
        return "evidence_cross_sentence"
    return None


_POSITIVE_EXPLOITED_CVE_LIST_RE = re.compile(
    r"\b(?:cyber\s+threat\s+|cyber\s+|threat\s+)?actors\s+"
    r"(?:have|has|had)\s+exploited\s+the\s+following\s+CVEs\b"
    r"[^.!?]{0,180}?\[(?P<tid>T\d{4}(?:\.\d{3})?)\][^:]{0,120}:",
    re.IGNORECASE,
)
_EXPLOIT_PUBLIC_FACING_ROW_RE = re.compile(
    r"\bExploit\s+Public-\s*Facing\s+Application\s+"
    r"(?P<tid>T\d{4}(?:\.\d{3})?)\b",
    re.IGNORECASE,
)
_UNAMBIGUOUS_EXPLOIT_TECHNIQUE_IDS = {
    "T1068", "T1190", "T1203", "T1210", "T1211"
}
_NEGATED_EXPLOIT_RE = re.compile(
    r"\b(?:not|never|without)\s+exploit(?:ed|ing|s|ation)?\b|"
    r"\bbut\s+not\s+exploit(?:ed|ing|s|ation)?\b",
    re.IGNORECASE,
)


def _explicit_exploited_by_facts(source_text: str):
    """Return CVE/technique pairs from two explicit exploit fact shapes.

    Supported structures are intentionally narrow: a positive "have exploited
    the following CVEs ... [Txxxx]:" heading governing consecutive CVE bullet
    items, and an ATT&CK ``Exploit Public-Facing Application`` row whose use
    sentence explicitly lists CVEs.  Negated acquisition lists and unrelated
    techniques elsewhere in the document cannot enter either pattern.
    """
    facts = []

    for heading in _POSITIVE_EXPLOITED_CVE_LIST_RE.finditer(source_text):
        technique_match = re.search(
            r"T\d{4}(?:\.\d{3})?", heading.group("tid"), re.IGNORECASE
        )
        if technique_match is None:
            continue
        technique_start = heading.start("tid")
        technique_end = heading.end("tid")
        markers = list(
            _INLINE_UNICODE_BULLET_RE.finditer(
                source_text,
                heading.end(),
                min(len(source_text), heading.end() + 2000),
            )
        )
        previous_marker_start = None
        for marker in markers:
            if marker.start() - heading.end() > 80 and previous_marker_start is None:
                break
            if (
                previous_marker_start is not None
                and marker.start() - previous_marker_start > 500
            ):
                break
            content_start = marker.end()
            whitespace = re.match(r"\s*", source_text[content_start:])
            content_start += whitespace.end() if whitespace else 0
            cve_match = _PDF_CVE_MENTION_RE.match(source_text, content_start)
            if cve_match is None:
                break
            canonical = _canonical_cve(cve_match.group(0))
            if canonical is None:
                break
            facts.append({
                "cve_id": canonical,
                "cve_start": cve_match.start(),
                "cve_end": cve_match.end(),
                "technique_id": heading.group("tid").upper(),
                "technique_start": technique_start,
                "technique_end": technique_end,
                "evidence_start": heading.start(),
                "evidence_end": cve_match.end(),
                "scope": "positive_exploited_cve_list",
            })
            previous_marker_start = marker.start()

    for row in _EXPLOIT_PUBLIC_FACING_ROW_RE.finditer(source_text):
        sentence_end = source_text.find(".", row.end())
        if sentence_end < 0 or sentence_end + 1 - row.start() > 800:
            continue
        sentence_end += 1
        row_text = source_text[row.start():sentence_end]
        if not EXPLOIT_TERMS.search(row_text):
            continue
        for cve_match in _PDF_CVE_MENTION_RE.finditer(
            source_text, row.end(), sentence_end
        ):
            canonical = _canonical_cve(cve_match.group(0))
            if canonical is None:
                continue
            facts.append({
                "cve_id": canonical,
                "cve_start": cve_match.start(),
                "cve_end": cve_match.end(),
                "technique_id": row.group("tid").upper(),
                "technique_start": row.start("tid"),
                "technique_end": row.end("tid"),
                "evidence_start": row.start(),
                "evidence_end": sentence_end,
                "scope": "exploit_public_facing_row",
            })

    tid_pattern = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
    for technique_match in tid_pattern.finditer(source_text):
        technique_id = technique_match.group(0).upper()
        if technique_id not in _UNAMBIGUOUS_EXPLOIT_TECHNIQUE_IDS:
            continue
        item = _explicit_fact_item_at(source_text, technique_match.start())
        if item is not None:
            _, block_start, block_end = item
        else:
            bounds = _sentence_bounds_at(source_text, technique_match.start())
            if bounds is None:
                continue
            block_start, block_end = bounds
        if block_end - block_start > 800:
            continue
        block = source_text[block_start:block_end]
        if _POSITIVE_EXPLOITED_CVE_LIST_RE.search(block):
            # A flattened PDF may place the heading after the previous bullet
            # without a newline.  Only the dedicated scoped-list rule may
            # connect CVEs around such a heading; the preceding item is not in
            # its forward scope.
            continue
        cve_matches = list(_PDF_CVE_MENTION_RE.finditer(block))
        technique_matches = list(tid_pattern.finditer(block))
        if len(cve_matches) != 1 or len(technique_matches) != 1:
            continue
        if not EXPLOIT_TERMS.search(block) or _NEGATED_EXPLOIT_RE.search(block):
            continue
        cve_match = cve_matches[0]
        global_cve_start = block_start + cve_match.start()
        global_cve_end = block_start + cve_match.end()
        if min(
            abs(global_cve_start - technique_match.end()),
            abs(technique_match.start() - global_cve_end),
        ) > 160:
            continue
        canonical = _canonical_cve(cve_match.group(0))
        if canonical is None:
            continue
        facts.append({
            "cve_id": canonical,
            "cve_start": global_cve_start,
            "cve_end": global_cve_end,
            "technique_id": technique_id,
            "technique_start": technique_match.start(),
            "technique_end": technique_match.end(),
            "evidence_start": block_start,
            "evidence_end": block_end,
            "scope": "unambiguous_local_exploit_fact",
        })

    unique = {}
    for fact in facts:
        key = (
            fact["cve_start"],
            fact["cve_end"],
            fact["technique_start"],
            fact["technique_end"],
        )
        unique.setdefault(key, fact)
    return list(unique.values())


def apply_postprocess(raw):
    """执行模式、证据、规范化与去重组成的完整质量控制流程。"""
    source_text = str(raw.get("text", ""))
    fact_block_index = _build_fact_block_index(source_text) if source_text else None
    entities, old2new, key_to_id = [], {}, {}
    rejected = Counter()
    raw_entities, recovered_local_cves = _recover_local_exploited_cve_mentions(
        raw, source_text
    )
    if recovered_local_cves:
        rejected["entity_vulnerability_relation_local_recovered"] += (
            recovered_local_cves
        )
    for e in raw_entities:
        typ = e.get("type")
        if typ not in LOWER_ENT_TYPES:
            rejected["entity_illegal_type"] += 1
            continue
        if typ == "Vulnerability" and not _canonical_cve(e.get("text")):
            rejected["entity_vulnerability_without_cve"] += 1
            continue
        if (
            typ == "AttackTechnique"
            and re.fullmatch(r"TA\d{4}", str(e.get("text", "")).strip(), re.IGNORECASE)
        ):
            rejected["entity_attack_tactic_as_technique"] += 1
            continue
        if (
            typ == "Weakness"
            and str(e.get("text", "")).strip().lower()
            in CONSEQUENCE_WEAKNESS_TERMS
        ):
            rejected["entity_consequence_as_weakness"] += 1
            continue
        variants = [e]
        if typ == "Configuration" and source_text:
            variants, actions = _configuration_entity_variants(source_text, e)
            rejected.update(actions)
        for variant in variants:
            normalized_id = _normalize_norm_id(variant, source_text)
            if typ == "Weakness" and not normalized_id:
                rejected["entity_weakness_without_local_explicit_cwe"] += 1
                continue
            key = (variant["start"], variant["end"], typ)
            if key in key_to_id:
                old2new.setdefault(e["id"], []).append(key_to_id[key])
                rejected["entity_duplicate_mention"] += 1
                continue
            nid = f"E{len(entities)+1}"
            old2new.setdefault(e["id"], []).append(nid)
            key_to_id[key] = nid
            entities.append({
                "id": nid,
                "text": variant["text"],
                "type": typ,
                "start": variant["start"],
                "end": variant["end"],
                "normalized_id": normalized_id,
            })
    for old_id, new_ids in old2new.items():
        old2new[old_id] = list(dict.fromkeys(new_ids))
    type_by_id = {e["id"]: e["type"] for e in entities}
    entity_by_id = {e["id"]: e for e in entities}
    def relation_rank(relation):
        """Prefer the most local evidence when the same mention pair is duplicated."""
        head = next(
            (e for e in raw.get("entities", []) if e.get("id") == relation.get("head")),
            None,
        )
        tail = next(
            (e for e in raw.get("entities", []) if e.get("id") == relation.get("tail")),
            None,
        )
        if head is None or tail is None:
            endpoint_gap = float("inf")
        else:
            endpoint_gap = max(
                0,
                max(head.get("start", 0), tail.get("start", 0))
                - min(head.get("end", 0), tail.get("end", 0)),
            )
        evidence_length = len(str(relation.get("evidence", ""))) or float("inf")
        return (endpoint_gap, evidence_length, str(relation.get("id", "")))

    rels, seen = [], set()
    for r in sorted(raw.get("relations", []), key=relation_rank):
        head_ids = old2new.get(r["head"], [])
        tail_ids = old2new.get(r["tail"], [])
        if not head_ids or not tail_ids:
            rejected["relation_missing_or_same_endpoint"] += 1
            continue
        relation_type = r.get("type")
        expected_pair = EXTRACTION_RELATION_ARGUMENT_TYPES.get(relation_type)
        evidence = r.get("evidence")
        if source_text:
            if not isinstance(evidence, str) or not evidence:
                rejected["relation_missing_evidence"] += 1
                continue
            if evidence not in source_text:
                rejected["relation_evidence_not_in_text"] += 1
                continue
        for h in head_ids:
            for t in tail_ids:
                pair_h, pair_t = h, t
                if pair_h == pair_t:
                    rejected["relation_missing_or_same_endpoint"] += 1
                    continue
                actual_pair = (type_by_id.get(pair_h), type_by_id.get(pair_t))
                if expected_pair != actual_pair:
                    rejected["relation_illegal_type_or_direction"] += 1
                    continue
                if source_text:
                    head_text = entity_by_id[pair_h]["text"]
                    tail_text = entity_by_id[pair_t]["text"]
                    if head_text not in evidence or tail_text not in evidence:
                        rejected["relation_evidence_missing_endpoint"] += 1
                        continue
                    evidence_interval = _relation_evidence_interval(
                        source_text,
                        r,
                        entity_by_id[pair_h],
                        entity_by_id[pair_t],
                    )
                    if evidence_interval is None:
                        rejected["relation_evidence_wrong_mention"] += 1
                        continue
                    if relation_type == "exploited_by":
                        local_head, local_tail, locally_reanchored = (
                            _locally_reanchor_equivalent_pair(
                                entity_by_id[pair_h],
                                entity_by_id[pair_t],
                                evidence_interval,
                                entity_by_id,
                            )
                        )
                    else:
                        local_head = entity_by_id[pair_h]
                        local_tail = entity_by_id[pair_t]
                        locally_reanchored = False
                    if locally_reanchored:
                        pair_h, pair_t = local_head["id"], local_tail["id"]
                        rejected["relation_endpoint_local_reanchored"] += 1
                        evidence_interval = _relation_evidence_interval(
                            source_text,
                            r,
                            entity_by_id[pair_h],
                            entity_by_id[pair_t],
                        )
                        if evidence_interval is None:
                            rejected["relation_evidence_wrong_mention"] += 1
                            continue
                    if relation_type == "affects":
                        fact_block_violation = _affects_fact_block_violation(
                            source_text,
                            evidence_interval,
                            entity_by_id[pair_h],
                            entity_by_id[pair_t],
                            fact_block_index,
                        )
                        if fact_block_violation:
                            rejected[
                                f"relation_affects_{fact_block_violation}"
                            ] += 1
                            continue
                    if (
                        relation_type == "exploited_by"
                        and not EXPLOIT_TERMS.search(evidence)
                    ):
                        rejected["exploited_by_missing_exploit_trigger"] += 1
                        continue
                    if (
                        relation_type == "exploited_by"
                        and _is_explicit_post_exploitation_sequence(evidence)
                    ):
                        rejected[
                            "exploited_by_explicit_post_exploitation_sequence"
                        ] += 1
                        continue
                else:
                    evidence_interval = None
                mention_key = (relation_type, pair_h, pair_t)
                if mention_key in seen:
                    rejected["relation_duplicate_mention_pair"] += 1
                    continue
                seen.add(mention_key)
                rel = {
                    "id": f"R{len(rels)+1}",
                    "type": relation_type,
                    "head": pair_h,
                    "tail": pair_t,
                }
                if evidence:
                    rel["evidence"] = evidence
                if evidence_interval is not None:
                    rel["evidence_start"], rel["evidence_end"] = evidence_interval
                rels.append(rel)
    for fact in _explicit_exploited_by_facts(source_text):
        head_id = key_to_id.get(
            (fact["cve_start"], fact["cve_end"], "Vulnerability")
        )
        tail_id = key_to_id.get(
            (
                fact["technique_start"],
                fact["technique_end"],
                "AttackTechnique",
            )
        )
        if not head_id or not tail_id:
            continue
        fact_evidence = source_text[
            fact["evidence_start"]:fact["evidence_end"]
        ]
        if _is_explicit_post_exploitation_sequence(fact_evidence):
            rejected[
                "exploited_by_explicit_post_exploitation_sequence"
            ] += 1
            continue
        mention_key = ("exploited_by", head_id, tail_id)
        if mention_key in seen:
            continue
        seen.add(mention_key)
        evidence_start = fact["evidence_start"]
        evidence_end = fact["evidence_end"]
        rels.append({
            "id": f"R{len(rels)+1}",
            "type": "exploited_by",
            "head": head_id,
            "tail": tail_id,
            "evidence": fact_evidence,
            "evidence_start": evidence_start,
            "evidence_end": evidence_end,
        })
        rejected[
            f"relation_exploited_by_{fact['scope']}_added"
        ] += 1
    linked_configurations = {
        relation["tail"]
        for relation in rels
        if relation["type"] == "affects"
    }
    candidate_entities = [dict(entity) for entity in entities]
    filtered_entities = []
    for entity in entities:
        if (
            entity["type"] == "Configuration"
            and entity["id"] not in linked_configurations
        ):
            rejected["entity_configuration_without_affects"] += 1
            continue
        filtered_entities.append(entity)
    entities = filtered_entities
    return {
        "candidate_entities": candidate_entities,
        "entities": entities,
        "relations": rels,
        "_postprocess": {
            "version": POSTPROCESS_VERSION,
            "input_entities": len(raw.get("entities", [])),
            "candidate_entities": len(candidate_entities),
            "output_entities": len(entities),
            "input_relations": len(raw.get("relations", [])),
            "output_relations": len(rels),
            "rejected": dict(sorted(rejected.items())),
        },
    }


_CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
_TID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_REGEX_IDS = [(_CWE_RE, "Weakness"), (_TID_RE, "AttackTechnique")]


def augment_with_regex(result, text):
    """正则兜底补回 LLM 漏抽的 CWE/ATT&CK ID（不补 CVE，避免参考文献误报）。"""
    existing = {(e["start"], e["end"], e["type"]) for e in result["entities"]}
    nid = len(result["entities"]) + 1
    for rx, etype in _REGEX_IDS:
        for m in rx.finditer(text):
            key = (m.start(), m.end(), etype)
            if key in existing:
                continue
            existing.add(key)
            result["entities"].append({"id": f"E{nid}", "text": m.group(0), "type": etype,
                                       "start": m.start(), "end": m.end(), "normalized_id": m.group(0).upper()})
            nid += 1
    return result


def predict_full(text, doc_id):
    """完整方法 = Multi-stage 原始预测 + 模式约束确定性后处理，不再次调用 API。"""
    raw_p = EXP_DIR / "results" / "raw_predictions" / "v6_multipass" / f"{doc_id}.json"
    if not raw_p.exists():
        raise FileNotFoundError(
            f"缺少 Multi-stage 原始预测 {raw_p}；请先运行 --method multipass"
        )
    raw = json.loads(raw_p.read_text(encoding="utf-8"))
    result = apply_postprocess(raw)
    result["_resource"] = {
        "source_method": "multipass",
        "source_prediction": str(raw_p),
        "source_resource": raw.get("resource"),
    }
    return result


def predict_llm_apo_full(text, doc_id):
    """APO 扩展完整方法 = APO 多阶段原始预测 + 模式约束确定性后处理，不再次调用 API。"""
    raw_p = EXP_DIR / "results" / "raw_predictions" / "v6_apo" / f"{doc_id}.json"
    if not raw_p.exists():
        raise FileNotFoundError(
            f"缺少 APO 原始预测 {raw_p}；请先运行 --method apo"
        )
    raw = json.loads(raw_p.read_text(encoding="utf-8"))
    result = apply_postprocess(raw)
    result["_resource"] = {
        "source_method": "apo",
        "source_prediction": str(raw_p),
        "source_resource": raw.get("resource"),
    }
    return result
from prompts.multipass_prompts import (
    MULTIPASS_FEWSHOT,
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
)

def _has_nonempty_model_response(value: object) -> bool:
    """Return whether a cached model response can safely be reused."""
    return isinstance(value, str) and bool(value.strip())


def _multipass_window_cacheable(prediction: dict) -> bool:
    """Only successful two-stage windows may enter the reusable cache.

    A transport failure is represented downstream as an empty extraction so
    evaluation can account for it.  It must not, however, be remembered as a
    deterministic model output; otherwise a transient proxy timeout can poison
    all later APO evaluations that reuse the same window cache.
    """
    trace = prediction.get("_trace") if isinstance(prediction, dict) else None
    if not isinstance(trace, dict):
        return False
    return all(
        _has_nonempty_model_response(
            (trace.get(stage) or {}).get("raw_response")
        )
        for stage in ("stage1", "stage2")
    )


def _stage2_contract_issues(
    text: str,
    raw_relations,
    entities: list[dict],
) -> list[str]:
    """Return schema/offset violations without consulting Gold annotations."""
    entity_by_id = {
        str(entity.get("id")): entity
        for entity in entities
        if isinstance(entity, dict) and entity.get("id") is not None
    }
    issues = []
    if not isinstance(raw_relations, list):
        return ["relations_not_a_list"]
    for index, relation in enumerate(raw_relations):
        prefix = f"relation[{index}]"
        if not isinstance(relation, dict):
            issues.append(f"{prefix}:not_an_object")
            continue
        relation_type = relation.get("type")
        source_id = _pick(relation, REL_HEAD_KEYS)
        target_id = _pick(relation, REL_TAIL_KEYS)
        source = entity_by_id.get(str(source_id))
        target = entity_by_id.get(str(target_id))
        expected = EXTRACTION_RELATION_ARGUMENT_TYPES.get(relation_type)
        if expected is None:
            issues.append(f"{prefix}:illegal_relation_type")
        if source is None:
            issues.append(f"{prefix}:unknown_source_id")
        if target is None:
            issues.append(f"{prefix}:unknown_target_id")
        if expected and source and target and (
            source.get("type"), target.get("type")
        ) != tuple(expected):
            issues.append(f"{prefix}:illegal_argument_types")
        start = relation.get("evidence_start")
        end = relation.get("evidence_end")
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(text)
        ):
            issues.append(f"{prefix}:invalid_evidence_offsets")
            continue
        if source and not (
            start <= int(source["start"]) < int(source["end"]) <= end
        ):
            issues.append(f"{prefix}:evidence_missing_source_mention")
        if target and not (
            start <= int(target["start"]) < int(target["end"]) <= end
        ):
            issues.append(f"{prefix}:evidence_missing_target_mention")
    return issues


def _predict_multipass_window(
    text: str,
    examples_str: str,
    stage1_guidance: str = "",
    stage2_guidance: str = "",
    stage1_cache_path: Path | str | None = None,
) -> dict:
    """在一个局部窗口内执行实体识别和基于候选的关系判定。"""
    validate_apo_guidance(stage1_guidance, stage2_guidance)
    ext = _ext()
    user_stage1 = STAGE1_USER_PROMPT.format(examples=examples_str, text=text)
    stage1_cache = Path(stage1_cache_path) if stage1_cache_path else None
    cached_stage1 = None
    if stage1_cache is not None and stage1_cache.exists():
        try:
            cached_stage1 = json.loads(
                stage1_cache.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            cached_stage1 = None
    if (
        isinstance(cached_stage1, dict)
        and _has_nonempty_model_response(cached_stage1.get("raw_response"))
        and isinstance(cached_stage1.get("trace"), dict)
    ):
        response_stage1 = cached_stage1["raw_response"]
        trace_stage1 = dict(cached_stage1["trace"])
        trace_stage1["cache_status"] = "hit"
        trace_stage1["cache_path"] = str(stage1_cache)
    else:
        response_stage1, trace_stage1 = _call_with_retries(
            ext,
            user_stage1,
            _append_apo_guidance(
                STAGE1_SYSTEM_PROMPT,
                stage1_guidance,
                "stage1",
            ),
        )
        trace_stage1["cache_status"] = "miss"
        if (
            stage1_cache is not None
            and _has_nonempty_model_response(response_stage1)
        ):
            stage1_cache.parent.mkdir(parents=True, exist_ok=True)
            trace_stage1["cache_path"] = str(stage1_cache)
            stage1_cache.write_text(
                json.dumps(
                    {
                        "created_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "raw_response": response_stage1,
                        "trace": trace_stage1,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    parsed_stage1 = _loose_json(response_stage1)
    entities, _ = parse_entity_mentions(
        text,
        parsed_stage1.get("entities", []),
    )
    public_entities = [
        {key: value for key, value in entity.items() if not key.startswith("_")}
        for entity in entities
    ]

    user_stage2 = STAGE2_USER_PROMPT.format(
        examples=examples_str,
        text=text,
        entities=json.dumps(public_entities, ensure_ascii=False),
    )
    response_stage2, trace_stage2 = _call_with_retries(
        ext,
        user_stage2,
        _append_apo_guidance(
            STAGE2_SYSTEM_PROMPT,
            stage2_guidance,
            "stage2",
        ),
    )
    trace_stage2["cache_status"] = "miss"
    parsed_stage2 = _loose_json(response_stage2)
    original_stage2_issues = _stage2_contract_issues(
        text,
        parsed_stage2.get("relations", []),
        public_entities,
    )
    semantic_repair = {
        "attempted": False,
        "selected": "original",
        "original_issues": original_stage2_issues,
        "repaired_issues": [],
    }
    if original_stage2_issues:
        semantic_repair["attempted"] = True
        repair_prompt = f"""{user_stage2}

The previous relation JSON violated the frozen output contract.
Previous JSON:
{response_stage2}

Contract issue codes:
{json.dumps(original_stage2_issues, ensure_ascii=False)}

Return the complete corrected relations JSON. Remove a relation when it cannot
be corrected from the supplied text and entity offsets. Do not invent an
endpoint or evidence interval. Return JSON only."""
        repaired_response, repaired_trace = _call_with_retries(
            ext,
            repair_prompt,
            _append_apo_guidance(
                STAGE2_SYSTEM_PROMPT,
                stage2_guidance,
                "stage2",
            ),
        )
        repaired_stage2 = _loose_json(repaired_response)
        repaired_issues = _stage2_contract_issues(
            text,
            repaired_stage2.get("relations", []),
            public_entities,
        )
        semantic_repair["repaired_issues"] = repaired_issues
        semantic_repair["repair_raw_response"] = repaired_response
        semantic_repair["repair_trace"] = repaired_trace
        trace_stage2["attempts"] = int(trace_stage2.get("attempts", 0)) + int(
            repaired_trace.get("attempts", 0)
        )
        trace_stage2["errors"] = list(trace_stage2.get("errors", [])) + list(
            repaired_trace.get("errors", [])
        )
        if len(repaired_issues) < len(original_stage2_issues):
            parsed_stage2 = repaired_stage2
            response_stage2 = repaired_response
            semantic_repair["selected"] = "repaired"
    identity_aliases = {entity["id"]: [entity["id"]] for entity in entities}
    relations = parse_relations(
        text,
        parsed_stage2.get("relations", []),
        entities,
        identity_aliases,
    )
    return {
        "entities": public_entities,
        "relations": relations,
        "_trace": {
            "stage1": {
                **trace_stage1,
                "raw_response": response_stage1,
                "parsed_nonempty": bool(entities),
            },
            "stage2": {
                **trace_stage2,
                "raw_response": response_stage2,
                "parsed_nonempty": bool(relations),
                "semantic_repair": semantic_repair,
            },
        },
    }


def predict_llm_multipass(
    text,
    doc_id,
    examples_str=MULTIPASS_FEWSHOT,
    stage1_guidance="",
    stage2_guidance="",
    window_cache_dir=None,
    stage1_cache_dir=None,
):
    """多阶段裸预测；与单阶段共享窗口、跨度和关系端点解析规则。"""
    windows = build_text_windows(text)
    _ext()  # 在进入线程池前完成共享客户端初始化。

    cache_root = Path(window_cache_dir) if window_cache_dir else None
    stage1_root = Path(stage1_cache_dir) if stage1_cache_dir else None

    def predict_window(window):
        cache_path = None
        local = None
        stage1_cache_path = None
        if stage1_root is not None:
            stage1_fingerprint = {
                "prompt_version": STAGE1_PROMPT_VERSION,
                "runtime_config": stage1_runtime_config(),
                "examples_sha256": hashlib.sha256(
                    examples_str.encode("utf-8")
                ).hexdigest(),
                "stage1_guidance": stage1_guidance,
                "window_start": window["start"],
                "window_end": window["end"],
                "window_text_sha256": hashlib.sha256(
                    window["text"].encode("utf-8")
                ).hexdigest(),
            }
            stage1_digest = hashlib.sha256(
                json.dumps(
                    stage1_fingerprint,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            stage1_cache_path = stage1_root / f"{stage1_digest}.json"
        if cache_root is not None:
            fingerprint = {
                "prompt_version": PROMPT_VERSION,
                "runtime_config": runtime_config(),
                "examples_sha256": hashlib.sha256(
                    examples_str.encode("utf-8")
                ).hexdigest(),
                "stage1_guidance": stage1_guidance,
                "stage2_guidance": stage2_guidance,
                "window_start": window["start"],
                "window_end": window["end"],
                "window_text_sha256": hashlib.sha256(
                    window["text"].encode("utf-8")
                ).hexdigest(),
            }
            digest = hashlib.sha256(
                json.dumps(
                    fingerprint,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            cache_path = cache_root / f"{digest}.json"
            if cache_path.exists():
                try:
                    cached_window = json.loads(
                        cache_path.read_text(encoding="utf-8")
                    )["prediction"]
                except (json.JSONDecodeError, KeyError, OSError):
                    cached_window = None
                if _multipass_window_cacheable(cached_window):
                    local = json.loads(
                        json.dumps(cached_window, ensure_ascii=False)
                    )
                    local.setdefault("_trace", {})["cache_status"] = "hit"
                    local["_trace"]["cache_path"] = str(cache_path)

        if (
            local is not None
            and stage1_cache_path is not None
            and not stage1_cache_path.exists()
        ):
            stage1_trace = dict(local.get("_trace", {}).get("stage1") or {})
            raw_stage1 = stage1_trace.get("raw_response", "")
            if raw_stage1:
                stage1_cache_path.parent.mkdir(parents=True, exist_ok=True)
                stage1_cache_path.write_text(
                    json.dumps(
                        {
                            "created_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "raw_response": raw_stage1,
                            "trace": stage1_trace,
                            "backfilled_from_window_cache": str(cache_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        if local is None:
            local = _predict_multipass_window(
                window["text"],
                examples_str,
                stage1_guidance=stage1_guidance,
                stage2_guidance=stage2_guidance,
                stage1_cache_path=stage1_cache_path,
            )
            local.setdefault("_trace", {})["cache_status"] = "miss"
            if (
                cache_path is not None
                and _multipass_window_cacheable(local)
            ):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                local["_trace"]["cache_path"] = str(cache_path)
                cache_path.write_text(
                    json.dumps(
                        {
                            "created_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "doc_id": doc_id,
                            "window_start": window["start"],
                            "window_end": window["end"],
                            "prediction": local,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        print(
            f"[{doc_id}] window {window['start']}:{window['end']} "
            f"cache_{local.get('_trace', {}).get('cache_status', 'disabled')}",
            flush=True,
        )
        return _offset_prediction(local, window["start"])

    if len(windows) > 1 and LLM_MAX_WORKERS > 1:
        with ThreadPoolExecutor(
            max_workers=min(LLM_MAX_WORKERS, len(windows))
        ) as executor:
            predictions = list(executor.map(predict_window, windows))
    else:
        predictions = [predict_window(window) for window in windows]
    return merge_window_predictions(predictions)


def predict_llm_apo(text, doc_id):
    """使用仅由固定开发集选出的冻结提示执行多阶段抽取。"""
    artifact = load_apo_prompt_artifact()
    result = predict_llm_multipass(
        text,
        doc_id,
        stage1_guidance=artifact["stage1_guidance"],
        stage2_guidance=artifact["stage2_guidance"],
    )
    result["_resource"] = {
        "apo_artifact": str(APO_ARTIFACT),
        "apo_artifact_sha256": hashlib.sha256(APO_ARTIFACT.read_bytes()).hexdigest(),
        "apo_algorithm": artifact.get("algorithm"),
        "apo_score": artifact.get("score"),
        "apo_selected_round": artifact.get("selected_round"),
        "apo_selected_candidate": artifact.get("selected_candidate"),
    }
    return result

