"""ProTeGi 的冻结任务契约与初始可优化 guidance。

标签语义、边界、示例和 JSON schema 属于实验契约，不参与提示词搜索。
ProTeGi 只能改写 ``<OPTIMIZABLE_GUIDANCE>`` 中的执行策略，避免模型通过
重定义 Gold 边界或复制训练样本来提高开发集分数。
"""

from __future__ import annotations

import re


IMMUTABLE_START = "<IMMUTABLE_CONTRACT>"
IMMUTABLE_END = "</IMMUTABLE_CONTRACT>"
GUIDANCE_START = "<OPTIMIZABLE_GUIDANCE>"
GUIDANCE_END = "</OPTIMIZABLE_GUIDANCE>"


ENTITY_FEWSHOT_P0 = """<example>
<text>
Attackers exploited CVE-2021-44228 in the Log4j component to gain initial access by exploiting a public-facing application (T1190).
</text>
<entities>
[
  {"id": "E1", "text": "CVE-2021-44228", "type": "Vulnerability", "start": 20, "end": 34, "normalized_id": "CVE-2021-44228"},
  {"id": "E2", "text": "Log4j", "type": "Configuration", "start": 42, "end": 47, "normalized_id": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"},
  {"id": "E3", "text": "T1190", "type": "AttackTechnique", "start": 124, "end": 129, "normalized_id": "T1190"}
]
</entities>
</example>

<example>
<text>
CVE-2021-26855 is a server-side request forgery (CWE-918) vulnerability in Microsoft Exchange Server.
</text>
<entities>
[
  {"id": "E1", "text": "CVE-2021-26855", "type": "Vulnerability", "start": 0, "end": 14, "normalized_id": "CVE-2021-26855"},
  {"id": "E2", "text": "server-side request forgery", "type": "Weakness", "start": 20, "end": 47, "normalized_id": "CWE-918"},
  {"id": "E3", "text": "CWE-918", "type": "Weakness", "start": 49, "end": 56, "normalized_id": "CWE-918"},
  {"id": "E4", "text": "Microsoft Exchange Server", "type": "Configuration", "start": 75, "end": 100, "normalized_id": "cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*"}
]
</entities>
</example>"""


RELATION_FEWSHOT_P0 = """<example>
<text>
Attackers exploited CVE-2021-44228 in the Log4j component to gain initial access by exploiting a public-facing application (T1190).
</text>
<entities>
[
  {"id": "E1", "text": "CVE-2021-44228", "type": "Vulnerability", "start": 20, "end": 34},
  {"id": "E2", "text": "Log4j", "type": "Configuration", "start": 42, "end": 47},
  {"id": "E3", "text": "T1190", "type": "AttackTechnique", "start": 124, "end": 129}
]
</entities>
<relations>
[
  {"source": "E1", "target": "E2", "type": "affects", "evidence_start": 20, "evidence_end": 47},
  {"source": "E1", "target": "E3", "type": "exploited_by", "evidence_start": 0, "evidence_end": 131}
]
</relations>
</example>

<example>
<text>
CVE-2021-26855 is a server-side request forgery (CWE-918) vulnerability in Microsoft Exchange Server.
</text>
<entities>
[
  {"id": "E1", "text": "CVE-2021-26855", "type": "Vulnerability", "start": 0, "end": 14},
  {"id": "E2", "text": "server-side request forgery", "type": "Weakness", "start": 20, "end": 47},
  {"id": "E3", "text": "CWE-918", "type": "Weakness", "start": 49, "end": 56},
  {"id": "E4", "text": "Microsoft Exchange Server", "type": "Configuration", "start": 75, "end": 100}
]
</entities>
<relations>
[
  {"source": "E1", "target": "E4", "type": "affects", "evidence_start": 0, "evidence_end": 100},
  {"source": "E1", "target": "E2", "type": "instantiates", "evidence_start": 0, "evidence_end": 47},
  {"source": "E1", "target": "E3", "type": "instantiates", "evidence_start": 0, "evidence_end": 56}
]
</relations>
</example>"""


ENTITY_IMMUTABLE_CONTRACT = f"""BOUNDARY_CONTRACT_VERSION: chapter3-boundary-sync-v2
TASK_STAGE: entity-only

Extract only these four entity types:
- Vulnerability: every explicit CVE identifier mention. Preserve PDF spacing or hyphen breaks in text and normalize the identifier.
- Weakness: an explicit CWE identifier, or a descriptive flaw phrase uniquely tied to an explicit CWE in the same local fact block. A description-only phrase without a locally explicit CWE is out of scope. Never infer a CWE from model knowledge or an external CVE-CWE mapping.
- Configuration: the Minimum Canonical Product Unit (MCPU) that can serve as the tail of a directly supported Vulnerability-affects->Configuration fact. Use the shortest continuous source span that preserves the canonical affected product, component, device, or complete CPE identity. Retain an explicitly adjacent vendor when it belongs to the same continuous product noun phrase, but never cross a table column, conjunction, list item, sentence, or non-contiguous text to add a vendor. Retain Server, Gateway, Manager, Appliance, Controller, Service, Driver and similar designators only when they belong to the official product or directly affected component identity; remove purely descriptive class nouns. Exclude ordinary release versions, updates, patches and builds. Preserve a full literal CPE URI and a lexicalized generation or device-model token required for product identity. Exclude incidental tools, services, filenames, IP addresses, ports, and generic product placeholders.
- AttackTechnique: an explicit ATT&CK T-code or a technique name that is uniquely normalizable from the local text. When a name and its T-code are adjacent, retain only the T-code. TA tactic IDs and S/G/C identifiers are out of scope.

Entity rules:
1. Entity text must be an exact substring of the input.
2. start/end are 0-based offsets and end is exclusive; text[start:end] must equal entity text.
3. Extract every distinct explicit mention; do not duplicate the same type and span.
4. normalized_id uses the canonical CVE, locally explicit CWE, verified CPE 2.3 URI, or ATT&CK T-code; otherwise null. For an ordinary prose Configuration mention, normalize unspecified release fields to * at family level. A non-wildcard version is permitted only when copied from a full literal CPE URI or required by a lexicalized generation/model identity.
5. Output entities only. Stage 1 must not extract, define, or output relations.
6. Return exactly these entity fields: id, text, type, start, end, normalized_id.

Frozen reference examples:
{ENTITY_FEWSHOT_P0}

Input text:
<text>
{{text}}
</text>

Return one JSON object only:
{{"entities": [{{"id": "E1", "text": "exact substring", "type": "Vulnerability", "start": 0, "end": 14, "normalized_id": "CVE-..."}}]}}"""


RELATION_IMMUTABLE_CONTRACT = f"""BOUNDARY_CONTRACT_VERSION: chapter3-boundary-sync-v2
TASK_STAGE: relation-only

The input contains source text and a frozen list of predicted entities. Extract only:
- affects: Vulnerability -> Configuration. The text directly states that the CVE affects, exists in, or applies to the product/component.
- instantiates: Vulnerability -> Weakness. The CVE and weakness are directly linked in the same fact block; an external CVE-CWE lookup is not evidence.
- exploited_by: Vulnerability -> AttackTechnique. The technique directly performs the named CVE exploitation or entry behavior; post-exploitation execution, persistence, privilege escalation, credential access, lateral movement, and C2 are excluded.

Relation rules:
1. Use only supplied entity IDs as source and target. Never create entities.
2. Co-occurrence is insufficient. Every relation requires direct local evidence.
3. evidence_start/evidence_end are 0-based, end-exclusive offsets for one continuous supporting span containing both endpoints and the relation wording.
4. Return an empty list when no relation is directly supported.
5. Return exactly these relation fields: source, target, type, evidence_start, evidence_end.

Frozen reference examples:
{RELATION_FEWSHOT_P0}

Input text:
<text>
{{text}}
</text>

Frozen predicted entities:
<entities>
{{entities}}
</entities>

Return one JSON object only:
{{"relations": [{{"source": "E1", "target": "E2", "type": "affects", "evidence_start": 0, "evidence_end": 100}}]}}"""


ENTITY_GUIDANCE_P0 = """Scan the entire input, including prose, tables, lists, captions, and line-broken identifiers. Apply the frozen definitions conservatively and check every emitted span against the source before returning JSON."""

RELATION_GUIDANCE_P0 = """Evaluate each eligible entity pair against one local evidence block. Prefer an empty relation list over a relation supported only by proximity or external knowledge."""


def build_prompt(immutable_contract: str, guidance: str) -> str:
    """Compose one task prompt with a byte-stable immutable contract."""
    return (
        f"{IMMUTABLE_START}\n{immutable_contract.strip()}\n{IMMUTABLE_END}\n\n"
        f"{GUIDANCE_START}\n{guidance.strip()}\n{GUIDANCE_END}\n\n"
        "Follow the immutable contract and the operational guidance. Output JSON only."
    )


def extract_immutable_contract(prompt_text: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(IMMUTABLE_START)}\s*(.*?)\s*{re.escape(IMMUTABLE_END)}",
        re.DOTALL,
    )
    match = pattern.search(prompt_text)
    return match.group(1).strip() if match else None


def extract_optimizable_guidance(prompt_text: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(GUIDANCE_START)}\s*(.*?)\s*{re.escape(GUIDANCE_END)}",
        re.DOTALL,
    )
    match = pattern.search(prompt_text)
    return match.group(1).strip() if match else None


def replace_optimizable_guidance(prompt_text: str, guidance: str) -> str:
    """Replace only the optimizable block; fail closed if the block is absent."""
    pattern = re.compile(
        rf"({re.escape(GUIDANCE_START)})\s*.*?\s*({re.escape(GUIDANCE_END)})",
        re.DOTALL,
    )
    if len(pattern.findall(prompt_text)) != 1:
        raise ValueError("prompt must contain exactly one optimizable guidance block")
    clean = guidance.strip()
    return pattern.sub(lambda m: f"{m.group(1)}\n{clean}\n{m.group(2)}", prompt_text)


ENTITY_PROMPT_P0 = build_prompt(ENTITY_IMMUTABLE_CONTRACT, ENTITY_GUIDANCE_P0)
RELATION_PROMPT_P0 = build_prompt(RELATION_IMMUTABLE_CONTRACT, RELATION_GUIDANCE_P0)
