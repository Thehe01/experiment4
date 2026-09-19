"""第四章实验使用的正式本体约束。

第三章正式版定义 6 类实体、5 类关系。第四章的文本抽取层只负责
Configuration、Vulnerability、Weakness、AttackTechnique 及三类具有
原文证据的关系；AttackTactic、KillChainPhase 和对应关系由权威映射
与本文阶段对齐规则补全。
"""

from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "chapter3-no-capec-v1"
ANNOTATION_PROTOCOL_VERSION: Final[str] = "4.6-mcpu-mention-fact-dual-layer-v1"
BOUNDARY_CONTRACT_VERSION: Final[str] = "chapter3-boundary-sync-v2"

ENTITY_TYPES: Final[set[str]] = {
    "Configuration",
    "Vulnerability",
    "Weakness",
    "AttackTechnique",
    "AttackTactic",
    "KillChainPhase",
}

RELATION_TYPES: Final[set[str]] = {
    "affects",
    "instantiates",
    "exploited_by",
    "implies",
    "belongs_to_phase",
}

RELATION_ARGUMENT_TYPES: Final[dict[str, tuple[str, str]]] = {
    "affects": ("Vulnerability", "Configuration"),
    "instantiates": ("Vulnerability", "Weakness"),
    "exploited_by": ("Vulnerability", "AttackTechnique"),
    "implies": ("AttackTechnique", "AttackTactic"),
    "belongs_to_phase": ("AttackTactic", "KillChainPhase"),
}

EXTRACTION_ENTITY_TYPES: Final[set[str]] = {
    "Configuration",
    "Vulnerability",
    "Weakness",
    "AttackTechnique",
}

EXTRACTION_RELATION_TYPES: Final[set[str]] = {
    "affects",
    "instantiates",
    "exploited_by",
}

EXTRACTION_RELATION_ARGUMENT_TYPES: Final[dict[str, tuple[str, str]]] = {
    relation: RELATION_ARGUMENT_TYPES[relation]
    for relation in EXTRACTION_RELATION_TYPES
}

MAPPING_RELATION_TYPES: Final[set[str]] = {
    "implies",
    "belongs_to_phase",
}
