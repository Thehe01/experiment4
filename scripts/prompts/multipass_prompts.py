"""第四章多阶段抽取提示词。

阶段 1 仅识别可由原文直接定位的下层实体；阶段 2 只判断具有文本证据的
下层关系。AttackTactic 与 KillChainPhase 由映射补全层生成，不作为文本
抽取目标。
"""

MULTIPASS_FEWSHOT = """<example>
<text>
Attackers exploited CVE-2021-44228 in the Log4j component to gain initial access by exploiting a public-facing application (T1190).
</text>
<entities>
[
  {"id":"E1","text":"CVE-2021-44228","type":"Vulnerability","start":20,"end":34,"normalized_id":"CVE-2021-44228"},
  {"id":"E2","text":"Log4j","type":"Configuration","start":42,"end":47,"normalized_id":"cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"},
  {"id":"E3","text":"T1190","type":"AttackTechnique","start":124,"end":129,"normalized_id":"T1190"}
]
</entities>
<relations>
[
  {"source":"E1","target":"E2","type":"affects","evidence_start":20,"evidence_end":57},
  {"source":"E1","target":"E3","type":"exploited_by","evidence_start":0,"evidence_end":131}
]
</relations>
</example>

<example>
<text>
CVE-2024-1000
Affected configurations:
- cpe:2.3:a:acme:gateway:*:*:*:*:*:*:*:*
</text>
<entities>
[
  {"id":"E1","text":"CVE-2024-1000","type":"Vulnerability","start":0,"end":13,"normalized_id":"CVE-2024-1000"},
  {"id":"E2","text":"cpe:2.3:a:acme:gateway:*:*:*:*:*:*:*:*","type":"Configuration","start":41,"end":79,"normalized_id":"cpe:2.3:a:acme:gateway:*:*:*:*:*:*:*:*"}
]
</entities>
<relations>
[
  {"source":"E1","target":"E2","type":"affects","evidence_start":0,"evidence_end":79}
]
</relations>
</example>

<example>
<text>
VendorCorp released an advisory mentioning CVE-2024-2000. Gateway is mentioned in a separate mitigation list.
</text>
<entities>
[
  {"id":"E1","text":"CVE-2024-2000","type":"Vulnerability","start":43,"end":56,"normalized_id":"CVE-2024-2000"}
]
</entities>
<relations>[]</relations>
</example>

<example>
<text>
CVE-2024-3000 (Acme Gateway Server and Data Center).
CVE-2024-4000 appears in a later list. Acme Gateway is mentioned as a target.
</text>
<entities>
[
  {"id":"E1","text":"CVE-2024-3000","type":"Vulnerability","start":0,"end":13,"normalized_id":"CVE-2024-3000"},
  {"id":"E2","text":"Acme Gateway","type":"Configuration","start":15,"end":27,"normalized_id":"cpe:2.3:a:acme:gateway:*:*:*:*:*:*:*:*"},
  {"id":"E3","text":"CVE-2024-4000","type":"Vulnerability","start":53,"end":66,"normalized_id":"CVE-2024-4000"}
]
</entities>
<relations>
[
  {"source":"E1","target":"E2","type":"affects","evidence_start":0,"evidence_end":27}
]
</relations>
</example>

<example>
<text>
CVE-2021-26855 is a server-side request forgery (CWE-918) vulnerability in Microsoft Exchange Server that allows an attacker to achieve Remote Code Execution.
</text>
<entities>
[
  {"id":"E1","text":"CVE-2021-26855","type":"Vulnerability","start":0,"end":14,"normalized_id":"CVE-2021-26855"},
  {"id":"E2","text":"server-side request forgery","type":"Weakness","start":20,"end":47,"normalized_id":"CWE-918"},
  {"id":"E3","text":"CWE-918","type":"Weakness","start":49,"end":56,"normalized_id":"CWE-918"},
  {"id":"E4","text":"Microsoft Exchange Server","type":"Configuration","start":75,"end":100,"normalized_id":"cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*"}
]
</entities>
<relations>
[
  {"source":"E1","target":"E4","type":"affects","evidence_start":0,"evidence_end":100},
  {"source":"E1","target":"E2","type":"instantiates","evidence_start":0,"evidence_end":57},
  {"source":"E1","target":"E3","type":"instantiates","evidence_start":0,"evidence_end":57}
]
</relations>
</example>


<example>
<text>
CVE-2024-5000 (Acme: Gateway).
</text>
<entities>
[
  {"id":"E1","text":"CVE-2024-5000","type":"Vulnerability","start":0,"end":13,"normalized_id":"CVE-2024-5000"},
  {"id":"E2","text":"Acme: Gateway","type":"Configuration","start":15,"end":28,"normalized_id":"cpe:2.3:a:acme:gateway:*:*:*:*:*:*:*:*"}
]
</entities>
<relations>
[
  {"source":"E1","target":"E2","type":"affects","evidence_start":0,"evidence_end":28}
]
</relations>
</example>"""



STAGE1_SYSTEM_PROMPT = """You are a cybersecurity intelligence annotation expert.
Perform named entity recognition using ONLY the four types below:

- Vulnerability: an explicit CVE identifier.
- Weakness: an explicit CWE mention, or a core flaw phrase only when the same
  local fact block explicitly supplies the matching CWE identifier for that
  phrase. A description-only flaw phrase without a local explicit CWE is a
  review candidate, not a formal extraction entity.
- Configuration: the concrete product, software, component, device, or full
  CPE URI that can serve as the target of a directly supported affects fact.
- AttackTechnique: annotate every explicit MITRE ATT&CK technique ID occurrence. A clearly named technique may be annotated only when no adjacent T ID is present and the name maps unambiguously to one ATT&CK ID. When a name and its T ID are adjacent, keep only the T-ID span.

Rules:
1. Every entity text MUST be an exact substring copied from the input.
2. Return 0-based start and end offsets relative to the provided input window. end is exclusive and text[start:end] MUST equal the entity text.
3. Return every explicit occurrence of CVE, CWE, and ATT&CK technique IDs. For
   Configuration, retain only the local mention needed to anchor a directly
   supported CVE-product fact. A local fact block is one sentence, one
   parenthetical CVE-product title, one table/list row, or an
   "Affected configurations" heading with its listed CPE lines when that
   heading is explicitly scoped to one CVE in the same local section. If the
   same product is tied to different CVEs, retain the locally supported mention
   for each distinct fact; do not always select the first occurrence. If the
   same normalized CVE-product fact is stated in multiple self-contained direct
   fact blocks, retain the local product mention in every such block. Omit only
   navigation, reference, alias-only, or contextual repeats that do not restate
   the direct relation locally.
4. Use the shortest independently identifiable product/component span. Remove
   versions, patches, builds, deployment qualifiers, and only non-identifying
   surrounding context. If coordinated editions share one product base and
   express one normalized product fact, keep the shared base: for example,
   retain "Acme Gateway" rather than "Acme Gateway Server and Data Center".
   When the text provides a full literal CPE URI in an affected-configuration
   list explicitly scoped to one CVE in the same local section, retain that
   exact CPE span even when no prose verb appears; prefer it over a nearby
   descriptive name for the same normalized fact, and retain distinct CPE
   product fields as distinct facts. A generation token may remain only when
   it is part of the conventional product identity required for unique
   normalization, such as SMBv1; do not retain an ordinary release number for
   that reason.
5. Do not treat impact phrases such as remote code execution, denial of service, or privilege escalation as Weakness unless the text explicitly uses them as a flaw mechanism.
6. Generic placeholders such as "multiple products", "various products", "affected systems", and "software" are NOT Configuration entities.
7. Do not use product names from navigation, tags, reference URLs, tool lists,
   mitigation lists, or general attack context unless the local text directly
   identifies that product as affected by an explicit CVE.
8. A labeled CVE-to-Vendor/Product table or list row can directly anchor an
   affects fact without a prose verb when the same row contains an explicit
   CVE and an independently identifiable product/component. Never return a
   bare vendor or publisher name as Configuration. In a "Vendor: Product"
   row, retain the identifiable product phrase from that row, not the vendor
   prefix alone. Do not use a CVE list in one sentence or row together with a
   product mentioned in another sentence or row. A statement that an actor
   did not exploit the row does not erase the CVE-product assignment.
9. Explicit negation such as unrelated, not affected, or does not affect
   blocks that CVE-product fact. A reference title is eligible only when the
   title itself states a specific CVE, product/component, and vulnerability or
   affected semantics.
10. Do not extract AttackPattern/CAPEC, AttackTactic, or KillChainPhase.
11. Configuration.normalized_id MUST be a canonical CPE 2.3 URI. Copy a full
   literal CPE; otherwise infer only the part/vendor/product identity supported
   by the named product and use * for unspecified fields. Never put a free-form
   product name in normalized_id. If vendor/product cannot be determined
   reliably, use null. Put the other standardized IDs in normalized_id when
   supported by the mention.
12. Weakness.normalized_id MUST come from a literal CWE in the entity span or
   the same local fact block. Do not infer a CWE from a general phrase, a CVE
   database lookup, or a model's background knowledge. If the local phrase can
   map to more than one CWE, omit it and leave it for review.
13. Deduplication: Extract each distinct entity mention only once per local context. Do NOT generate repetitive duplicate entries for the same entity name.

Return one valid JSON object:
{"entities":[{"id":"E1","text":"exact substring","type":"Vulnerability","start":0,"end":15,"normalized_id":"CVE-..."}]}
Return JSON only."""


STAGE1_USER_PROMPT = """Extract entities from the input text.

Reference example:
{examples}

Input text:
<text>
{text}
</text>

Return JSON only."""


STAGE2_SYSTEM_PROMPT = """You are a cybersecurity relation annotation expert.
You are given a source text and a fixed list of extracted entities. Use ONLY those entity IDs.

Allowed relations:
- affects: Vulnerability -> Configuration. The text states that the CVE affects or exists in the concrete product/component.
- instantiates: Vulnerability -> Weakness. The text states that the CVE is an instance of the flaw mechanism or CWE.
- exploited_by: Vulnerability -> AttackTechnique. The text explicitly links exploitation of the specified CVE to that ATT&CK technique as the direct vulnerability-exploitation behavior.

Evidence and boundary rules:
1. For each relation, return evidence_start and evidence_end as 0-based,
   end-exclusive offsets relative to the provided input window. The program
   will create evidence by slicing text[evidence_start:evidence_end]; do not
   copy or paraphrase the evidence text in the JSON.
2. The selected interval must be one continuous supporting span containing the
   exact surface text of both the source and target entities plus the wording
   that states the relation. It may cover multiple clauses or a full paragraph
   only when that paragraph is one fact block; do not cross a sentence, list,
   or table fact boundary. Never concatenate non-contiguous spans.
   Before returning, mechanically verify all four inequalities using the
   supplied entity offsets:
   evidence_start <= source.start, evidence_start <= target.start,
   evidence_end >= source.end, and evidence_end >= target.end.
   An identical surface string at another position does not satisfy this rule.
3. Co-occurrence is not enough. Do not create a relation merely because two entities occur in the same document.
4. For affects, both endpoints must belong to one explicit fact block: the
   same sentence, parenthetical CVE-product title, table/list row, or one
   "Affected configurations" heading with its CPE lines when the heading is
   explicitly scoped to one CVE in the same local section. A labeled
   CVE-to-Vendor/Product row is direct structural evidence when that same row
   contains both endpoints and an identifiable product/component. Do not pair
   a CVE from a nearby list or earlier sentence with a product in a different
   list item or later sentence. A product phrase followed by an explicit list
   of CVEs in that same sentence (for example, "vulnerabilities in Product:
   CVE-... and CVE-...") supports one relation for each listed CVE. Vendor-only
   or flaw-description-only cells are insufficient. Explicit unrelated/not
   affected/does not affect wording blocks the relation. A statement that an
   actor did not exploit a CVE does not negate a separately explicit
   CVE-product assignment.
   Emit one mention-level relation for every self-contained direct fact block,
   even when another block states the same normalized CVE-product fact. Do not
   globally collapse relations by normalized IDs; fact-level deduplication is
   performed separately by the evaluator and knowledge-graph projection.
5. For exploited_by, reject techniques describing reconnaissance, persistence, credential theft, command and control, or other activity that occurs before or after exploitation unless the text directly identifies that technique as the way the specified CVE is exploited. Direct positive constructions include "exploit/ing CVE by/using/via <technique>", "<technique> used to exploit CVE", and "exploitation of CVE through <technique>" when the wording makes that technique the exploitation mechanism. A technique merely mentioned in the same paragraph, or used only after a separate CVE exploitation, is not enough. For example, "exploited the CVE and then ran a RAT for C2" does not support an exploited_by edge to the command-and-control technique.
6. Do not infer upper-layer implies or belongs_to_phase relations; they are completed from controlled mappings.
7. If no single continuous supporting interval contains both endpoints and directly supports the relation, output no relation.
Return one valid JSON object:
{"relations":[{"source":"E1","target":"E2","type":"affects","evidence_start":0,"evidence_end":120}]}
Return JSON only."""


STAGE2_USER_PROMPT = """Extract evidence-supported relations between the provided entities.

Reference example:
{examples}

Input text:
<text>
{text}
</text>

Entities:
<entities>
{entities}
</entities>

Return JSON only."""
