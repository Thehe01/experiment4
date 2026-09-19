"""ProTeGi 优化模型提示词模板定义。

模块同时提供两个预注册实验臂：

``constrained``
    冻结任务契约，只允许批评和改写 operational guidance。
``unconstrained``
    允许批评和改写完整语义提示（包括任务定义与 few-shot 示例），但要求
    保留评价程序所需的输入占位符和 JSON 接口。
"""

from __future__ import annotations

GRADIENT_TEMPLATE = """I'm optimizing the operational guidance for a cybersecurity knowledge extraction task.

The complete current prompt is shown for context:

<START_PROMPT>
{prompt}
<END_PROMPT>

The prompt produced incorrect outputs on the following examples:

{error_examples}

Give {num_feedbacks} different reasons why the current prompt may have caused these errors.

The IMMUTABLE_CONTRACT block is the frozen experimental contract. It is not editable.
Focus only on deficiencies, ambiguities, missing decision procedures, or unclear instructions
inside OPTIMIZABLE_GUIDANCE. Generalize across the error group instead of copying an input,
Gold span, identifier, or reference-example answer into the guidance.

Do not simply solve each individual example.

Each reason should describe a direction in which the prompt could be improved.

Wrap every reason with:
<START>
...
<END>"""

EDIT_TEMPLATE = """I'm optimizing only the operational guidance of a cybersecurity knowledge extraction prompt.

Frozen immutable contract (read-only):

<IMMUTABLE_CONTRACT>
{immutable_contract}
</IMMUTABLE_CONTRACT>

Current editable guidance:

<OPTIMIZABLE_GUIDANCE>
{guidance}
</OPTIMIZABLE_GUIDANCE>

The prompt produced the following errors:

{error_examples}

The identified deficiency is:

{gradient}

Rewrite only the editable guidance to address this deficiency.

Requirements:
1. Return only revised operational guidance, not a complete prompt.
2. Do not repeat, reinterpret, weaken, or extend the immutable contract.
3. Do not define entity/relation labels, directions, JSON fields, placeholders, or examples.
4. Do not copy sample-specific text, identifiers, spans, or Gold answers.
5. Give an executable decision/checking procedure rather than commentary about the prompt.
6. Keep the guidance concise and generally applicable to unseen documents.

Wrap the revised guidance with:
<START>
...
<END>"""

PARAPHRASE_TEMPLATE = """Generate one semantic variation of the editable operational guidance below.

You must preserve:
- the decision procedure and its strictness;
- every inclusion/exclusion preference expressed by the guidance.

Do not define or repeat the immutable task contract, label schema, relation directions,
JSON fields, input placeholders, examples, identifiers, spans, or sample-specific answers.
Do not add a new extraction target or relax evidence requirements.

Editable guidance:

<OPTIMIZABLE_GUIDANCE>
{guidance}
</OPTIMIZABLE_GUIDANCE>

Return only the paraphrased guidance wrapped with:
<START>
...
<END>"""


UNCONSTRAINED_GRADIENT_TEMPLATE = """I'm optimizing the complete prompt for a cybersecurity knowledge extraction task.

The current prompt is:

<START_PROMPT>
{prompt}
<END_PROMPT>

The prompt produced incorrect outputs on the following examples:

{error_examples}

Give {num_feedbacks} different reasons why the complete prompt may have caused these errors.

This is the unconstrained semantic-prompt arm: task wording, label explanations, decision
rules, and reference examples may all be criticized. Generalize across the error group
instead of copying an input, Gold span, identifier, or answer.

The runtime interface is still fixed: the rewritten prompt must retain the literal input
placeholders and the JSON field names expected by the evaluator. Do not propose changing
the evaluator, Gold annotations, data split, metric, or output parser.

Do not simply solve each individual example.

Each reason should describe a direction in which the complete prompt could be improved.

Wrap every reason with:
<START>
...
<END>"""


UNCONSTRAINED_EDIT_TEMPLATE = """I'm optimizing the complete semantic prompt for a cybersecurity knowledge extraction task.

Current complete prompt:

<CURRENT_PROMPT>
{prompt}
</CURRENT_PROMPT>

The prompt produced the following errors:

{error_examples}

The identified deficiency is:

{gradient}

Rewrite the complete prompt to address this deficiency. You may revise task wording,
label explanations, decision rules, and reference examples.

Runtime-interface requirements:
1. Return one complete prompt, not commentary or a patch.
2. Preserve every literal input placeholder listed here: {required_placeholders}.
3. Require one JSON object whose top-level key and item fields remain: {required_output_fields}.
4. Do not copy sample-specific text, identifiers, spans, or Gold answers from the error examples.
5. Do not mention the optimization process, Train/Dev/Test, Gold labels, or evaluation scores
   in the rewritten task prompt.
6. Keep the result generally applicable to unseen documents.

Wrap the complete rewritten prompt with:
<START>
...
<END>"""


UNCONSTRAINED_PARAPHRASE_TEMPLATE = """Generate one semantic variation of the complete extraction prompt below.

This is the unconstrained semantic-prompt arm: wording, organization, label explanations,
decision rules, and reference examples may be changed. Preserve the extraction task's
overall purpose, but do not assume any block is immutable.

Runtime-interface requirements:
1. Preserve every literal input placeholder listed here: {required_placeholders}.
2. Require one JSON object whose top-level key and item fields remain: {required_output_fields}.
3. Do not introduce text from hidden error examples, optimization metadata, or evaluation scores.

Complete prompt:

<CURRENT_PROMPT>
{prompt}
</CURRENT_PROMPT>

Return only the complete paraphrased prompt wrapped with:
<START>
...
<END>"""
