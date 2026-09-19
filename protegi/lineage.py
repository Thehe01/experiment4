"""Prompt 演化谱系追踪与可视化导出模块 (Prompt Lineage)。

记录每个 Candidate 节点及其父子关系、突变类型与奖励，
支持导出 JSON 谱系树与 Graphviz DOT 拓扑图。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from protegi.entity_cache import compute_prompt_hash
from protegi.models import PromptCandidate


class PromptLineageTracker:
    """Prompt 谱系跟踪器。"""

    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []

    def register_candidate(
        self,
        candidate: PromptCandidate,
        gradient_text: Optional[str] = None,
    ) -> None:
        """注册或更新 Candidate；保留完整提示词和可审计文本梯度。"""
        existing = self.nodes.get(candidate.candidate_id, {})
        if gradient_text is None:
            gradient_text = existing.get("gradient_text")
        self.nodes[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "round_idx": candidate.round_idx,
            "generation_type": candidate.generation_type,
            "gradient_id": candidate.gradient_id,
            "gradient_text": gradient_text,
            "estimated_reward": round(candidate.estimated_reward, 6),
            "selection_status": candidate.selection_status,
            "num_evaluations": candidate.num_evaluations,
            "samples_seen": candidate.samples_seen,
            "tp": candidate.tp,
            "fp": candidate.fp,
            "fn": candidate.fn,
            "ucb_score": candidate.ucb_score,
            "metrics": candidate.metrics,
            "prompt_sha256": compute_prompt_hash(candidate.prompt_text),
            "prompt_text": candidate.prompt_text,
            "prompt_text_preview": candidate.prompt_text[:120].replace("\n", " ") + "...",
        }
        if candidate.parent_id:
            edge = {
                "source": candidate.parent_id,
                "target": candidate.candidate_id,
                "generation_type": candidate.generation_type,
                "gradient_id": candidate.gradient_id,
            }
            if edge not in self.edges:
                self.edges.append(edge)

    def export_json(self, output_path: Path, final_candidate_id: Optional[str] = None) -> None:
        """导出完整的谱系追踪 JSON 文件。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_to_root = []
        if final_candidate_id and final_candidate_id in self.nodes:
            curr = final_candidate_id
            while curr:
                trace_to_root.append(curr)
                curr = self.nodes.get(curr, {}).get("parent_id")
            trace_to_root.reverse()

        payload = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "final_candidate_id": final_candidate_id,
            "trace_to_root": trace_to_root,
            "nodes": self.nodes,
            "edges": self.edges,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_dot(self, output_path: Path, final_candidate_id: Optional[str] = None) -> None:
        """导出用于 Graphviz 绘图的 .dot 拓扑图文件。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["digraph PromptLineage {", '  rankdir="LR";', '  node [shape="box", style="rounded,filled", fontname="Arial"];']

        final_trace = set()
        if final_candidate_id and final_candidate_id in self.nodes:
            curr = final_candidate_id
            while curr:
                final_trace.add(curr)
                curr = self.nodes.get(curr, {}).get("parent_id")

        for cid, node in self.nodes.items():
            f1_str = f"{node['estimated_reward']:.4f}"
            label = f"{cid}\\n(R{node['round_idx']}, F1:{f1_str})\\n{node['generation_type']}"
            color = "#ffcccc" if cid == final_candidate_id else ("#e6f2ff" if cid in final_trace else "#f9f9f9")
            lines.append(f'  "{cid}" [label="{label}", fillcolor="{color}"];')

        for edge in self.edges:
            edge_style = 'color="#0066cc", penwidth=2.0' if edge["target"] in final_trace else 'color="#aaaaaa"'
            label = edge["generation_type"]
            lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [label="{label}", {edge_style}];')

        lines.append("}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
