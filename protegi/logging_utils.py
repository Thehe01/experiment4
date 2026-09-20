"""ProTeGi 实验日志与中间产物写入模块。

每轮持久化记录：
- round_X/beam.json
- round_X/gradients.json
- round_X/error_examples.json
- round_X/gradient_minibatches.json
- round_X/candidates.json
- round_X/generated_candidates.json
- round_X/selector_history.json
- optimization_curve.csv
- final_dev_evaluations.json / final_beam_dev.json
- summary.json
- artifact_manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from protegi.models import CallStats, PromptCandidate, PromptGradient


class ProTeGiLogger:
    """ProTeGi 实验日志与过程归档管理器。"""

    def __init__(self, output_dir: Path, stage: str, method: str):
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.method = method
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.curve_csv_path = self.output_dir / "optimization_curve.csv"
        self._init_curve_csv()

    def _init_curve_csv(self) -> None:
        """若 optimization_curve.csv 不存在，则初始化表头。"""
        if not self.curve_csv_path.exists():
            with open(self.curve_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage",
                    "method",
                    "round",
                    "best_score",
                    "mean_beam_score",
                    "num_generated_candidates",
                    "num_evaluated_candidates",
                    "task_model_calls",
                    "optimizer_model_calls",
                ])

    def log_round(
        self,
        round_idx: int,
        beam: List[PromptCandidate],
        candidates: List[PromptCandidate],
        gradients: List[PromptGradient],
        selector_history: Optional[List[dict]] = None,
        call_stats: Optional[CallStats] = None,
        generated_candidates: Optional[List[PromptCandidate]] = None,
        error_examples: Optional[List[dict]] = None,
        gradient_minibatches: Optional[List[dict]] = None,
        stability: Optional[dict] = None,
    ) -> Path:
        """持久化保存单轮所有搜索与评估产物。"""
        round_dir = self.output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. beam.json
        beam_data = [c.to_dict() for c in beam]
        (round_dir / "beam.json").write_text(
            json.dumps(beam_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 2. candidates.json
        cands_data = [c.to_dict() for c in candidates]
        (round_dir / "candidates.json").write_text(
            json.dumps(cands_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 2b. 所有生成项，包括未抽样与契约拒绝项。
        generated_source = generated_candidates if generated_candidates is not None else candidates
        generated_data = [c.to_dict() for c in generated_source]
        (round_dir / "generated_candidates.json").write_text(
            json.dumps(generated_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 3. gradients.json
        grads_data = [g.to_dict() for g in gradients]
        (round_dir / "gradients.json").write_text(
            json.dumps(grads_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        (round_dir / "error_examples.json").write_text(
            json.dumps(error_examples or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (round_dir / "gradient_minibatches.json").write_text(
            json.dumps(gradient_minibatches or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 4. selector_history.json
        if selector_history is not None:
            (round_dir / "selector_history.json").write_text(
                json.dumps(selector_history, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # 4b. stability.json（运行稳定性快照，加性文件）。
        if stability is not None:
            (round_dir / "stability.json").write_text(
                json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # 5. optimization_curve.csv 追加记录
        best_score = max((c.estimated_reward for c in beam), default=0.0)
        mean_beam_score = (
            sum(c.estimated_reward for c in beam) / len(beam) if beam else 0.0
        )
        task_calls = call_stats.task_model_calls if call_stats else 0
        opt_calls = call_stats.optimizer_model_calls if call_stats else 0

        with open(self.curve_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.stage,
                self.method,
                round_idx,
                round(best_score, 6),
                round(mean_beam_score, 6),
                len(generated_source),
                len(candidates),
                task_calls,
                opt_calls,
            ])

        return round_dir

    def prune_rounds_from(self, round_idx: int) -> None:
        """Resume 支持：删除 >= round_idx 的轮次目录与曲线行，避免 redo 重复。

        检查点本身与 eval_cache.json 不受影响。
        """
        for child in sorted(self.output_dir.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("round_"):
                continue
            try:
                number = int(name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if number >= round_idx:
                for path in sorted(child.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                child.rmdir()
        if not self.curve_csv_path.is_file():
            return
        with open(self.curve_csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header, data = rows[0], rows[1:]
        kept = [row for row in data if not (len(row) > 2 and row[2].isdigit() and int(row[2]) >= round_idx)]
        with open(self.curve_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)

    def save_final_prompt(self, prompt_text: str, filename: str = "final_prompt.txt") -> Path:
        """保存最终选定的最优提示词文本。"""
        target_path = self.output_dir / filename
        with open(target_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(prompt_text.replace("\r\n", "\n").replace("\r", "\n"))
        return target_path

    def save_json(self, payload: Any, filename: str) -> Path:
        """保存通用 JSON 审计产物。"""
        target_path = self.output_dir / filename
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target_path

    def save_summary(self, summary_data: dict, filename: str = "summary.json") -> Path:
        """保存搜索总结元数据。"""
        target_path = self.output_dir / filename
        target_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target_path

    def create_artifact_manifest(
        self,
        *,
        final_prompt_filename: str,
        canonical_prompt_sha256: str,
    ) -> Path:
        """最后写入全目录原始字节哈希，封存本次实验的完整可追溯产物。"""
        manifest_path = self.output_dir / "artifact_manifest.json"
        files: Dict[str, dict] = {}
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path == manifest_path:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rel_path = path.relative_to(self.output_dir).as_posix()
            files[rel_path] = {
                "sha256_raw_bytes": digest,
                "size_bytes": path.stat().st_size,
            }
        payload = {
            "stage": self.stage,
            "method": self.method,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "final_prompt_file": final_prompt_filename,
            "final_prompt_sha256_canonical_lf": canonical_prompt_sha256,
            "files": files,
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest_path
