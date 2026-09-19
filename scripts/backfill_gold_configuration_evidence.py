"""Backfill and normalize evidence for all Configuration entities in Gold annotations.

Logic:
1. Traverse all 105 Gold documents in experiments/v6/data/annotations/gold/.
2. For each Configuration entity:
   - If bound as target (tail) of one or more 'affects' relations, prioritize the
     best self-contained fact evidence from the affects relation(s), ensuring
     word boundaries are cleanly healed without mid-word cuts.
   - If not bound to an 'affects' relation, adaptively extract a self-contained
     natural language sentence from doc['text'] covering [start, end].
3. Preserve original JSON structure, formatting (indent=2, ensure_ascii=False),
   field ordering, and all other entities and relations.
4. Synchronize the 105 documents to data/annotations/gold/ if independent directory.
5. Re-freeze the v6 dataset manifest to update SHA-256 hashes.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = EXP_DIR.parent.parent
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
ROOT_GOLD_DIR = ROOT_DIR / "data" / "annotations" / "gold"


def clean_evidence_span(text: str, s: int | None, e: int | None, raw: str) -> str:
    """Heal truncated word boundaries, balance brackets/parens, and capture closing sentence punctuation."""
    if s is None or e is None or s < 0 or e > len(text) or s >= e:
        return raw.strip()

    # Expand backwards if sliced mid-word
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] in "-_"):
        s -= 1

    # Expand forwards if sliced mid-word
    while e < len(text) and (text[e].isalnum() or text[e] in "-_"):
        e += 1

    # Heal unclosed parentheses if closed within reasonable forward window (<= 80 chars)
    # without crossing newline, bullet or punctuation boundary
    curr_e = e
    while curr_e < len(text) and curr_e - e < 80:
        if text[s:curr_e].count("(") <= text[s:curr_e].count(")"):
            break
        if text[curr_e] in ("\n", "\r", "\uf0a7", "\uf0b7", "•"):
            break
        curr_e += 1
    if text[s:curr_e].count("(") == text[s:curr_e].count(")"):
        e = curr_e

    # Heal unclosed brackets if closed within reasonable forward window
    curr_e = e
    while curr_e < len(text) and curr_e - e < 80:
        if text[s:curr_e].count("[") <= text[s:curr_e].count("]"):
            break
        if text[curr_e] in ("\n", "\r", "\uf0a7", "\uf0b7", "•"):
            break
        curr_e += 1
    if text[s:curr_e].count("[") == text[s:curr_e].count("]"):
        e = curr_e

    # Trim trailing dangling artifacts like ':[' or ': [' or trailing colons
    while e > s and text[s:e].endswith((": [", ":[", ":")):
        trimmed = text[s:e].rstrip(": [")
        e = s + len(trimmed)

    # Include immediately following sentence-ending punctuation if appropriate
    if e < len(text) and text[e] in ".?!。":
        # Do not include period if it looks like a decimal number
        if not (e > 0 and text[e - 1].isdigit() and e + 1 < len(text) and text[e + 1].isdigit()):
            e += 1
            while e < len(text) and text[e] in '")]}\'':
                e += 1

    res = text[s:e].strip()
    return res if res else raw.strip()


def score_evidence(ev: str, ent_text: str) -> float:
    """Score candidate evidence string to select the most self-contained fact clause."""
    score = 0.0
    s_ev = ev.strip()
    if not s_ev:
        return -100.0

    if ent_text in ev:
        score += 50.0

    # Penalty if evidence starts with space or lower case (unless cpe:)
    if ev.startswith(" ") or (len(s_ev) > 0 and s_ev[0].islower() and not s_ev.startswith("cpe:")):
        score -= 30.0

    # Bonus for clean sentence-ending punctuation
    if s_ev.endswith((".", "!", "?", '."', ".'", ')"', "].", "。")):
        score += 20.0
    elif s_ev.endswith("\n"):
        score += 10.0

    # Length preferences: prefer concise, self-contained clauses (50..350 chars)
    if 50 <= len(ev) <= 350:
        score += 25.0
    elif len(ev) > 500:
        score -= 20.0

    score += min(len(ev), 350) * 0.1
    return score


def extract_sentence_for_span(text: str, start: int, end: int) -> str:
    """Adaptively extract a complete self-contained natural language sentence."""
    if start < 0 or end > len(text) or start >= end:
        return text[max(0, start):min(len(text), end)].strip()

    # Search backwards for sentence start
    left = start
    while left > 0:
        c = text[left - 1]
        if c == "\n":
            break
        if c in ".?!。;；":
            is_abbrev = False
            if left > 1 and text[left - 2].isdigit() and left < len(text) and text[left].isdigit():
                is_abbrev = True
            elif left >= 4 and text[left - 4:left] in ("U.S.", "e.g.", "i.e."):
                is_abbrev = True
            elif c in ";；":
                amp_idx = text.rfind("&", max(0, left - 10), left)
                if amp_idx != -1 and text[amp_idx:left].isalnum():
                    is_abbrev = True
            if not is_abbrev:
                if left < len(text) and text[left] in " \t\r\n\"'":
                    break
        left -= 1

    # Expand backwards if sliced mid-word
    while left > 0 and (text[left - 1].isalnum() or text[left - 1] in "-_"):
        left -= 1

    while left < start and text[left] in " \t\r\n•-*\uf0a7\uf0b7":
        left += 1

    # Search forwards for sentence end
    right = end
    while right < len(text):
        c = text[right]
        if c == "\n":
            break
        if c in ".?!。;；":
            is_abbrev = False
            if right > 0 and text[right - 1].isdigit() and right + 1 < len(text) and text[right + 1].isdigit():
                is_abbrev = True
            elif right >= 3 and text[right - 3:right + 1] in ("U.S.", "e.g.", "i.e."):
                is_abbrev = True
            elif c in ";；":
                amp_idx = text.rfind("&", max(0, right - 8), right)
                if amp_idx != -1 and text[amp_idx + 1:right].isalnum():
                    is_abbrev = True
            if not is_abbrev:
                right += 1
                while right < len(text) and text[right] in "\"\')]}":
                    right += 1
                break
        right += 1

    # Expand forwards if sliced mid-word
    while right < len(text) and (text[right].isalnum() or text[right] in "-_"):
        right += 1

    # Expand right to heal unclosed parentheses or brackets within [left, right]
    curr_r = right
    while curr_r < len(text) and curr_r - right < 80:
        if text[left:curr_r].count("(") <= text[left:curr_r].count(")"):
            break
        if text[curr_r] in ("\n", "\r", "\uf0a7", "\uf0b7", "•"):
            break
        curr_r += 1
    if text[left:curr_r].count("(") == text[left:curr_r].count(")"):
        right = curr_r

    curr_r = right
    while curr_r < len(text) and curr_r - right < 80:
        if text[left:curr_r].count("[") <= text[left:curr_r].count("]"):
            break
        if text[curr_r] in ("\n", "\r", "\uf0a7", "\uf0b7", "•"):
            break
        curr_r += 1
    if text[left:curr_r].count("[") == text[left:curr_r].count("]"):
        right = curr_r

    return text[left:right].strip()


def select_best_evidence(doc: Dict[str, Any], ent: Dict[str, Any]) -> Tuple[str, str]:
    """Select the best evidence for a Configuration entity.

    Returns (evidence, source_type), where source_type is 'affects' or 'adaptive_sentence'.
    """
    eid = ent["id"]
    ent_text = ent.get("text", "")
    text = doc.get("text", "")

    affects_rels = [
        r for r in doc.get("relations", [])
        if r.get("type") == "affects" and r.get("tail") == eid and r.get("evidence")
    ]

    if affects_rels:
        scored: List[Tuple[float, str]] = []
        for r in affects_rels:
            raw_ev = r["evidence"]
            cleaned_ev = clean_evidence_span(
                text, r.get("evidence_start"), r.get("evidence_end"), raw_ev
            )
            sc = score_evidence(cleaned_ev, ent_text)
            scored.append((sc, cleaned_ev))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], "affects"

    # Fallback to adaptive sentence extraction
    extracted = extract_sentence_for_span(text, ent.get("start", 0), ent.get("end", 0))
    return extracted, "adaptive_sentence"


def update_entity_with_evidence(ent: Dict[str, Any], evidence: str) -> Dict[str, Any]:
    """Update entity with evidence field preserving existing key order."""
    new_ent: Dict[str, Any] = {}
    had_evidence = "evidence" in ent

    for k, v in ent.items():
        if k == "evidence":
            new_ent["evidence"] = evidence
        elif k == "notes" and not had_evidence:
            new_ent["evidence"] = evidence
            new_ent["notes"] = v
        else:
            new_ent[k] = v

    if "evidence" not in new_ent:
        new_ent["evidence"] = evidence

    return new_ent


def process_gold_annotations() -> Dict[str, Any]:
    gold_files = sorted([f for f in os.listdir(GOLD_DIR) if f.endswith(".json")])
    stats = {
        "documents_processed": len(gold_files),
        "configuration_entities_updated": 0,
        "source_affects_count": 0,
        "source_adaptive_sentence_count": 0,
    }

    for fname in gold_files:
        path = GOLD_DIR / fname
        with open(path, "r", encoding="utf-8") as fp:
            doc = json.load(fp)

        modified = False
        new_entities = []
        for ent in doc.get("entities", []):
            if ent.get("type") == "Configuration":
                best_ev, source = select_best_evidence(doc, ent)
                updated_ent = update_entity_with_evidence(ent, best_ev)
                new_entities.append(updated_ent)
                modified = True
                stats["configuration_entities_updated"] += 1
                if source == "affects":
                    stats["source_affects_count"] += 1
                else:
                    stats["source_adaptive_sentence_count"] += 1
            else:
                new_entities.append(ent)

        doc["entities"] = new_entities

        if modified:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(doc, fp, ensure_ascii=False, indent=2)
                fp.write("\n")

    # Sync to ROOT_GOLD_DIR if it exists and is independent
    if ROOT_GOLD_DIR.exists() and not os.path.samefile(GOLD_DIR, ROOT_GOLD_DIR):
        print(f"Synchronizing 105 Gold files to {ROOT_GOLD_DIR}...")
        for fname in gold_files:
            src = GOLD_DIR / fname
            dst = ROOT_GOLD_DIR / fname
            shutil.copy2(src, dst)
        stats["root_gold_synced"] = True

    return stats


def main() -> None:
    print(f"Processing Gold annotations in {GOLD_DIR}...")
    stats = process_gold_annotations()
    print("Backfill completed with stats:", json.dumps(stats, indent=2))

    # Re-freeze manifest
    freeze_script = EXP_DIR / "scripts" / "freeze_dataset_manifest_v6.py"
    print(f"Running freeze script: {freeze_script}...")
    import subprocess
    import sys
    subprocess.run([sys.executable, str(freeze_script)], check=True)
    print("Manifest re-frozen successfully.")


if __name__ == "__main__":
    main()
