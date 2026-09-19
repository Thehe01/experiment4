"""Comprehensive verification script for Gold Configuration evidence backfill."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = EXP_DIR.parent.parent
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
ROOT_GOLD_DIR = ROOT_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
MANIFEST_FILE = EXP_DIR / "data" / "dataset_freeze_manifest_v6.json"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_all() -> None:
    print("=== STARTING COMPREHENSIVE GOLD EVIDENCE VERIFICATION ===")

    # 1. Check split and document count
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    all_split_docs = split["train"] + split["dev"] + split["test"]
    assert len(all_split_docs) == 105, f"Expected 105 docs in split, got {len(all_split_docs)}"

    gold_files = sorted([f for f in os.listdir(GOLD_DIR) if f.endswith(".json")])
    assert len(gold_files) == 105, f"Expected 105 json files in {GOLD_DIR}, got {len(gold_files)}"

    # 2. Check root gold sync
    if ROOT_GOLD_DIR.exists() and not os.path.samefile(GOLD_DIR, ROOT_GOLD_DIR):
        print(f"Verifying root gold sync in {ROOT_GOLD_DIR}...")
        for fname in gold_files:
            v6_path = GOLD_DIR / fname
            root_path = ROOT_GOLD_DIR / fname
            assert root_path.is_file(), f"Missing file in root gold: {root_path}"
            assert sha256_file(v6_path) == sha256_file(root_path), f"Hash mismatch for {fname}"
        print("Root gold sync verified: 105/105 files identical.")

    # 3. Check entity counts, relation counts, and Configuration evidence
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    config_entities_checked = 0
    hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()

    for fname in gold_files:
        path = GOLD_DIR / fname
        doc_id = fname[:-5]
        h = sha256_file(path)
        hashes[doc_id] = h

        with open(path, "r", encoding="utf-8") as fp:
            doc = json.load(fp)

        # Check required doc keys
        for required_key in ("doc_id", "schema_version", "annotation_protocol_version", "text", "entities", "relations"):
            assert required_key in doc, f"Missing key '{required_key}' in {fname}"

        text = doc["text"]

        # Check relations integrity
        for rel in doc.get("relations", []):
            rel_type = rel["type"]
            relation_counts[rel_type] += 1
            assert "id" in rel and "head" in rel and "tail" in rel, f"Malformed relation in {fname}: {rel}"
            assert "evidence" in rel and "evidence_start" in rel and "evidence_end" in rel, f"Missing evidence in relation {rel['id']} in {fname}"

        # Check entities integrity
        for ent in doc.get("entities", []):
            etype = ent["type"]
            entity_counts[etype] += 1
            assert "id" in ent and "text" in ent and "start" in ent and "end" in ent, f"Malformed entity in {fname}: {ent}"

            # Verify offsets match text
            s, e = ent["start"], ent["end"]
            assert text[s:e] == ent["text"], f"Span mismatch in {fname} entity {ent['id']}: text[{s}:{e}]={repr(text[s:e])} != {repr(ent['text'])}"

            if etype == "Configuration":
                config_entities_checked += 1
                assert "evidence" in ent, f"Missing evidence field in {fname} entity {ent['id']}"
                ev = ent["evidence"]
                assert isinstance(ev, str), f"Evidence is not a string in {fname} entity {ent['id']}"
                assert len(ev.strip()) > 0, f"Evidence is empty in {fname} entity {ent['id']}"

                # Verify no mechanical evidence == text
                t = ent["text"]
                assert ev.strip() != t.strip(), f"Evidence == text in {fname} entity {ent['id']}: {repr(ev)}"

                # Verify entity text is within evidence
                assert t in ev, f"Entity text {repr(t)} not found in evidence {repr(ev)} in {fname} entity {ent['id']}"

                # Verify evidence length is substantially self-contained
                assert len(ev.strip()) > len(t.strip()), f"Evidence shorter or equal to entity text in {fname} entity {ent['id']}"

                # Verify evidence is an exact substring of document text
                assert ev in text, f"Evidence not found in document text in {fname} entity {ent['id']}: {repr(ev)[:80]}"

                # Verify evidence occurrence covers entity offsets
                ev_spans = []
                find_idx = 0
                while True:
                    find_idx = text.find(ev, find_idx)
                    if find_idx == -1:
                        break
                    ev_spans.append((find_idx, find_idx + len(ev)))
                    find_idx += 1
                assert any(sp[0] <= s and e <= sp[1] for sp in ev_spans), (
                    f"Evidence does not span entity [{s}:{e}] in {fname} entity {ent['id']}"
                )

                # Verify balanced parentheses
                assert ev.count("(") == ev.count(")"), (
                    f"Unbalanced parentheses ({ev.count('(')} != {ev.count(')')}) in {fname} entity {ent['id']}: {repr(ev)}"
                )

                # Verify balanced brackets (excluding known source-document typo in aa25-071a E2)
                if not (fname == "aa25-071a.json" and ent["id"] == "E2"):
                    assert ev.count("[") == ev.count("]"), (
                        f"Unbalanced brackets ({ev.count('[')} != {ev.count(']')}) in {fname} entity {ent['id']}: {repr(ev)}"
                    )

                # Verify no dangling delimiter artifacts like ':[' or ':('
                assert not ev.rstrip().endswith((": [", ":[", ":(")), (
                    f"Dangling delimiter in {fname} entity {ent['id']}: {repr(ev)}"
                )

    for doc_id in sorted(hashes):
        aggregate.update(f"{doc_id}\0{hashes[doc_id]}\n".encode("utf-8"))

    # Assert expected entity & relation counts
    print("Entity counts across Gold:", dict(entity_counts))
    print("Relation counts across Gold:", dict(relation_counts))

    assert config_entities_checked == 406, f"Expected 406 Configuration entities, found {config_entities_checked}"
    assert entity_counts["Configuration"] == 406, f"Expected 406 Configuration entities, got {entity_counts['Configuration']}"
    assert entity_counts["Vulnerability"] == 1131, f"Expected 1131 Vulnerability entities, got {entity_counts['Vulnerability']}"
    assert entity_counts["Weakness"] == 142, f"Expected 142 Weakness entities, got {entity_counts['Weakness']}"
    assert entity_counts["AttackTechnique"] == 3702, f"Expected 3702 AttackTechnique entities, got {entity_counts['AttackTechnique']}"

    assert relation_counts["affects"] == 487, f"Expected 487 affects relations, got {relation_counts['affects']}"
    assert relation_counts["instantiates"] == 140, f"Expected 140 instantiates relations, got {relation_counts['instantiates']}"
    assert relation_counts["exploited_by"] == 54, f"Expected 54 exploited_by relations, got {relation_counts['exploited_by']}"

    # 4. Check manifest matches current gold
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    assert manifest["gold_aggregate_sha256"] == aggregate.hexdigest(), (
        f"Manifest aggregate sha256 mismatch: {manifest['gold_aggregate_sha256']} != {aggregate.hexdigest()}"
    )
    for doc_id, digest in hashes.items():
        assert manifest["gold_document_sha256"][doc_id] == digest, f"Doc sha256 mismatch for {doc_id}"

    print("=== ALL COMPREHENSIVE VERIFICATION CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    verify_all()
