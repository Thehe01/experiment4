"""在 exploited_by 裁决完成后生成文档组级分层切分。

同一 AA 通告的全文与专题子文档被视为一个文档组，避免近重复内容跨越训练
集和测试集。当前 105 篇语料按约 80%/20% 划分为 84/21。
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
GOLD = EXP_DIR / "data" / "annotations" / "gold"
DEFAULT_OUTPUT = EXP_DIR / "data" / "train_test_split_v5.json"
RELATIONS = ("affects", "instantiates", "exploited_by")


def group_id(doc_id):
    match = re.match(r"^(aa\d{2}-\d{3}[a-z])(?:-|$)", doc_id, re.IGNORECASE)
    return match.group(1).lower() if match else doc_id


def shingles(text, size=5):
    tokens = re.findall(r"[A-Za-z0-9_.:/-]+", text.lower())
    return {
        tuple(tokens[index:index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_documents():
    documents = {}
    for path in sorted(GOLD.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        relation_edges = Counter(
            relation["type"]
            for relation in data.get("relations", [])
            if relation.get("type") in RELATIONS
        )
        documents[path.stem] = {
            "source": data.get("source_type", "unknown"),
            "relation_edges": relation_edges,
            "relation_docs": {key for key, value in relation_edges.items() if value},
            "fingerprint": shingles(data.get("text", "")),
        }
    return documents


def build_groups(documents, near_duplicate_threshold):
    """合并同一 AA 编号组以及内容高度相似但命名不同的专题文档。"""
    ids = sorted(documents)
    parent = {doc_id: doc_id for doc_id in ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    aa_groups = defaultdict(list)
    for doc_id in ids:
        aa_groups[group_id(doc_id)].append(doc_id)
    for values in aa_groups.values():
        for doc_id in values[1:]:
            union(values[0], doc_id)

    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            if find(left) == find(right):
                continue
            similarity = jaccard(
                documents[left]["fingerprint"],
                documents[right]["fingerprint"],
            )
            if similarity >= near_duplicate_threshold:
                union(left, right)

    groups = defaultdict(list)
    for doc_id in ids:
        groups[find(doc_id)].append(doc_id)
    return list(groups.values())


def summary(doc_ids, documents):
    sources = Counter()
    relation_edges = Counter()
    relation_docs = Counter()
    for doc_id in doc_ids:
        item = documents[doc_id]
        sources[item["source"]] += 1
        relation_edges.update(item["relation_edges"])
        relation_docs.update(item["relation_docs"])
    return sources, relation_edges, relation_docs


def distance(test_ids, documents, target_size):
    all_ids = list(documents)
    total_sources, total_edges, total_docs = summary(all_ids, documents)
    test_sources, test_edges, test_docs = summary(test_ids, documents)
    ratio = target_size / len(all_ids)
    score = 0.0
    for source, total in total_sources.items():
        target = total * ratio
        score += ((test_sources[source] - target) / max(1.0, target)) ** 2
    for relation in RELATIONS:
        edge_target = total_edges[relation] * ratio
        doc_target = total_docs[relation] * ratio
        score += 2.0 * ((test_edges[relation] - edge_target) / max(1.0, edge_target)) ** 2
        score += 3.0 * ((test_docs[relation] - doc_target) / max(1.0, doc_target)) ** 2
        if total_docs[relation] >= 5 and test_docs[relation] == 0:
            score += 1000.0
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.75)
    args = parser.parse_args()

    documents = load_documents()
    _, total_edges, total_relation_docs = summary(list(documents), documents)
    if total_edges["exploited_by"] == 0:
        raise SystemExit("Gold 中尚无 exploited_by；请先完成候选裁决，不能按候选自动切分。")

    group_values = build_groups(documents, args.near_duplicate_threshold)
    rng = random.Random(args.seed)
    best_ids = None
    best_score = math.inf

    for _ in range(args.trials):
        order = group_values[:]
        rng.shuffle(order)
        selected = []
        size = 0
        for group in order:
            if size + len(group) > args.test_size:
                continue
            # 小幅随机跳过，增加候选切分多样性。
            if rng.random() < 0.18 and size < args.test_size - 3:
                continue
            selected.extend(group)
            size += len(group)
            if size == args.test_size:
                break
        if size != args.test_size:
            continue
        score = distance(selected, documents, args.test_size)
        if score < best_score:
            best_ids = sorted(selected)
            best_score = score

    if best_ids is None:
        raise SystemExit("未找到满足文档组约束的切分，请提高 --trials 或调整 --test-size。")

    test = best_ids
    train = sorted(set(documents) - set(test))
    train_set, test_set = set(train), set(test)
    leaked_groups = [
        group
        for group in group_values
        if train_set.intersection(group) and test_set.intersection(group)
    ]
    if leaked_groups:
        raise AssertionError(f"文档组泄漏: {leaked_groups}")

    split = {
        "schema_version": "chapter3-no-capec-v1",
        "seed": args.seed,
        "strategy": (
            "grouped random search stratified by source and relation distribution; "
            "AA groups and >=0.75 five-token-shingle near duplicates kept together"
        ),
        "train": train,
        "test": test,
    }
    output = Path(args.output)
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(train)}/{len(test)} 切分：{output}")
    for name, ids in (("train", train), ("test", test)):
        sources, edges, relation_docs = summary(ids, documents)
        print(f"{name}: sources={dict(sources)}")
        print(f"{name}: relation_edges={dict(edges)}")
        print(f"{name}: relation_docs={dict(relation_docs)}")
    print("请人工核对分布后，再在 v5 data/review_status.json 中将 split_validated 设为 true。")


if __name__ == "__main__":
    main()

