"""基于漏洞事实锚定的论元条件回填模块 (Vulnerability-Anchored Argument Backfilling).

本模块为 Stage 1 实体抽取提供确定性后处理与文档内条件补全增强：
1. 跨度尾缀修饰词修剪 (normalize_configuration_entity)；
2. 文档级产品种子收集 (harvest_document_configuration_seeds)；
3. 同句 CVE 约束回填 (vulnerability_anchored_backfill)。

严格遵循第三章冻结契约：Configuration 特指充当 Vulnerability-affects-Configuration 关系客体
的受害资产，绝不在脱离 CVE 语境的情况下全局泛化回填。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from protegi.document_context import extract_explicit_abbreviation_pairs

CVE_RE = re.compile(r"\bCVE\s*-\s*\d{4}-\d{4,}\b", re.IGNORECASE)

TRAILING_NOISE_RE = re.compile(
    r"\s+(server|servers|application|applications|"
    r"devices|instance|instances|data center)$",
    re.IGNORECASE,
)

LOG_NOISE_RE = re.compile(
    r"(\.log|\.exe|httpproxy|log file|event log|audit log)",
    re.IGNORECASE,
)

GENERIC_EXCLUSIONS: Set[str] = {
    "host",
    "system",
    "server",
    "application",
    "devices",
    "endpoint",
    "network",
    "agency",
    "database",
    "cpg 1.e",
    "ms-isac",
    "sdlc",
    "software development lifecycle",
    "remote desktop protocol",
    "local security authority subsystem service",
    "windows remote management (winrm)",
    "virtual private servers",
    "vpns",
    "national institute of standards and technology",
    "national institute for standards and technology",
    # 协议与通用技术缩写排除项 (防止3字符种子误召回)
    "rdp",
    "ssh",
    "vpn",
    "smb",
    "cve",
    "cwe",
    "cpe",
    "log",
    "app",
    "web",
    "ftp",
    "tls",
    "ssl",
    "api",
    "tcp",
    "udp",
    "ioc",
    "c2",
    "cnc",
    "dns",
    "ips",
    "ids",
    "nat",
    "lan",
    "wan",
    "url",
    "uri",
    "sql",
    "xml",
    "pdf",
    "doc",
    "exe",
    "dll",
    "zip",
    "tar",
    "apt",
    "csa",
    "tac",
    "soc",
    "mit",
    "poc",
    "dos",
    "ddos",
    "http",
    "https",
}

VENDOR_PREFIXES: Set[str] = {
    "microsoft",
    "fortinet",
    "adobe",
    "atlassian",
    "apache",
    "cisco",
    "vmware",
    "ivanti",
    "citrix",
    "oracle",
    "sap",
}

PROTECTED_SERVER_PRODUCTS: Set[str] = {
    "windows server",
    "exchange server",
    "http server",
    "apache http server",
    "weblogic server",
    "sql server",
    "sharepoint server",
    "sap content server",
    "content server",
    "wso2 identity server",
    "identity server",
    "confluence server",
    "netweaver application server",
    "data center and server",
}


def is_protected_configuration(text: str) -> bool:
    """判断实体文本是否属于受保护的官方服务器类产品全称（豁免尾缀修剪）。"""
    norm = text.strip().lower()
    for prod in PROTECTED_SERVER_PRODUCTS:
        if norm == prod or norm.endswith(" " + prod):
            return True
    return False


def normalize_configuration_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    """对 Configuration 实体进行确定性尾缀修整。

    若实体尾部带有常见修饰名词（如 application、instance、devices 等），
    自动修剪尾部并拉齐 end 字符坐标，消除表面跨度主观切分带来的伪漏报。
    但对官方合法产品全称（如 Windows Server, Exchange Server）实施白名单保护。
    """
    if entity.get("type") != "Configuration":
        return dict(entity)

    new_entity = dict(entity)
    raw_text = new_entity.get("text", "")
    if is_protected_configuration(raw_text):
        return new_entity

    match = TRAILING_NOISE_RE.search(raw_text)
    if match:
        cut_len = len(match.group(0))
        new_text = raw_text[:-cut_len]
        new_entity["text"] = new_text
        if "end" in new_entity and "start" in new_entity:
            new_entity["end"] = new_entity["start"] + len(new_text)
    return new_entity


def split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """将文本切分为句子区间列表 [(start, end), ...]，保留字符偏移。"""
    sentence_spans: List[Tuple[int, int]] = []
    cur_start = 0
    for match in re.finditer(r"([.!?]\s+|\n{2,})", text):
        sentence_spans.append((cur_start, match.start()))
        cur_start = match.end()
    if cur_start < len(text):
        sentence_spans.append((cur_start, len(text)))
    return sentence_spans


def is_valid_configuration_seed(token: str) -> bool:
    """检查候选资产种子是否符合长度和排除项安全门禁。"""
    clean = token.strip()
    if not clean:
        return False
    cl = clean.lower()
    if cl in GENERIC_EXCLUSIONS or cl.startswith("cpe:2.3"):
        return False
    # 允许 4 字符及以上有效词，或 3 字符大写英文字母缩写 (如 ZCS)
    if len(clean) >= 4:
        return True
    if len(clean) == 3 and clean.isupper() and clean.isalpha():
        return True
    return False


def extract_abbreviation_mapping_from_items(
    items: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]
) -> Dict[str, Tuple[str, str]]:
    """从样本集提取文档内显式缩写对 (short_form <-> long_form)。"""
    mapping: Dict[str, Tuple[str, str]] = {}
    for sample, _ in items:
        abbr_ctx = sample.get("document_abbreviations")
        if abbr_ctx and isinstance(abbr_ctx, str):
            for line in abbr_ctx.splitlines():
                line = line.strip()
                if line.startswith("- ") and " = " in line:
                    parts = line[2:].split(" = ", 1)
                    if len(parts) == 2:
                        sf, lf = parts[0].strip(), parts[1].strip()
                        if sf and lf:
                            mapping[sf.lower()] = (sf, lf)
                            mapping[lf.lower()] = (sf, lf)

        pairs = sample.get("document_abbreviation_pairs")
        if isinstance(pairs, list):
            for p in pairs:
                if isinstance(p, dict):
                    sf = str(p.get("short_form", "")).strip()
                    lf = str(p.get("long_form", "")).strip()
                    if sf and lf:
                        mapping[sf.lower()] = (sf, lf)
                        mapping[lf.lower()] = (sf, lf)

        w_text = sample.get("text", "")
        if w_text and not mapping:
            for p in extract_explicit_abbreviation_pairs(w_text):
                sf = str(p.get("short_form", "")).strip()
                lf = str(p.get("long_form", "")).strip()
                if sf and lf:
                    mapping[sf.lower()] = (sf, lf)
                    mapping[lf.lower()] = (sf, lf)

    return mapping


def harvest_document_configuration_seeds(
    items: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
) -> Dict[str, Optional[str]]:
    """从单篇文档的所有窗口预测中收集高置信度受影响资产种子。

    输出字典: {产品清洗名: normalized_id}。
    特性支持:
    1. Item 1: 3 字符大写缩写识别与文档显式缩写对双向关联 (如 Zimbra Collaboration Suite <-> ZCS)；
    2. Item 2: Server 类产品双形态基础种子派生 (如 Exchange Server -> Exchange) 及厂商前缀剥离。
    """
    seeds: Dict[str, Optional[str]] = {}
    abbr_mapping = extract_abbreviation_mapping_from_items(items)

    def _register_seed(token: str, nid: Optional[str]) -> None:
        t_clean = token.strip()
        if is_valid_configuration_seed(t_clean):
            if t_clean not in seeds or (nid and not seeds[t_clean]):
                seeds[t_clean] = nid
            # 缩写映射双向互补 (Long Form <-> Short Form)
            tl = t_clean.lower()
            if tl in abbr_mapping:
                sf, lf = abbr_mapping[tl]
                paired = sf if tl == lf.lower() else lf
                if is_valid_configuration_seed(paired):
                    if paired not in seeds or (nid and not seeds[paired]):
                        seeds[paired] = nid

    for _, pred_list in items:
        for pr in pred_list:
            if pr.get("type") == "Configuration":
                raw_t = pr.get("text", "").strip()
                if is_protected_configuration(raw_t):
                    clean_t = raw_t
                else:
                    clean_t = TRAILING_NOISE_RE.sub("", raw_t).strip()

                nid = pr.get("normalized_id")
                _register_seed(clean_t, nid)

                # Item 2: Server 产品基础形态派生 (如 Exchange Server -> Exchange)
                if clean_t.lower().endswith(" server"):
                    base_server = clean_t[:-7].strip()
                    _register_seed(base_server, nid)

                # 厂商前缀剥离 (如 Microsoft Exchange Server -> Exchange Server / Exchange)
                parts = clean_t.split(maxsplit=1)
                if len(parts) == 2 and parts[0].lower() in VENDOR_PREFIXES:
                    base_prod = parts[1].strip()
                    _register_seed(base_prod, nid)
                    if base_prod.lower().endswith(" server"):
                        _register_seed(base_prod[:-7].strip(), nid)

    return seeds


def vulnerability_anchored_backfill(
    samples: List[Dict[str, Any]],
    predictions: List[List[Dict[str, Any]]],
    enabled: bool = True,
) -> List[List[Dict[str, Any]]]:
    """执行基于漏洞事实锚定的论元条件回填。

    流程:
    1. 首先对所有现有预测执行 normalize_configuration_entity 尾缀裁剪；
    2. 按 doc_id 进行单文档内分组；
    3. 收集单文档内的合法资产种子库；
    4. 仅在包含 CVE 的句子中搜索种子，满足安全门禁后执行子串回填。
    """
    if not enabled:
        return predictions

    # 1. 先行跨度修剪
    normalized_predictions: List[List[Dict[str, Any]]] = []
    for pred_list in predictions:
        normalized_predictions.append([
            normalize_configuration_entity(ent) for ent in pred_list
        ])

    if len(samples) != len(normalized_predictions):
        return normalized_predictions

    # 2. 按 doc_id 分组
    docs: Dict[str, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]] = {}
    for sample, pred_list in zip(samples, normalized_predictions):
        doc_id = sample.get("doc_id", "unknown")
        docs.setdefault(doc_id, []).append((sample, pred_list))

    # 3. 逐文档执行回填
    sample_to_updated: Dict[str, List[Dict[str, Any]]] = {}

    for doc_id, items in docs.items():
        seeds = harvest_document_configuration_seeds(items)
        # 最长词优先，避免包含关系冲突（如 Fortinet FortiOS 优先于 Fortinet）
        sorted_seeds = sorted(seeds.keys(), key=len, reverse=True)

        for sample, pred_list in items:
            sid = sample.get("sample_id", "")
            w_text = sample.get("text", "")
            occupied_spans = [
                (e["start"], e["end"])
                for e in pred_list
                if "start" in e and "end" in e
            ]
            new_entities = list(pred_list)

            sentence_spans = split_sentence_spans(w_text)
            counter = len(pred_list) + 1

            for sent_s, sent_e in sentence_spans:
                sent_txt = w_text[sent_s:sent_e]
                # 关键门禁：仅当同句中显式包含 CVE 编号时才激活回填
                if not CVE_RE.search(sent_txt):
                    continue

                seen_in_sentence: Set[str] = set()
                for seed in sorted_seeds:
                    pattern = re.compile(
                        r"\b" + re.escape(seed) + r"\b", re.IGNORECASE
                    )
                    for match in pattern.finditer(sent_txt):
                        m_s = sent_s + match.start()
                        m_e = sent_s + match.end()

                        # 门禁 1：单句单产品去重（每个 CVE 句仅采纳第 1 次出现）
                        if seed.lower() in seen_in_sentence:
                            continue

                        # 门禁 2：跨度无重叠
                        if any(
                            max(m_s, o_s) < min(m_e, o_e)
                            for o_s, o_e in occupied_spans
                        ):
                            continue

                        # 门禁 3：排除日志、可执行文件与取证噪音
                        local_ctx = w_text[
                            max(0, m_s - 30) : min(len(w_text), m_e + 30)
                        ]
                        if LOG_NOISE_RE.search(local_ctx):
                            continue

                        # 准入通过，构造回填实体
                        seen_in_sentence.add(seed.lower())
                        exact_str = w_text[m_s:m_e]
                        new_entities.append({
                            "id": f"E_bf_{counter}",
                            "text": exact_str,
                            "type": "Configuration",
                            "start": m_s,
                            "end": m_e,
                            "normalized_id": seeds[seed],
                            "_source": "cve_anchored_backfill",
                        })
                        occupied_spans.append((m_s, m_e))
                        counter += 1

            new_entities.sort(key=lambda x: x.get("start", 0))
            sample_to_updated[sid] = new_entities

    # 按原始样本顺序返回
    final_predictions: List[List[Dict[str, Any]]] = []
    for sample in samples:
        sid = sample.get("sample_id", "")
        final_predictions.append(sample_to_updated.get(sid, []))

    return final_predictions
