# 标注协议 v4.6——MCPU 文本提及与规范化事实双层 Gold

本协议服务于第四章文本抽取实验，正式模式版本为 `chapter3-no-capec-v1`，当前实体关系边界契约为 `chapter3-boundary-sync-v2`。文本抽取层只标注能够回指原文的 `Configuration`、`Vulnerability`、`Weakness`、`AttackTechnique` 及 `affects`、`instantiates`、`exploited_by`；`AttackTactic`、`KillChainPhase`、`implies`、`belongs_to_phase` 由映射补全层生成，不计入文本抽取 Gold。

`data/annotations/gold/` 中现有 105 篇 Gold 保留其 v4.5 人工复标与分歧裁决来源，并已按 `chapter3-boundary-sync-v2` 对 406 个 Configuration mention 实施预测盲法的 MCPU 全量审计：修订 15 个跨度、7 个普通版本 CPE，涉及 17 个不重复 mention。既有 test/APO 分数绑定旧边界，只能作为历史受控结果。由于本次规则形成使用了全量 406 项审计，现有 test 不再承担一次性最终检验；正式结论须另行预留未参与协议形成的新 test。Gold 来源与人工 IAA 分开记录：边界复裁完成不等于双标注者一致性已经完成。

## 1. 基本原则

1. 每个实体必须具有与解码后原文完全一致的 0-based、右端不包含字符区间。
2. 每条关系必须引用能够同时覆盖两个端点并直接支持关系的原文证据；篇内共现不等于关系。
3. 标准库只能用于核对规范化标识，不得据此补造原文没有表达的关系。
4. 文本提及层与规范化事实层分开：前者回答“原文哪一处表达了什么关系”，后者回答“文档最终包含哪些去重后的知识图谱事实”。
5. Gold 修订只能依据冻结协议和原文完成，不得依据待评价模型的预测增删标签。
6. 证据不足或存在多种合理端点解释的候选进入盲审区，不直接进入正式 Gold。
7. 正式标注单位是“文本提及关系”：同一规范化事实在不同、自足、直接的证据块中重复出现时，必须保留各处合法 mention；规范化事实层只负责确定性去重，不反向删除文本提及。
8. 关系端点不得按证据区间内的实体做笛卡尔积。多个 CVE、多个产品或多个 ATT&CK 技术同时出现在一个区间内时，只有语法或表格结构明确授权的端点对才能入 Gold。

## 2. 实体类型与边界

| 实体类型 | 标注对象 | `normalized_id` | 边界要求 |
|---|---|---|---|
| Vulnerability | 原文明示的具体漏洞 | 规范 CVE 编号 | 每次显式出现均按位置标注，兼容 PDF 空格断裂 |
| Weakness | 明示 CWE；或与同一事实块中的明示 CWE 唯一对应的根因缺陷短语 | 该事实块中明示的 CWE 编号 | 描述性短语不得仅凭模型知识、外部 CVE—CWE 映射或临时词典晋级 |
| Configuration | 能够作为直接 `affects` 证据尾端的具体产品、组件、设备或完整 CPE URI | 规范 CPE 2.3 URI | 标注最小规范产品单元（MCPU）；保留同一连续产品名词短语中的紧邻厂商与必要产品标识，删除普通版本及描述性类名词 |
| AttackTechnique | 明示 ATT&CK 技术编号或可唯一归一化的技术名称 | T 编号 | 相邻名称与 T 编号并存时只标 T 编号；显式 T 编号逐次标注 |

不标注 `AttackPattern`/CAPEC、`AttackTactic`、`KillChainPhase`，也不把 “multiple products”“various products”“affected systems”“software”等泛化占位词标成 `Configuration`。remote code execution、denial of service、information disclosure 等结果性短语通常不标成 `Weakness`。

### 2.1 Configuration 统一规则

1. 只有能够参与一条直接文本支持的 `Vulnerability-affects->Configuration` 关系时，产品提及才进入最终 Gold。安全工具、攻击平台、缓解产品、厂商名和一般背景产品不因共现而进入 Gold。
2. 边界采用“最小规范产品单元”（Minimum Canonical Product Unit, MCPU）：选择能够保持受影响产品、组件或设备规范身份的最短连续原文跨度。同一连续产品名词短语中紧邻产品的显式厂商必须保留；不得跨表格列、并列项、列表项、句子或非连续文本补入厂商。
3. `Server`、`Gateway`、`Manager`、`Appliance`、`Controller`、`Service`、`Driver` 等词只有在属于正式产品名或原文明示的受影响组件身份时才保留；`software library`、`email servers`、`appliances`、`webmail clients` 等仅描述产品类别或部署形态的后缀应删除。不能仅凭停用词表机械裁剪，最终跨度仍须能够独立指称目标产品。
4. “Product Server and Data Center”等共享基名表达同一规范化产品时标共享产品基名；只有不同形态对应不同 CPE 或不同漏洞事实时才分别保留。
5. 完整 CPE URI 在同一局部事实块中明确受某 CVE 影响时，优先把完整 URI 作为表面跨度，不再重复保留附近的描述性名称；不同 CPE 产品字段分别保留。
6. 同一产品与不同 CVE 构成不同事实时，必须分别保留能够局部支持各事实的产品 mention，不得固定取第一次。
7. 同一规范化 CVE—产品事实在不同位置形成多个自足、直接事实块时，文本提及层分别标注各组合法端点和关系；参考文献、导航、URL、非自足简称和纯粹重复标题不因此进入 Gold。
8. 同一事实的多个 mention 在规范化事实层按 `文档＋关系类型＋规范化头＋规范化尾` 去重，但不得在文本提及层提前去重。
9. `Configuration.normalized_id` 必须是可由官方 NVD CPE 字典或相关 CVE 配置核验的 CPE 2.3 URI；`part`、`vendor`、`product` 不得使用自造简称或把操作系统、固件误写为应用。普通自然语言产品提及按产品族规范化，原文未给出的版本字段用 `*`，不得把附近版本信息静默写入 CPE。
10. NVD 配置只用于规范化已由原文直接支持的 `Configuration`，不得据此外推或新增 `affects`。原文所指组件没有独立 CPE 时，只有在相关 CVE 的官方配置能够唯一确定一个受影响父产品族时，才可投影到该父族；若仍有多个互不等价的产品或型号，实体与关系进入隔离区，不得任选代表 CPE。
11. 普通发布版本、更新、补丁和 build 不进入表面跨度，也不进入普通自然语言 mention 的 CPE 版本字段。只有完整字面 CPE URI，或已经成为产品身份组成部分的代际/型号标识（如 `SMB version 1`），才允许保留非通配版本或代际信息。

### 2.2 边界判定示例

- `Microsoft: Windows Server`：若冒号前为表格厂商列，标 `Windows Server`；若完整字符串是正式产品名，保留完整名。
- `Progress Telerik user interface (UI) for ASP.NET AJAX`：`Progress` 与 `Telerik` 位于同一连续产品名词短语，保留完整跨度；不能从其他列或前句补入厂商。
- `Microsoft Exchange email servers`：标 `Microsoft Exchange`；`Microsoft` 是同一连续产品名词短语中的厂商，`email servers` 只是类别描述。
- `Red Hat: Polkit Privilege Escalation`：厂商和漏洞标题不属于产品核心，标 `Polkit`。
- `Cisco IOS XE Web management user interface`：若事实影响 IOS XE 整体，标 `Cisco IOS XE`；若原文明示只影响该组件且组件可独立规范化，才保留组件。
- `Ivanti CSA` 与 `Ivanti Cloud Service Appliances`：各自只有在所在位置形成直接事实块时才成为 mention；规范化后可以指向同一 CPE 事实。

## 3. 关系语义

| 关系 | 方向 | 正例条件 | 排除条件 |
|---|---|---|---|
| affects | Vulnerability→Configuration | 原文说明具体 CVE 影响、存在于或适用于具体产品/组件；或同一结构行明确给出 CVE—产品 | 泛化产品、厂商共现、跨行/跨条目拼接、明确否定 |
| instantiates | Vulnerability→Weakness | 原文明确说明该 CVE 属于或源于某缺陷，或在同一事实句、同一表格行/单元格或明确支配的列表项中直接对应 CWE | 跨相邻句、仅凭共现或仅凭外部 CVE—CWE 映射推断 |
| exploited_by | Vulnerability→AttackTechnique | 同一事实块中明确把指定 CVE 与该 ATT&CK 技术绑定，且技术直接承担该 CVE 的利用或入口行为 | 只有附近标题、表头、脚注或共用区间；利用后的执行、持久化、凭据访问、横向移动和命令控制；仅共现 |

`instantiates` 中“明确支配的列表项”限定为：列表标题、引导句或表格结构明确声明该 CWE 适用于后续指定 CVE，且没有新的语义单元终止支配范围。严禁跨句推断或仅凭外部知识库补全弱点。

当前 v4.6 冻结批次启用的 `Weakness.normalized_id` 受控词表为原文明确给出的 `CWE-*` 标识。原文只有描述性缺陷短语而没有显式 CWE 或项目已注册的概念标识时，可记录为复核候选，但不得通过外部 CVE-CWE 映射或临时自造 ID 晋级为 Gold；后续扩充概念词表须提升协议版本并重新冻结。

描述性 Weakness 只有在同一事实块中出现明确 CWE，且句法、括注、同一表格行或明确列表支配能够唯一确定短语与该 CWE 的对应关系时，才能与显式 CWE 采用相同 `normalized_id`。`authentication bypass` 等可能对应多个 CWE 的宽泛短语，在没有局部明示 CWE 时必须留在候选区。

`exploited_by` 必须同时满足以下条件：

1. 证据中能够唯一确定漏洞端点和 ATT&CK 技术端点；
2. 同一谓词、同一表格行或明确的列表支配语句直接绑定二者；
3. 关系语义是该 CVE 的利用或入口，而不是利用成功后的后续行为。

下列情况不得建立正式关系：

- 一个 CVE 在“using/exploiting”从句中出现，其他 CVE 只在后续“also exploited”枚举中出现，而 ATT&CK 技术没有明确覆盖后续枚举；
- 只有段落标题、表头、脚注或邻近行出现 ATT&CK 技术，正文没有明确说明其支配范围；
- 攻击者利用 CVE 后执行 PowerShell、持久化、凭据访问、横向移动或命令控制，仅凭共现不能建立关系。
- “利用 CVE 后运行 RAT 并通过 C2 通信”只说明后续命令控制活动，不能建立该 CVE 指向命令控制技术的 `exploited_by`；“在利用 CVE 后获得任意代码执行”也不能仅凭技术标签建立客户端执行关系。

如果 `T1068` 等技术本身在同一事实块中被明确写成该 CVE 的利用方式，可以建立关系；“先利用 CVE、随后发生提权”不满足该条件。

## 4. 事实块与证据边界

1. 普通正文以一个自足关系句或紧邻从句为事实块。
2. 项目列表以单个 bullet 为事实块；不得跨 bullet 配对。
3. 表格以单行或同一单元格中的明确支配结构为事实块；不得因为相邻行、表头或脚注而跨行配对。
4. CVE 枚举默认只标注实体。只有枚举前后的支配语句明确说明“以下全部 CVE 均通过该技术被利用”时，才可把该技术逐一映射到列表中的 CVE。
5. 一个技术标签出现在列表标题或表头时，不能自动支配下面的普通段落；必须存在同一行/条目绑定，或存在明确、可回指的支配句。
6. `Affected configurations`、`Exploited CVE information` 等标题只有在同一局部小节明确作用于一个 CVE 时才能向下支配；遇到新 CVE、标题、表格行或新的语义单元即终止。
7. 同一共享句可以支持多个关系，但每条关系必须有独立的端点配对依据。若区间包含多个同类型候选端点而无法由语法或结构唯一消歧，所有相关关系进入复核，不得批量展开。
8. 证据区间应是能够覆盖端点和关系语义的最短连续事实块。超过 200 字符、包含多个 CVE、多个 Configuration 或多个 ATT&CK 技术时，必须记录作用域依据；没有作用域依据的关系不得晋级正式 Gold。
9. `unrelated`、`not affected`、`does not affect` 等明确否定阻断 `affects`；“未利用”只否定 `exploited_by`，不自动否定同一结构行明示的产品归属。

## 5. 文档级完整性与双层 Gold

### 5.1 文本提及层

- 标注正文中全部显式 CVE、CWE 和 ATT&CK T 编号。
- 标注所有处于直接关系事实块中的 Configuration mention。
- 每个直接、可独立定位的关系 mention 对分别建立关系；同一规范化事实在不同证据块重复出现时不得提前去重。
- 独立复标必须使用同一 mention-level 规则，不能一方保留全部合法 mention、另一方只选一个“代表位置”。
- 主体评价使用严格实体跨度 F1 和严格关系端点 F1；该指标反映文本定位一致性，不单独代表规范化事实一致性。

### 5.2 规范化事实层

- 由文本提及层确定性投影得到，不单独补造关系。
- 事实键为 `文档＋关系类型＋规范化头实体ID＋规范化尾实体ID`。
- 同一事实的多个 mention 关系只计一个规范化事实。
- 规范化事实 F1、分关系类型事实 F1 和证据歧义率必须与严格指标同时报告；它们用于解释别名与重复 mention，不替代严格指标。
- 规范化事实评价必须按 `文档＋关系类型＋规范化头＋规范化尾` 去重；不能把 mention-level 关系数量直接当成事实数量。
- 关系一致性报告必须同时给出 mention-level 和 fact-level 结果，不能只报告其中一个并将其称为总体关系一致性。
- 图谱入库在该层去重；文本提及及证据位置作为 provenance 保留。

## 6. 数据格式与元数据规范

每个 Gold JSON 文件必须具备以下顶级字段与元数据结构：

```json
{
  "doc_id": "aa20-259a-iran-citrix-vpn-cve-19781",
  "schema_version": "chapter3-no-capec-v1",
  "annotation_protocol_version": "4.6-mcpu-mention-fact-dual-layer-v1",
  "boundary_contract_version": "chapter3-boundary-sync-v2",
  "annotation_status": "v4.5_human_reannotated_adjudicated",
  "gold_review": {
    "kind": "human reannotation and adjudication",
    "formal_human_iaa": false,
    "prediction_outputs_used": false
  },
  "text": "...",
  "entities": [...],
  "relations": [...]
}
```

- **实体字段**：`id`, `text`, `type`, `start`, `end`, `normalized_id`。
- **关系字段**：`id`, `type`, `head`, `tail`, `evidence`, `evidence_start`, `evidence_end`。
- **审计字段**：关系进入证据复核后可附带 `adjudication_basis`，用于记录基于原文句、表格行或明确列表作用域的裁决依据；该字段不得引用待评价模型预测，也不能替代关系证据本身。
- 双层事实视图由评测程序根据规范化 ID 投影，不在 Gold 中复制一套可能漂移的关系数组。任何 Gold 文件缺失标准元数据均视为结构不合规。

## 7. 质量控制与版本门禁

1. 结构审计：跨度、模式、方向、悬空端点、证据偏移、CPE 格式和孤立 Configuration 必须为 0 错误；CPE 语义审计还必须确认 `part/vendor/product` 可核验，未决项必须为 0。
2. 策略审计：检查重复产品表面词、规范化重复事实、跨事实块证据、多个同类型候选端点、超长证据以及多 CVE/多技术作用域。
3. 语义复核：策略审计候选必须在不查看待评模型输出的条件下依据原文裁决；“结构无硬错误”不等于“语义审计通过”。`exploited_by` 必须按证据中的行为角色复核，不得用固定 T 编号黑名单代替语义判断。
4. 关系晋级门禁：没有明确端点作用域、直接关系谓词或表格绑定依据的关系必须留在候选/复核区，不得因模型一致、外部知识或篇内共现晋级 Gold。
5. 审计结果分三类记录：`review_candidates` 表示尚未解决的关系证据候选，必须为 0；`resolved_review_candidates` 表示已有原文作用域裁决依据的关系提示；`informational_flags` 表示重复 mention 等不改变标注合法性的提示，不计入未解决候选。
6. 版本一致：train/dev/test、每个 Gold 文件、盲标包和评测结果必须记录同一协议版本；不得用单一版本号掩盖某个划分仍沿用旧政策。
7. 测试集：在协议冻结后进行盲法复标；不得因待评价模型的错误分析修改测试 Gold。v4.6 的 MCPU 规则由全量 406 项审计收敛而来，因此现有 test 只可作开发审计集；必须另行预留未参与规则形成的新 test，冻结后只评价一次。
8. 一致性：人工 IAA 必须由两名人工在裁决前独立标注同一批文档后计算，并同时报告 mention-level 与 fact-level 指标；历史模型间一致性不能写成人工 IAA。
9. 现有 Gold 的 v4.5 人工复标和分歧裁决记录仍作为历史 provenance 保留；`chapter3-boundary-sync-v2` 的 MCPU 迁移只依据协议与原文，不读取模型预测。可核验人工 IAA 与新未见 test 尚未完成，因此既有实验结果只能称为历史受控结果，新边界下的运行也不得扩大为最终正式结论。
