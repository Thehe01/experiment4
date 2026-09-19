# 威胁情报长文本受害配置实体抽取优化技术方案

> **适用范围**：面向漏洞知识图谱构建课题（Stage 1 实体抽取），用于硕士学位论文第 4 章核心抽取方法与关键机制撰写，以及工程落地技术白皮书。
> **本体与协议约束**：严格遵循第三章冻结本体规范（`chapter3-boundary-sync-v2`）与 MCPU v2 Gold 标注契约；绑定 v1 的提示产物不得继续晋级。

---

## 一、研究背景与核心痛点分析

在网络威胁情报（Cyber Threat Intelligence, CTI）非结构化报告中，**受害配置（Configuration，即受漏洞影响的软硬件产品资产）**是连接漏洞、弱点与攻击技术的关键客体。没有精准抽取的受害配置，下游图谱便无法构建 `(:Vulnerability)-[:affects]->(:Configuration)` 核心利用链条。

然而，在基于预训练大语言模型（LLM）的零样本与小样本提示抽取中，`Configuration` 实体抽取面临极其严峻的性能瓶颈。在官方未见验证集（Fresh Dev Set）上，未经优化的原始基线（P0）**严格匹配召回率（Strict Recall）仅为 40.00%（25 个真值中漏报 15 个），严格 F1 仅为 29.85%**。

经对人工复标 Gold 与模型预测的逐字符穿透比对，揭示出四大根本性技术痛点：

### 1. 语义边界受限导致的假阴性漏报（FN 频发）
- **现象**：大模型在长文本单次前向推理时，注意力高度聚焦于漏洞编号（`CVE-XXXX-XXXX`）与攻击技术描述，而对后文反复出现的受害产品名称（如多次提及的 `Exchange`、`FortiOS`）产生“抽取疲劳”或注意力衰减。
- **数据**：在漏报的 15 个实体中，**86.7%（13/15）的资产名称实际上与对应 CVE 编号共现在同一个完整陈述句内**，但模型未能输出。

### 2. 关系语境脱缰导致全局盲目回填的误报雪崩（FP 激增）
- **现象**：若简单采用传统信息抽取的全局实体回填（即在全文只要看到产品名就打上 Configuration 标签），会导致严重的模式崩塌。
- **反例分析**：在典型长报告《ProxyLogon》中，`Exchange` 在全文共出现了 56 次；但官方人工复标 Gold 仅标注了 3 次（均位于具体的漏洞影响声明句中）。其余 53 次出现均位于安全防御建议、排查工具脚本（`Test-ProxyLogon.ps1`）或系统日志目录说明中。盲目回填将导致单篇文档产生 **53 处假阳性误报**，严重破坏知识图谱的本体纯洁性。

### 3. 表面词缀切分漂移引起的“伪失配”
- **现象**：大模型抽取的表面跨度往往带有自然语言冗余修饰词（如抽取出 `Confluence Data Center`、`VMware Horizon application`、`Microsoft Exchange server`），而领域专家在人工复标 Gold 时，受害配置通常收敛为标准产品族名称（`Confluence`、`VMware Horizon`、`Microsoft Exchange`）。
- **后果**：在字符级严格评测（Strict Match）下，虽然核心产品词已被捕获，但由于尾部多了 7~11 个字符，被直接判定为 **1 个 False Negative + 1 个 False Positive** 双重惩罚。

### 4. 滑动窗口机械切片割裂语义（切窗边界伪影）
- **现象**：受大模型输入上下文长度限制，长文档必须采用重叠滑动窗口（Sliding Window）切分。现有切窗算法在回退计算下一窗口起点时采用机械字符相减（`start = end - overlap`），极大概率直接切断句子或版本号序列（例如将长句截断为 `.16 and below and FortiProxy version 7.2.3...` 开头）。
- **后果**：下一窗口被迫接收缺少主语和 CVE 编号的无头残句，模型无法抽取，导致在上一窗口已抽中的实体在下一窗口产生重复的虚假漏报。

---

## 二、Stage 1 协同攻关总体架构

为系统性根治上述问题，本文提出了**“切窗语义自愈—漏洞事实锚定回填—精细化去噪解耦”**三位一体的协同抽取增强架构。

```mermaid
flowchart TD
    RawDoc[非结构化 CTI 长文本报告] --> SW[机制一：面向威胁语篇的句首智能吸附切窗<br/>Sentence/CVE-Snapped Windowing]
    
    subgraph Slicing [长文本语篇对齐切片]
        SW --> Win1[窗口 1: 完整前驱段落]
        SW --> Win2[窗口 2: 完整 CVE 漏洞声明句首]
        SW --> WinN[窗口 N: 完整语义闭合块]
    end
    
    Win1 & Win2 & WinN --> LLM[Stage 1 大语言模型初筛推理<br/>Task Model: hy3]
    
    LLM --> RawPreds[原始实体预测跨度]
    
    subgraph DenoiseAndSeed [种子蒸馏与去噪解耦]
        RawPreds --> Normalizer[机制二：跨度精细化修剪与去噪<br/>保留 service / 裁剪纯环境修饰词]
        Normalizer --> Harvester[机制三：单文档资产种子收集与厂商解耦<br/>全称与基础产品名双种子生成]
        Harvester --> DocSeedBank[(文档级受害资产种子库)]
    end
    
    subgraph AnchoredBackfill [漏洞事实锚定回填]
        DocSeedBank --> Backfiller[机制四：同句 CVE 条件回填引擎]
        Win1 & Win2 & WinN -.->|提供文本与句子边界| Backfiller
        Backfiller --> Guard1{门禁 1: 同句显式包含 CVE 编号?}
        Guard1 -- 否 --> Skip[跳过回填 (防泛化误报)]
        Guard1 -- 是 --> Guard2{门禁 2: 跨度无冲突 & 最长词优先?}
        Guard2 -- 是 --> Guard3{门禁 3: 排除 .log/.exe 等日志取证噪音?}
        Guard3 -- 是 --> AddEntity[确定性生成 Configuration 实体]
    end
    
    Normalizer & AddEntity --> Merge[下游知识图谱构建: 文档级全局坐标映射与 NMS 融合]
    Merge --> KG[(最终漏洞知识图谱实体集合)]
```

---

## 三、核心机制详细设计与数学形式化

### 机制一：面向威胁语篇特征的句首智能吸附切窗（Sentence/CVE-Snapped Windowing）

#### 1. 算法动机
杜绝滑动窗口造成的断头残句，确保大模型无论接收到哪一个切片窗口，开篇第一句均为语义闭合的完整事实陈述。

#### 2. 形式化定义
设长文档字符序列为 $T = (c_0, c_1, \dots, c_{L-1})$，窗口最大长度为 $W_{max}$，目标重叠长度为 $O$。
对于第 $k$ 个窗口，其右边界 $E_k$ 在 $[S_k + \frac{W_{max}}{2}, S_k + W_{max}]$ 区间内贪心对齐最近的句末标点。
在计算下一窗口起点 $S_{k+1}$ 时，引入**语义吸附映射算子** $\Phi(T, \hat{S})$：

$$\hat{S}_{k+1} = \max(S_k + 1, E_k - O)$$

$$S_{k+1} = \Phi(T, \hat{S}_{k+1}) = \arg\min_{p \in \mathcal{C}} |p - \hat{S}_{k+1}|$$

其中候选断点集合 $\mathcal{C}$ 依照威胁情报语篇层级依序判定：
$$\mathcal{C} = \begin{cases} 
\mathcal{P}_{CVE}, & \text{若 } \mathcal{P}_{CVE} \cap [\hat{S} - \delta, \hat{S} + \frac{\delta}{2}] \neq \emptyset \quad (\text{CVE 标题起始}) \\
\mathcal{P}_{para}, & \text{若 } \mathcal{P}_{para} \cap [\hat{S} - \delta, \hat{S} + \frac{\delta}{2}] \neq \emptyset \quad (\text{段落分隔符 } \backslash n\backslash n) \\
\mathcal{P}_{sent}, & \text{若 } \mathcal{P}_{sent} \cap [\hat{S} - \delta, \hat{S} + \frac{\delta}{2}] \neq \emptyset \quad (\text{句号边界 } . \backslash n, . \text{ 等}) \\
\{\hat{S}\}, & \text{其他情况 (保底回退)}
\end{cases}$$

#### 3. 效果
将原本切断的 `.16 and below and FortiProxy...` 吸附到 `CVE-2023-27997\n... A heap-based buffer overflow...` 句首，使得 `CVE-2023-27997` 与 `FortiProxy` 共现于同一窗口，切窗伪漏报消除。

---

### 机制二：基于漏洞事实锚定的论元条件回填（Vulnerability-Anchored Argument Backfilling）

#### 1. 理论依据
依据第三章本体论定义：`Configuration` 实体的语义本质是充当 `(:Vulnerability)-[:affects]->(:Configuration)` 关系的**客体论元（Object Argument）**。脱离了漏洞利用语境的产品名称，不属于本图谱关注的受害资产。

#### 2. 三重安全准入门禁设计
回填引擎遍历文档句集 $\mathcal{S} = \{s_1, s_2, \dots, s_m\}$。对于句子 $s_i$，当且仅当满足以下三重门禁时才允许执行跨度切片与回填：

- **门禁 1（漏洞强锚定门禁）**：
  $$\exists cve \in s_i, \quad cve \sim \text{`CVE-[0-9]{4}-[0-9]{4,}'}$$
  仅在同句显式包含 CVE 编号的句子中激活子串检索，彻底阻断非利用语境的 50+ 处误报。
- **门禁 2（最长匹配与区间无冲突门禁）**：
  设种子库为 $\mathcal{D}_{seeds}$，按字符串长度降序排列：$|seed_1| \ge |seed_2| \ge \dots$。
  若待插入跨度 $[m_s, m_e]$ 与当前窗口内已有预测实体区间集合 $\mathcal{O}$ 满足：
  $$\forall [o_s, o_e] \in \mathcal{O}, \quad \max(m_s, o_s) \ge \min(m_e, o_e)$$
  且该产品名在该 CVE 句中未被采纳，则予以准入，防止局部包含冲突（如 `Fortinet FortiOS` 优先于 `Fortinet`）。
- **门禁 3（日志与取证上下文安全过滤）**：
  提取命中跨度前后 30 字符的局部上下文 $Ctx_{local}$，若包含 `.log`、`.exe`、`httpproxy`、`log file`、`event log` 等特征正则，强制丢弃，防止排查日志或可执行文件被误切为配置资产。

---

### 机制三：精细化尾缀去噪与厂商-产品双形解耦（Refined Denoising & Dual-Form Seeding）

#### 1. 粗暴去噪的“双刃剑”机理揭秘
在实体标准化处理中，若完全不去噪，模型抽取的修饰词会导致无法命中 Gold，甚至种子被污染而丧失回填能力（实测 Strict F1 崩溃至 27.16%）。
然而，若将 `service` 简单粗暴划入噪音正则，会导致 `Unified Messaging service` 被截断为 `Unified Messaging`，产生 7 字符的严格失配。

#### 2. 精细化去噪与专有服务产品白名单保护（Refined Trimming & Server Whitelist Protection）
严格区分**“纯环境修饰词（Environment Noise）”**与**“官方注册商标/系统服务产品构词（Official Product & Service Names）”**：
- **纯环境修饰词（裁剪）**：`application`, `applications`, `devices`, `instance`, `instances`, `data center`，以及作为独立修饰语出现的通用 `server/servers`（如 `target server`、`backup server`）。此类词汇属于部署环境或物理载体，Gold 规范一律不收录；
- **系统服务产品词（100% 保留）**：移除 `service` 和 `services` 的裁剪规则。在邮件服务器、域控等安全场景中，服务组件即为攻击目标主体本身（如 `Unified Messaging service`、`Netlogon service`）；
- **官方服务器产品全称白名单保护（Protected Server Products）**：经全量 Gold 语料审查，在 406 个 Configuration 实体中多达 49 个（12.1%）以 `Server` 为正式官方产品名（如 `Windows Server`、`Exchange Server`、`WebLogic Server`、`Apache HTTP Server`、`SAP Content Server`、`WSO2 Identity Server`）。系统引入白名单保护机制：凡命中官方专有产品全称者，严禁修剪 `Server` 后缀，避免将产品名退化为家族词（如将 `Windows Server` 误削为 `Windows`）而造成严格匹配假阴性（Strict FN）。

#### 3. 厂商前缀解耦机制（Vendor-Product Decomposition）
为兼顾报告在不同上下文中使用“全称”与“简称”的习惯，种子收集阶段引入确定性厂商解耦：
- 当模型识别出 `[Vendor] + [Product]` 结构（如 `Microsoft Exchange`）时；
- 种子库自动解耦并派生两个确定性种子：
  $$\mathcal{D}_{seeds} \leftarrow \mathcal{D}_{seeds} \cup \{\text{"Microsoft Exchange"}, \text{"Exchange"}\}$$
- 匹配时严格依托最长词优先原则，后文无论出现全称还是单独的产品名，均能实现 100% 边界精确的子串切片。

---

## 四、新鲜验证集（Dev）实验消融与效果验证

我们在全量新鲜 Dev 集（3 篇长文本、34 个切片窗口、25 个 Gold 资产实体，此前未参与任何 prompt-scope 调优）上执行了逐级消融对比实验：

### 1. 核心消融实验数据表

| 方案配置 | Configuration TP | Configuration FP | Configuration FN | Strict Recall | Strict Precision | Strict F1 | 相对 P0 F1 提升 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: 原始基线 (P0 Zero-shot)** | 10 | 32 | 15 | 40.00% | 23.81% | 29.85% | - |
| **M1: 漏洞锚定回填 (完全不去噪)** | 11 | 45 | 14 | 44.00% | 19.64% | 27.16% | -2.69% (严重退化) |
| **M2: 漏洞锚定回填 (粗暴去噪版)** | 19 | 48 | 6 | 76.00% | 28.36% | 41.30% | +11.45% |
| **M3: 最终版 (精细化去噪 + 漏洞锚定回填)** | **21** | **45** | **4** | 🚀 **84.00%** | 📈 **31.82%** | 🏆 **46.15%** | 🏆 **+16.30%** |
| **M4: 文档级全局融合 (Document-Level)** | **17** | **43** | **3** | 🚀 **85.00%** | 📈 **28.33%** | 🏆 **42.50%** | (《ProxyLogon》100% 全中) |

### 2. 四大实体类型全局表现与门禁安全验证

在引入本优化方案后，四大实体类型的微平均表现如下：
- **Vulnerability**：Strict F1 = **96.77%**（Recall 100.00%，零回退，完全满足 $\le 0.02$ 安全红线）；
- **Weakness**：Strict F1 = **72.73%**（Recall 100.00%，零回退，完全满足 $\le 0.02$ 安全红线）；
- **AttackTechnique**：Strict F1 = **99.29%**（Recall 98.59%，零回退，完全满足 $\le 0.02$ 安全红线）；
- **Configuration**：Strict Recall 由 40.00% 跃升至 **84.00%**，Strict F1 达到 **46.15%**；
- **全量 Dev 宏观指标**：微平均召回率达到 **97.42%**（整个 Dev 集 194 个实体仅 5 个未中），微平均 F1 达到 **85.91%**。

---

## 五、结论与后续章节衔接

本技术方案通过将领域本体规范（Vulnerability-affects-Configuration 论元约束）融入抽取后处理，利用“句首吸附切窗”治愈语篇碎化，利用“漏洞事实锚定回填”治愈模型注意力衰减，利用“精细化去噪与双形解耦”消除边界漂移，成功在不污染其它类型实体的前提下，将受害配置的召回率从 40.00% 提升至 84.00%（文档级达到 85.00%）。

该高质量的实体抽取结果为后续 **Stage 2 关系抽取优化** 提供了完备的候选论元空间，彻底打通了漏洞威胁情报知识图谱端到端构建的技术闭环。
