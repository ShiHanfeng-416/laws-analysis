"""law_rag.schemas — 全项目通用的数据模型（L1 基础层，零框架依赖）。

【这个文件解决什么问题】
项目是双引擎的（LangChain / LlamaIndex），两家的文档类型分别是 Document 与
TextNode。如果 L2 数据层直接用其中任何一家，另一家就永久"从属"，共享层也被迫
依赖某个框架（架构 ADR-D1）。所以这里定义一组**自有**的 frozen dataclass 作为
全流水线的"普通话"：loaders 产出 LawDocument，splitters 产出 LawChunk，
store 读写 LawChunk，引擎侧各自做一行转换回框架类型。

【数据流】
    loaders ─▶ LawDocument ─splitters─▶ LawChunk ─store─▶ 磁盘三件套
    引擎问答 ─▶ AnswerResult（含 Citation 列表）      审查 ─▶ ReviewFinding

【三条铁律】
  1. 本模块不 import langchain / llama_index / faiss（架构 R2），可被任意层依赖；
  2. 不含任何 I/O（不读文件、不发请求）——纯数据形状定义；
  3. 字段是不可变"事实"：frozen dataclass，运行期篡改当场抛错。
     注意 frozen 是浅冻结：字段绑定不可改，但 list 内容仍可 append，
     深层不可变靠约定（构造后不再修改容器内容）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 固定免责声明：AC-6 要求所有回答必带，与其各处复制字符串不如单点定义。
DISCLAIMER = (
    "本回答由本地法规库检索生成，仅供参考，不构成正式法律意见；"
    "重大事项请咨询执业律师。"
)


@dataclass(frozen=True)
class LawDocument:
    """一部法规的完整内容 + 元数据（loaders 的输出）。

    字段与 data/laws/*.meta.json 一一对应；meta 缺失或为 null 的字段
    由 loaders 兜底成 "未知"，保证下游永远拿到完整字符串。
    """

    doc_id: str          # 文件名去扩展名，如 "zhufang_zulin_tiaoli"
    title: str           # 法规名，如 "住房租赁条例"
    law_type: str        # 法律/行政法规/司法解释/部门规章
    issued_by: str       # 制定机关
    publish_date: str    # 公布日期（YYYY-MM-DD 或 "未知"）
    effective_date: str  # 施行日期（引用展示用，AC-6）
    source_url: str      # 官方来源链接（引用展示用，可回溯核对）
    text: str            # 全文正文


@dataclass(frozen=True)
class LawChunk:
    """一个切分后的法条块（splitters 的输出、store 的写入单元）。

    text 是"给人看的"纯正文（展示与引用），embed_text 是"给模型看的"
    向量化输入——两者刻意分离：LawVault 训练格式带 title 前缀，直接把
    带前缀的串存进 docstore 会污染引用展示（AC-2 要求引用逐字可查）。
    """

    chunk_id: str        # f"{doc_id}:{article_no 或 preface}:{seq}"，全局唯一
    doc_id: str          # 所属法规
    law_title: str       # 法规名（冗余存一份，检索结果可直接组装引用）
    article_no: str      # 如 "第七百零四条"；非条文块为 ""
    section: str         # 所属章节名，如 "第二编 合同"；无章节为 ""
    text: str            # 纯正文
    # 以下两个 doc 级字段冗余进 chunk（架构 §6：docstore 记录含施行日期与来源，
    # 引用组装 Citation 时直接从检索结果取，免去再查一遍元数据）
    effective_date: str = "未知"   # 施行日期（AC-6）
    source_url: str = ""           # 官方来源链接（FR-5.3）

    @property
    def embed_text(self) -> str:
        """向量化输入：逐字对齐 LawVault 训练格式（架构 ADR-D4）。

        训练语料形如 "title: 中华人民共和国XX法 第五十条 | text: 第五十条 …"，
        推理侧必须同样拼接，否则向量空间错位、检索质量崩塌。
        """
        return f"title: {self.law_title} | text: {self.text}"


@dataclass(frozen=True)
class Citation:
    """一条引用（FR-5.3）：回答里的每个结论都要能落到这个结构。"""

    law_title: str
    article_no: str
    quote: str           # 原文摘要，必须能在 data/laws/ 逐字查到（AC-2）
    source_url: str
    effective_date: str  # 施行日期（AC-6：让读者判断法规时效）


@dataclass(frozen=True)
class AnswerResult:
    """双引擎统一输出（FR-5.6）：结构一致才能直接 diff 对比（AC-5）。"""

    conclusion: str                       # 结论
    citations: list[Citation] = field(default_factory=list)  # 依据引用
    action_steps: list[str] = field(default_factory=list)    # 行动建议
    disclaimer: str = DISCLAIMER          # 固定免责声明（AC-6）
    engine: str = "langchain"             # "langchain" | "llamaindex"，标记来源


@dataclass(frozen=True)
class ReviewFinding:
    """合同审查的一条风险发现（FR-6.2）：规则命中或模型检出的"坑"。"""

    clause_text: str              # 条款原文
    risk_level: str               # "高" | "中" | "低"
    explanation: str              # 问题说明
    suggestion: str               # 建议改法
    # 带默认值的字段必须排在无默认字段之后（dataclass 规则）
    legal_basis: list[Citation] = field(default_factory=list)  # 依据法条
