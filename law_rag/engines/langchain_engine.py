"""law_rag.engines.langchain_engine — LCEL 实现的问答引擎（L3）。

【链路】
    检索（L2 store.search，双引擎共读同一份索引）
    → ChatPromptTemplate 注入法条片段 + 对话历史 + 问题
    → ChatOpenAI（OpenAI 兼容协议，DeepSeek/智谱/本地 Ollama 均可）
    → StrOutputParser
    → 自研解析器拆成 AnswerResult（结论/引用/建议/免责声明）

【反幻觉设计（本项目立项动机，三道闸）】
  1. 检索分数兜底：Top-1 余弦低于 NO_EVIDENCE_SCORE 就不调 LLM，
     直接返回"无依据"——省钱、省时、且不给模型编的机会；
  2. Prompt 硬约束：只准依据片段作答，引用必须标片段编号；
  3. 引用白名单：LLM 标了 [n] 才进 citations，n 超出检索结果范围一律丢弃
     ——模型嘴里的"法条"不允许绕过检索结果直接出现在引用列表里（AC-2 的根基）。

【为什么检索不走 LangChain 的 FAISS retriever】
索引三件套是自管格式（ADR-D2），LC/LI 只是"读"它；检索统一走 L2 的
store.search，双引擎拿到同一份候选再各自编排——对比评测（AC-5）才公平。
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from law_rag.config import get_settings
from law_rag.embeddings import get_embedder
from law_rag.engines.base import QAEngine
from law_rag.retriever import HybridRetriever
from law_rag.schemas import AnswerResult, Citation
from law_rag.store import VectorStore

# 第一道闸的阈值：向量 Top-1 余弦 + 融合结果数量双条件（单看分数会误杀
# "BM25 精确命中但向量分不高"的查询，如口语"押金"类问题）
NO_EVIDENCE_SCORE = 0.12

_SYSTEM = (
    "你是一名严谨的中国法律助理。只依据 <法条片段> 回答问题："
    "禁止引用片段之外的法条，禁止凭记忆补充条文内容。"
    "引用时必须标注片段编号（如 [1]、[3]）。"
)

_HUMAN = """<法条片段>
{context}
</法条片段>

{history_line}问题：{question}

严格按以下格式输出，四段一个都不能少：
【结论】（直接回答，两三句话）
【依据】（用到的片段编号及一句话说明，如：[1] 租赁合同的内容；片段与问题无关时只写"无"）
【行动建议】（可操作的编号步骤；无则写"暂无"）

再次强调：若片段与问题无关，【结论】必须写"知识库中没有与该问题相关的依据"。"""


class LangChainEngine(QAEngine):
    name = "langchain"

    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._embedder = get_embedder()
        self._store = VectorStore()
        self._store.load(self._embedder.name, self._embedder.dim)  # 指纹校验在内
        self._retriever = HybridRetriever(self._store, self._embedder)
        self._llm: Runnable = (
            ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
            | ChatOpenAI(
                base_url=s.llm_base_url, api_key=s.llm_api_key,
                model=s.llm_model, temperature=0.1,
            )
            | StrOutputParser()
        )

    # ---- 检索与上下文 ----
    def _retrieve(self, question: str):
        return self._retriever.search(question, self._settings.top_k)

    @staticmethod
    def _format_context(hits) -> str:
        return "\n\n".join(
            f"[{i}] 《{c.law_title}》{c.article_no}（{c.section}）\n{c.text}"
            for i, (c, _score) in enumerate(hits, 1)
        )

    # ---- 输出解析 ----
    @staticmethod
    def _section(md: str, tag: str) -> str:
        m = re.search(rf"【{tag}】\s*(.*?)(?=【|$)", md, re.S)
        return (m.group(1) if m else "").strip()

    def _parse(self, md: str, hits) -> AnswerResult:
        conclusion = self._section(md, "结论") or md.strip()
        basis = self._section(md, "依据")
        steps = [
            re.sub(r"^\d+[\.、\)]\s*", "", line).strip()
            for line in self._section(md, "行动建议").splitlines()
            if line.strip() and line.strip() != "暂无"
        ]
        # 引用白名单：只收 LLM 实际标注且在检索结果范围内的编号
        used = {int(n) for n in re.findall(r"\[(\d+)\]", basis)}
        citations: list[Citation] = []
        for n in sorted(used):
            if 1 <= n <= len(hits):
                c = hits[n - 1][0]
                citations.append(Citation(
                    law_title=c.law_title, article_no=c.article_no,
                    quote=c.text[:80].strip(),
                    source_url=c.source_url, effective_date=c.effective_date,
                ))
        return AnswerResult(
            conclusion=conclusion, citations=citations,
            action_steps=steps, engine=self.name,
        )

    # ---- 主流程 ----
    def ask(self, question: str,
            history: list[tuple[str, str]] | None = None) -> AnswerResult:
        hits = self._retrieve(question)

        vec_top = self._retriever.vector_top_score(question)
        if not hits or (vec_top < NO_EVIDENCE_SCORE and len(hits) < 3):
            return AnswerResult(  # 第一道闸：不调 LLM，直接无依据（FR-5.2）
                conclusion="知识库中没有与该问题相关的依据。",
                citations=[], action_steps=[], engine=self.name,
            )

        history_line = ""
        if history:
            turns = history[-2:]  # 只带最近两轮，控制上下文长度
            rendered = "\n".join(f"用户：{q}\n助手：{a[:200]}" for q, a in turns)
            history_line = f"<对话历史>\n{rendered}\n</对话历史>\n\n"

        md = self._llm.invoke({
            "context": self._format_context(hits),
            "history_line": history_line,
            "question": question,
        })
        return self._parse(md, hits)
