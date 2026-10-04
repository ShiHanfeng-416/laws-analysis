"""law_rag.engines.llamaindex_engine — LlamaIndex 实现的问答引擎（L3，M6）。

【链路】
    检索：共享 HybridRetriever（向量+BM25，与 LC 引擎同源——AC-5 对比的
          是"编排与生成"，检索一致才有可比性），适配成 LI 的 BaseRetriever
    编排：RetrieverQueryEngine + text_qa_template（与 LC 同款输出格式）
    生成：OpenAILike（OpenAI 兼容协议）
    引用：source_nodes 白名单 + 法条名文本回退

【为什么检索不走 FaissVectorStore（踩过坑后的决策）】
0.12 里包装外部裸 faiss 索引是可行的：TextNode 以 str(faiss_id) 对齐、
SimpleDocumentStore.add_documents 喂节点、再手动补 index_struct.nodes_dict
（retriever.py:166 按 id 反查，官方 load_index_from_storage 会重建该映射，
绕过官方存档就必须自己补）——三关都过了，检索能跑。
但这样 LI 是"纯向量"通道而 LC 是"向量+BM25"混合通道：口语化问题（如
"押金"类）LI 召回不到关键条文直接拒答，AC-5 实测重合度只有 36%。
统一走共享混合检索后，对比的才是编排与生成本身（LCEL vs QueryEngine）。

【与 LC 引擎的对照】
              LangChain 引擎              LlamaIndex 引擎
    检索      HybridRetriever（共享）      HybridRetriever（共享）
    编排      LCEL 管道                    QueryEngine/ResponseSynthesizer
    生成      ChatOpenAI                   OpenAILike
    引用      [n] 编号白名单               source_nodes 白名单+文本回退
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.llms.openai_like import OpenAILike
from pydantic import PrivateAttr

from law_rag.config import get_settings
from law_rag.embeddings import get_embedder
from law_rag.engines.base import QAEngine
from law_rag.retriever import HybridRetriever
from law_rag.schemas import AnswerResult, Citation
from law_rag.store import VectorStore

NO_EVIDENCE_SCORE = 0.12  # 与 LC 引擎同阈值，行为对齐

_QA_TMPL = PromptTemplate(
    "你是一名严谨的中国法律助理。只依据以下法条片段回答问题："
    "禁止引用片段之外的法条，禁止凭记忆补充条文内容。引用时标注片段编号（如 [1]）。\n"
    "---------------------\n{context_str}\n---------------------\n"
    "问题：{query_str}\n\n"
    "严格按以下格式输出，三段一个都不能少：\n"
    "【结论】（直接回答，两三句话）\n"
    "【依据】（用到的片段编号及一句话说明；片段与问题无关时只写「无」）\n"
    "【行动建议】（可操作的编号步骤；无则写「暂无」）\n"
    "若片段与问题无关，【结论】必须写「知识库中没有与该问题相关的依据」。"
)


class _HybridLIRetriever(BaseRetriever):
    """把共享的 HybridRetriever（L2）适配成 LI 的 BaseRetriever。

    返回 NodeWithScore 列表；node id 直接用 chunk_id（不再需要对齐 faiss
    内部 id——那是 FaissVectorStore 路径的约束，见模块 docstring）。
    """

    _hybrid: Any = PrivateAttr()
    _node_of: Any = PrivateAttr()
    _top_k: int = PrivateAttr(default=6)

    @classmethod
    def class_name(cls) -> str:
        return "LawHybridRetriever"

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        hits = self._hybrid.search(query_bundle.query_str, self._top_k)
        return [NodeWithScore(node=self._node_of[c.chunk_id], score=s)
                for c, s in hits]


class LlamaIndexEngine(QAEngine):
    name = "llamaindex"

    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._embedder = get_embedder()

        # ---- 1) 共享索引：读同一份三件套（指纹校验复用 L2）----
        store = VectorStore()
        store.load(self._embedder.name, self._embedder.dim)

        # ---- 2) TextNode 化（引用元数据挂 node.metadata）----
        def _to_node(cid: str, rec: dict) -> TextNode:
            return TextNode(
                id_=cid,
                text=f"《{rec['law_title']}》{rec['article_no']}（{rec['section']}）\n{rec['text']}",
                metadata={
                    "chunk_id": cid,
                    "doc_id": rec["doc_id"],
                    "law_title": rec["law_title"],
                    "article_no": rec["article_no"],
                    "effective_date": rec.get("effective_date", "未知"),
                    "source_url": rec.get("source_url", ""),
                },
            )

        node_of = {cid: _to_node(cid, rec) for cid, rec in store.docstore.items()}

        hybrid = HybridRetriever(store, self._embedder)
        retriever = _HybridLIRetriever()
        retriever._hybrid = hybrid
        retriever._node_of = node_of
        retriever._top_k = s.top_k

        llm = OpenAILike(
            api_base=s.llm_base_url, api_key=s.llm_api_key,  # 注意是 api_base 不是 base_url
            model=s.llm_model, temperature=0.1, is_chat_model=True,
            timeout=120,  # LLM 生成较长回答时 60s 默认值偶发超时
        )
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever, llm=llm, text_qa_template=_QA_TMPL,
        )
        # 无依据判断用向量分数（与 LC 引擎一致的兜底语义）
        self._plain_store = store

    def ask(self, question: str,
            history: list[tuple[str, str]] | None = None) -> AnswerResult:
        qv = self._embedder.embed_query(question)
        # 无依据判定与 LC 引擎同构：向量分 + 命中数双条件，行为对齐才可对比
        probe = self._plain_store.search(qv, 3)
        if not probe or (probe[0][1] < NO_EVIDENCE_SCORE and len(probe) < 3):
            return AnswerResult(
                conclusion="知识库中没有与该问题相关的依据。",
                citations=[], action_steps=[], engine=self.name,
            )

        # 多轮：LI 侧把最近两轮拼进 query_str（简化对齐 LC 行为）
        q = question
        if history:
            turns = history[-2:]
            ctx = "\n".join(f"（此前问答）用户：{a}\n助手：{b[:200]}" for a, b in turns)
            q = f"{ctx}\n（当前问题）{question}"

        response = self._query_engine.query(q)
        md = str(response)

        def _section(tag: str) -> str:
            m = re.search(rf"【{tag}】\s*(.*?)(?=【|$)", md, re.S)
            return (m.group(1) if m else "").strip()

        conclusion = _section("结论") or md.strip()
        basis = _section("依据")
        steps = [
            re.sub(r"^\d+[\.、\)]\s*", "", line).strip()
            for line in _section("行动建议").splitlines()
            if line.strip() and line.strip() != "暂无"
        ]
        # 引用白名单：source_nodes 即检索结果，LLM 标注的 [n] 只是筛选依据
        used = {int(n) for n in re.findall(r"\[(\d+)\]", basis)}
        source_nodes = list(response.source_nodes or [])
        cited: list[Citation] = []
        for n in sorted(used):
            if 1 <= n <= len(source_nodes):
                node = source_nodes[n - 1].node
                if node.metadata.get("article_no"):
                    cited.append(self._to_citation(node))

        # 回退：LI 合成器的 context 拼装未必保留 [n] 标号，此时按
        # 【依据】段出现的"《法规名》条号"对 source_nodes 做文本匹配
        if not cited and source_nodes:
            for node in source_nodes:
                m = node.metadata
                if not m.get("article_no"):
                    continue  # preface 块无条号，不作为引用
                if m.get("law_title", "") in basis or m.get("article_no", "") in basis:
                    cited.append(self._to_citation(node))
        return AnswerResult(
            conclusion=conclusion, citations=cited,
            action_steps=steps, engine=self.name,
        )

    @staticmethod
    def _to_citation(node) -> Citation:
        md_ = node.metadata
        return Citation(
            law_title=md_.get("law_title", ""),
            article_no=md_.get("article_no", ""),
            quote=node.text[:80].strip(),
            source_url=md_.get("source_url", ""),
            effective_date=md_.get("effective_date", "未知"),
        )
