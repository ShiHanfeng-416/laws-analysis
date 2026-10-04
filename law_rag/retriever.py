"""law_rag.retriever — 混合检索（向量 + BM25，RRF 融相）（L2，FR-4.3）。

【为什么需要它：真实踩坑】
上线 M4 首测即翻车：问"房东不退押金怎么办"，纯向量 top6 全是"租金/转租"
条文（余弦 0.5+），而库里 10 条真正讲"押金"的条文（《住房租赁条例》第十条等）
只拿到 0.29~0.36——口语问句与法言法语的相似度拼不过"同章节近邻"。
这是纯向量检索的结构性弱点：**关键词的精确命中**（押金、定金、条文号）它做不好。

【方案：双通道 + RRF】
    向量通道（语义）：store.search（LawVault 余弦）
    BM25 通道（词面）：字符 bigram 中文分词 + 手写 BM25（零依赖）
    融合：RRF（Reciprocal Rank Fusion），score = Σ 1/(k+rank)，k=60
选 RRF 而非加权分数求和：两通道的分数量纲不可比（余弦 0~1 vs BM25 无界），
RRF 只用排名，天然免调参——工程上"少一个要调的参数"本身就是优点。

【为什么 BM25 自实现】
rank_bm25 一行 pip 能解决，但中文 bigram 分词 + 40 行 BM25 是本项目
"共享层零框架依赖"（R2）的自然延伸，也顺便把 BM25 公式留在代码库里可审计。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from law_rag.embeddings import EmbeddingBackend
from law_rag.schemas import LawChunk
from law_rag.store import VectorStore

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文按字符 bigram（"押金"→['押金']，单字弃掉），ASCII 按词。

    bigram 而非逐字：单字（"的""人"）在中文里停用词化严重，bigram 恰好
    对齐大多数法律关键词（押金/定金/违约金/三包），且无需分词依赖。
    """
    out: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    out.extend(w.lower() for w in _WORD.findall(text))
    return out


class BM25Index:
    """Okapi BM25（k1=1.5, b=0.75），语料规模 3k+ 篇档，纯 python 秒级构建。"""

    def __init__(self, docs: list[list[str]]) -> None:
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        self.doc_len = [len(d) for d in docs]
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def search(self, query: list[str], top_k: int) -> list[tuple[int, float]]:
        """返回 (doc 序号, bm25 分) top_k。"""
        scores: list[float] = [0.0] * self.n
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counter in enumerate(self.tf):
                f = counter.get(term, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                scores[i] += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / self.avgdl))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked if s > 0][:top_k]


class HybridRetriever:
    """向量 + BM25 双通道检索，RRF 融合。engines 的统一检索入口。"""

    RRF_K = 60  # RRF 标准平滑常数

    def __init__(self, store: VectorStore, embedder: EmbeddingBackend) -> None:
        self._store = store
        self._embedder = embedder
        # BM25 索引按 docstore 快照构建；增量更新后需 rebuild（build_index 负责钩住）
        self._chunks: list[LawChunk] = []
        corpus = [tokenize(c.text) for c in self._snapshot()]
        self._bm25 = BM25Index(corpus)

    def _snapshot(self) -> list[LawChunk]:
        self._chunks = [
            LawChunk(
                chunk_id=cid, doc_id=r["doc_id"], law_title=r["law_title"],
                article_no=r["article_no"], section=r["section"], text=r["text"],
                effective_date=r.get("effective_date", "未知"),
                source_url=r.get("source_url", ""),
            )
            for cid, r in self._store.docstore.items()
        ]
        return self._chunks

    def rebuild_bm25(self) -> None:
        """增量更新（增删法规）后重建 BM25 语料（秒级）。"""
        corpus = [tokenize(c.text) for c in self._snapshot()]
        self._bm25 = BM25Index(corpus)

    def vector_top_score(self, query: str) -> float:
        """仅供无依据短路判断使用。"""
        hits = self._store.search(self._embedder.embed_query(query), 1)
        return hits[0][1] if hits else 0.0

    def search(self, query: str, top_k: int) -> list[tuple[LawChunk, float]]:
        widen = top_k * 3  # 各通道多召回，融合后截断
        vec_hits = self._store.search(self._embedder.embed_query(query), widen)
        bm_hits = self._bm25.search(tokenize(query), widen)

        rrf: dict[str, float] = {}
        for rank, (c, _s) in enumerate(vec_hits, 1):
            rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (self.RRF_K + rank)
        by_id = {c.chunk_id: (c, s) for c, s in [*vec_hits, *self._pick(bm_hits)]}
        for rank, (i, _s) in enumerate(bm_hits, 1):
            cid = self._chunks[i].chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)

        ordered = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        return [(by_id[cid][0], score) for cid, score in ordered]

    def _pick(self, bm_hits) -> list[tuple[LawChunk, float]]:
        return [(self._chunks[i], s) for i, s in bm_hits if i < len(self._chunks)]
