"""law_rag.store — 裸 FAISS 索引存取（L2，双引擎共享索引的根基，ADR-D2）。

【这个文件解决什么问题】
LangChain 的 FAISS 包装落盘 index.pkl，LlamaIndex 落盘自有 docstore.json 结构，
两家格式互不兼容——"同一份索引文件给两引擎复用"（FR-3.5）在框架体系内无解。
解法是绕开两框架，直接操作裸 faiss 库，自管"三件套"：

    index.faiss      IndexIDMap2(IndexFlatIP) + 归一化向量 = 余弦相似度
    docstore.json    chunk_id → {文本, 法规名, 条号, 章节, faiss_id, …}
    index_meta.json  模型指纹（name/dim）+ 切分参数 + 各源文件 hash（增量依据）

IndexIDMap2 而非裸 IndexFlatIP：前者支持 remove_ids——增量更新"删旧 chunk
再合新向量"的前提（FR-1.6）。

【指纹校验（FR-3.4）】
load 时传入当前的模型名/维度，与 index_meta.json 不一致 → RuntimeError 拒绝加载。
原因：768 维索引混进 1024 维查询向量，faiss 直接段错误级别崩溃；即使维度碰巧
相同，不同模型的向量空间也毫无意义。宁可让用户重建，不给错误答案。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from law_rag.config import get_settings
from law_rag.schemas import LawChunk

_DOC_FIELDS = ("doc_id", "law_title", "article_no", "section", "text",
               "effective_date", "source_url")


def file_sha256(path: Path) -> str:
    """源文件指纹：增量更新的判断依据（内容级，不受 mtime 触碰影响）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class VectorStore:
    """索引三件套的持有者：建库时 add，检索时 search，更新时 remove+add。"""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.docstore: dict[str, dict] = {}          # chunk_id -> chunk 数据（含 faiss_id）
        self._faiss_ids: dict[int, str] = {}          # faiss_id -> chunk_id（搜索结果反查）
        self._next_id: int = 0
        self.meta: dict = {}

    # ---------- 生命周期 ----------
    @property
    def index_dir(self) -> Path:
        return get_settings().index_dir

    def create(self, dim: int, embedding_name: str) -> None:
        """新建空索引（建库入口）。"""
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.docstore, self._faiss_ids, self._next_id = {}, {}, 0
        s = get_settings()
        self.meta = {
            "embedding_model": embedding_name,
            "dim": dim,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {},   # 源文件名 -> sha256
        }

    def save(self) -> None:
        assert self.index is not None, "空索引无需保存"
        d = self.index_dir
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        (d / "docstore.json").write_text(
            json.dumps(self.docstore, ensure_ascii=False), encoding="utf-8")
        (d / "index_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self, embedding_name: str, dim: int) -> None:
        """加载并校验指纹。不存在 → FileNotFoundError（调用方决定建新库）；
        指纹不符 → RuntimeError（模型/维度变了，旧索引必须重建）。"""
        d = self.index_dir
        for name in ("index.faiss", "docstore.json", "index_meta.json"):
            if not (d / name).exists():
                raise FileNotFoundError(f"索引不完整，缺少 {name}（{d}）")

        self.meta = json.loads((d / "index_meta.json").read_text(encoding="utf-8"))
        if (self.meta.get("embedding_model") != embedding_name
                or self.meta.get("dim") != dim):
            raise RuntimeError(
                f"索引指纹不符：索引为 {self.meta.get('embedding_model')}/"
                f"{self.meta.get('dim')} 维，当前 {embedding_name}/{dim} 维。"
                f"Embedding 模型或维度已变化，请删除 {d} 后全量重建。"
            )
        if (self.meta.get("chunk_size") != get_settings().chunk_size
                or self.meta.get("chunk_overlap") != get_settings().chunk_overlap):
            raise RuntimeError(
                "切分参数（CHUNK_SIZE/CHUNK_OVERLAP）与建库时不一致，"
                "请恢复配置或删除索引重建。"
            )

        self.index = faiss.read_index(str(d / "index.faiss"))
        self.docstore = json.loads((d / "docstore.json").read_text(encoding="utf-8"))
        self._faiss_ids = {rec["faiss_id"]: cid for cid, rec in self.docstore.items()}
        self._next_id = (max(self._faiss_ids) + 1) if self._faiss_ids else 0

    # ---------- 写入 / 删除 ----------
    def add_chunks(self, chunks: list[LawChunk], vectors: np.ndarray) -> None:
        """向量与 chunk 一起入库（vectors 必须已归一化、与 chunks 等长同序）。"""
        assert self.index is not None, "先 create() 或 load()"
        if not chunks:
            return
        # 防重：chunk_id 是 docstore 的 key，重复会静默覆盖成"向量孤儿"
        #（faiss 里两条向量指向同一 chunk_id）。曾因语料文件正文重复两遍触发，
        # 这类 bug 检索结果"看起来正常"，必须在建库入口拦下。
        clashed = [c.chunk_id for c in chunks if c.chunk_id in self.docstore]
        if clashed:
            raise ValueError(f"chunk_id 重复入库：{clashed[:3]}…共 {len(clashed)} 个")
        ids = np.arange(self._next_id, self._next_id + len(chunks), dtype=np.int64)
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)
        for chunk, fid in zip(chunks, ids.tolist()):
            rec = {k: getattr(chunk, k) for k in _DOC_FIELDS}
            rec["faiss_id"] = fid
            self.docstore[chunk.chunk_id] = rec
            self._faiss_ids[fid] = chunk.chunk_id
        self._next_id += len(chunks)

    def remove_docs(self, doc_ids: list[str]) -> int:
        """按法规 doc_id 删除其全部 chunk（增量更新：旧版法规整部下架）。"""
        assert self.index is not None
        victims = [cid for cid, rec in self.docstore.items() if rec["doc_id"] in doc_ids]
        if not victims:
            return 0
        fids = np.array([self.docstore[c]["faiss_id"] for c in victims], dtype=np.int64)
        self.index.remove_ids(fids)
        for cid in victims:
            self._faiss_ids.pop(self.docstore[cid]["faiss_id"], None)
            self.docstore.pop(cid)
        return len(victims)

    def set_file_hash(self, filename: str, sha: str) -> None:
        self.meta["files"][filename] = sha

    def file_hashes(self) -> dict[str, str]:
        return dict(self.meta.get("files", {}))

    # ---------- 检索 ----------
    def search(self, query_vec: np.ndarray, top_k: int,
               doc_ids: set[str] | None = None) -> list[tuple[LawChunk, float]]:
        """余弦检索 top_k。doc_ids 非空时做元数据过滤（FR-4.2）：
        拉多一些候选在 docstore 侧过滤后截断——比逐条重排省事且正确。"""
        assert self.index is not None, "索引未加载"
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        if doc_ids is not None:
            k = min(self.index.ntotal, max(top_k * 4, top_k))
        else:
            k = min(self.index.ntotal, top_k)
        if k == 0:
            return []
        scores, ids = self.index.search(q, k)
        results: list[tuple[LawChunk, float]] = []
        for score, fid in zip(scores[0].tolist(), ids[0].tolist()):
            cid = self._faiss_ids.get(int(fid))
            if cid is None:
                continue  # IDMap 删除后的空洞
            rec = self.docstore[cid]
            if doc_ids is not None and rec["doc_id"] not in doc_ids:
                continue
            results.append((LawChunk(
                chunk_id=cid, doc_id=rec["doc_id"], law_title=rec["law_title"],
                article_no=rec["article_no"], section=rec["section"], text=rec["text"],
                effective_date=rec.get("effective_date", "未知"),
                source_url=rec.get("source_url", ""),
            ), float(score)))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self.docstore)
