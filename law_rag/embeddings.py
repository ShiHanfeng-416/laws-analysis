"""law_rag.embeddings — Embedding 双通道工厂（L2，引擎中立，不 import 框架）。

【这个文件解决什么问题】
架构 ADR-D4：本地 LawVault（离线/免费/法律微调）与 OpenAI 兼容 API（bge-m3）
二选一，由 EMBED_PROVIDER 切换。本模块把两条通道收敛成同一个后端接口，
下游（store 建库、engines 检索）完全不感知差别。

【最容易踩的坑：训练格式对齐】
LawVault 基于 EmbeddingGemma，训练时文档/查询带不同前缀（模型自带
config_sentence_transformers.json 的 prompts 字段）：
  - 文档侧 "title: {标题} | text: {正文}" —— LawChunk.embed_text 已拼好，
    encode 时**必须关闭** prompt，否则会叠成 "title: none | text: title: …"；
  - 查询侧 "task: search result | query: {问题}" —— 用 prompt_name="query"
    让 sentence-transformers 自动加前缀。
两侧前缀错配是检索质量隐形杀手：不报错，只是 Top-1 掉十几个点。

【依赖隔离】
sentence-transformers/torch 只在 local-embed 特性环境里存在（约 5GB），
所以 import 必须延迟到 local 分支内部——base 环境（跑 CLI/UI）不装 torch 也能
import 本模块（只要不用 local 通道）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from law_rag.config import get_settings


class EmbeddingBackend(Protocol):
    """统一后端接口：两通道、两引擎共用的最小契约。"""

    name: str   # 模型指纹（写入 index_meta.json，换模型=指纹变=全量重建）
    dim: int    # 向量维度（建 IndexFlatIP 前必须知道）

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """文档侧批量向量化，返回 (n, dim) 的 float32 **归一化**矩阵。"""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """查询侧单条向量化，返回 (dim,) 的 float32 归一化向量。"""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2 归一化：内积（IndexFlatIP）在归一化后等价余弦相似度。"""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LocalLawVaultBackend:
    """本地 LawVault（sentence-transformers）。需在 pixi -e local-embed 环境运行。"""

    def __init__(self, model_path: str, batch_size: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # base 环境没装 torch，给出可操作的提示
            raise RuntimeError(
                "当前环境缺少 sentence-transformers；"
                "本地 Embedding 请用：pixi run -e local-embed …"
            ) from e
        self._model = SentenceTransformer(str(model_path))
        self._batch = batch_size
        # 指纹用模型目录名而非绝对路径：换机器/换盘符不该导致"指纹不符须重建"
        self.name = f"local:{Path(model_path).name}"
        # sentence-transformers 5.x 改名 get_embedding_dimension，旧名兼容 4.x
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # prompt_name 刻意不传：调用方传入的已是 "title: … | text: …" 训练格式
        vecs = self._model.encode(
            texts, batch_size=self._batch, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return _normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        # 查询侧加训练时前缀 "task: search result | query: "（模型配置自带）
        vec = self._model.encode(
            [query], prompt_name="query",
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return _normalize(vec)[0]


class ApiEmbeddingBackend:
    """OpenAI 兼容 /embeddings 接口（bge-m3 等）。无特殊前缀，原样送编码。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 batch_size: int = 64) -> None:
        if not (base_url and api_key and model):
            raise ValueError(
                "EMBED_PROVIDER=api 需要同时配置 EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL"
            )
        from openai import OpenAI  # 惰性 import：openai 是 langchain-openai 的传递依赖
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._batch = batch_size
        self.name = f"api:{model}"
        self.dim: int = 0  # 首次调用时探测（bge-m3=1024，不同模型不同）

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = self._client.embeddings.create(model=self._model, input=texts[i:i + self._batch])
            out.extend(d.embedding for d in resp.data)
        if not self.dim:
            self.dim = len(out[0])
        return _normalize(np.array(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query])[0]


def get_embedder() -> EmbeddingBackend:
    """工厂入口：按 Settings 造后端。"""
    s = get_settings()
    if s.embed_provider == "local":
        return LocalLawVaultBackend(str(s.embed_model_path), s.embed_batch_size)
    return ApiEmbeddingBackend(s.embed_base_url, s.embed_api_key, s.embed_model)
