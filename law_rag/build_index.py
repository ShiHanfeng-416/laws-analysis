"""law_rag.build_index — 建库 CLI（L4，组装 L2 流水线）。

用法：
    pixi run -e local-embed build-index            # 增量（默认）
    pixi run -e local-embed build-index --rebuild  # 忽略指纹与 hash，全量重建

【编排流程】
    load_laws ─▶ split_documents ─▶ embedder.embed_texts ─▶ store.add_chunks ─▶ save

【增量更新（FR-1.6 / AC-1）】
按源文件 sha256 与 index_meta.json 记录比对：
    新文件 → 整部入库；hash 变了 → 删旧 chunk 再入库；没变 → 跳过。
切分参数或 Embedding 指纹与索引不符 → 拒绝并提示重建（fail fast，防混库）。

【为什么在 local-embed 环境跑】
本地 LawVault 依赖 torch（约 5GB），只在 pixi 的 local-embed 特性环境安装；
API 通道（EMBED_PROVIDER=api）则任何环境都能跑。
"""

from __future__ import annotations

import argparse
import sys
import time

from law_rag.config import get_settings
from law_rag.loaders import load_laws
from law_rag.splitters import split_document
from law_rag.store import VectorStore, file_sha256


def build(rebuild: bool = False) -> VectorStore:
    settings = get_settings()
    print(f"[build] 语料目录 {settings.laws_dir}")

    docs = load_laws()
    if not docs:
        print("[build] 没有可用的法规语料（检查 data/laws/）")
        sys.exit(1)
    total_chars = sum(len(d.text) for d in docs)
    print(f"[build] 已加载 {len(docs)} 部法规，共 {total_chars / 10000:.1f} 万字")

    # 延迟 import：build_index 本身不触发模型加载（--help 等场景保持轻量）
    from law_rag.embeddings import get_embedder
    t0 = time.time()
    embedder = get_embedder()
    print(f"[build] Embedding 后端 {embedder.name}（{embedder.dim} 维，加载 {time.time() - t0:.1f}s）")

    store = VectorStore()
    if rebuild:
        store.create(embedder.dim, embedder.name)
    else:
        try:
            store.load(embedder.name, embedder.dim)
            print(f"[build] 已有索引：{len(store)} chunks")
        except FileNotFoundError:
            print("[build] 无既有索引，全新建库")
            store.create(embedder.dim, embedder.name)
        except RuntimeError as e:
            print(f"[build] {e}")
            sys.exit(2)

    known = store.file_hashes()
    stats = {"new": 0, "changed": 0, "unchanged": 0, "chunks": 0}
    t0 = time.time()
    for i, doc in enumerate(docs, 1):
        txt_path = settings.laws_dir / f"{doc.doc_id}.txt"
        sha = file_sha256(txt_path)
        if doc.doc_id + ".txt" in known and known[doc.doc_id + ".txt"] == sha:
            stats["unchanged"] += 1
            continue

        if doc.doc_id + ".txt" in known:  # hash 变化：先整部下架旧 chunk
            removed = store.remove_docs([doc.doc_id])
            stats["changed"] += 1
            tag = f"更新（删 {removed} 旧 chunk）"
        else:
            stats["new"] += 1
            tag = "新增"

        chunks = split_document(doc)
        vectors = embedder.embed_texts([c.embed_text for c in chunks])
        store.add_chunks(chunks, vectors)
        store.set_file_hash(doc.doc_id + ".txt", sha)
        stats["chunks"] += len(chunks)
        print(f"[build] ({i}/{len(docs)}) {doc.title}：{tag}，+{len(chunks)} chunks")

    store.save()
    dt = time.time() - t0
    print(
        f"[build] 完成：新增 {stats['new']} 部 / 更新 {stats['changed']} 部 / "
        f"未变 {stats['unchanged']} 部；本次入库 {stats['chunks']} chunks；"
        f"耗时 {dt:.0f}s；索引总量 {len(store)} chunks → {settings.index_dir}"
    )
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 / 增量更新 FAISS 向量索引")
    parser.add_argument("--rebuild", action="store_true",
                        help="忽略既有索引，全量重建")
    args = parser.parse_args()
    build(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
