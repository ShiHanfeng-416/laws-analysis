"""law_rag.loaders — 法规语料加载（L2 数据层第一站）。

【这个文件解决什么问题】
data/laws/ 下是 33 部法规的成对文件：`{slug}.txt`（正文）+ `{slug}.meta.json`
（爬虫抓的元数据）。下游 splitters/store 需要的是"正文 + 结构化元数据"合体，
本模块负责扫描、配对、合并、兜底，输出 list[LawDocument]。

【实现要点】
  1. 元数据兜底：meta.json 里 issued_by / effective_date 等键可能为 null
     （部门规章来源杂），缺失时兜底成 "未知"——下游（引用展示、按层级过滤）
     拿到的永远是完整 str，不必到处判空；
  2. 编码显式 utf-8：Windows 默认 GBK，不显式声明会在中文正文上炸
     UnicodeDecodeError 或静默乱码；
  3. txt 首行是法规标题，与 meta.title 重复，加载时剥掉，避免切分后每个
     chunk 头部出现两次标题；
  4. 孤儿文件（有 txt 无 meta，或反之）记警告、跳过——语料目录是"配置"，
     坏文件不该让整库构建崩溃；
  5. _manifest.json 是爬虫的下载清单，不是某部法规的 meta，按命名约定排除。
"""

from __future__ import annotations

import json
from pathlib import Path

from law_rag.config import get_settings
from law_rag.schemas import LawDocument

# meta.json 里值为 null / 空串时统一兜底成这个
UNKNOWN = "未知"


def _load_meta(meta_path: Path) -> dict:
    """读一个 meta.json，损坏时抛出带文件名的错误（方便定位是哪部法规坏了）。"""
    try:
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"元数据文件损坏：{meta_path.name}（{e}）") from e


def _clean(value: object) -> str:
    """null / 空白 / 非字符串 → "未知"；正常字符串去首尾空白。"""
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN
    return value.strip()


def load_laws(laws_dir: Path | None = None) -> list[LawDocument]:
    """扫描语料目录，加载全部法规。目录不存在时抛 FileNotFoundError（fail fast，
    多半是 LAWS_DIR 配错，提示用户比静默返回空列表更省排查时间）。"""
    settings = get_settings()
    root = laws_dir or settings.laws_dir
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{root}（检查 LAWS_DIR 配置）")

    docs: list[LawDocument] = []
    for txt_path in sorted(root.glob("*.txt")):
        # xxx.txt → xxx.meta.json（不用 with_suffix：slug 本身可能含点）
        meta_path = txt_path.with_name(txt_path.stem + ".meta.json")
        if not meta_path.exists():
            print(f"[loaders] 警告：{txt_path.name} 缺少同名 meta.json，已跳过")
            continue

        meta = _load_meta(meta_path)
        text = txt_path.read_text(encoding="utf-8").strip()

        # 剥掉与 meta.title 重复的首行标题（fetch 阶段写入，此处冗余）
        first_line = text.split("\n", 1)[0].strip()
        if meta.get("title") and first_line == _clean(meta.get("title")):
            text = text.split("\n", 1)[1].strip() if "\n" in text else ""

        docs.append(
            LawDocument(
                doc_id=txt_path.stem,
                title=_clean(meta.get("title")) if meta.get("title") else txt_path.stem,
                law_type=_clean(meta.get("law_type")),
                issued_by=_clean(meta.get("issued_by")),
                publish_date=_clean(meta.get("publish_date")),
                effective_date=_clean(meta.get("effective_date")),
                source_url=_clean(meta.get("source_url")),
                text=text,
            )
        )
    return docs
