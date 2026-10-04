"""law_rag.splitters — 法条感知切分（L2 数据层第二站，本项目检索质量的地基）。

【这个文件解决什么问题】
法律文本结构性极强：一"条"就是一个完整语义单元（检索命中它、引用引用它）。
通用 RecursiveCharacterTextSplitter 按 600 字滑窗切，会把"第七百零四条"
拦腰斩断——检索侧命中半句话，引用侧凑不出完整条文，可溯源（AC-2）无从谈起。
所以这里不用框架的切分器，自己按法律文本的真实结构切：

    编 → 章 → 节 → 条（首选边界）→ 句（超长条兜底）→ 字符（最后兜底）

【实测语料里的坑（都处理了）】
  1. 民法典开头是"目 录"+整份目录（目录行也是"第一章 …"样式）→ 统统落进
     preface 区，article_no 为空，不会污染条文 chunk；
  2. 民法典有"编"层级（第一编 总则）→ section 记"编 > 章 > 节"完整链；
  3. "附  则"是不带编号的章级标题 → 特判为 section，不并入上一条；
  4. 条内换行多为（一）（二）列举 → 行合并为空格，quote 才是单行字符串；
  5. 条号是中文数字（"第一千二百六十条"）→ 正则覆盖 零〇一…千万。

【切分粒度决策】
一 chunk = 一条（短条不合并）。理由：引用的原子单位是条，合并会让 article_no
失去意义；114 万字按条切约数千 chunk，FAISS IndexFlat 完全无压力。
仅当单条超过 CHUNK_SIZE（默认 600）才按句切并带 OVERLAP 重叠。
"""

from __future__ import annotations

import re

from law_rag.config import get_settings
from law_rag.schemas import LawChunk, LawDocument

# 中文数字条号：覆盖"第一条"到"第一千二百六十条"乃至更长（组1不含"第"，拼回即可）
_CN_NUM = r"[零〇一二三四五六七八九十百千万]+"
_ARTICLE_RE = re.compile(rf"^第({_CN_NUM}条)\s*(.*)$")
# 编/章/节标题行："第一编 总则"、"第二章 自然人"、"第三节 宣告失踪…"
_SECTION_RE = re.compile(rf"^(第{_CN_NUM}[编章节])\s*(.*)$")
# 不带编号的章级标题："附  则"（中间可能有全角/半角空白）
_APPENDIX_RE = re.compile(r"^附\s*则\s*$")
# 目录标记（民法典等开头的整份目录）
_TOC_RE = re.compile(r"^目\s*录\s*$")
# 句边界（用于超长条兜底切分）：句号/分号/问叹号，保留分隔符
_SENT_RE = re.compile(r"(?<=[。；！？])")

_SECTION_LABEL = {"编": None, "章": None, "节": None}  # 类型提示用，见 _SectionTracker


class _SectionTracker:
    """跟踪"编 > 章 > 节"层级链。遇到新"编"时重置章/节（民法典多编结构）。"""

    def __init__(self) -> None:
        self.parts: dict[str, str] = {}

    def update(self, label: str, title: str) -> None:
        kind = label[-1]  # "编" / "章" / "节"
        self.parts[kind] = f"{label} {title}".strip()
        # 新编重置下层；新章重置节
        if kind == "编":
            self.parts.pop("章", None)
            self.parts.pop("节", None)
        elif kind == "章":
            self.parts.pop("节", None)

    def set_appendix(self) -> None:
        self.parts = {"章": "附则"}

    def current(self) -> str:
        return " > ".join(self.parts[k] for k in ("编", "章", "节") if k in self.parts)


def _split_long_text(text: str, max_len: int, overlap: int) -> list[str]:
    """超长条兜底：按句边界贪心组装，段间保留尾部 overlap 字符。

    为什么在句边界切而不是字符滑窗：切断句子会制造"半句话 embedding"，
    语义向量质量下降；重叠的意义是让边界句在相邻块都完整出现。
    """
    if len(text) <= max_len:
        return [text]
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    pieces: list[str] = []
    buf = ""
    for sent in sentences:
        # 单句本身就超长（极罕见）：硬切
        while len(sent) > max_len:
            pieces.append(sent[:max_len])
            sent = sent[max_len - overlap:] if overlap < max_len else ""
        if len(buf) + len(sent) > max_len and buf:
            pieces.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            # 从上一个块的尾部恢复上下文，但不吞掉半句：找不到句界就整段带过来
            buf = tail + sent if tail else sent
        else:
            buf += sent
    if buf.strip():
        pieces.append(buf)
    return pieces


def _merge_lines(lines: list[str]) -> str:
    """条内多行合并：换行→空格，连续空白折叠成一个。"""
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_document(doc: LawDocument) -> list[LawChunk]:
    """把一部法规切成 LawChunk 列表（一 chunk = 一条；首部说明为 preface）。"""
    settings = get_settings()
    max_len, overlap = settings.chunk_size, settings.chunk_overlap

    chunks: list[LawChunk] = []
    tracker = _SectionTracker()

    preface_lines: list[str] = []
    current_art: str | None = None   # 当前条号；None 表示还在首部
    current_lines: list[str] = []

    def _emit(article_no: str, lines: list[str], section: str) -> None:
        """把累积的行组装成 1..n 个 chunk。"""
        text = _merge_lines(lines)
        if not text:
            return
        for i, piece in enumerate(_split_long_text(text, max_len, overlap)):
            chunks.append(
                LawChunk(
                    chunk_id=f"{doc.doc_id}:{article_no or 'preface'}:{i}",
                    doc_id=doc.doc_id,
                    law_title=doc.title,
                    article_no=article_no,
                    section=section,
                    text=piece,
                    effective_date=doc.effective_date,
                    source_url=doc.source_url,
                )
            )

    for raw in doc.text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        if _TOC_RE.match(line):
            # 目录标记本身没内容；目录行会继续落入 preface，不产条文污染
            preface_lines.append(line)
            continue

        sec_m = _SECTION_RE.match(line)
        if sec_m:
            # 章节行始终更新 tracker：目录里的章节行无害（正文同名行随后覆盖），
            # 而正文"第一条"之前的编/章行恰好靠它生效——否则首条 section 为空。
            tracker.update(sec_m.group(1), sec_m.group(2))
            if current_art is None:
                preface_lines.append(line)  # 目录里的章节行：留给 preface
            continue

        if _APPENDIX_RE.match(line):
            if current_art is not None:
                tracker.set_appendix()
            continue

        art_m = _ARTICLE_RE.match(line)
        if art_m:
            # 新条开始：先把上一条吐出来
            if current_art is not None:
                _emit(current_art, current_lines, tracker.current())
            else:
                _emit("", preface_lines, "")  # 首部（含目录）一次性吐出
                preface_lines = []
            current_art = f"第{art_m.group(1)}"
            current_lines = [art_m.group(2)] if art_m.group(2) else []
            continue

        # 普通行：归属当前条，否则归属首部
        (current_lines if current_art is not None else preface_lines).append(line)

    # 收尾：最后一条
    if current_art is not None:
        _emit(current_art, current_lines, tracker.current())
    elif preface_lines:
        _emit("", preface_lines, "")

    return chunks


def split_documents(docs: list[LawDocument]) -> list[LawChunk]:
    """批量切分（build_index 的入口）。"""
    out: list[LawChunk] = []
    for d in docs:
        out.extend(split_document(d))
    return out
