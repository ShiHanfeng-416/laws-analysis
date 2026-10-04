"""law_rag.review — 合同风险审查 CLI（L4，FR-6）。

用法：
    pixi run -e local-embed review examples/rental_contract.txt
    pixi run -e local-embed review contract.docx --out report.md

【流程：规则先行，RAG 佐证】
    读合同（.txt/.docx）→ rules.yaml 逐规则正则扫描 → 命中片段截上下文
    → 每条命中的 law_query 走混合检索取依据法条 → ReviewFinding 列表
    → rich 表格输出 + 可选 Markdown 报告

【设计取舍：为什么是"正则 + 检索"而不是"整份合同扔给 LLM"】
  1. 可解释：每条风险对应明文规则与库内法条，不依赖模型自由发挥；
  2. 可控成本：LLM 只在需要语义判断时介入（本版规则已足够锋利，未启用）；
  3. 依据可信：law_query 检索到的引用与问答侧同一套白名单逻辑（AC-4 的根基）。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from law_rag.config import get_settings
from law_rag.schemas import Citation, ReviewFinding

console = Console()

_RISK_STYLE = {"高": "red bold", "中": "yellow", "低": "dim"}


def load_rules() -> list[dict]:
    """加载规则库。规则文件是"配置"：坏了直接 fail fast。"""
    path = Path(__file__).with_name("rules.yaml")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rules = data.get("rules") or []
    for r in rules:
        r["_re"] = re.compile(r["pattern"])   # 预编译，扫描循环外做一次
    return rules


def read_contract(path: Path) -> str:
    """.txt 直读；.docx 用 python-docx 抽段落（PDF 支持留待演进项）。"""
    if path.suffix.lower() == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


def _clip(text: str, start: int, end: int, width: int = 48) -> str:
    """命中片段连同前后文各取一句范围，报告里"能定位到条款"比精确边界更重要。"""
    s = max(0, start - width)
    e = min(len(text), end + width)
    return ("…" if s > 0 else "") + text[s:e].replace("\n", " ") + ("…" if e < len(text) else "")


def review(contract_text: str) -> list[ReviewFinding]:
    """核心审查：规则扫描 + 检索佐证（检索器惰性初始化，纯规则场景零模型开销）。"""
    findings: list[ReviewFinding] = []
    rules = load_rules()
    hit_queries: set[str] = set()

    for rule in rules:
        for m in rule["_re"].finditer(contract_text):
            findings.append(ReviewFinding(
                clause_text=_clip(contract_text, m.start(), m.end()),
                risk_level=rule["level"],
                explanation=f"[{rule['id']}] {rule['name']}：{rule['explain'].strip()}",
                suggestion=rule["advice"].strip(),
            ))
            hit_queries.add(rule["law_query"])

    if not findings:
        return []

    # 每个不同 law_query 检索一次，同 query 的多命中共享依据（省时省钱）
    citations_by_query: dict[str, list[Citation]] = {}
    if hit_queries:
        from law_rag.embeddings import get_embedder
        from law_rag.retriever import HybridRetriever
        from law_rag.store import VectorStore
        embedder = get_embedder()
        store = VectorStore()
        store.load(embedder.name, embedder.dim)
        retriever = HybridRetriever(store, embedder)
        top_k = get_settings().top_k
        for q in hit_queries:
            hits = retriever.search(q, 2)
            citations_by_query[q] = [
                Citation(
                    law_title=c.law_title, article_no=c.article_no,
                    quote=c.text[:80].strip(), source_url=c.source_url,
                    effective_date=c.effective_date,
                ) for c, _s in hits if c.article_no
            ]

    # 把依据挂回对应规则（按规则 id 反查 law_query）
    rules_by_id = {r["id"]: r for r in rules}
    findings2: list[ReviewFinding] = []
    for f in findings:
        rid = f.explanation.split("]")[0].lstrip("[")
        q = rules_by_id[rid]["law_query"]
        findings2.append(ReviewFinding(
            clause_text=f.clause_text, risk_level=f.risk_level,
            explanation=f.explanation, suggestion=f.suggestion,
            legal_basis=citations_by_query.get(q, []),
        ))
    order = {"高": 0, "中": 1, "低": 2}
    findings2.sort(key=lambda f: order.get(f.risk_level, 3))
    return findings2


def render(findings: list[ReviewFinding], source: str) -> None:
    if not findings:
        console.print(Panel("未命中规则库风险条款（不代表合同无风险，重大事项请咨询律师）",
                            title="审查结果", border_style="green"))
        return
    table = Table(title=f"合同审查：{source}（命中 {len(findings)} 条）")
    table.add_column("等级", width=4)
    table.add_column("条款", max_width=40)
    table.add_column("问题", max_width=44)
    table.add_column("依据", max_width=24)
    for f in findings:
        basis = "；".join(f"{c.law_title[:6]}{c.article_no}" for c in f.legal_basis[:2]) or "—"
        table.add_row(f"[{_RISK_STYLE.get(f.risk_level, '')}]{f.risk_level}[/]",
                      f.clause_text, f.explanation[:80], basis)
    console.print(table)


def export_markdown(findings: list[ReviewFinding], source: str, out: Path) -> None:
    lines = [
        f"# 合同风险审查报告",
        f"",
        f"- 审查对象：`{source}`",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 风险条数：{len(findings)}（高 {sum(1 for f in findings if f.risk_level=='高')}"
        f" / 中 {sum(1 for f in findings if f.risk_level=='中')}"
        f" / 低 {sum(1 for f in findings if f.risk_level=='低')}）",
        "",
    ]
    for i, f in enumerate(findings, 1):
        lines += [f"## {i}. [{f.risk_level}] {f.clause_text}", "",
                  f"- **问题**：{f.explanation}",
                  f"- **建议**：{f.suggestion}"]
        if f.legal_basis:
            lines += ["- **依据**："]
            lines += [f"  - 《{c.law_title}》{c.article_no}（施行 {c.effective_date}）：{c.quote}" for c in f.legal_basis]
        lines.append("")
    lines += ["---", "*本报告由规则引擎与本地法规库检索生成，仅供参考，不构成正式法律意见。*"]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="合同风险审查（规则扫描 + 法条依据）")
    parser.add_argument("contract", help="合同文件路径（.txt / .docx）")
    parser.add_argument("--out", type=Path, default=None, help="导出 Markdown 报告路径")
    args = parser.parse_args()

    path = Path(args.contract)
    if not path.exists():
        console.print(f"[red]文件不存在：{path}[/red]")
        sys.exit(1)

    text = read_contract(path)
    findings = review(text)
    render(findings, path.name)
    if args.out:
        export_markdown(findings, path.name, args.out)
        console.print(f"[green]报告已导出：{args.out}[/green]")


if __name__ == "__main__":
    main()
