"""law_rag.ask — 问答 CLI（L4 入口，不含业务逻辑）。

用法：
    pixi run -e local-embed ask 房东不退押金怎么办
    pixi run -e local-embed ask                      # 交互式多轮对话（exit 退出）
    pixi run -e local-embed ask "…" --engine lc      # 指定引擎（M6 后支持 li）

本地 Embedding 需要 torch，请在 local-embed 环境运行（EMBED_PROVIDER=api 时随意）。
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from law_rag.schemas import AnswerResult

console = Console()

_ENGINES: dict[str, str] = {
    "lc": "law_rag.engines.langchain_engine:LangChainEngine",
    "li": "law_rag.engines.llamaindex_engine:LlamaIndexEngine",  # M6 交付
}


def get_engine(spec: str):
    """按标识惰性加载引擎类（import 放最后：CLI --help 不触发模型加载）。"""
    if spec not in _ENGINES:
        raise ValueError(f"未知引擎 {spec!r}，可选：{', '.join(_ENGINES)}")
    module_path, cls_name = _ENGINES[spec].split(":")
    from importlib import import_module
    cls = getattr(import_module(module_path), cls_name)
    return cls()


def render(result: AnswerResult) -> None:
    console.print(Panel(Markdown(result.conclusion), title="结论", border_style="cyan"))

    if result.citations:
        table = Table(title="依据（可在 data/laws/ 原文逐字核对）", show_lines=False)
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("法规", style="bold")
        table.add_column("条文", style="yellow")
        table.add_column("施行", style="dim")
        for i, c in enumerate(result.citations, 1):
            table.add_row(str(i), c.law_title, c.article_no, c.effective_date)
        console.print(table)
        for i, c in enumerate(result.citations, 1):
            console.print(f"  [{i}] {c.law_title}{c.article_no}（施行 {c.effective_date}）")
            console.print(f"      {c.quote}", style="dim")
            if c.source_url:
                console.print(f"      来源 {c.source_url}", style="dim blue")
    else:
        console.print("[yellow]（无引用：知识库中无相关依据）[/yellow]")

    if result.action_steps:
        console.print(Panel("\n".join(f"{i}. {s}" for i, s in enumerate(result.action_steps, 1)),
                             title="行动建议", border_style="green"))
    console.print(f"[dim italic]{result.disclaimer}[/dim italic]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="法律问答（答案可溯源到本地法条）")
    parser.add_argument("question", nargs="*", help="要问的问题；留空进入交互模式")
    parser.add_argument("--engine", default="lc", choices=list(_ENGINES),
                        help="引擎：lc=LangChain，li=LlamaIndex")
    args = parser.parse_args()

    try:
        engine = get_engine(args.engine)
    except ModuleNotFoundError:
        console.print("[red]LlamaIndex 引擎将在 M6 交付，当前请用 --engine lc[/red]")
        sys.exit(1)

    if args.question:
        render(engine.ask(" ".join(args.question)))
        return

    # 交互式多轮（FR-5.7）：保留最近几轮作为上下文
    console.print("[bold]法律问答（输入 exit 退出，多轮对话已启用）[/bold]")
    history: list[tuple[str, str]] = []
    while True:
        try:
            q = console.input("[bold green]你：[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        with console.status("检索法条并生成回答…"):
            result = engine.ask(q, history=history)
        console.print("[bold blue]助手：[/bold blue]")
        render(result)
        history.append((q, result.conclusion))
        history = history[-5:]  # 只留最近 5 轮


if __name__ == "__main__":
    main()
