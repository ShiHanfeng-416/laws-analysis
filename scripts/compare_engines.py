"""scripts/compare_engines.py — 双引擎对比评测（AC-5）。

用法：
    pixi run -e local-embed compare-engines            # 默认题集
    pixi run -e local-embed compare-engines --n 5      # 只跑前 5 题（省钱）

指标：
    每题两引擎各答一次，比较"引用法规集合"的重合度（Jaccard 与 双向召回），
    附结论级差异（长度 / 是否拒答）。重合度按题平均后须 ≥ 70%（AC-5）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from law_rag.ask import get_engine  # noqa: E402

console = Console()

QUESTIONS = [
    "房东不退押金怎么办",
    "网购商品七天无理由退货的条件",
    "经营者收取押金不退怎么处理",
    "租房合同没写维修责任谁来修",
    "买到有质量问题的汽车可以要求退换吗",
    "格式条款什么情况下无效",
    "经营者最终解释权条款合法吗",
    "承租人可以提前退租吗",
    "定金最多能收多少",
    "消费者和商家发生争议有哪些解决途径",
]


def cite_set(result) -> set[str]:
    return {f"{c.law_title}|{c.article_no}" for c in result.citations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="只跑前 n 题")
    args = parser.parse_args()
    questions = QUESTIONS[:args.n] if args.n else QUESTIONS

    console.print("[bold]加载双引擎（共享同一份索引）…[/bold]")
    t0 = time.time()
    lc = get_engine("lc")
    li = get_engine("li")
    console.print(f"引擎就绪 {time.time() - t0:.0f}s；开始评测 {len(questions)} 题\n")

    rows, overlaps = [], []
    for i, q in enumerate(questions, 1):
        console.print(f"[dim]({i}/{len(questions)}) {q} …[/dim]")
        t0 = time.time()
        r1 = lc.ask(q)
        console.print(f"[dim]  LC 完成（{time.time()-t0:.0f}s）[/dim]")
        t1 = time.time()
        r2 = li.ask(q)
        dt = time.time() - t0
        console.print(f"[dim]  LI 完成（{time.time()-t1:.0f}s）[/dim]")
        s1, s2 = cite_set(r1), cite_set(r2)
        union = s1 | s2
        inter = s1 & s2
        overlap = len(inter) / len(union) if union else 1.0
        overlaps.append(overlap)
        rows.append((
            q, len(s1), len(s2), len(inter),
            f"{overlap:.0%}", f"{dt:.1f}s",
            "拒" if "没有与该问题相关" in r1.conclusion else "答",
            "拒" if "没有与该问题相关" in r2.conclusion else "答",
        ))

    table = Table(title="双引擎对比（引用法规集合重合度）")
    for col in ("问题", "LC引用", "LI引用", "交集", "Jaccard", "耗时", "LC", "LI"):
        table.add_column(col)
    for r in rows:
        table.add_row(*[str(x) for x in r])
    console.print(table)

    avg = sum(overlaps) / len(overlaps)
    console.print(f"\n[bold]平均重合度 {avg:.0%}（AC-5 门槛 70%）"
                  f"：{'通过' if avg >= 0.7 else '未通过'}[/bold]")

    out = Path("notes/engine_compare.jsonl")
    out.parent.mkdir(exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "avg_overlap": avg},
                           ensure_ascii=False) + "\n")
    console.print(f"[dim]结果已追加 {out}[/dim]")


if __name__ == "__main__":
    main()
