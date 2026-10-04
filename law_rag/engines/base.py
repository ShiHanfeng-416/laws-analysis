"""law_rag.engines.base — QAEngine 抽象接口（L3，双引擎的共同契约）。

应用层（ask.py / app.py / 对比脚本）只认这个接口，不认识具体引擎
（架构 R4）——切换引擎、并行对比、替换实现都不动上层代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from law_rag.schemas import AnswerResult


class QAEngine(ABC):
    """一问一答引擎。实现类负责：检索 → 组装上下文 → 生成 → 结构化输出。"""

    name: str  # 引擎标识，写入 AnswerResult.engine

    @abstractmethod
    def ask(self, question: str,
            history: list[tuple[str, str]] | None = None) -> AnswerResult:
        """回答一个问题。

        history: 最近几轮 (用户问题, 助手结论) 元组，用于多轮对话（FR-5.7）。
        检索仍基于当前问题本身——问题改写（query rewrite）留作演进项。
        """
        raise NotImplementedError
