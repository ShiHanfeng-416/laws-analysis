"""app.py — Streamlit Web UI（L4 入口，M7）。

启动：
    pixi run -e full app        # 本地 LawVault + UI 同环境（torch + streamlit）

三个页面：
    问答      多轮对话，答案带引用折叠面板（可逐条展开核对原文）
    合同审查  粘贴合同文本 → 规则扫描 + 法条依据 → 可下载 Markdown 报告
    关于      项目说明与免责声明

工程要点：
    - 引擎用 @st.cache_resource 单例：模型加载 ~2 分钟只发生一次，
      会话刷新/多轮对话复用同一实例；
    - UI 只做展示与编排，业务全部下沉 law_rag 包（R4：UI 可替换）；
    - 反幻觉语义与 CLI 完全一致（同一个引擎对象）。
"""

from __future__ import annotations

import time

import streamlit as st

from law_rag.schemas import AnswerResult, ReviewFinding

st.set_page_config(page_title="个人法律助手", page_icon="⚖️", layout="wide")

_ENGINES = {"LangChain": "lc", "LlamaIndex": "li"}


@st.cache_resource(show_spinner="加载 Embedding 模型与 FAISS 索引（首次约 1-2 分钟）…")
def get_engine(spec: str):
    from law_rag.ask import get_engine as _ge
    return _ge(spec)


def render_answer(result: AnswerResult) -> None:
    st.markdown(result.conclusion)

    if result.citations:
        st.caption(f"依据（{len(result.citations)} 条，可点开核对原文）")
        for i, c in enumerate(result.citations, 1):
            with st.expander(f"[{i}] 《{c.law_title}》{c.article_no}"
                             f"（施行 {c.effective_date}）"):
                st.write(c.quote)
                if c.source_url:
                    st.markdown(f"[官方来源]({c.source_url})")
    else:
        st.info("知识库中没有与该问题相关的依据。")

    if result.action_steps:
        st.markdown("**行动建议**")
        for i, s in enumerate(result.action_steps, 1):
            st.markdown(f"{i}. {s}")
    st.caption(result.disclaimer)


def page_ask() -> None:
    engine_choice = st.sidebar.selectbox("引擎", list(_ENGINES))
    st.sidebar.caption("两引擎共享同一份 FAISS 索引与混合检索，仅编排与生成实现不同。")

    st.title("⚖️ 法律问答")
    st.caption("答案仅基于本地法规库（33 部 / 约 114 万字），引用可逐字核对。")

    spec = _ENGINES[engine_choice]
    if "engine_spec" not in st.session_state or st.session_state.engine_spec != spec:
        st.session_state.engine_spec = spec
        st.session_state.messages = []
    engine = get_engine(spec)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_answer(m["content"])

    if q := st.chat_input("例如：房东不退押金怎么办"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("检索法条并生成回答…"):
                t0 = time.time()
                msgs = st.session_state.messages[:-1]  # 刚 append 的用户消息除外
                pairs = [
                    (u["content"], a["content"].conclusion)
                    for u, a in zip(
                        (m for m in msgs if m["role"] == "user"),
                        (m for m in msgs if m["role"] == "assistant"),
                    )
                ]
                result = engine.ask(q, history=pairs[-2:])
                st.caption(f"{time.time() - t0:.1f}s · 引擎 {result.engine}")
                render_answer(result)
        st.session_state.messages.append({"role": "assistant", "content": result})


def page_review() -> None:
    st.title("📝 合同风险审查")
    st.caption("规则库扫描 + 本地法规检索佐证；不联网、不上传，文本只在本机处理。")

    text = st.text_area("粘贴合同文本", height=260,
                        placeholder="将合同全文粘贴到这里…")
    if st.button("开始审查", type="primary", disabled=not text.strip()):
        with st.spinner("扫描规则并检索依据…"):
            from law_rag.review import review
            findings = review(text)
        st.session_state.findings = findings

    findings: list[ReviewFinding] = st.session_state.get("findings", [])
    if findings:
        color = {"高": "🔴", "中": "🟡", "低": "⚪"}
        n_high = sum(1 for f in findings if f.risk_level == "高")
        st.markdown(f"**命中 {len(findings)} 条风险（高风险 {n_high} 条）**")
        for i, f in enumerate(findings, 1):
            with st.expander(f"{color.get(f.risk_level, '')} {i}. "
                             f"[{f.risk_level}] {f.clause_text[:38]}…"):
                st.markdown(f"**问题**：{f.explanation}")
                st.markdown(f"**建议**：{f.suggestion}")
                if f.legal_basis:
                    st.markdown("**依据**")
                    for c in f.legal_basis:
                        st.markdown(f"- 《{c.law_title}》{c.article_no}"
                                    f"（施行 {c.effective_date}）：{c.quote}")
        from law_rag.review import export_markdown
        export_markdown(findings, "粘贴文本", __import__("pathlib").Path("report.md"))
        with open("report.md", encoding="utf-8") as fh:
            st.download_button("下载 Markdown 报告", fh.read(),
                               file_name="合同审查报告.md", mime="text/markdown")
    elif "findings" in st.session_state:
        st.success("未命中规则库风险条款（重大事项仍建议咨询执业律师）")


def page_about() -> None:
    st.title("关于本应用")
    st.markdown(
        "- 语料：33 部常用法规（民法典、消费者权益保护法、住房租赁条例等），"
        "约 114 万字，来自政府官网公开文本\n"
        "- 链路：法条感知切分 → LawVault 本地向量化（768 维）→ FAISS 平坦索引"
        "（余弦）+ BM25 混合检索 → 双引擎（LangChain / LlamaIndex）生成\n"
        "- 原则：答案只允许引用检索到的法条（引用白名单），知识库外一律回答"
        "「没有相关依据」，杜绝编造法条\n"
        f"- 局限：法规库更新至 2025 年下半年，不含地方性法规与判例"
    )
    st.warning("本应用输出由检索生成，仅供参考，不构成正式法律意见；"
               "重大事项请咨询执业律师。")


def main() -> None:
    with st.sidebar:
        st.title("⚖️ 个人法律助手")
        st.caption("本地法规库 RAG · 可溯源问答 · 合同审查")
    page = st.sidebar.radio("功能", ["问答", "合同审查", "关于"],
                            label_visibility="collapsed")
    if page == "问答":
        page_ask()
    elif page == "合同审查":
        page_review()
    else:
        page_about()


if __name__ == "__main__":
    main()
