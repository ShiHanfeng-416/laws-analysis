# laws_analysis — 个人法律助手（本地 RAG）

33 部常用法规（约 114 万字）+ 本地向量化 + FAISS 混合检索 + 双引擎编排：
答案强制可溯源（引用逐字来自语料），知识库外问题一律回答"没有相关依据"。

## 功能

- **问答**（`ask`）：多轮对话，结论 / 引用 / 行动建议 / 免责声明四段式输出
- **合同审查**（`review`）：10 条风险规则扫描 + 法条依据佐证 + Markdown 报告
- **双引擎对比**（`compare-engines`）：LangChain 与 LlamaIndex 读同一份索引，
  输出结构一致可直接 diff
- **Web UI**（`app`）：Streamlit 对话 + 审查界面

## 快速开始

```powershell
cp .env.example .env          # 填入 LLM_API_KEY（OpenAI 兼容协议，如智谱）
pixi install

# 1. 建库（本地 LawVault 模型 1.2GB，首次自动下载到 models/）
pixi run -e local-embed build-index

# 2. 问答 / 审查
pixi run -e local-embed ask 房东不退押金怎么办
pixi run -e local-embed review examples/rental_contract.txt --out report.md

# 3. Web UI（Embedding + UI 同环境）
pixi run -e full app
```

不想下载本地模型：`.env` 里设 `EMBED_PROVIDER=api` 并填 `EMBED_API_KEY`
（默认走 SiliconFlow 的 bge-m3），此后所有命令无需 `-e local-embed`。

## 常用任务

| 命令 | 说明 |
| --- | --- |
| `pixi run -e local-embed build-index` | 建库 / 增量更新（按源文件 sha256 对账） |
| `pixi run -e local-embed ask` | 交互式多轮问答 |
| `pixi run -e local-embed review <file>` | 合同审查 |
| `pixi run -e local-embed compare-engines` | 双引擎对比评测 |
| `pixi run -e full app` | Streamlit Web UI |

## 环境说明

- `default`：问答 / 审查 / 对比（API Embedding 时零重依赖）
- `local-embed`：+ torch / sentence-transformers（本地 LawVault，约 5GB）
- `full`：local-embed + streamlit（Web UI 一体）

设计文档见 `docs/`（需求、架构、实施清单）。
