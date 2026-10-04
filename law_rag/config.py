"""law_rag.config — 全项目唯一的配置入口。

【这个文件解决什么问题】
没有它，loaders 要读 LAWS_DIR、store 要读 INDEX_DIR、build 要读 CHUNK_SIZE……
os.getenv 撒满全项目，三个具体痛点：
  1. 散落：改一个键名要全局搜索替换，需求文档"可维护"一栏要求配置集中、无硬编码；
  2. 类型：os.getenv 返回的全是 str，TOP_K=6 拿到的是 "6"，直接传 faiss.search(q, k)
     会炸 TypeError——所以在这里集中转换一次，别处拿到的直接是 int / Path；
  3. 重复读：环境变量每个进程读一次就够（见 get_settings 的 lru_cache）。

【三条铁律】
  1. 本模块是整个项目**唯一**允许读环境变量的地方（全部 os.getenv 集中在下面
     三个 _env_* 工具函数里，全项目再无第二处）。
  2. 其他模块要用配置，只准写：
         from law_rag.config import get_settings
         settings = get_settings()
  3. 测试想注入假配置：改完环境变量后调 get_settings.cache_clear() 再重新调用。

【边界：这个文件刻意不做什么】
  - 不验证 API key 是否有效——那是 embeddings / LLM 客户端连接时的事；
  - 不 import langchain / llama-index 等任何框架——它是 L1 基础层，
    被所有上层模块依赖，必须保持零框架依赖（否则框架升级会波及全项目）；
  - 不含任何业务逻辑。
"""

from __future__ import annotations  # 让类型注解支持 "延迟求值"，3.11 下非必需但是好习惯

import os  # noqa: 使用 os.getenv 是本模块的特权，别的模块不允许 import os 来读环境变量
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

# 第三方依赖仅此一个（python-dotenv），功能单一：把 .env 文件读进环境变量。
# 不用 pydantic-settings 这类重型方案——教学项目里手写 60 行比引一个框架更好懂。
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 第一步：定位项目根 + 加载 .env
# ---------------------------------------------------------------------------
# 为什么不用 os.getcwd()？cwd 是"你在哪个目录敲的命令"，同一份代码换个目录
# 启动就指向别处；而 __file__ 是这个源码文件自身的位置，永远可靠。
# parents[1]：parents[0] 是 law_rag/（本文件所在包），parents[1] 才是项目根。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 显式传路径而不是裸 load_dotenv()：裸调用会从 cwd 往上找 .env，
# 可能找到别的项目的；显式锚定到本项目根，行为确定。
# 注意：.env 不存在时它静默跳过（返回 False，不抛错）——CI/容器里常没有 .env，
# 全靠下方字段默认值兜底，这是特性不是疏漏。返回值赋给 _ 表示"知道但不在意"。
_ = load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# 第二步：三个私有工具函数（全项目仅有的 os.getenv 调用点）
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str = "") -> str:
    """读字符串环境变量；未设置或为空时返回默认值。

    为什么不能直接用 os.getenv(name, default)：
    设置了但值是空的（EMBED_API_KEY= 回车）时 getenv 返回 "" 而不是 None，
    语义上同样算"没配"，统一回退默认值，省得每个调用处都要判空串。
    """
    value = os.getenv(name)
    return value if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    """读整数环境变量；值不合法时回退默认值并打印警告。

    两个痛点一次解决：
    1. 类型：os.getenv 拿到的是 str（"6"），直接用于 faiss.search(q, k) 会炸；
    2. 容错：.env 里手滑写成 TOP_K=6o 时，不该让整个程序在启动时崩溃，
       打句警告、退回默认值，程序还能跑，问题也能被发现。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():  # 未设置 / 空白 → 用默认值，静默即可
        return default
    try:
        return int(raw.strip())
    except ValueError:  # "6o"、"six" 之类 → 警告 + 回退，不让配置手滑搞崩程序
        print(f"[config] 警告：{name}={raw!r} 不是合法整数，已回退默认值 {default}")
        return default


def _env_path(name: str, default: str) -> Path:
    """读路径环境变量；相对路径一律基于 PROJECT_ROOT 解析成绝对路径。

    为什么必须锚定：Path("models/lawvault") 相对的是 cwd——
    在项目根跑是一个路径，在别的目录跑又是另一个，同一份代码跑出两种行为。
    锚定到 PROJECT_ROOT 后，从任何目录启动都指向同一处。
    resolve() 顺便把 "models/../models" 这类冗余段清干净，日志里路径清爽。
    """
    path = Path(_env_str(name, default))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# 第三步：Settings —— 配置的"形状"
# ---------------------------------------------------------------------------
# frozen=True 是本模块的教学点之一：配置应该是"事实"（fact）而不是"状态"（state）。
# 运行期谁写 s.top_k = 99 会当场抛 FrozenInstanceError——配置被意外篡改的
# bug 立刻暴露，而不是悄悄改变行为、留一个难查的隐患。
@dataclass(frozen=True)
class Settings:
    """全局配置。字段与 .env.example 的键一一对应（全小写命名）。

    类型即文档：字段类型就是"别处拿到什么"的合同——
    top_k 是 int、laws_dir 是 Path，调用方不需要再转换或猜测。
    """

    # ---- 大模型（生成侧三件套，OpenAI 兼容协议，换供应商只改这三行）----
    llm_base_url: str      # .env: LLM_BASE_URL
    llm_api_key: str       # .env: LLM_API_KEY（只存放/传递，有效性由客户端连接时验证）
    llm_model: str         # .env: LLM_MODEL

    # ---- Embedding（向量化）----
    # Literal 是"只允许这几个字符串"的类型。坑：它只约束类型检查器，
    # 运行时不校验！所以 get_settings() 里还有一道手写校验兜底（见下）。
    embed_provider: Literal["local", "api"]  # .env: EMBED_PROVIDER
    embed_model_path: Path                   # .env: EMBED_MODEL_PATH（provider=local 时生效，已是绝对路径）
    embed_batch_size: int                    # .env: EMBED_BATCH_SIZE（CPU 建库批大小，内存紧张就调小）
    embed_base_url: str                      # .env: EMBED_BASE_URL（以下三项仅 provider=api 时有意义）
    embed_api_key: str                       # .env: EMBED_API_KEY
    embed_model: str                         # .env: EMBED_MODEL

    # ---- 检索参数 ----
    top_k: int              # .env: TOP_K（检索返回条数，直接喂 faiss.search 的 k）
    chunk_size: int         # .env: CHUNK_SIZE（切块字符数）
    chunk_overlap: int      # .env: CHUNK_OVERLAP（相邻块重叠字符数）

    # ---- 目录 ----
    # .env 里没有这两个键也能跑：data/ 是原始法规目录，index/ 由 build_index 创建。
    laws_dir: Path   # .env: LAWS_DIR，默认 "data"
    index_dir: Path  # .env: INDEX_DIR，默认 "index"


# ---------------------------------------------------------------------------
# 第四步：get_settings —— 全项目统一的配置入口（单例）
# ---------------------------------------------------------------------------
# lru_cache(maxsize=1)：首次调用真正构造，之后所有调用拿同一个缓存对象。
# 好处有二：
#   1. 性能——环境变量一个进程读一次就够；
#   2. 一致性——全进程看到同一份配置快照，运行中改 .env 不会半程生效造成混乱。
# 测试钩子：monkeypatch 环境变量后调 get_settings.cache_clear() 即可"重造"。
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """构造并缓存全局配置。其他模块的唯一入口：
    from law_rag.config import get_settings
    """
    # provider 校验为什么放这里：Literal 注解运行时不生效（见 Settings 内注释），
    # 与其等 embeddings 初始化时神秘报错，不如在构造入口尽早抛错（fail fast），
    # 错误信息里带上当前值和合法值，一眼看懂怎么改。
    provider = _env_str("EMBED_PROVIDER", "local").strip().lower()
    if provider not in ("local", "api"):
        raise ValueError(
            f"EMBED_PROVIDER 必须是 'local' 或 'api'，当前值：{provider!r}"
        )

    return Settings(
        # 生成侧三件套：没有默认值——没配就该在用到时报得明明白白，不在这里编造
        llm_base_url=_env_str("LLM_BASE_URL"),
        llm_api_key=_env_str("LLM_API_KEY"),
        llm_model=_env_str("LLM_MODEL"),
        # type: ignore[arg-type]：provider 此刻是 str，类型检查器不知道上面
        # 的 if 已把它收敛到 Literal 的两个值之内，人工确认后压掉这条误报
        embed_provider=provider,  # type: ignore[arg-type]
        # 路径字段全部经 _env_path：相对路径已锚定成绝对路径
        embed_model_path=_env_path("EMBED_MODEL_PATH", "models/lawvault"),
        embed_batch_size=_env_int("EMBED_BATCH_SIZE", 32),
        embed_base_url=_env_str("EMBED_BASE_URL"),
        embed_api_key=_env_str("EMBED_API_KEY"),
        embed_model=_env_str("EMBED_MODEL"),
        # 数字字段全部经 _env_int：拿到的是 int，手滑值回退默认并警告
        top_k=_env_int("TOP_K", 6),
        chunk_size=_env_int("CHUNK_SIZE", 600),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        laws_dir=_env_path("LAWS_DIR", "data"),
        index_dir=_env_path("INDEX_DIR", "data/index"),  # 对齐架构 §6 与 .gitignore：索引落 data/index/
    )
