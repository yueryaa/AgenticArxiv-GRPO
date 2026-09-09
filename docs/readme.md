# AgenticArXiv-RL 项目深度解析

> 本文基于当前仓库代码进行静态分析。仓库当前以 **Agentic RL 训练环境** 为主线，同时保留一套可运行的 Vue + FastAPI + MySQL Web 应用。阅读架构时应将两条运行路径分开：它们复用 Agent、工具和数据模型，但部署依赖与副作用策略不同。

## 1. 项目简介与核心技术栈组合

### 1.1 项目定位

AgenticArXiv-RL 将 arXiv 论文检索、PDF 下载、PDF 翻译和缓存查询建模为一组 Agent 工具，并在这些工具之上实现 ReAct 循环、可验证奖励、离线环境回放以及 SFT/DPO/GRPO/PPO/OPD 训练流程。

项目可以从两个视角理解：

1. **Agentic RL 研究环境（当前主线）**
   - 将“任务描述 + 对话历史 + 工具结果”视为状态。
   - 将工具调用或 `FINISH` 视为动作。
   - 使用规则化的五分量奖励，无需单独训练 Reward Model。
   - 通过冻结的 arXiv 快照离线回放，保证训练和评测可重复。
   - 使用 JSON/JSONL 保存数据集和轨迹，不依赖 Web、MySQL 或在线 arXiv。

2. **论文管理 Web 应用（兼容/历史路径）**
   - 提供聊天、论文搜索、PDF 下载、翻译、缓存管理和日志查看页面。
   - FastAPI 暴露 REST API，并通过 SSE 推送 Agent 步骤和翻译进度。
   - SQLAlchemy + MySQL 保存会话、论文资产、翻译任务和执行日志。

这两条路径复用以下核心抽象：

- `BaseAgent` 定义通用 ReAct 生命周期。
- `ToolRegistry` 提供统一工具发现和执行契约。
- `SideEffectManager` 隔离数据库、SSE、线程和会话记忆等副作用。
- `StoreProxy` 在 MySQL Store 与 MemoryStore 之间切换。
- `TaskSpec`、`TaskMetrics` 和 `RewardCalculator` 为训练、benchmark 和阶段验证提供一致的评价口径。

### 1.2 技术栈总览

| 领域 | 技术 | 在项目中的作用 |
|---|---|---|
| 后端语言 | Python 3.9+ | Agent、工具、API、数据访问、训练和评测的主要实现语言 |
| Web API | FastAPI、Uvicorn、Pydantic 2 | REST/SSE 接入、请求校验、响应序列化、应用生命周期管理 |
| Web 前端 | Vue 3、TypeScript、Pinia、Axios、Vite | SPA 页面、全局状态、REST 请求、开发构建 |
| 实时通信 | Server-Sent Events、`EventSource` | 后端向浏览器单向推送 Agent 步骤和翻译进度 |
| 关系数据库 | MySQL 8、SQLAlchemy 2、PyMySQL | Web 模式下保存会话、资产、任务、聊天和步骤日志 |
| Agent 模式 | ReAct、MCP、Skill/CLI | 分别提供进程内、stdio JSON-RPC 和 CLI 子进程三种工具执行方式 |
| LLM 接入 | OpenAI-compatible Chat Completions、Hugging Face Transformers | 远程 API 推理或本地 Causal LM 推理 |
| 工具能力 | `arxiv`、`requests`、`pdf2zh` | 论文搜索、PDF 网络下载和全文翻译 |
| 训练框架 | PyTorch、Transformers、Datasets、TRL、Accelerate | 模型加载、数据处理及 SFT/DPO/GRPO/PPO/OPD 训练 |
| 高效微调 | PEFT/LoRA、bitsandbytes QLoRA | 4-bit 量化和参数高效微调，降低显存需求 |
| 训练观测 | TensorBoard，可选 W&B | 记录 loss、reward、KL、奖励分量和 rollout 健康度 |
| 数据格式 | JSON、JSONL、CSV、本地模型目录 | 快照、任务切分、训练集、轨迹、报告和 checkpoint |
| 测试 | Python `unittest` | 工具、环境、奖励、训练守卫和数据生成回归测试 |

### 1.3 技术栈如何协同

#### Web 运行链路

1. Vue 页面调用 Pinia action。
2. Pinia 使用 Axios 请求 FastAPI，例如 `POST /chat`。
3. FastAPI 根据 `agent_type` 创建 `ReActAgent`、`MCPAgent` 或 `SkillAgent`。
4. Agent 通过 OpenAI-compatible API 或本地 Transformers 模型生成 Action。
5. Action 通过 ToolRegistry、MCP server 或 CLI 子进程执行。
6. 工具访问 arXiv、下载 PDF、调用 `pdf2zh`，并通过 Store 更新数据。
7. `MySQLSideEffectManager` 把聊天和步骤写入 MySQL，同时向 EventBus 发布事件。
8. FastAPI 的 `/events` 将 EventBus 中的事件转换为 SSE，浏览器用 `EventSource` 接收并刷新 Pinia 状态。

#### RL 运行链路

1. `benchmark/tasks*.py` 和 `splits.py` 提供任务与训练/验证切分。
2. `scripts/` 生成 SFT demonstrations、DPO preference pairs 等 JSONL 数据。
3. Transformers 加载 tokenizer 和策略模型，Datasets 加载 JSONL。
4. TRL Trainer 执行对应训练算法。
5. GRPO/PPO/多轮 OPD 在生成 Action 后调用 `MockArxivEnv`，把真实或快照 Observation 写回上下文。
6. `benchmark/metrics.py` 从完整轨迹提取工具序列、参数、指代解析和终止指标。
7. `RewardCalculator` 计算 format/tool/argument/process/outcome 五分量奖励。
8. 奖励进入 GRPO/PPO 更新；完整轨迹写入 JSONL，并由 Canary、StageVerifier 和 benchmark 继续验证。

### 1.4 需要特别注意的技术事实

- 项目**没有使用 Redis 或 Celery**。Web 异步翻译由 `threading.Thread` 执行，事件分发使用进程内 `queue.Queue`。因此任务状态和 SSE 订阅不会跨多个后端进程共享。
- Web 应用启动时会通过 FastAPI lifespan 调用 `init_db()`，因此 Web 模式需要 SQLAlchemy、PyMySQL 和可连接的 MySQL。
- RL 入口会设置 `STORE_BACKEND=memory`，以避免训练过程意外连接数据库或产生线上副作用。
- MCP 路径在代码中使用 Python MCP SDK，但当前两个 requirements 文件没有显式声明 `mcp`；启用前应单独确认依赖可导入。
- 主 requirements 是 RL 核心依赖；FastAPI、SQLAlchemy、PyMySQL 和 pdf2zh 位于 `requirements-extra.txt`。

## 2. 分层架构设计与目录职责

### 2.1 仓库级目录

```text
AgenticArXiv-RL/
├─ AgenticArxiv/          Python 核心包：Agent、工具、API、Store、RL、benchmark
├─ AgenticArxivWeb/       Vue 3 + TypeScript 前端
├─ scripts/               SFT/DPO 数据生成、增强与混合脚本
├─ data/                  任务切分、数据集、快照及历史统计
├─ eval/                  Bad Case 捕获与回放入口
├─ traces/                rollout 轨迹，通常为 JSONL
├─ outputs/               本地训练输出和 checkpoint，运行后生成
├─ artifacts/             已归档实验的轨迹、摘要和报告
├─ docs/                  奖励、训练和项目设计文档
├─ draw/                  评测结果绘图脚本及图片
├─ archive/               已归档的早期独立项目/服务
├─ bin/                   Linux 前后端启停脚本
├─ Makefile               Linux 开发快捷命令
└─ README.md              当前 RL 主线说明
```

### 2.2 接入与展示层

#### `AgenticArxivWeb/src/`

| 路径 | 职责 |
|---|---|
| `main.ts` | 创建 Vue 应用并安装 Pinia |
| `App.vue` | 页面容器，通过动态组件和 `KeepAlive` 切换功能页 |
| `components/ChatPanel.vue` | 对话、Agent 思考步骤和任务反馈 |
| `components/PapersPanel.vue` | 搜索结果和论文操作入口 |
| `components/AssetsPanel.vue` | 原文/译文 PDF、翻译任务和缓存管理 |
| `components/LogsPanel.vue` | 会话、消息及 Agent Step 日志展示 |
| `components/SettingsPanel.vue` | session、Agent 模式和连接状态设置 |
| `stores/appStore.ts` | 页面共享状态、REST 调用、SSE 重连和事件归并 |
| `api/client.ts` | Axios 实例和 API base URL |
| `api/sse.ts` | EventSource 建连及事件名分发 |
| `api/types.ts` | 前后端交互类型定义 |

#### `AgenticArxiv/api/`

| 文件 | 职责 |
|---|---|
| `app.py` | 创建 FastAPI、配置 CORS、注册路由；lifespan 中初始化数据库和本地目录 |
| `endpoints.py` | HTTP 请求模型、REST 路由、SSE 输出、PDF 预览与安全删除 |

API 层原则上只做参数校验、Agent/服务选择和返回值封装。业务动作最终下沉到 Agent、ToolRegistry、Store 或 TranslateRunner。

### 2.3 Agent 编排层

#### `AgenticArxiv/agents/`

| 文件 | 职责 |
|---|---|
| `base_agent.py` | 通用 ReAct 循环、上下文注入、LLM 计时、token 统计、工具调度、日志/SSE 钩子 |
| `agent_engine.py` | 默认 `ReActAgent`，使用正则 + JSON 解析 Action，并在进程内调用 Registry |
| `prompt_templates.py` | ReAct Prompt 和工具描述格式化 |
| `context_manager.py` | 对话上下文管理辅助对象 |
| `side_effects.py` | NoOp、Local、MySQL 三种副作用策略 |

`BaseAgent` 是最重要的模板方法实现。子类只需提供：

- `discover_tools()`：工具发现。
- `build_messages()`：Prompt 构造。
- `parse_response()`：模型输出解析。
- `invoke_tool()`：实际工具调用。

循环、终止、日志、SSE、会话记忆和翻译入队由基类统一处理。

#### `AgenticArxiv/mcp_protocol/`

- `server.py`：把 ToolRegistry 暴露为 MCP stdio server。
- `mcp_agent.py`：创建 MCP 子进程、发现工具并通过 `tools/call` 调用。
- MCP event loop 与同步 `BaseAgent.run()` 通过线程执行器桥接。

#### `AgenticArxiv/skill_cli/`

- `SKILL.md`：面向模型的 CLI 工具说明。
- `skill_agent.py`：把模型生成的命令解析为 Registry 工具和参数。
- `tool_cli.py`：基于 Python Fire 的 CLI 工具入口。

### 2.4 工具与外部能力层

#### `AgenticArxiv/tools/`

| 文件 | 提供的能力 |
|---|---|
| `tool_registry.py` | 全局工具元数据和 callable 注册表 |
| `bootstrap.py` | 独立导入每个工具模块、检查缺失工具、为可信 benchmark 提供硬失败 |
| `arxiv_tool.py` | 最近提交查询、关键词/标题/作者检索 |
| `pdf_download_tool.py` | ref 解析、下载、校验、缓存资产更新 |
| `pdf_translate_tool.py` | 自动准备原 PDF、调用 pdf2zh、更新译文资产、上报进度 |
| `cache_status_tool.py` | 聚合原 PDF 和译文 PDF 的缓存状态 |

工具模块在 import 时调用 `registry.register_tool(...)`。`bootstrap.register_all_tools()` 逐模块导入，避免某个第三方依赖缺失导致整个工具集合为空。

#### `AgenticArxiv/utils/`

| 文件 | 职责 |
|---|---|
| `llm_client.py` | 远程 Chat Completions 客户端和本地 Transformers 兼容客户端 |
| `pdf_downloader.py` | URL 标准化、下载、锁、哈希和文件名安全处理 |
| `pdf_translator.py` | pdf2zh 子进程调用和进度解析 |
| `file_writer.py` | 搜索结果文件输出 |
| `logger.py` | Loguru 日志配置 |

### 2.5 应用服务与副作用层

#### `AgenticArxiv/services/`

| 文件 | 职责 |
|---|---|
| `event_bus.py` | 按 session 保存订阅者队列并广播 JSON 事件 |
| `translate_runner.py` | 创建翻译任务、启动后台线程、更新状态、发布进度/成功/失败事件 |
| `runtime.py` | 创建全局 `event_bus` 和 `translate_runner` 实例 |
| `log_service.py` | 写入和查询聊天日志、Agent Step 和会话摘要 |

`SideEffectManager` 将这些 Web 副作用与 Agent 核心循环隔离：

- `LocalSideEffectManager`：用于 RL；不写数据库、不启动真实翻译，只保留内存会话状态。
- `MySQLSideEffectManager`：用于 Web；写日志、发布 SSE、启动翻译线程。
- `NoOpSideEffectManager`：完全关闭副作用，仅适用于无状态单步任务。

### 2.6 数据与持久化层

#### `AgenticArxiv/models/`

| 文件 | 职责 |
|---|---|
| `schemas.py` | `Paper`、`TranslateTask`、`PdfAsset`、日志等 Pydantic 模型 |
| `db.py` | SQLAlchemy Engine、Session factory、`Base.metadata.create_all()` |
| `orm.py` | 7 张 MySQL 表的 ORM 映射 |
| `store_mysql.py` | SQLAlchemy 数据访问和 session/ref 解析 |
| `store_memory.py` | 与 MySQL Store 尽量一致的内存实现 |
| `store.py` | 惰性 Store 代理及 backend 选择 |

MySQL 表职责：

| 表 | 内容 |
|---|---|
| `pdf_assets` | 原始 PDF 的 URL、路径、状态、大小和 SHA256 |
| `translate_assets` | 翻译输入/输出路径、状态、服务和错误 |
| `sessions` | 会话最近操作论文及活动时间 |
| `session_papers` | 某次搜索写入会话的论文列表和位置 |
| `translate_tasks` | 后台翻译任务、进度和元数据 |
| `chat_logs` | 用户/助手消息、模型和 Agent 类型 |
| `agent_steps` | Thought、Action、Observation、LLM/工具耗时 |

### 2.7 RL、奖励与评测层

#### `AgenticArxiv/rl/`

| 模块组 | 关键文件 | 职责 |
|---|---|---|
| 环境 | `env.py`、`multiturn_env.py` | 快照 record/replay/auto、多轮工具接口、独立 rollout 状态 |
| 奖励 | `reward.py`、`grpo_reward.py` | 五分量奖励、课程权重、TRL reward/rollout 适配 |
| 轨迹 | `trajectory.py`、`rollout.py` | ReAct 轨迹结构、JSONL 读写、离线数据收集 |
| 训练 | `train_sft.py`、`train_dpo.py`、`train_grpo.py`、`train_ppo.py`、`train_opd.py` | 五条训练路线 |
| OPD | `opd_multiturn.py` | 多轮学生 rollout、Observation mask、GKD loss 适配 |
| 质量守卫 | `canary.py`、`stage_verifier.py` | 训练中退化早停和阶段产物最低质量验证 |
| 工程适配 | `precision.py`、`observability.py`、`trl_compat.py` | 混合精度、训练日志和版本兼容 |
| 快照 | `build_snapshot.py` | 唯一需要联网的标准快照构建入口 |

#### `AgenticArxiv/benchmark/`

- `task_spec.py`：让标准工具序列和标准参数从同一 TaskSpec 派生。
- `tasks.py` / `tasks_expanded.py`：基础与扩展任务集。
- `splits.py`：模板级 train/dev/iid/ood 切分，降低数据泄漏。
- `metrics.py`：严格工具序列、参数、指代、false finish 等指标。
- `runner.py`：驱动不同模型后端和 Agent 类型执行任务。
- `report.py`：输出 Markdown、JSON 和 CSV 报告。
- `baselines.py`：用退化策略验证奖励是否具有区分度。
- `badcases.py`：捕获和回放奖励漏洞或失败轨迹。

## 3. 核心数据流转图

### 3.1 总体模块拓扑

```mermaid
flowchart TB
    subgraph Web[Web 在线应用]
        UI[Vue 3 页面]
        Pinia[Pinia Store]
        API[FastAPI Endpoints]
        SSE[SSE / EventSource]
        UI --> Pinia
        Pinia -->|Axios REST| API
        API -->|StreamingResponse| SSE
        SSE --> Pinia
    end

    subgraph Agent[Agent 编排]
        Selector{Agent 类型}
        React[ReActAgent]
        MCP[MCPAgent]
        Skill[SkillAgent]
        Base[BaseAgent ReAct 循环]
        LLM[远程 Chat Completions或本地 Transformers]
        Selector --> React
        Selector --> MCP
        Selector --> Skill
        React --> Base
        MCP --> Base
        Skill --> Base
        Base <--> LLM
    end

    subgraph Tooling[工具与外部能力]
        Registry[ToolRegistry]
        Arxiv[arXiv 搜索]
        Download[PDF 下载]
        Translate[pdf2zh 翻译]
        Cache[缓存查询]
        Registry --> Arxiv
        Registry --> Download
        Registry --> Translate
        Registry --> Cache
    end

    subgraph Data[状态与副作用]
        SideFx[SideEffectManager]
        Store{StoreProxy}
        MySQL[(MySQL / SQLAlchemy)]
        Memory[(MemoryStore)]
        EventBus[EventBus]
        Runner[TranslateRunner 线程]
        Files[(PDF 文件与本地缓存)]
        SideFx --> Store
        Store --> MySQL
        Store --> Memory
        SideFx --> EventBus
        SideFx --> Runner
        Download --> Files
        Translate --> Files
    end

    subgraph RL[Agentic RL 主线]
        Tasks[TaskSpec / splits]
        Dataset[JSONL Dataset]
        Trainer[TRL Trainers]
        Policy[Policy Model]
        MockEnv[MockArxivEnv]
        Metrics[Benchmark Metrics]
        Reward[RewardCalculator]
        Trace[Trajectory JSONL]
        Verify[Canary / StageVerifier]
        Tasks --> Dataset --> Trainer
        Trainer <--> Policy
        Policy --> MockEnv
        MockEnv --> Registry
        MockEnv --> Metrics --> Reward --> Trainer
        Metrics --> Trace --> Verify
    end

    API --> Selector
    Base --> Registry
    Base --> SideFx
    EventBus --> API
    Runner --> Translate
```

### 3.2 一次 Web 聊天请求的数据流

```mermaid
sequenceDiagram
    actor User as 用户
    participant Vue as Vue/Pinia
    participant API as FastAPI /chat
    participant Agent as BaseAgent
    participant LLM as LLM Client
    participant Tool as ToolRegistry或MCP/CLI
    participant Store as MySQL Store
    participant Bus as EventBus/SSE

    User->>Vue: 输入论文任务
    Vue->>API: POST /chat
    API->>Agent: run(task, model, session_id)
    Agent->>Store: 读取会话论文上下文

    loop 最多 max_iterations 次
        Agent->>LLM: messages + tools + history
        LLM-->>Agent: Thought + Action
        Agent->>Tool: execute_tool(name, args)
        Tool->>Store: 更新论文/资产/last_active
        Tool-->>Agent: Observation
        Agent->>Store: 写入 Agent Step
        Agent->>Bus: 发布 agent_step
        Bus-->>Vue: SSE 实时步骤
    end

    Agent-->>API: history + final_observation + timing
    API->>Store: 读取论文、资产和任务快照
    API-->>Vue: ChatResponse
    Vue-->>User: 回复及最新页面状态
```

### 3.3 GRPO 多轮训练数据流

```mermaid
flowchart LR
    Split[固定任务切分] --> Prompt[Prompt Dataset]
    Snapshot[冻结 arXiv Snapshot] --> EnvFactory[Environment Factory]
    Prompt --> GRPO[GRPOTrainer]
    Policy[当前策略模型] --> GRPO
    GRPO --> Generate[同一 Prompt 多次采样]
    Generate --> Parse[解析 Thought / Action]
    Parse --> EnvFactory
    EnvFactory --> Observe[执行工具并返回 Observation]
    Observe --> Generate
    Generate --> Trajectory[完整多轮轨迹]
    Trajectory --> Metrics[TaskMetrics]
    Metrics --> Reward[五分量可验证奖励]
    Reward --> Advantage[组内相对优势]
    Advantage --> GRPO
    GRPO --> Checkpoint[模型/Adapter Checkpoint]
    Trajectory --> Logs[奖励分量与健康度日志]
    Checkpoint --> Verify[StageVerifier / Canary]
```

## 4. 关键函数 API 与参数字典

### 4.1 Agent 核心 API

#### `BaseAgent.__init__`

```python
BaseAgent(
    llm_client,
    side_effect_mgr=None,
    env=None,
    max_iterations=5,
    llm_extra=None,
)
```

| 参数 | 类型/默认值 | 含义 |
|---|---|---|
| `llm_client` | `LLMClient` | 必填；需实现 `chat_completions()` |
| `side_effect_mgr` | `SideEffectManager | None` | 默认 Local；Web 应显式传 MySQL 实现 |
| `env` | `Any | None` | 注入后工具优先经 `env.execute_tool()`，RL 常传 MockEnv |
| `max_iterations` | `int = 5` | ReAct 最大轮数，耗尽后追加 `FORCE_STOP` |
| `llm_extra` | `dict | None` | 透传给 LLM，例如 temperature、stop、chat template 参数 |

#### `BaseAgent.run`

```python
run(task: str, agent_model: str | None = None, session_id: str = "default") -> dict
```

| 参数 | 说明 |
|---|---|
| `task` | 用户任务文本 |
| `agent_model` | 可选模型名；为空时读取 `settings.models.agent_model` |
| `session_id` | 会话隔离键，也用于 ref/last_active 解析 |

关键返回字段：

| 字段 | 含义 |
|---|---|
| `msg_id` | 本次运行唯一 ID |
| `history` | Thought/Action/Observation 步骤数组 |
| `final_observation` | 最后一步 Observation |
| `iteration_count` | 轨迹步数，包含终止步骤 |
| `agent_type` | regex/mcp/skill_cli |
| `timing` | 总 LLM、工具、框架开销及逐步耗时 |
| `token_usage` | prompt/completion/total token 统计 |

### 4.2 ToolRegistry API

| 函数 | 参数 | 返回/行为 |
|---|---|---|
| `register_tool(name, description, parameter_schema, func)` | 名称、描述、JSON Schema、callable | 注册或覆盖同名工具 |
| `get_tool(name)` | 工具名 | 返回包含元数据和 callable 的内部字典，未找到返回 `None` |
| `list_tools()` | 无 | 返回适合注入 Prompt/API 的工具元数据，不暴露 callable |
| `execute_tool(name, arguments)` | 工具名、参数字典 | 等价于 `func(**arguments)`；统一包装不存在、参数和执行错误 |
| `register_all_tools()` | 无 | 逐模块导入工具，返回 `{module: error}` |
| `require_all_tools(context)` | 运行场景描述 | 工具不完整时直接终止，适合 benchmark/RL |

### 4.3 业务工具参数字典

#### `get_recently_submitted_cs_papers`

```python
get_recently_submitted_cs_papers(
    max_results=50,
    aspect="*",
    days=7,
    output_path=None,
    save_to_file=True,
) -> list[dict]
```

| 参数 | 含义 |
|---|---|
| `max_results` | 最大论文数 |
| `aspect` | CS 子领域后缀，如 `AI`、`LG`、`CL`；`*` 表示全部 CS |
| `days` | 从当前 UTC 时间向前查询的天数 |
| `output_path` | 自定义文本输出路径 |
| `save_to_file` | 是否写入论文列表文件 |

#### `search_arxiv_papers`

```python
search_arxiv_papers(query: str, max_results=10, days=None) -> list[dict]
```

| 参数 | 含义 |
|---|---|
| `query` | 裸文本，或 `all:`、`ti:`、`au:` 前缀查询 |
| `max_results` | 最大返回数量 |
| `days` | 可选提交时间窗口；必须大于 0 |

#### `download_arxiv_pdf`

```python
download_arxiv_pdf(session_id="default", ref=1, force=False) -> dict
```

| 参数 | 含义 |
|---|---|
| `session_id` | 用于读取最近搜索结果和更新 last_active |
| `ref` | 1-based 序号、arXiv ID、标题子串；`None` 表示最近操作论文 |
| `force` | 为 `True` 时忽略已有文件并重新下载 |

主要返回：`paper_id`、`pdf_url`、`local_path`、`status`、`existed`、`size_bytes`、`sha256`。

#### `translate_arxiv_pdf`

```python
translate_arxiv_pdf(
    session_id="default",
    ref=None,
    force=False,
    service=None,
    threads=None,
    keep_dual=False,
    paper_id=None,
    pdf_url=None,
    input_pdf_path=None,
    progress_cb=None,
) -> dict
```

| 参数 | 含义 |
|---|---|
| `session_id` / `ref` | 与下载工具相同的会话指代机制 |
| `force` | 强制重新翻译 |
| `service` | pdf2zh 翻译服务；为空时取配置 |
| `threads` | pdf2zh 工作线程数 |
| `keep_dual` | 是否保留双语 PDF；默认仅保留中文单语版 |
| `paper_id` / `pdf_url` | 后台任务执行时可绕过 ref 直接指定论文 |
| `input_pdf_path` | 直接翻译本地文件 |
| `progress_cb` | 内部进度回调，取值约定为 0～1 |

#### `get_paper_cache_status`

```python
get_paper_cache_status(session_id="default", ref=None, paper_id=None) -> dict
```

返回 `paper_id`、`pdf`、`translate`、`pdf_ready` 和 `translated_ready`。

### 4.4 离线环境与奖励 API

#### `MockArxivEnv`

```python
MockArxivEnv(
    snapshot_path=None,
    mode="auto",
    offline_download=True,
    snapshot_tools=None,
)
```

| 参数 | 含义 |
|---|---|
| `snapshot_path` | JSON 快照路径 |
| `mode` | `replay`：只读快照；`record`：真实执行并记录；`auto`：命中回放、未命中联网 |
| `offline_download` | 是否使用离线 PDF 下载桩 |
| `snapshot_tools` | 需要快照化的工具集合 |

核心函数：

```python
execute_tool(tool_name: str, args: dict) -> list | dict | str
reset_runtime_state() -> None
save_snapshot() -> None
describe() -> str
```

正式离线评测建议使用 `mode="replay"`。`auto` 的快照 miss 会访问真实工具，不应被误认为完全离线。

#### `RewardCalculator`

```python
RewardCalculator(
    curriculum_steps=30,
    early_correctness_scale=1/3,
    weights=None,
)
```

默认权重：

```python
{
    "format": 1.0,
    "tool": 3.0,
    "argument": 2.0,
    "process": 1.0,
    "outcome": 3.0,
}
```

```python
compute_reward(
    task_def,
    result,
    agent_type="regex",
    trial=0,
    session_id="rl_train",
    training_step=0,
) -> tuple[float, TaskMetrics]
```

`compute_reward_breakdown(...)` 参数相同，但返回 `RewardBreakdown` 和 `TaskMetrics`。每个分量与总奖励都限制在 `[-1, 1]`。前 `curriculum_steps` 步将 tool/argument/outcome 权重乘以 `early_correctness_scale`，先强化结构和可解析性，再逐步恢复语义奖励。

### 4.5 主要 HTTP API

| 方法与路径 | 核心参数 | 作用 |
|---|---|---|
| `GET /health` | 无 | 健康检查 |
| `GET /tools` | 无 | 返回已注册工具及 JSON Schema |
| `POST /tools/execute` | `name`, `args` | 直接执行指定 Registry 工具 |
| `POST /arxiv/recent` | `session_id`, `max_results`, `aspect`, `days`, `save_to_file` | 查询近期论文并写入会话记忆 |
| `GET /sessions/{session_id}/papers` | path: `session_id` | 获取当前会话最近论文列表 |
| `POST /agent/run` | `session_id`, `task`, `agent_model?` | 固定使用 ReActAgent 执行任务 |
| `POST /chat` | `session_id`, `message`, `agent_model?`, `agent_type` | Web 主聊天入口，支持三种 Agent |
| `POST /pdf/download` | `session_id`, `ref`, `force` | 同步下载 PDF |
| `POST /pdf/translate` | 翻译参数字典 | 同步翻译 PDF |
| `POST /pdf/translate/async` | 翻译参数字典 | 创建后台翻译任务 |
| `POST /translate/tasks` | 翻译参数字典 | 另一翻译任务创建入口 |
| `GET /translate/tasks/{task_id}` | path: `task_id` | 查询任务状态 |
| `GET /events?session_id=...` | `session_id` | SSE 事件流 |
| `GET /pdf/assets` | 无 | 列出原 PDF 资产 |
| `GET /translate/assets` | 无 | 列出译文资产 |
| `GET /pdf/view/raw/{paper_id}` | `session_id` | 浏览器内联预览原 PDF |
| `GET /pdf/view/translated/{paper_id}` | `variant`, `session_id` | 预览 mono/dual 译文 |
| `DELETE /pdf/assets/{paper_id}` | `session_id` | 删除原 PDF 文件和索引 |
| `DELETE /translate/assets/{paper_id}` | `session_id` | 删除译文文件和索引 |
| `GET /logs/sessions` | 分页参数 | 会话日志摘要 |
| `GET /logs/sessions/{session_id}/messages` | session + 分页 | 会话消息 |
| `GET /logs/messages/{msg_id}/steps` | `msg_id` | Agent 执行步骤 |

### 4.6 训练入口速查

| 入口 | 输入 | 主要输出 |
|---|---|---|
| `python -m AgenticArxiv.rl.rollout` | 任务、模型/API、快照 | `traces/**/*.jsonl` |
| `python -m AgenticArxiv.rl.train_sft` | SFT JSONL | SFT 模型或 LoRA adapter |
| `python -m AgenticArxiv.rl.train_dpo` | chosen/rejected JSONL + SFT 模型 | DPO 模型 |
| `python -m AgenticArxiv.rl.train_grpo` | 任务集、快照、策略模型 | GRPO 模型、训练指标 |
| `python -m AgenticArxiv.rl.train_ppo` | 任务集、快照、策略模型 | 带策略更新的 PPO 模型 |
| `python -m AgenticArxiv.rl.train_opd` | 学生模型、教师模型、任务集 | 蒸馏后的学生模型 |
| `python -m AgenticArxiv.rl.build_snapshot` | 真实 arXiv 工具 | `mock_arxiv_snapshot.json` |

## 5. 项目启动与调试建议

### 5.1 环境准备

以下命令默认从仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r AgenticArxiv\requirements.txt
```

在 `AgenticArxiv/.env` 配置远程 LLM：

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
MODEL=gpt-4-turbo

STORE_BACKEND=memory
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
```

不要提交包含真实密钥的 `.env`。

### 5.2 优先验证离线 RL 主链

先执行测试：

```powershell
Set-Location AgenticArxiv
python -m unittest discover -s tests -p "test_*.py"
Set-Location ..
```

执行单任务 rollout：

```powershell
$env:STORE_BACKEND = "memory"
python -m AgenticArxiv.rl.rollout search_01 traces/train
```

如果仓库中存在 `data/mock_arxiv_snapshot.json`，rollout 会优先使用离线快照。正式可复现实验应显式传递快照：

```powershell
python -m AgenticArxiv.rl.rollout search_01 traces/train `
  --snapshot data/mock_arxiv_snapshot.json
```

执行 benchmark：

```powershell
Set-Location AgenticArxiv
python -m benchmark.run_benchmark `
  --backend transformers `
  --model C:\path\to\model `
  --agents regex `
  --task-set expanded `
  --offline `
  --split ..\data\splits\v2_62.json:dev
```

建议始终显式指定带版本的 split 文件和分组，不要只传裸 `train/dev/iid_test`，以免历史默认切分改变实验口径。

### 5.3 启动 Web 应用

安装额外依赖：

```powershell
pip install -r AgenticArxiv\requirements-extra.txt
Set-Location AgenticArxivWeb
npm install
Set-Location ..
```

准备 MySQL 数据库，并更新 `AgenticArxiv/.env`：

```dotenv
STORE_BACKEND=mysql
MYSQL_URI=mysql+pymysql://arxiv:your_password@127.0.0.1:3306/agentic_arxiv?charset=utf8mb4
```

无需手动建表；FastAPI lifespan 会调用 `Base.metadata.create_all()`。但它不等价于数据库迁移系统：已有表结构变化仍需手动迁移或引入 Alembic。

终端一：

```powershell
Set-Location AgenticArxiv
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

终端二：

```powershell
Set-Location AgenticArxivWeb
$env:VITE_API_BASE = "http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 5173
```

浏览器访问 `http://127.0.0.1:5173`，API 文档位于 `http://127.0.0.1:8000/docs`。

仓库的 `Makefile` 和 `bin/*.sh` 面向 Linux/bash；Windows PowerShell 建议分别启动两个终端。

### 5.4 推荐调试顺序

#### 工具注册异常

症状：Prompt 中工具列表为空，或模型编造不存在的工具名。

排查：

```powershell
Set-Location AgenticArxiv
python -m tests.test_tool_registry list
```

benchmark/RL 入口应调用 `require_all_tools()`，不要忽略注册失败。重点检查 `arxiv` 和可选的 `pdf2zh`/MCP 依赖。

#### LLM 输出解析失败

- 检查模型是否严格输出 `Thought:` 和 `Action:`。
- Action 建议使用 `{"name": "...", "args": {...}}`。
- 检查 stop sequence 是否包含 `Observation:`，防止模型伪造环境反馈。
- 查看轨迹中的 `parse_failed`、`termination_type` 和原始 observation。

#### 离线评测意外联网

- 显式指定 `--offline` 和 `--snapshot`。
- 使用 `MockArxivEnv(mode="replay")`，不要使用 `auto`。
- 查看 `env.describe()` 中的 `hit`、`miss`、`real_calls` 和 `offline_stubs`。
- `real_calls > 0` 不一定代表联网：缓存查询等纯本地工具也走 Registry；需结合具体工具判断。

#### Web 启动失败

依次检查：

1. `GET /health` 是否返回 200。
2. MySQL 服务、数据库、账号权限和 `MYSQL_URI`。
3. FastAPI 启动日志中的 `init_db()` 错误。
4. `GET /tools` 是否包含完整工具集合。
5. 前端 `VITE_API_BASE` 和浏览器 CORS/Network 面板。

#### SSE 无消息或断连

```powershell
curl.exe -N "http://127.0.0.1:8000/events?session_id=demo1"
```

- 确认发起 `/chat` 和订阅 `/events` 使用相同 `session_id`。
- `/events` 每 15 秒发送 ping；无业务消息不代表连接失效。
- 不要使用多个 Uvicorn worker：EventBus 是进程内对象，不同 worker 无法共享订阅和翻译事件。

#### 翻译任务卡住

- 检查 `pdf2zh` 命令能否在启动 Uvicorn 的同一环境中执行。
- 检查原 PDF 是否已下载、路径是否位于配置目录内。
- 查看 `translate_tasks.error`、后端日志和 SSE `task_failed` 事件。
- 重启后端会终止进程内线程；当前实现没有 Celery 式持久任务恢复。

#### 训练无有效梯度

- SFT 先用 `--inspect_only` 检查数据 manifest 和 token 长度，不加载大模型。
- GRPO 重点观察 `frac_reward_zero_std`、`reward_std` 和五个 reward component。
- 若组内奖励长期同值，说明任务过易、过难或 Action 无法解析；RewardVarianceGuard 会在开局阶段中止静默空转。
- 开启 TensorBoard：

```powershell
python -m AgenticArxiv.rl.train_grpo --report_to tensorboard
tensorboard --logdir outputs\grpo\logs
```

- 训练结束启用 `--verify`，检查生成的 `verification_report.json`，不要只看 loss。

### 5.5 开发扩展建议

- **新增工具**：在 `tools/` 实现函数并注册，同时更新 `bootstrap.TOOL_MODULES`、Prompt/Skill 文档、MockEnv 和测试。
- **新增 Agent 模式**：继承 `BaseAgent` 的四个抽象方法，在 `/chat` 的选择逻辑中注册；副作用通过 manager 注入，不要直接写数据库或发 SSE。
- **修改数据库表**：当前只有 `create_all()`，不会修改已存在列。持续演进时建议引入 Alembic。
- **扩展异步能力**：若需要多实例、失败重试或任务恢复，再考虑 Redis + Celery/RQ；目前线程模型适合单机开发。
- **保证实验可复现**：固定模型版本、snapshot、task split、seed、TRL/Transformers 版本和数据 manifest，并保存完整参数到 artifact 目录。
- **避免训练/测试泄漏**：切分应在任务模板层完成；dev 用于调试，iid/ood 留作最终评估，不要根据其轨迹反向修改训练数据。

## 附录：推荐阅读顺序

1. `README.md`：确认当前 RL 目标与训练路线。
2. `AgenticArxiv/agents/base_agent.py`：理解运行时控制流。
3. `AgenticArxiv/tools/tool_registry.py` 与 `tools/bootstrap.py`：理解动作空间。
4. `AgenticArxiv/rl/env.py`：理解离线、确定性工具环境。
5. `AgenticArxiv/benchmark/task_spec.py` 与 `metrics.py`：理解任务标准答案和指标。
6. `AgenticArxiv/rl/reward.py`：理解可验证奖励。
7. `AgenticArxiv/rl/train_grpo.py`：理解多轮 Agentic RL 如何接入 TRL。
8. `AgenticArxiv/models/store.py` 与 `agents/side_effects.py`：理解 Web 与 RL 的解耦边界。
9. `AgenticArxiv/api/endpoints.py` 与 `AgenticArxivWeb/src/stores/appStore.ts`：理解 Web 数据流。

