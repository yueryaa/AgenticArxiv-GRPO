# AgenticArXiv-RL 项目梳理

## 1. 项目概览

AgenticArXiv-RL 将 arXiv 论文检索、下载、翻译和缓存查询任务改造成一个可训练、可验证、可离线复现的 Agentic RL 环境。项目的核心目标不是构建生产级论文应用，而是研究小型语言模型如何通过 SFT 和强化学习掌握 ReAct 工具调用、多步规划、会话状态保持以及正确终止任务的能力。

整体研究链路为：

```text
Qwen2.5-1.5B-Instruct
        ↓
Base 模型基线评测
        ↓
构建可执行专家数据
        ↓
4-bit QLoRA SFT
        ↓
分析 SFT 后 Bad Case
        ↓
冻结策略，审计逐 prompt 奖励方差
        ↓
筛选适合 GRPO 的任务
        ↓
QLoRA GRPO
        ↓
IID / OOD 对比评测
```

当前实际进度如下：

- Base 模型评测、版本化数据切分、SFT 数据构建和单卡 QLoRA SFT 已完成。
- GRPO 多轮 rollout、五分量奖励、训练守卫和奖励方差审计已完成。
- setup-aware v5 已冻结包含 7 个 train-only 任务的正式 GRPO 切分，并完成 30 步 smoke。
- v5 正式 GRPO 请求训练 120 步，在第 47 步因连续零方差采样窗口提前停止并保存模型。
- 已完成同协议的 SFT/GRPO `rl_train` 与 dev 对比；GRPO 获得真实但较窄的定向提升。
- 在后续全量 train 重扫中又发现并修复了“相同用户文本、不同隐藏环境状态”的 observation aliasing。因为输入协议已经改变，下一轮训练需要从冻结 SFT adapter 重新采样。
- 最终 IID/OOD 盲测尚未完成。
- DPO、PPO 和 OPD 已提供实现或实验入口，但当前完整实验主线是 `Base → SFT → GRPO`。

因此，目前更准确的项目定位是：训练链路、数据治理、GRPO 正式运行和开发集对照均已完成，已经证明部分定向能力可被规则奖励纠正；但提升集中在少数任务，且新输入协议仍需重跑，尚不能宣称获得广泛或最终的 IID/OOD 泛化提升。

## 2. 模型

### 2.1 基座模型

项目主模型为：

```text
Qwen2.5-1.5B-Instruct
```

选择该模型的主要原因是：

1. Instruct 模型已经具备基础指令遵循能力。SFT 可以集中学习 ReAct 格式、工具语义和状态转换，而不必从头学习语言能力。
2. 1.5B 参数规模适合在单张 RTX 4090 24 GB 上完成推理、SFT、参考策略约束和多候选 GRPO rollout。
3. 项目任务主要使用中文描述，并要求模型生成严格 JSON 工具调用。Qwen 系列与中文指令和结构化生成场景较匹配。

第三点是根据任务特点作出的工程选择。项目目前没有完成多基座模型消融，因此不能据此断言 Qwen2.5-1.5B-Instruct 是该任务上的最优模型。

### 2.2 QLoRA 配置

SFT 和 GRPO 主线采用 4-bit QLoRA：

| 配置项 | 取值 |
|---|---|
| 量化方式 | NF4 |
| Double Quantization | 开启 |
| 计算精度 | BF16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Attention 目标层 | `q_proj`、`k_proj`、`v_proj`、`o_proj` |
| MLP 目标层 | `gate_proj`、`up_proj`、`down_proj` |
| 优化器 | `paged_adamw_8bit` |

实际 SFT 中可训练 LoRA 参数为 18,464,768。相对于原始约 15.4 亿参数，真实可训练比例约为 1.2%。量化模型的 `numel()` 可能按底层打包存储统计，因此日志中以量化存储为分母得到的约 2% 不能直接解释为模型结构参数比例。

训练入口还会检查：

- 模型是否以 4-bit 方式加载；
- 是否只有名称包含 `lora_` 的参数参与训练；
- CUDA、BF16 和关键依赖是否可用；
- 训练 loss 是否为有限数；
- 最终 adapter 文件是否完整生成。

### 2.3 已完成的 SFT 配置

已完成的一轮正式 SFT 使用：

| 参数 | 取值 |
|---|---:|
| Epoch | 1 |
| Micro batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Learning rate | `1e-4` |
| Max length | 4096 |
| Scheduler | Cosine |
| Seed | 42 |
| Train loss | 约 0.0647 |

低训练 loss 只能说明模型较好地拟合了训练数据，不能代替 held-out Agent 成功率。项目后续使用严格工具执行指标、参数指标和 IID/OOD 评测判断实际能力。

## 3. 数据

### 3.1 原始任务定义

项目数据不是一般问答语料，而是带有可执行标准答案的 Agent 任务规格。一个任务通常包含：

```text
task                  用户请求
setup                 正式执行前需要建立的环境状态
steps                 标准工具调用步骤
expected_tools        预期工具序列
expected_tool_args    每一步的预期参数
expected_termination  预期终止状态
category              任务类别
```

任务覆盖：

- 最近提交论文检索；
- 按关键词、标题或作者检索；
- PDF 下载；
- PDF 翻译；
- 缓存状态查询；
- 多步组合调用；
- 会话状态和“上一条”“第 N 篇”等指代；
- 可选参数；
- 不可执行请求；
- 长工具调用链。

当前扩展基准集实际包含 62 条任务。部分旧说明仍写 59 条，应以 `data/splits/v2_62.json` 和运行时代码为准。

### 3.2 数据切分

62 条任务采用版本化切分：

| 集合 | 数量 | 用途 |
|---|---:|---|
| train | 36 | 构建 SFT、DPO 和 GRPO 数据 |
| dev | 8 | 开发诊断和调试 |
| IID test | 14 | 测试相同能力模板下的参数泛化 |
| OOD test | 4 | 测试未见过的多步组合模板 |

切分不是随机拆分 JSONL 行，而是考虑任务模板和工具链长度：

- IID 测试保留训练中出现过的能力结构，但使用不同参数实例。
- OOD 测试整体留出三步、四步组合模板。
- dev、IID 和 OOD 不能进入 SFT 数据生成或 GRPO 任务筛选。
- GRPO 候选只能根据 train 轨迹和 train Bad Case 产生。

这种切分能避免语言扩增后的同义样本同时落入训练集和测试集，降低语义级数据泄漏风险。

### 3.3 离线 arXiv 环境

真实 arXiv 查询结果被固化在：

```text
data/mock_arxiv_snapshot.json
```

真实联网主要发生在构建快照阶段。后续专家数据生成、benchmark 和 GRPO rollout 使用 `MockArxivEnv` 离线回放。这样能够保证：

- 同一个工具调用返回稳定结果；
- 实验不受 arXiv 每日更新影响；
- 网络错误和接口限流不会进入训练信号；
- Base、SFT 和 GRPO 在同一个环境中比较；
- “模型是否操作了正确论文”具有确定、可复核的答案。

每条多轮 rollout 都使用独立环境实例，避免不同 generation 之间共享会话状态。

### 3.4 SFT 专家 seed

SFT 专家数据不是让另一个 LLM 自由生成，而是根据 `TaskSpec` 的标准步骤，在离线环境中确定性执行：

```text
TaskSpec 标准步骤
        ↓
执行 setup，建立会话状态
        ↓
逐步执行标准工具和参数
        ↓
记录真实 Mock Observation
        ↓
追加 FINISH
        ↓
严格校验完整轨迹
        ↓
将每个决策点拆成一条 SFT 样本
```

第 `k` 个训练样本的 prompt 包含前 `k-1` 步完整的 Thought、Action 和 Observation。因此，模型是在完整状态下学习下一步动作，而不是只学习孤立的工具 JSON。

36 个 train 任务最终产生 85 条逐决策 seed。多步任务会被拆成多条训练样本，因此任务数和 JSONL 行数并不相等。

### 3.5 语言等价扩增

第一类合成数据是确定性的语言等价扩增。每条 seed 使用：

- 6 种任务表达包装；
- 2 种 Thought 表达。

Action、工具参数、Observation 和步骤顺序保持不变，因此得到：

```text
85 × 6 × 2 = 1020 行
```

这类扩增增加了语言表达多样性，但没有增加新的决策问题。1020 行仍然只来源于 36 个底层语义任务，不能表述为 1020 个独立任务。

### 3.6 参数化任务扩增

第二类合成数据会真正改变工具参数和环境状态，包括：

- 论文方向；
- 时间范围；
- 最大返回数量；
- 目标论文序号；
- 多篇论文组合；
- 有效或无效引用；
- 可选参数。

参数化任务必须保留 train 父任务的工具拓扑，并在离线环境中真实执行。项目由 train 模板派生出 65 个新语义任务，生成 159 条逐决策 seed，再做 12 倍语言扩增：

```text
159 × 12 = 1908 行
```

### 3.7 最终 SFT 数据

最终训练数据为：

```text
原始 train 语言扩增       1020 行 / 36 个语义任务
参数化任务语言扩增        1908 行 / 65 个语义任务
-------------------------------------------------
总计                      2928 行 / 101 个语义任务
```

两部分虽然行数约为 1:1.87，但每个语义任务的平均行数接近，因此项目保留全部数据，而没有为了行数比例强制下采样。

### 3.8 数据有效性保障

项目通过以下机制保证数据有效性和可追溯性：

1. **真实执行**：专家标准动作必须在 `MockArxivEnv` 中成功执行。
2. **严格轨迹校验**：终止状态、工具序列、参数、指代解析和工具执行结果必须符合标准答案。
3. **训练血缘检查**：SFT seed 必须恰好覆盖 train 来源，不能混入 dev、IID 或 OOD。
4. **拓扑限制**：参数化数据只能从 train 父任务派生，并保留父任务的工具链结构，不能生成留出的 OOD 拓扑。
5. **样本去重**：每条扩增样本计算规范化 SHA256；出现重复样本时直接报错。
6. **文件哈希**：数据文件、父样本和切分文件均记录 SHA256。
7. **Manifest 检查**：正式 QLoRA SFT 默认只接受经混合脚本生成、类型为 `qlora_sft_train_mix` 的数据。
8. **训练前长度检查**：验证 `max_length` 能否覆盖数据，避免无意截断关键动作。

这些机制能较好保证结构正确、执行有效、数据无明显泄漏。当前数据仍存在一个质量上限：部分 Thought 是规则模板，例如“需要调用某工具来处理任务步骤 1”，模型可能学习固定表达，而不是形成自然、可迁移的推理过程。

## 4. 算法

### 4.1 ReAct 与 MDP

项目将 Agent 行为建模为：

| MDP 元素 | 项目定义 |
|---|---|
| State | 用户任务、历史 Thought/Action/Observation、会话论文状态 |
| Action | 搜索、下载、翻译、缓存查询或 FINISH |
| Transition | 执行工具并获得 Observation |
| Reward | 完整轨迹的五分量可验证奖励 |

模型输出采用 ReAct 格式：

```text
Thought: 分析当前状态
Action: {"name": "工具名称", "args": {...}}
Observation: 环境返回结果
Thought: 根据结果决定下一步
Action: FINISH
```

### 4.2 SFT

SFT 主要解决：

- ReAct 格式；
- 工具名称和 JSON 参数格式；
- 基础工具路由；
- 根据 Observation 继续执行；
- 适时输出 FINISH。

完整 Base 评测中，train 严格成功率约为 14.8%。SFT 后对现有轨迹做语义重评分，train 严格成功率约为 77.8%。同时观察到 format 基本满分、解析错误基本消失，但工具语义、参数填写和跨步状态仍有错误。

这说明 SFT 已经解决了“会输出 ReAct”的主要问题，剩余问题更接近序列决策和结果优化，适合继续研究 Agentic RL。

### 4.3 DPO

DPO 数据由 SFT 模型对每个任务多次 rollout 后构造。项目比较不同轨迹的可验证奖励，在首次产生不同决策的位置提取：

```text
chosen    来自更高奖励轨迹的动作
rejected  来自更低奖励轨迹的动作
```

只有奖励差大于默认阈值 0.05，且两条轨迹确实存在决策分叉时才生成偏好对。首次分叉设计可以减少长轨迹中后续状态不同造成的归因混乱。

DPO 已有实现，但当前仓库中没有与正式 SFT 相同完整度的 DPO 实验 manifest 和最终评测证据，因此应将其视为可选路线，而不是已经验证完成的主线阶段。

### 4.4 GRPO reward

GRPO 使用五分量可验证奖励，每个分量位于 `[-1, 1]`：

| 分量 | 默认权重 | 含义 |
|---|---:|---|
| format | 1 | 动作是否为合法 JSON 工具调用或终止符 |
| tool | 3 | 实际和预期工具序列的顺序感知 LCS-F1 |
| argument | 2 | 工具参数和论文指代是否正确 |
| process | 1 | 合法步骤奖励，以及解析失败、工具失败、多余调用惩罚 |
| outcome | 3 | 是否正确完成任务以及最终业务结果是否正确 |

总奖励为有效分量的加权平均：

```text
R = Σ(weight_i × component_i) / Σ|weight_i|
```

项目原有 30 步 reward curriculum：训练早期将 tool、argument 和 outcome 权重乘以 `1/3`，让模型先学习结构。但这也会导致 total reward 随权重变化。正式实验建议显式设置 `reward_curriculum_steps=0`，或者同时记录原始分量和当前权重。

### 4.5 GRPO 采样与 advantage

对于同一个 prompt，当前策略以随机采样方式生成 `G=4` 条完整轨迹：

```text
相同初始 prompt
├── rollout 1
├── rollout 2
├── rollout 3
└── rollout 4
```

模型使用 `do_sample=True`。`temperature=1.0` 表示按照模型原始概率分布采样，并不表示确定性输出。四条轨迹会使用独立随机采样和独立环境状态，工具动作不同后，后续 Observation 和上下文也会逐渐分叉。

四条轨迹分别获得奖励后，在同一个 prompt 的组内计算相对 advantage：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

如果四条轨迹奖励完全相同，则组内标准差为 0，所有 advantage 为 0，该组基本不提供奖励驱动的策略梯度。

### 4.6 GRPO loss

项目使用 TRL `GRPOTrainer` 计算 loss，核心为裁剪策略目标和参考策略 KL 约束：

```text
ratio = π_current(token | state) / π_old(token | state)

policy_loss = -min(
    ratio × advantage,
    clip(ratio, 1-ε, 1+ε) × advantage
)

token_loss = policy_loss + β × KL(policy || reference)
```

项目默认 `beta=0.04`。KL 项限制策略过快偏离 SFT/reference 行为。

多轮 rollout 中：

- 模型生成的 Thought 和 Action token 参与 GRPO loss；
- 用户 prompt 不参与生成 loss；
- 环境 Observation token 使用 `env_mask=0`，只作为后续决策上下文；
- 完整轨迹获得一个轨迹级总 reward；
- 同一条轨迹的生成 token 共享相应的轨迹 advantage。

### 4.7 为什么选择 GRPO

该任务适合 GRPO，主要因为：

- 工具调用结果可以用规则自动验证，不需要训练额外 reward model；
- GRPO 使用同一 prompt 的组内奖励作为相对基线，不需要 value model；
- 相比 PPO，单卡显存和工程复杂度更低；
- 完整工具轨迹可以在线采样，适合优化 SFT 后仍存在的多步决策错误。

### 4.8 GRPO 任务筛选

项目没有将全部 36 个 train 任务直接投入 GRPO，而是从 SFT 后仍稳定失败的 train Bad Case 中选择 8 个任务，冻结 SFT 策略并进行奖励方差审计：

```text
每个任务 8 个 prompt 采样组
每组 4 条 rollout
每个任务 32 条 rollout
总计 256 条 rollout
```

最终统计为：

```text
64 个 prompt 采样组
35 个非零方差组
informative_group_fraction = 35 / 64 = 54.69%
```

首次 v4 审计曾保留 7 个任务并剔除一个近乎恒定的长链任务。保存原始 rollout 后发现旧 GRPO 环境没有执行 `TaskSpec.setup`，导致部分 reference 任务出现环境失败与奖励倒置，因此旧 v4 统计作废。

修复 setup 和 outcome 奖励后，项目从原始 SFT adapter 重新做 v5 审计。8 个候选任务共 64 组；一个任务的 8 组全部满分，被移入 `ceiling_control`。其余 7 个任务共 56 组，其中 34 组非零方差：

```text
informative_group_fraction = 34 / 56 = 60.7%
```

这 7 个任务构成 `data/splits/v5_grpo_train.json:rl_train`。

这一步的判断依据不是全局 reward 标准差，而是逐 prompt 的组内标准差。不同任务平均 reward 不同，并不能为 GRPO 提供同组相对 advantage。

## 5. 资源

### 5.1 当前硬件方案

当前主实验资源为：

```text
GPU：单张 RTX 4090 24 GB
基座：Qwen2.5-1.5B-Instruct
模型权重：4-bit NF4
训练参数：LoRA
计算精度：BF16
优化器：Paged AdamW 8-bit
Gradient checkpointing：开启
GRPO group size：4
环境执行：CPU 离线快照
```

已记录的峰值显存：

| 阶段 | 峰值显存 |
|---|---:|
| BF16 模型推理 | 约 2.9 GiB |
| QLoRA SFT | 约 5.73 GiB |
| QLoRA GRPO G=2 探针 | 约 5.42 GiB |

这些结果说明当前模型的参数训练显存不是主要瓶颈，后续主要压力来自：

- 多条 rollout 的生成时间；
- 多轮工具调用和重复编码；
- KV cache 和长上下文；
- 有效采样组比例；
- GRPO 语义任务数量；
- 奖励是否具有可靠区分度。

### 5.2 为什么 G 从 2 增加到 4

对于成功概率约为 0.5 的二元结果任务：

- `G=2` 出现一好一坏的概率约为 50%；
- `G=4` 出现至少一种不同结果的概率约为 87.5%。

项目的 G=2 审计只有约 31.25% 的任务组具有非零方差。首次 G=4 v4 审计得到约 54.69%，修复 setup 后的 v5 候选审计达到 60.7%。由于显存仍有余量，增加 group size 比扩大普通 batch 更直接地改善 GRPO 的组内比较信号。

## 6. 调优建议

### 6.1 当前正式 GRPO 结果与下一轮闭环

setup-aware v5 已完成 30 步 smoke：

```text
task split：v5_grpo_train.json:rl_train
task count：7
num_generations：4
reward_curriculum_steps：0
```

Smoke 共生成 240 条 rollout、60 个 prompt 组，其中 30 组具有非零方差；训练 loss 为 0.011773，峰值显存约 9.02 GiB，没有格式异常，证明模型加载、环境、奖励、反向传播、adapter 保存和审计链路可以工作。

正式运行从原始 SFT adapter 启动，请求 120 个 optimizer step，实际在第 47 步触发奖励方差保护后停止：

| 指标 | 结果 |
|---|---:|
| Runtime | 515.10 秒 |
| 实际 optimizer step | 47 |
| Rollout | 376 |
| Prompt group | 94 |
| Informative group | 30 |
| 累计 informative fraction | 31.9% |
| Train loss | 0.0002868 |
| 峰值显存 | 约 8.75 GiB |
| 异常样本 | 15，均为工具执行错误 |

连续 5 个日志窗口 reward 为 1.0 且组内方差为 0，只能说明当时的随机采样窗口已经饱和，不能证明 7 个任务全部收敛。

同协议评测显示：

- v5 `rl_train` 严格成功率由 SFT 的 4.76% 提升到 GRPO 的 33.33%；
- dev 严格成功率由 41.67% 提升到 50%；
- 提升主要来自 `infeasible_no_session` 和 `infeasible_zero_index`；
- `infeasible_unknown_id`、长链完整性和部分 reference 指代仍未解决；
- dev 增益只来自一个关键词搜索任务，应视为正迁移证据，不能视为广泛泛化证明。

后续全量 train 重扫又修复了相同文本对应不同初始状态的 observation aliasing。下一轮正式实验必须从冻结 SFT adapter 按新可见状态协议重新采样，不能直接续训旧 v5 checkpoint。

重点检查：

- `reward_components/tool`
- `reward_components/argument`
- `reward_components/outcome`
- `frac_reward_zero_std`
- `informative_group_fraction`
- `kl`
- `clip_ratio`
- `grad_norm`
- `rollout/finished`
- `parse_error_rate`
- `tool_error_rate`

下一轮应先重复短 smoke，确认不存在 NaN、KL 爆炸、连续全零方差和多轮循环，再根据新的 middle band 决定训练步数。

### 6.2 不要只看 total reward

如果训练跨过 reward curriculum 边界，即使模型行为不变，total reward 也会因为权重调整而变化。正式实验应：

- 显式关闭 curriculum，固定完整权重；或
- 同时记录五个原始 reward 分量和每个分量的当前权重。

模型选择应基于严格任务成功率、工具和参数准确率，而不是只比较 total reward。

### 6.3 扩大 GRPO 语义覆盖

正式 GRPO 当前只有 7 个语义任务。即使重复产生大量 rollout，仍容易过拟合任务文本并快速进入 ceiling。

建议沿这 7 个任务的 train 血缘继续生成参数化变体，重点改变：

- 初始环境状态；
- 目标论文位置；
- 工具参数；
- 多个候选对象之间的选择；
- 合法与非法边界；
- 同一工具拓扑下的不同语言表达。

每个新任务仍需满足：

- 不引入留出的 OOD 工具拓扑；
- 能在离线环境中真实执行；
- 有明确、可验证的 oracle；
- 先用冻结策略做 G=4 多组方差审计；
- 只有持续提供组内排序信号的任务才进入正式训练。

相比单纯增加同义改写，增加新的状态和参数决策更有价值。

### 6.4 建议的消融实验

| 实验 | 需要回答的问题 |
|---|---|
| SFT vs SFT+GRPO | GRPO 是否产生额外收益 |
| G=2 vs G=4 | 增加 generations 是否提高有效梯度比例 |
| Curriculum 30 vs 0 | 奖励课程是否有帮助 |
| `beta=0.01/0.04/0.1` | KL 约束强度是否合适 |
| `temperature=0.7/1.0/1.2` | 探索多样性与轨迹质量如何平衡 |
| Outcome-only vs 五分量 reward | 分层奖励是否改善学习效率 |
| 原始数据 vs 参数化混合数据 | 参数化合成是否提升泛化 |

每个实验应固定：

- 初始化模型；
- 数据切分及 SHA256；
- 离线快照及 SHA256；
- 随机种子；
- rollout 解码参数；
- 评测解码参数；
- 评测任务与重复次数。

### 6.5 提高正式评测门槛

当前 StageVerifier 的阈值主要用于发现训练是否彻底失败，例如 SFT parse rate 低于 0.3 或 GRPO mean reward 低于 -0.2。它们不足以证明模型获得实际能力。

正式实验建议报告：

- 严格任务成功率；
- 工具序列准确率；
- 参数准确率；
- 论文指代解析准确率；
- 正常 FINISH 比例；
- 工具执行错误率；
- 平均轨迹长度；
- IID 和 OOD 分别表现；
- 各任务类别表现；
- 至少 3 个随机种子的均值和波动。

### 6.6 工程风险

当前需要持续关注：

1. 部分 README 统计仍停留在旧版 59 条任务，与当前 62 条不一致。
2. DPO、PPO 路线没有达到当前 QLoRA SFT/GRPO 主线的实验成熟度。
3. PPO 当前主要对单段生成结果合成最小轨迹，与 GRPO 的完整多轮 rollout 还不是完全同口径。
4. GRPO 正式训练集语义任务较少，重复 rollout 不能替代任务多样性。
5. SFT Thought 存在模板化，可能限制推理表达和迁移能力。
6. 当前只有单一基座模型，缺少同规模模型对照。
7. 训练 loss 和平均 reward 都不能代替 held-out 严格成功率。
8. v5 GRPO 已证明局部定向收益，但新状态协议重跑和最终 IID/OOD 实验未完成，不能提前宣称广泛泛化收益。

## 7. 算法选择、替代路线与适用场景

### 7.1 项目实际使用了哪些算法

项目同时实现或规划了 SFT、DPO、GRPO、PPO 和 OPD，但它们在当前工程中的成熟度不同：

| 方法 | 学习信号 | 是否在线采样 | 额外模型 | 当前项目状态 |
|---|---|---:|---|---|
| SFT | 专家下一动作的 token 监督 | 否 | 无 | 正式完成，是主线起点 |
| DPO | chosen/rejected 动作偏好 | 数据生成时采样 | 参考策略 | 已实现，未形成主线完整实验 |
| GRPO | 同 prompt 多轨迹的规则 reward | 是 | 参考策略，无 critic | 正式 v5 已运行并评测，是当前 RL 主线 |
| PPO | reward + value advantage | 是 | 参考策略和 value head | 有训练入口，尚未达到多轮 GRPO 同等验证程度 |
| OPD | 强教师逐 token 分布 | 是 | 教师模型 | 可选实验路线，依赖和显存要求更高 |

### 7.2 为什么先 SFT，再做 GRPO

Base 模型的早期 Bad Case 包含错误工具、错误参数、错误指代、失败后重复调用和假完成。如果模型连稳定的 ReAct/JSON 格式都不会，GRPO 对同一 prompt 采样的轨迹可能全部落在相同低分区，组内 advantage 接近 0，训练会空转。

因此项目先用 SFT 学习动作格式和基础工具语义。SFT 后 format 分量稳定为 1、解析错误接近 0，但 tool、argument 和 outcome 仍明显偏低，说明问题已经从“不会表达动作”转成“动作选择和多步决策不够好”。这时再使用 GRPO，奖励更可能作用在真正的策略差异上。

### 7.3 为什么主线选择 GRPO

当前场景具备三个关键条件：

1. 完整轨迹可以由工具序列、参数、实际论文 ID 和终止状态自动判分。
2. 同一任务可以低成本重复采样多条轨迹。
3. 单张 4090 不适合同时承担大 policy、reference 和独立 value model 的高开销训练。

GRPO 用同一 prompt 的组内平均和标准差构造相对 advantage，不需要 critic/value model；规则 reward 又使项目不需要训练独立 reward model。因此它在这个项目中比 PPO 更轻，也比离线偏好训练更直接地利用在线工具执行结果。

它的代价是必须有组内 reward 方差。任务过难时四条都错，任务过易时四条都对，两种情况都没有有效相对信号。这也是项目先做 frozen-policy 方差审计、只选择 middle band 任务的原因。

### 7.4 DPO、GRPO、PPO 和 OPD 的区别

**DPO** 适合已经有稳定偏好对、但不方便在线运行环境的场景。它实现简单、训练稳定，但性能受离线 chosen/rejected 覆盖限制；策略变强后不会自动探索新的更优轨迹。当前项目可以用首次分叉处的高低 reward 动作构造 DPO 对，但完整实验尚未验证。

**GRPO** 适合有可验证结果、能够针对同一输入生成多条候选的场景。它能在线发现新的正确行为，不需要 value model；缺点是 rollout 昂贵，并且很依赖 group size、采样温度和组内 reward 方差。

**PPO** 适合长时序 credit assignment、需要 value baseline 或 reward 比较连续的环境。它的 actor-critic 结构理论上更通用，但需要训练 value head，工程和显存成本更高。当前 PPO 路径主要对单段输出合成最小轨迹，还没有与 setup-aware 多轮 GRPO 建立完全一致的环境协议，因此暂时不适合作为主实验结论。

**OPD** 适合存在明显更强的本地教师模型、规则 reward 较稀疏或难写的场景。教师可为每个 student token 提供稠密分布信号，但教师与学生需要共同占用显存，外部 API 通常又拿不到完整 token 分布。项目曾遇到 Torch 2.5.1 与 TRL experimental GKD 所需 `FSDPModule` 不兼容，因此决定把 OPD 放入独立 Torch 2.6+ 环境，避免破坏已经验证的 SFT/GRPO 主环境。

### 7.5 是否尝试过别的算法

项目不是通过完整公平实验得出“GRPO 一定优于所有方法”，而是基于资源和信号条件选择主线：

- SFT 已完成并显著改善基础格式和工具路由。
- DPO 数据构建和训练代码存在，但缺少与正式 SFT/GRPO 同口径的完整结果。
- PPO 有可运行入口，但多轮环境和 value-head 资源方案尚未充分验证。
- OPD 已进行兼容性探索，因依赖和教师显存问题暂未并入主环境。
- GRPO 已完成 reward probe、smoke、正式运行和开发集对照，因此是当前证据最完整的 RL 路线。

对外介绍时应说“根据当前可验证奖励和单卡资源约束选择 GRPO”，而不是声称已经通过大规模消融证明 GRPO 全面最优。

### 7.6 对当前任务的理解

这个任务不是传统单轮 QA，而是一个小型、部分可观测的状态决策问题。模型不仅要理解用户文字，还要知道当前会话是否已有论文、最近操作的是哪一篇、工具执行是否成功，以及下一步是继续行动还是停止。

最关键的学习目标包括：

- 正确区分搜索、下载、翻译和缓存查询；
- 精确填写 `days`、`max_results`、`ref`、`keep_dual` 等参数；
- 在多轮链中保持论文列表和 last-active 状态；
- 工具失败后基于 Observation 恢复，而不是重复相同失败动作；
- 对不可执行请求拒绝调用工具；
- 完成所有必要步骤后及时 FINISH；
- 不通过裸 FINISH、冗余调用或奖励漏洞获取高分。

因此，SFT 负责建立行为先验，GRPO 负责在在线执行中修正局部决策稳定性，独立 benchmark 负责判断这种训练收益是否真正转化为业务成功。

## 8. 资源、训练问题与解决过程

### 8.1 使用的资源

主实验使用单张 RTX 4090 24 GB。软件环境的已记录版本包括：

```text
PyTorch       2.5.1 + CUDA 12.1
Transformers  4.57.6
TRL           0.29.1
PEFT          0.17.1
bitsandbytes  0.45.5
```

SFT 采用 micro batch 1、梯度累积 8；v5 GRPO smoke 采用 batch size 4、梯度累积 2、G=4。QLoRA、8-bit optimizer、BF16 和 gradient checkpointing 共同控制显存。

### 8.2 系统盘空间不足

服务器根分区一度只剩约 5.4 GB，安装依赖、下载模型和保存 checkpoint 都可能失败。原因是 Conda 环境、pip 缓存和项目集中在系统盘。

处理方式是将环境、项目、模型、Hugging Face/torch/pip 缓存和临时目录统一迁移到大容量数据盘，并清理 pip 缓存。根分区可用空间恢复到约 13 GB。这说明训练预算必须同时计算模型、缓存、优化器状态和 checkpoint 的磁盘占用，不能只计算 GPU 显存。

### 8.3 外部网络不稳定

服务器访问 GitHub、Hugging Face、ModelScope、NVIDIA PyPI 和 arXiv 时出现过连接超时。项目采用：

- 在可联网环境手动下载源码、模型和 arXiv 快照，再同步到服务器；
- 对模型检查 SHA256、tensor 数、参数量和真实离线加载；
- 使用可访问的 CUDA 12.1 wheel 源；
- 训练和评测统一使用离线快照。

手动下载只解决可达性问题，哈希、结构检查和离线加载才保证文件可信。

### 8.4 依赖和版本兼容

项目遇到过三类依赖问题：

1. IDE 刷新包列表失败，同时 PyTorch 提示缺少 NumPy。CUDA 矩阵计算实际正常，最终定位为缺少 NumPy，固定 `numpy==1.26.4` 并执行 `pip check` 后解决。
2. TRL 0.29.1 的 `SFTConfig` 不接受新版示例里的 `use_cache` 参数。项目将 `use_cache=False` 放回 `model.config`，并在构造 dataclass 前按当前版本字段过滤兼容参数。
3. OPD 导入需要当前 Torch 没有的 `FSDPModule`。项目没有升级主环境，而是将 OPD 隔离为未来独立环境。

### 8.5 初始脚本会误入全参数训练

早期 SFT/GRPO 脚本直接加载完整模型，没有量化、LoRA 配置和可训练参数检查。在单张 4090 上直接使用较大 batch 和长上下文会有明显 OOM 风险，也与单卡可迁移方案不一致。

修复后：

- 4-bit NF4 加载冻结基座；
- 调用 `prepare_model_for_kbit_training`；
- 关闭训练时 KV cache；
- 使用 Qwen attention 和 MLP 全套 LoRA 目标层；
- 使用 paged 8-bit optimizer；
- 开启非 reentrant gradient checkpointing；
- 只保存 adapter、tokenizer 和 manifest；
- 训练前断言只有 LoRA 参数可训练。

### 8.6 离线快照时间漂移

新生成的“最近 30 天”快照会把任务中固定引用的论文挤出前 5，导致正确轨迹在环境中找不到目标论文。这会把数据问题误判成模型错误。

项目在快照构建中加入固定锚点论文、顺序检查和构建后 oracle 校验；任何网络查询失败默认抛错，避免写出看似成功的空快照。

### 8.7 随机种子仍不能保证复现

早期模型调用随机种子依赖全局调用序号。前一个任务多生成一轮，会改变后续所有任务的随机数流。修复为根据 `task_id + trial` 稳定派生 seed，并在每条轨迹开始前重置调用计数。

随后双跑仍有差异，最终发现离线下载桩读取真实磁盘文件是否已存在，并写入墙上时间。第二次运行因此看到不同 Observation。项目改为只使用当前 task/trial 内的下载记录，并在 trial 开始时清空运行状态，从而消除跨运行磁盘状态泄漏。

### 8.8 setup 缺失导致奖励倒置

首次保存 GRPO 原始 rollout 后发现，部分 reference 任务大量收到“未找到论文；请先搜索”。原因是 benchmark 会静默执行 `TaskSpec.setup` 建立会话状态，而旧 GRPO rollout 只重置了空环境。

这造成错误激励：

- 模型按标准答案直接下载时，环境执行失败；
- 模型为了让环境可用而额外搜索时，又因工具序列多一步被扣分；
- 某些失败轨迹的平均 reward 反而高于环境干净的轨迹。

修复包括：

- 每条 rollout reset 后执行所属任务的 setup；
- setup 只进入环境和审计元数据，不冒充策略动作参与 reward 或 loss；
- prompt 无法唯一映射任务、setup 工具未知或执行失败时立即报错；
- outcome 增加 `tool_exec_failures > 0 → -1` 硬约束，FINISH 不能覆盖已发生的工具失败。

修复后 32 条验证 rollout 的异常率从 53.125% 降到 9.375%，三个 setup-dependent 任务的相关异常从 14/16 降到 0/12。

### 8.9 相同文本对应不同隐藏状态

全量 train 重扫时发现两个任务具有完全相同的用户文本“把刚才那篇论文翻译一下”，但一个会话为空，另一个已经通过 setup 保存论文。旧实现只按 prompt 文本映射 task ID，会把两种相反状态混淆。

第一步修复是在结构化消息中加入模型不可见的 `_task_id`，用于路由正确环境；tokenization 前删除，防止模型看到答案标签。但只修复环境路由还不够：如果模型看到的 token 完全相同，却分别被奖励“不调用工具”和“调用翻译”，就会产生 observation aliasing 和相反梯度。

最终修复是将业务合法的初始状态写入可见 ReAct history，例如：

```text
空会话：当前没有既有论文，ref=null 无法解析
有状态：此前已检索并操作某篇论文，ref=null 将指向该论文
```

隐藏 `_task_id` 继续只用于环境路由；标准答案、expected tools 和 reward 不会暴露给模型。Benchmark 和 GRPO 使用同一份初始状态构造逻辑，确保首轮 prompt 完全一致。

## 9. 训练慢时怎么办

### 9.1 当前速度判断

v5 正式运行 47 个 optimizer step 用时约 515 秒，每步约 11 秒。每步对应 8 条 rollout，因此共生成 376 条轨迹。对 1.5B 模型而言，主要耗时已经不是 LoRA 反向传播，而是多轮自回归生成、重复 prompt 编码和工具交互。

### 9.2 已经采用的措施

- 使用 4-bit QLoRA，减少参数加载和反向传播开销。
- 使用 BF16 和 paged 8-bit optimizer。
- 使用 gradient accumulation，在显存允许范围内形成有效 batch。
- 将真实网络工具替换为离线 `MockArxivEnv`，避免网络延迟。
- 将环境 Observation token 从 loss 中屏蔽，避免对外部文本计算无意义策略梯度。
- 将 `max_completion_length` 从早期 2048 调低到更符合实际输出的范围。实测 completion 最大约 115 token，盲目保留 2048 只会增加生成风险和缓存预算。
- 根据标准最长工具链设置最小必要的 `max_turns`，不通过无限加轮数容忍冗余动作。
- rollout trace 支持 `rollout_trace_max_samples`，限制原文持久化量，同时保留全量汇总计数。
- 正式长跑支持周期 checkpoint、数量上限和 `resume_from_checkpoint`，避免中断后从头开始。

### 9.3 推荐的优化顺序

1. **先缩短无效生成**：根据真实 token 分布设置 `max_completion_length`，并用 stop condition 在 `Observation:` 或 FINISH 后及时结束单轮生成。
2. **控制 max_turns**：设置为标准最长链加 FINISH 所需轮数。增加轮数无法修复模型反复调用错误工具的问题，只会增加成本。
3. **提高有效 rollout 比例**：优先训练 frozen-policy 审计中有组内方差的 middle-band 任务。减少零方差采样比单纯提高 GPU 吞吐更有价值。
4. **利用显存余量做批量生成**：在不改变 G=4 语义的情况下增加 generation batch；不要为了速度随意把 G 降到 1，否则算法不再具有组内比较信号。
5. **评估关闭 gradient checkpointing**：当前峰值显存约 9 GiB，存在较大余量。可以做短 smoke 比较关闭 checkpointing 后的速度和峰值显存，再决定是否用于 1.5B 模型。该项是建议，尚未记录正式消融结果。
6. **缓存静态 prompt tokenization**：工具说明和初始 prompt 大量重复，可缓存不变前缀的 tokenization。多轮 Observation 之后的动态后缀仍需重新处理。
7. **引入高吞吐生成后端**：vLLM 或独立 rollout worker 能提高采样吞吐，但必须保留每条轨迹独立环境、setup、env mask、原始 logprob 和 task metadata。当前项目尚未完成这项迁移。
8. **更长远采用异步 rollout**：长轨迹场景可以研究 SAO/异步策略，但需要处理策略陈旧度和 importance sampling，不能直接把同步 GRPO 改成无约束异步。

优化速度时应同时报告 wall-clock、tokens/s、rollout/s、有效 prompt group/s 和峰值显存。只提高 rollout/s，但生成的大多是零方差组，对训练没有实际收益。

## 10. 如何监控训练过程

### 10.1 标准训练指标

TRL/Hugging Face 日志至少关注：

| 指标 | 含义 | 异常信号 |
|---|---|---|
| `loss` | 当前策略目标和 KL 的聚合 | NaN/Inf 或长期异常震荡 |
| `reward` | 当前 batch 平均总奖励 | 单独上涨不能证明成功 |
| `reward_std` | 当前奖励离散程度 | 可能只是不同任务均值不同 |
| `frac_reward_zero_std` | 零方差 prompt 组比例 | 长期接近 1 表示 GRPO 空转 |
| `kl` | 当前策略偏离 reference 的程度 | 快速爆炸表示更新过强 |
| `grad_norm` | 梯度规模 | 爆炸、NaN 或长期为 0 |
| `clip_ratio` | 被策略裁剪的 token 比例 | 过高可能表示更新过激 |
| `learning_rate` | 当前学习率 | 用于对应 scheduler 阶段 |

项目曾发现 `clipped_ratio` 与 `1-rollout/finished` 同步。根因不是 512 token 太短，而是模型在 `max_turns` 内没有主动 FINISH。这个指标必须与完成率和 token 长度一起解释。

### 10.2 项目自定义指标

项目额外记录：

```text
reward_components/format
reward_components/tool
reward_components/argument
reward_components/process
reward_components/outcome

reward_weights/*

rollout/turns
rollout/finished
rollout/parse_error_rate
rollout/tool_error_rate

reward_by_task/<task_id>/mean
reward_by_task/<task_id>/std
reward_by_task/<task_id>/zero_std
```

必须同时观察 reward 分量和当前权重，否则 reward curriculum 切换可能被误读为模型性能变化。必须观察逐任务组内 std，因为全局 reward_std 可能只是不同任务的平均分不同。

### 10.3 TensorBoard、W&B 和控制台

训练入口支持：

```text
--report_to none
--report_to auto
--report_to tensorboard
--report_to wandb
```

推荐单机离线实验使用 TensorBoard。指定的日志后端未安装时，程序会在加载模型之前失败，避免训练结束后才发现没有曲线。

即使 `report_to=none`，自定义 tracker 仍会接入 TRL 的 `_metrics["train"]` 并输出控制台统计。早期实现只有启用外部 backend 才绑定 tracker，导致指标积压在内存且不落盘；该问题已经修复。

### 10.4 原始 rollout 审计

聚合 reward 不能解释模型为什么失败。开启 `--save_rollout_traces` 后，每条样本保存：

- `task_id`、`group_index` 和 `generation_index`；
- 模型每轮原始输出 `raw_assistant_turns`；
- 环境解析后的 trajectory；
- policy/environment token mask；
- 五项 reward breakdown 和 total reward；
- group reward mean/std；
- 是否截断、是否跑满轮数；
- 活跃异常标签。

异常检测覆盖：Action 缺失或无法解析、Thought 缺失、多个 Action、自行编造 Observation、未知工具、工具失败、未 FINISH、截断和达到 `max_turns`。

JSONL 每批 flush；即使 OOM 或训练中断，仍尽量写出 summary。这条链路曾直接发现 setup 缺失造成的奖励倒置，说明原始轨迹审计比只看平均 reward 更重要。

### 10.5 自动保护机制

项目包含四级保护：

1. **RewardVarianceGuard**：训练早期连续零方差视为故障；训练后期出现连续满分窗口时正常提前停止并保存模型。
2. **CanaryEvaluator/Callback**：周期性在固定任务上评估，连续低于阈值时停止训练。
3. **StageVerifier**：阶段结束后检查 SFT parse rate 或 RL mean reward 是否达到最低门槛。
4. **Checkpoint/manifest**：记录请求步数、实际步数、提前停止原因、模型来源、数据切分、seed、超参数、依赖版本和峰值显存。

保护器只能发现异常或采样窗口饱和，不能证明泛化。最终结论必须来自冻结解码配置下的独立 benchmark。

## 11. 调优与 Bad Case 闭环

### 11.1 调优原则

项目没有从大量超参数盲扫开始，而是遵循：

```text
先固定环境和评测
    ↓
跑 Base 并保存逐条轨迹
    ↓
区分模型错误、环境错误和指标错误
    ↓
用 train Bad Case 构造 SFT 数据
    ↓
SFT 后按同协议复测
    ↓
只对仍有组内方差的问题做 GRPO
    ↓
用独立 dev 验证是否退化
```

这种顺序避免把环境 bug、奖励 bug 或数据泄漏误认为模型提升。

### 11.2 Base Bad Case 分类

Base 模型主要出现：

1. **工具选择错误**：该下载时搜索，该查询缓存时下载。
2. **参数错误**：字段放错工具、数值错误、漏掉业务参数。
3. **指代错误**：无法理解标题子串、中文序号、last-active 或 `ref=null`。
4. **状态错误**：未使用已有会话论文列表，或者错误地重新搜索。
5. **失败恢复错误**：工具已经明确报错，模型仍重复同一个动作。
6. **假完成**：必要步骤没做完就 FINISH。
7. **冗余调用**：完成任务后继续调用工具，或插入无关缓存查询。
8. **不可行请求处理错误**：会话为空、索引为 0、论文 ID 不存在时仍强行调用工具。

早期 24 条基线中没有解析失败，但 FINISH 率约 88%、严格成功率只有约 21%，说明最大问题不是 JSON 格式，而是错误地判断任务已经完成。

### 11.3 如何把 Bad Case 转成训练数据

对于格式和基础工具语义问题，使用 SFT：

- 为每个错误场景构造严格成功的标准轨迹；
- setup 在生成前真实执行；
- 每一个决策点拆为带完整历史的监督样本；
- 对不可行请求加入“不调用工具并解释原因”的正例；
- 对有状态/无状态的同文本任务提供不同、合法的可见初始状态。

对于局部动作偏好，可使用 DPO：

- 对同一任务采样多条轨迹；
- 找到首次动作分叉；
- 以 reward 更高的动作作为 chosen；
- reward gap 太小或没有真实分叉时不生成偏好对。

对于 SFT 后仍存在、且同 prompt 多采样能产生好坏轨迹的问题，使用 GRPO在线优化。稳定全错的问题继续补 SFT；稳定全对的问题作为 ceiling 回归样本；只有 middle band 任务进入 RL。

### 11.4 已解决的模型和评测问题

**离线 fallback 污染会话。** 未知关键词过去返回一个确定性回退论文池，BaseAgent 又把这些论文写入 `last_papers`，后续 ref 看似可以执行。修复为明确返回工具失败，不让回退论文进入会话状态。

**FINISH 被误认为成功。** 项目建立严格成功判据：必须 FINISH、工具序列和参数正确、指代论文正确，且没有解析或工具执行失败。

**参数字符串等价误判。** `ref=1`、标题子串和 `ref=null` 可能最终解析到同一论文。项目增加离线语义 oracle，比对工具实际返回的 `paper_id`；同时保留对 `force`、`threads`、`keep_dual` 等业务参数的严格检查，避免整体放宽指标。

**全局随机流耦合。** 改为每个 task/trial 独立派生 seed。

**磁盘和墙上时间泄漏。** 离线桩改为 trial 内纯状态，并在每条轨迹前清理。

**GRPO setup 缺失。** 每条独立 rollout 执行正确 setup，并将 setup 与策略动作、reward history 和 loss 隔离。

**相同 prompt 的隐藏状态冲突。** 使用隐藏 task metadata 选择环境，并把合法初始状态作为模型可见 Observation，避免相反标签作用于相同 token 输入。

### 11.5 GRPO 调参过程

**Group size。** G=2 审计中有效组比例约 31.25%，存在小样本假阴性。显存允许后改为 G=4，setup-aware v5 候选中的有效组比例达到 60.7%。因此 G=4 是基于信号质量和资源实测选择的，不是任意默认值。

**Temperature。** 训练使用随机采样以获得多样轨迹。`temperature=1.0` 使用模型原始概率分布；评测则固定 seed 和解码协议。温度过低会增加完全相同或同分轨迹，过高会增加无效动作，应通过有效组比例、异常率和平均 reward 联合选择。

**Learning rate。** QLoRA SFT 使用 `1e-4`，因为 adapter 训练通常可以使用高于全参数训练的学习率。GRPO 使用更小学习率，避免在线奖励驱动下快速偏离 SFT 策略，并通过 KL、clip ratio 和 dev 指标监控。

**KL beta。** 当前默认 0.04。beta 太小可能导致策略漂移和 reward hacking，太大会让模型难以偏离 SFT。建议围绕 0.01、0.04、0.1 做小规模消融，并固定 rollout 协议比较严格成功率。

**Reward curriculum。** 旧探针前 30 步降低语义奖励权重，适合格式尚未稳定的模型；SFT 后 format 已经稳定，因此 v5 正式实验使用 `reward_curriculum_steps=0`，从第一步启用完整的 `1/3/2/1/3` 权重。

**生成长度。** 早期 `max_completion_length=2048`，实际生成远短于上限。后续根据真实轨迹将预算压缩到约 512 或满足标准动作的最小安全值。遇到未 FINISH 时优先分析循环和动作冗余，而不是继续增加 token 上限。

**训练步数。** 不机械跑满设定步数。v5 请求 120 步，在第 47 步连续 5 个零方差满分窗口时停止。后续应重扫当前策略的 train middle band，而不是用 `allow_zero_variance` 在已饱和任务上强行续训。

### 11.6 v5 训练后的 Bad Case

v5 GRPO 的新增成功集中在：

- `infeasible_no_session`：从空会话仍调用翻译工具，变为说明无法完成并 FINISH；
- `infeasible_zero_index`：从错误下载默认论文，变为拒绝非法第 0 篇；
- dev 的 `search_kw_agentic_rl`：从偶尔遗漏 `days=30`，变为三次都正确补全参数。

仍未解决：

- `infeasible_unknown_id`：仍将不存在的 ID 直接交给下载工具；
- `chain_ro5_dl_three`：仍少下载一篇便提前完成；
- `ref_stress_image_restoration`：仍可能使用错误 query；
- `ref_stress_state_across`：仍可能使用空或错误 ref；
- 长链中的完整性、冗余动作和最终停止仍不稳定。

针对这些问题，下一步不应直接在旧 7 任务上堆更多 step，而应：

1. 用新状态协议从冻结 SFT adapter 重新生成基线；
2. 对 v2 train 全量冻结重扫，重新划分 floor/middle/ceiling；
3. 为 unknown ID、长链完成检查和标题指代生成更多 train-only 参数化任务；
4. floor 问题补 SFT 正例和困难负例；
5. middle 问题进入新一轮 GRPO；
6. ceiling 问题降采样或只作为回归控制；
7. 保持 dev/IID/OOD 不参与选题和调参；
8. 最后用相同快照、seed、repeat 和解码参数比较 SFT 与新 GRPO。

## 12. 推理链路

### 12.1 推理过程总览

一次完整推理不是“输入 prompt，模型一次性输出最终答案”，而是一个多轮 ReAct 环：

```text
用户任务 + 工具描述 + 初始会话状态 + 已执行历史
                         ↓
                    构造 Prompt
                         ↓
              Qwen2.5-1.5B-Instruct
                         ↓
               Thought + 单个 Action
                         ↓
             解析并校验工具名称和参数
                         ↓
                  真实/离线工具执行
                         ↓
                    Observation
                         ↓
             追加到下一轮 Prompt 上下文
                         ↓
          FINISH / ERROR / FORCE_STOP
```

这条链路的核心原则是：模型负责选择动作，环境负责产生事实。模型不能自行生成 Observation，也不能一次生成整段看似合理但没有真正执行的工具轨迹。

### 12.2 输入是什么

模型每轮输入是一条动态构造的 ReAct prompt，包含五部分：

1. **角色和目标**：模型是可以获取 arXiv 计算机科学论文的研究助手。
2. **工具说明**：运行时从工具注册表生成工具名、功能、参数类型、默认值和枚举值。
3. **当前用户任务**：例如搜索论文、下载第 N 篇、翻译最近操作的论文。
4. **初始会话状态**：任务开始前是否已有论文列表、最近操作论文以及 `ref=null` 能否解析。
5. **ReAct 历史**：此前每一步的 Thought、Action 和由环境返回的 Observation。

简化示例：

```text
你是一个 AI 研究助手，可以使用以下工具：
- search_arxiv_papers(query, max_results, days)
- download_arxiv_pdf(ref, force)
- translate_arxiv_pdf(ref, service, threads, keep_dual)
- get_paper_cache_status(ref)

当前任务：把刚才那篇论文翻译一下

[任务开始前的会话状态]
- 此前已下载 ref=2 的论文；它是最近操作论文，ref=null 会指向它。

请严格输出：
Thought: ...
Action: {"name":"...","args":{...}}

每轮只能执行一个动作；完成时输出 Action: FINISH。
```

初始状态是输入中很重要的一部分。项目曾经有两个用户文本完全相同的任务：一个会话为空，正确行为是拒绝执行；另一个已经存在最近论文，正确行为是调用翻译。如果只输入相同用户文字，任何确定性模型都无法同时答对。将合法会话状态显式放入 prompt 后，这才成为一个可学习的决策问题。

### 12.3 Prompt 是否调优过

项目做过 Prompt 调整，但方式是**基于真实 Bad Case 的人工迭代**，不是自动 Prompt 搜索，也没有形成严格的多版本 A/B 消融结果。

主要调整包括：

| 发现的问题 | Prompt 或协议调整 | 目的 |
|---|---|---|
| JSON 中出现 `True/False/None`、尾随逗号 | 明确要求严格 JSON 和小写 `true/false/null` | 提高解析稳定性 |
| 一次生成多个动作 | 强调每轮只能执行一个 Action | 保持 Action→Observation 闭环 |
| 模型自行编造 Observation | 将 `Observation:` 设为 stop sequence | 强制事实来自工具环境 |
| 不理解“上一条操作的论文” | 明确无显式 ID/标题时使用 `ref=null` | 对齐会话状态语义 |
| 翻译后反复轮询缓存 | 说明翻译为异步任务，提交后直接 FINISH | 减少冗余调用和循环 |
| setup 状态对模型不可见 | 加入不泄漏标准答案的可见初始状态 | 消除 observation aliasing |
| 训练和评测 prompt 不一致 | SFT、GRPO、Benchmark 共用 `get_react_prompt` 和初始状态构造 | 降低 train/inference mismatch |

此外，SFT 数据对用户任务做了 6 种语言包装、对 Thought 做了 2 种表达扩增。这训练的是模型对表达变化的鲁棒性，不等于推理时自动搜索最佳 Prompt。

### 12.4 Prompt 调优思路

项目采用的思路可以概括为：

```text
保存失败轨迹
    ↓
判断问题来自输入缺信息、格式歧义、模型能力还是环境错误
    ↓
只有输入协议确实不完整时才改 Prompt
    ↓
保持 SFT / GRPO / Benchmark 使用同一 Prompt 构造器
    ↓
用固定模型、固定 seed 和固定任务复测
```

不是所有错误都应该通过堆 Prompt 解决：

- 缺少会话状态属于输入不可观测，应补充状态。
- JSON 解析失败属于输出协议不明确，可以强化格式约束。
- 长链少执行一步属于策略能力问题，应通过数据或训练解决。
- 工具返回错误属于环境或工具问题，不应改 Prompt 掩盖。
- Reward 错误属于评测问题，更不能通过提示模型迎合错误规则。

下一步若要系统优化 Prompt，应建立版本化实验：

1. 固定同一模型 checkpoint、任务、环境快照、seed 和解码参数。
2. 只改变一个 Prompt 因素，例如是否包含示例、工具参数描述详细度、状态表达格式。
3. 每个版本重复多次，比较严格成功率、参数准确率、平均 token、工具失败和循环率。
4. 在 dev 上选 Prompt，IID/OOD 保持盲测。
5. 保存 prompt hash，避免模型变化和 Prompt 变化同时发生而无法归因。

### 12.5 为什么不同 Prompt 会有不同效果

语言模型根据当前 token 上下文预测下一 token。Prompt 改变后，模型看到的条件分布也随之改变。对于 1.5B 小模型，这种影响通常更明显：

- 明确列出工具名和参数 schema，会提高合法工具 token 的条件概率。
- 给出严格 JSON 示例，会把输出引导到训练中熟悉的结构模式。
- 明确“一次一个动作”，可以抑制模型一次生成整段虚构轨迹。
- 提供真实会话状态，能够消除同一句用户请求背后的状态歧义。
- 把关键约束放在 Action 附近，比埋在很长背景文字里更容易影响当前生成。
- Prompt 太长、规则重复或包含冲突示例，也可能分散小模型注意力并提高 token 成本。

效果较好的 Prompt 不是因为文字更华丽，而是因为它做到了：动作空间明确、参数约束明确、状态可观测、事实与模型生成分离、结束条件明确，并与训练时的数据格式一致。

### 12.6 推理模型如何加载

项目支持两种推理后端：

1. **API 后端**：通过兼容 OpenAI Chat Completions 的接口调用远程模型；
2. **Transformers 本地后端**：使用 Hugging Face `AutoTokenizer` 和 `AutoModelForCausalLM` 加载模型。

本地推理会应用模型的 chat template，调用 `model.eval()`，并在 `torch.inference_mode()` 下生成；设备可由 `device_map="auto"` 自动分配，精度支持 BF16、FP16 和 FP32。任务和 trial 还会派生稳定随机种子，使同一实验配置更容易复现。

训练产物是 LoRA adapter，因此部署时通常需要同时加载基座 Qwen 模型和 adapter。若线上框架更适合单模型权重，可以在离线验证通过后再合并 adapter；当前项目的产物清单仍保留“基座路径 + adapter 路径”，便于追踪训练来源和回退。

### 12.7 推理解码策略：温度应该怎么设

普通 Agent 推理的默认温度约为 `0.1`，目的是在保留少量灵活性的同时减少工具选择和参数格式的随机波动。本地后端在 `temperature <= 0` 时关闭采样，采用贪心解码；温度大于 0 时才启用采样。

训练、评测和线上部署的温度承担不同职责：

| 场景 | 温度目的 | 本项目中的典型做法 |
|---|---|---|
| SFT 后的普通推理 | 稳定完成工具调用 | 低温，如 `0` 或 `0.1` |
| GRPO rollout | 产生同一 Prompt 下的不同轨迹，形成组内相对优势 | 较高温度，训练配置为 `1.0` |
| 严格评测 | 降低抽样噪声，便于比较 checkpoint | 固定温度、种子和任务快照 |
| 创造性需求 | 增加候选多样性 | 适当提高，但必须加验证和重排 |

温度只改变解码时的 token 抽样分布，不会修改模型参数。GRPO 中对同一 Prompt 采样多条轨迹，正是为了利用这种随机性找到好坏不同的行为；若四条轨迹都完全相同或最终奖励相同，该组的标准差为零，无法提供有效的相对学习信号。

### 12.8 部署和推理加速

项目当前已经使用的推理优化包括：

- BF16/FP16 降低显存和计算开销；
- `torch.inference_mode()` 关闭梯度记录；
- `device_map="auto"` 简化设备放置；
- 在 `Observation:` 处设置停止词，避免模型自行编造工具返回值；
- 限制单轮 `max_tokens` 和总迭代次数，防止无止境生成；
- 记录每步 LLM、工具和框架耗时以及 token 用量，便于定位瓶颈；
- 离线 benchmark 使用固定环境快照，避免网络抖动污染模型延迟和成功率。

当前仓库仍以 Transformers `generate` 为主要本地后端；README 中的 vLLM 高吞吐 rollout/服务化属于后续方向，并非已经完成的能力。本地 client 也没有实现流式输出。因此不能把“支持 vLLM、连续批处理或生产级并发”写成项目现状。

若要进一步加速，可以按下面的顺序实施：

1. 先缩短静态 Prompt、限制无效 Thought 长度，并通过停止词尽早结束单轮生成；
2. 统计真实链路中 LLM 与工具耗时占比，优先优化占比最大的部分；
3. 对多个独立请求做动态批处理，避免逐条串行 `generate`；
4. 在质量验证后采用量化、Flash Attention 或合并后的 adapter；
5. 将本地后端迁移到 vLLM 等连续批处理引擎，并补充并发、吞吐、首 token 延迟和端到端延迟测试；
6. 对重复的静态系统提示和工具描述使用前缀/KV cache（取决于服务框架能力）。

Agent 是多轮推理链路，总延迟近似等于各轮 LLM 生成时间、工具执行时间和框架开销之和。减少无效轮次、重复查询和错误后的盲目重试，往往比只提高单次 token 生成速度更有效。

### 12.9 输出是什么

单轮 LLM 输出不是最终自然语言答案，而是可执行的 ReAct 决策：

```text
Thought: 当前要解决什么、下一步为什么调用该工具
Action: {"name": "工具名", "args": {"参数": "值"}}
```

任务完成时输出 `FINISH`。框架解析 Action、调用真实工具，再把真实 `Observation` 放回历史，驱动下一轮推理。这样把“模型决策”和“环境事实”分开，防止模型把自己生成的内容误当成工具结果。

Agent 最终返回的不只是文本，还包括：

- 完整 Thought/Action/Observation 轨迹；
- 最终 observation 和面向用户的 reply；
- 迭代次数和总耗时；
- 每步 LLM、工具、框架耗时；
- token 使用量；
- `FINISH`、`ERROR` 或 `FORCE_STOP` 等终止状态。

正常情况下，面向用户的 reply 取最后一个有效的非终止 observation。翻译任务是异步工具调用：工具返回 `task_id/PENDING` 后 Agent 应立即结束，线上由 SSE 等机制通知后续结果；离线训练和评测使用确定性的模拟环境。

### 12.10 如何保证输出质量

质量控制应覆盖生成前、执行中和执行后，而不能只检查最终文本是否通顺。

**生成前：**

- Prompt 明确工具 schema、JSON 规则、引用语义和任务状态；
- 训练与推理共用同一模板，避免模板漂移；
- 使用可见 setup context 消除“文本相同但隐藏状态不同”的歧义；
- 固定模型版本、adapter、解码参数和工具版本。

**执行中：**

- 每轮只允许一个 Action；
- 在 `Observation:` 前停止生成，环境负责产生 observation；
- 工具异常作为 observation 返回，允许模型依据真实错误修正；
- 设置最大迭代数，避免无限循环；
- 对工具名、参数类型、枚举、必填项和资源引用进行校验。

**执行后：**

- 严格 verifier 同时检查是否正常 `FINISH`、工具路径、参数、引用对象和工具执行结果；
- GRPO reward 分解为多个可解释分量，区分格式、选工具、填参数、执行和完成任务；
- Benchmark 同时统计 strict success、工具失败、参数错误、迭代数、延迟和 token；
- 保存完整轨迹，badcase 可以回溯到具体一步。

“文本看起来合理”不等于 Agent 成功。例如模型可能说已经下载论文，但没有执行下载工具；也可能调用了正确工具，却使用了错误论文 ID。项目的严格成功标准需要行为轨迹和环境结果共同成立。

### 12.11 输出效果不好时如何处理

建议先按失败发生的位置分类，再决定修复手段：

| 失败类型 | 常见表现 | 优先处理方法 |
|---|---|---|
| 输入状态缺失 | 同一任务在不同环境下动作混乱 | 补充可见 setup context，消除 observation aliasing |
| 格式错误 | JSON 缺引号、额外文字、字段不完整 | 收紧 schema，加入约束解码；最多做一次格式修复 |
| 工具选择错误 | 搜索后错误调用翻译或下载 | 增加对比样本和负例，检查工具描述是否重叠 |
| 参数错误 | ID、索引、枚举或布尔值错误 | 执行前做 JSON Schema 校验，将具体错误返回模型 |
| 工具临时失败 | 超时、网络错误、服务异常 | 仅对可重试错误限次重试，并采用退避策略 |
| 重复循环 | 相同工具和参数反复调用 | 检测重复 `(tool, args)`，达到阈值后停止或要求改写计划 |
| 过早结束 | 尚未完成便输出 `FINISH` | 用状态机/任务 verifier 检查完成条件，不满足则返回缺失步骤 |
| 任务不可执行 | 输入缺少必要信息或资源不存在 | 明确返回不可执行原因，不伪造成功结果 |
| 模型能力不足 | 多步规划持续失败 | 先补 SFT/DPO 数据；只有组内奖励有方差时再用 GRPO |
| 评分器错误 | 正确轨迹被判错或错误轨迹获高分 | 修复 oracle/verifier，并重新验证旧实验结论 |

项目已经具备的兜底包括：工具异常转 observation、LLM 异常返回 `ERROR`、超过轮数返回 `FORCE_STOP`、离线后端失败时明确报错而不伪造成功，以及保留最后一个有效 observation 作为用户可见信息。

当前解析器仍有一个需要重点修复的风险：完全无法解析 Action 时可能返回空 Action，而主循环会把空 Action 当成 `FINISH`。这会把“格式错误”误判为“任务完成”。更稳妥的做法是让解析器显式返回 `PARSE_ERROR`，进行一次受约束的 JSON 修复；仍然失败则安全终止，并把错误状态交给 verifier 和监控系统。执行工具前还应统一做 schema 校验，避免依赖宽松文本解析猜测参数。

兜底不应静默补全可能改变任务含义的关键参数。涉及论文 ID、文件目标或异步任务类型时，应返回清楚的缺失项或冲突原因，让上层请求补充信息。

### 12.12 对推理环节的整体理解

这个项目的推理质量可以概括为四个因素的乘积：

```text
推理质量 ≈ 输入状态可观测性 × 模型策略能力 × 工具环境真实性 × 结果验证严格度
```

Prompt 的作用是把任务、状态和动作空间表达清楚；SFT、DPO、GRPO 的作用是让模型在这个动作空间中形成更可靠的策略；工具环境决定 observation 是否真实；verifier 决定“成功”是否经得起检查。任何一项接近零，最终自然语言回复再流畅也不能代表任务真正完成。

因此，推理调优的正确顺序通常是：先修输入歧义和环境问题，再修解析与校验，然后用 badcase 补监督数据，最后才用 GRPO 优化仍有探索空间的决策。这样可以避免模型替错误的系统设计背锅，也能减少无效 rollout 和错误奖励带来的成本。

## 13. 项目亮点总结

项目最有价值的部分不是简单调用 TRL 训练模型，而是围绕 Agentic RL 建立了较完整的数据和验证闭环：

1. 将 arXiv 多步工具调用建模为可离线复现的 ReAct 环境。
2. 从纯 train 任务确定性执行专家轨迹，而不是依赖无法验证的自由生成答案。
3. 通过语言等价扩增和 train-only 参数化扩展得到 2928 行、101 个语义任务实例。
4. 使用单张 RTX 4090 完成 Qwen2.5-1.5B-Instruct 的 4-bit QLoRA SFT。
5. 从 SFT 后真实 Bad Case 构造 GRPO 候选，而不是将全部任务直接投入 RL。
6. 使用冻结策略对同一 prompt 做 G=4 多组采样，以组内奖励方差判断任务是否能提供 GRPO 信号。
7. 通过版本化 split、离线快照、严格 oracle、SHA256 manifest 和 IID/OOD 隔离保证实验可追溯。
8. 将 reward 拆分为 format、tool、argument、process 和 outcome，便于定位 reward hacking 和训练退化。
9. 保存原始 GRPO rollout，在计分边界关联模型原文、环境轨迹、token mask 和 reward，从而发现并修复 setup 缺失导致的奖励倒置。
10. v5 GRPO 在目标训练集上将严格成功率从 4.76% 提升到 33.33%，dev 从 41.67% 提升到 50%，同时明确将其限定为少数能力上的定向提升。

适合在项目介绍中使用的表述是：

> 项目先将 arXiv 多步工具调用改造成带可验证标准答案的离线 ReAct 环境，再从 36 个纯 train 任务构建并严格执行专家轨迹，通过语言扩增和参数化扩展形成 2928 条 SFT 数据。在 Qwen2.5-1.5B-Instruct 上完成单卡 4090 的 QLoRA SFT 后，从真实 Bad Case 中构造 GRPO 候选，并以冻结策略对同一 prompt 进行 G=4 多采样，根据组内奖励方差筛选真正能提供相对优势信号的任务。原始 rollout 审计进一步发现并修复了 setup 缺失造成的奖励倒置；setup-aware v5 GRPO 在目标训练集和 dev 上取得真实但较窄的提升。整个过程通过版本化切分、离线快照、严格 oracle、数据哈希和 IID/OOD 隔离保证可复现与防泄漏。
