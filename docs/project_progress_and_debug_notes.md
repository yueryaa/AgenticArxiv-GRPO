# AgenticArXiv-RL 项目进度与工程问题笔记

> 更新时间：2026-09-08  
> 硬件：单张 RTX 4090 24 GB  
> 基座模型：Qwen2.5-1.5B-Instruct  
> 目标链路：Base 评测 → 数据构建 → QLoRA SFT → Bad Case 分析 → RL 数据与奖励 → GRPO → IID/OOD 对比

## 1. 当前处于什么阶段

项目目前处于**正式 GRPO 前的奖励方差审计与训练集冻结阶段**。Base 基线、v2 切分、
QLoRA SFT、SFT 后语义重评分以及单卡 QLoRA GRPO 冒烟均已完成。当前不是继续验证“代码能否
启动”，而是在回答更关键的问题：哪些 SFT 后 Bad Case 在随机多采样下具有稳定的组内奖励
方差，值得进入正式 GRPO；哪些任务接近 ceiling 或奖励几乎恒定，应删除、降权或重构。

当前进度：

| 环节 | 状态 | 结果 |
| --- | --- | --- |
| 硬件与 Python 环境 | 已完成 | 4090、CUDA、BF16 和矩阵计算正常 |
| 本地模型完整性与推理 | 已完成 | 1.5B 模型可离线加载，峰值显存约 2.9 GiB |
| 代码与离线环境测试 | 已完成 | 快照锚点、会话状态、语义 oracle、严格成功判定和可复现性回归均已接通 |
| arXiv 离线快照 | 已完成 | 冻结快照已同步；Benchmark 与训练均为零真实网络调用 |
| Base 模型 62 任务评测 | 已完成 | 186 条轨迹；严格成功率 18.8%，OOD 为 0% |
| v2 数据切分 | 已完成 | train/dev/IID/OOD 固定，训练数据只来自 train 血缘 |
| SFT 数据构建 | 已完成 | 2928 行、101 个语义任务实例；来源与 SHA256 可审计 |
| 单卡 QLoRA SFT | 已完成 | train loss 约 0.0647，峰值显存约 5.73 GiB |
| SFT 后语义重评分 | 已完成 | train 严格成功率 77.78%；保留 8 个稳定失败任务做 RL 探针 |
| QLoRA GRPO 冒烟与方差审计 | 已完成 | 峰值显存约 5.42 GiB；G=4 最终审计获得 64 个任务采样组 |
| 正式 GRPO | 待冒烟 | v4 已冻结 7 个 train-only 任务；主实验显式使用完整语义权重 |
| SFT+GRPO IID/OOD 对比 | 未开始 | 保持盲测，正式训练后统一揭盲评测 |

一句话定位：**完整 Base→数据→SFT→Bad Case→在线奖励审计链路已经打通，正在用组内奖励方差
而不是主观挑题来冻结正式 GRPO 数据。**

## 2. 已获得的关键实验结果

### 2.1 模型与显存

- 模型参数量：1,543,714,304。
- `model.safetensors` 已验证可读，包含 338 个 tensor。
- 服务器离线加载类型：`Qwen2ForCausalLM`。
- 加载显存约 2.875 GiB，单次推理峰值约 2.887 GiB。
- 一轮 QLoRA SFT 峰值显存约 5.73 GiB。
- QLoRA GRPO（G=2）冒烟峰值显存约 5.42 GiB；4090 具备继续测试 G=4 的资源余量。
- CUDA、BF16 和实际 GPU 计算均正常。

这说明 1.5B 模型的推理、QLoRA SFT 和 QLoRA GRPO 在 24 GB 显存上均已被实测验证，
不再只是资源估算。正式 GRPO 的主要约束已经从显存转为任务多样性、组内奖励方差和奖励设计。

### 2.2 奖励函数敏感性

在 62 个扩展任务上运行了不调用 LLM 的合成退化策略：

| 策略 | 平均奖励 | 含义 |
| --- | ---: | --- |
| reference | 1.000 | 标准工具轨迹稳定得到满分 |
| always_finish | -0.126 | 无脑结束被惩罚 |
| always_search | 0.005 | 无脑搜索接近零分 |
| random_tool | -0.064 | 随机调用工具被惩罚 |
| wrong_args | 0.450 | 工具顺序正确但参数错误，仍得到部分过程分 |

健康检查和逐类别检查均为 `PASS`，说明奖励函数目前能区分正确轨迹和明显退化策略。

仍需关注 `wrong_args=0.450`：模型可能学会“工具选对即可”，但不认真填写参数。
当前不立即改奖励权重，因为还没有真实模型轨迹作为证据；应在 Base/SFT 的 Bad Case
中统计参数错误率，再决定是否提高参数正确性权重或增加参数级惩罚。

## 3. 测试过程中遇到的问题与代码处理

### 3.1 系统盘已满

**现象**：根分区只剩约 5.4 GB，安装依赖和下载模型有失败风险。  
**根因**：Conda 环境、pip 缓存和既有项目集中在系统盘。  
**处理**：把新环境、项目、模型、Hugging Face/torch/pip 缓存和临时目录统一放到
`/mnt/disk4/gaojiayu/`。清理 pip 缓存后根分区恢复到约 13 GB。  
**工程意义**：训练任务不能只估算显存，还要规划模型权重、数据、checkpoint 和缓存的磁盘预算。

### 3.2 外部网络访问不稳定

**现象**：GitHub、Hugging Face、ModelScope、NVIDIA PyPI 和 arXiv 在服务器上出现连接超时。  
**处理**：

- 源码和模型使用浏览器手动下载，再上传服务器；
- 对模型做 SHA256、tensor 数量、参数量和真实离线加载四层验证；
- PyTorch 使用可访问的官方 CUDA 12.1 wheel 源；
- arXiv 快照在能联网的 Windows 环境生成，再同步服务器。

**工程意义**：手动下载不能等同于“文件可信”，必须有来源、哈希、结构和加载验证。

### 3.3 依赖安装完成，但 IDE 刷新包列表报错

**现象**：IDE 执行 `pip list --format=json` 失败；PyTorch 同时提示缺少 NumPy。  
**定位**：实际 CUDA 矩阵计算成功，说明核心 PyTorch 没坏；问题是环境中缺少 NumPy，
IDE 报错属于包列表刷新失败。  
**处理**：安装并固定 `numpy==1.26.4`，检查核心包版本，再运行 `python -m pip check`。  
**结果**：依赖检查显示 `No broken requirements found`。

### 3.4 OPD 测试导入失败

**现象**：测试导入 TRL experimental GKD/OPD 路径时，当前 Torch 2.5.1 缺少其需要的
`FSDPModule`。  
**根因**：这是 OPD 实验功能与 Torch/TRL 的版本兼容问题，不是 SFT/GRPO 主链路故障。  
**决策**：不为一个可选实验盲目升级当前稳定环境。主环境继续用于 QLoRA SFT 和 GRPO；
若后续做 OPD 消融，单独创建 Torch 2.6+ 的环境。  
**工程意义**：隔离可选实验依赖，避免修复次要功能时破坏已经验证的主链路。

### 3.5 扩展任务从 59 条变成 62 条，v1 切分失效

**现象**：两个切分测试失败；`v1.json` 缺少三个新关键词任务：

- `search_kw_agentic_rl`
- `search_kw_llm`
- `search_kw_rag`

README 和部分测试注释仍写 59 条，运行时代码实际返回 62 条。  
**根因**：任务集合更新了，但固定切分和文档没有同步，属于数据版本漂移。  
**决策**：不把三个 ID 直接追加到 v1，也不覆盖历史成功率。应先让 Base 模型在全部
62 条任务上重复评测，用实测成功率生成并冻结 `v2.json`。  
**原因**：GRPO 需要有组内奖励方差；过易或过难的任务成功率接近 1 或 0，采样几乎不产生有效梯度。

### 3.6 仓库缺少离线快照

**现象**：代码要求 `data/mock_arxiv_snapshot.json`，下载的仓库中没有该文件；服务器也无法访问 arXiv。  
**处理**：在 Windows 建立只含 `arxiv`、`loguru`、`pydantic` 的轻量环境，调用官方
arXiv API 构建 7 个学科池和 3 个关键词池，再验证离线执行时 `real_calls=0`。  
**结果**：快照包含 10 个查询记录；论文池大小为 50、50、50、50、50、50、50、47、50、50；
离线环境测试 6/6 通过。

### 3.7 新快照与固定评测答案发生时间漂移

**现象**：普通回放测试通过，但进一步检查发现，任务硬编码的 8 个 ID/标题引用中只有
`Handover` 能在新快照中命中，其余引用已被滚动的“最近 30 天”结果挤出前 5。  
**风险**：正确的模型轨迹会因为环境找不到论文而失败，最终把数据问题误判为模型能力问题。  
**代码处理**：修改 `rl/build_snapshot.py`：

1. 给 AI/CV 论文池加入固定锚点论文；
2. 保留短标题歧义对照所需的顺序；
3. 构建后检查固定 ID 和标题是否能在 setup 的前 5 篇中解析；
4. 任一网络查询失败时默认抛错，不再以退出码 0 写出空快照；
5. 提供 `--allow-partial` 和 `--no-pin-references`，仅供研究原始数据时显式选择。

**验证**：修复后的 AI/CV 前 5 篇包含全部固定引用，离线回放仍为 6/6。

这是目前最有价值的工程改进之一：它把“时间变化导致的隐式标签污染”变成了可检测、可阻断的问题。

### 3.8 本地 Transformers 客户端存在，但 Benchmark 没有使用

**现象**：`utils/llm_client.py` 已实现 `TransformersLLMClient`，能直接加载 Hugging Face
目录；但 `benchmark/runner.py` 的 `llm_client` 属性仍固定调用 `get_env_llm_client()`。
因此 `--model /mnt/.../Qwen2.5-1.5B-Instruct` 只会把本地路径作为模型名发给远程 API，
不会加载本地模型。  
**代码处理**：为 Benchmark 增加了明确的 backend 参数：

```text
--backend api          # 保持原行为
--backend transformers # 加载本地模型目录
```

同时加入 dtype、device 和 seed 参数。本地生成会在 `Observation:` 处真正停止，
避免模型继续伪造环境反馈并浪费生成 token。无模型单元测试已通过；同步服务器后，
先用 4 条不同类型的任务做真实模型冒烟测试，再跑 62 条完整基线。

### 3.9 初始训练脚本尚未实现 QLoRA（已解决）

**现象**：`train_sft.py` 和 `train_grpo.py` 都直接使用
`AutoModelForCausalLM.from_pretrained(model)` 加载模型，Trainer 中也没有传入
`BitsAndBytesConfig`、`LoraConfig` 或 `peft_config`。当前脚本保存的是完整模型，
而不是 LoRA adapter。  
**风险**：如果按默认参数在单张 4090 上直接运行，会进入全参数训练；尤其是
`batch_size=4`、`max_length=3072` 的 SFT，以及多候选 rollout 的 GRPO，显存风险很高，
也不符合本项目选择的“单卡 QLoRA”实验设定。  
**下一步代码处理**：

1. 用 4-bit NF4 加载 Qwen 基座，并设置 BF16 compute dtype；
2. 调用 `prepare_model_for_kbit_training`，关闭 `use_cache`，开启 gradient checkpointing；
3. 为 Qwen 的 attention/MLP 线性层配置 LoRA target modules；
4. SFT 只保存 adapter、tokenizer 和训练配置；
5. GRPO 重新加载同一个 4-bit Qwen 基座，再加载 SFT adapter 为可训练状态；
6. 分别保留 Base、SFT adapter 和 SFT+GRPO adapter，避免覆盖，确保三阶段可比较。

**正确关系**：Qwen 是三阶段共同的基座；RL 应从已经完成 SFT 的 adapter 继续优化，
而不是重新从原始 Qwen 开始，也不是默认进行全参数 RL。

## 4. 为什么当时不能直接训练（历史决策）

如果没有 Base 基线就直接 SFT，会缺少四项关键证据：

1. 不知道模型原本会什么、不会什么；
2. 无法从真实失败中倒推 SFT 数据结构；
3. 无法判断 SFT 是否已经解决主要问题；
4. 无法证明 GRPO 的收益来自 RL，而不是数据泄漏或评测变化。

所以当前正确顺序是：先把本地模型接入评测，固定快照和解码参数，对 62 条任务重复采样，
保存逐条轨迹；再根据任务级成功率和 Bad Case 设计 SFT 数据。

## 5. 原定阶段计划与完成标准（现已推进至阶段 C）

### 阶段 A：Base 模型评测

1. 将修复后的快照和构建脚本同步服务器；
2. 给 Benchmark 增加本地 Transformers backend；
3. 先跑 2～3 个任务，确认工具格式、结束条件和离线命中；
4. 对全部 62 个任务重复 3～5 次，保存 traces；
5. 输出总体、分类和逐任务成功率，以及参数错误率。

完成标准：同一配置可重复运行；快照零真实网络调用；每次失败都能追溯到完整轨迹。

### 阶段 B：冻结 v2 与构建 SFT 数据

1. 按模板分组切分，避免同模板参数变体泄漏；
2. 保留 OOD 模板；
3. 根据 Base 成功率选择训练难度；
4. 冻结 `v2.json`、数据生成脚本和数据统计报告；
5. 从真实 Bad Case 反推监督轨迹，而不是只下载通用开源数据。

### 阶段 C：QLoRA SFT → Bad Case → GRPO

1. QLoRA SFT 后使用同一 v2、快照和解码配置复测；
2. 判断剩余问题是否主要是序列决策、长链状态或参数精确性；
3. 只有当 SFT 收益趋于饱和且任务仍有可学习的奖励方差时才进入 GRPO；
4. 用 Base、SFT、SFT+GRPO 三组结果做 IID/OOD 和分类对比。

## 6. 面试时可以怎样讲

### 60 秒版本

> 我复现的是一个面向科研团队的 arXiv 工具调用 Agent，目标是让小模型完成检索、下载、
> 缓存查询、翻译和多步状态链。项目不是简单跑训练脚本，而是建立可归因的 Base—SFT—RL
> 链路：先冻结离线快照和 train/dev/IID/OOD 切分，Base 在 186 条轨迹上的严格成功率为
> 18.8%；再从 train Bad Case 反推数据，构造 2928 行、101 个语义任务实例，用单张 4090
> 完成 QLoRA SFT。语义重评分后 train 严格成功率达到 77.78%。我随后只从 train 的 8 个稳定
> 失败任务构造 GRPO 探针，并发现全局 reward_std 会掩盖逐 prompt 零方差，所以增加了逐任务
> 组内方差、奖励分量和轨迹健康度统计。冻结 SFT 策略的 G=4 审计表明约 54.7% 的任务采样组
> 有有效相对优势，下一步据此冻结正式 RL 任务集，再做 SFT 与 SFT+GRPO 的盲测对比。

### 常见追问

**为什么不直接训练？**  
没有冻结的 Base、数据切分和评测环境，训练后的提升不可归因，也无法判断是否发生数据泄漏。

**为什么不直接修改 v1？**  
v1 的成功率来自旧任务集。直接追加新任务会混合不同模型、数据和采样条件，破坏历史实验可复现性。

**如何判断该不该做 RL？**  
先看 SFT 后的 Bad Case。如果问题仍是长链决策、状态保持或局部动作选择，并且同一任务多次采样
能得到有差异的奖励，GRPO 才可能有效；若奖励恒为 0 或 1，应先改任务、数据或奖励。

**你做了哪些代码层面的改进？**  
增加快照锚点固定、任务—数据一致性校验和 fail-fast；把本地 Transformers 模型接入
Benchmark；修复离线搜索到下载之间的 session 状态断裂；增加语义指代 oracle，避免把等价
ref 误判为错误；实现单卡 NF4 QLoRA SFT/GRPO；并补充逐任务组内奖励方差、五项奖励分量、
轨迹健康度和 JSON 训练审计产物。OPD 的 Torch 版本冲突被隔离到可选实验，不影响主链路。

**奖励函数有没有风险？**  
有。Base 合成退化测试中 `wrong_args` 平均仍能获得 0.450；真实 SFT 后 GRPO 探针中 format
稳定为 1、parse error 为 0，但 argument 分量仍为负、工具失败率约一半。奖励还带 30 步课程，
早期会把 tool/argument/outcome 权重压到三分之一，所以 total reward 的变化不一定等于策略变强。
因此必须同时观察奖励分量、当前权重、组内方差和最终严格成功率，而不能只看一条总奖励曲线。

## 7. 第一次真实模型冒烟测试（Qwen2.5-1.5B Base）

配置：单张 RTX 4090，Transformers + BF16，本地 Qwen2.5-1.5B-Instruct，
离线快照回放，regex Agent，2 个任务各运行 1 次。基础链路已成功：模型加载、
ReAct 解析、工具回放、指标聚合和 `traces.jsonl` 落盘均无异常。

严格成功为 0/2。`ref_stress_image_restoration` 撞到迭代上限：任务 setup 已经
把 5 篇 CV 论文放进会话，正确动作应直接调用
`download_arxiv_pdf(ref="Image Restoration")`；Base 模型却调用 recent-search，
还混入了只属于 `search_arxiv_papers` 的 `query` 参数，随后用 `ref=null` 下载，
失败后重复搜索。这暴露出工具选择、参数归属、指代解析和失败恢复四类问题。

`infeasible_no_session` 应识别空会话并不调用工具，Base 模型却直接翻译。旧的本地
翻译 mock 还创建了 `paper_id=None` 的 PENDING 任务，导致模型收到虚假成功反馈。
同时终端把任何 `FINISH` 都打印成 PASS，因此出现 `PASS` 与 `accurate=False`
同屏的误导结果。

针对评测器已补四项修复：

1. 引入严格成功判据：必须 FINISH，且工具序列、参数、指代均正确，无解析/执行失败；
2. 终端将“FINISH 但结果错误”显示为 `FAIL(WRONG_RESULT)`；
3. replay 环境按真实 Python 函数签名校验参数，避免错误动作在 mock 中假成功；
4. 空会话翻译无法解析论文时直接失败，不再生成 `paper_id=None` 的任务。

这次测试的关键结论不是“小模型不行”，而是完成了评测链路验收，并用真实轨迹同时
找到了 Base 能力缺口和评测环境漏洞。修复后应复跑同样两题，确认观测语义可信，再扩大样本。

## 8. v2 冒烟复测：评测链路验收通过

### 8.1 运行配置与总体结论

复测仍使用单张 RTX 4090、本地 Qwen2.5-1.5B-Instruct、Transformers + BF16、
regex Agent 和离线快照回放。两个任务各采样 1 次，输出写入
`artifacts/base_smoke_v2`。

结果如下：

| 任务 | 终止状态 | 严格成功 | 主要失败形态 |
|---|---|---:|---|
| `ref_stress_image_restoration` | `FORCE_STOP` | 否 | 错工具、错参数、错指代、重复失败动作 |
| `infeasible_no_session` | `FINISH` | 否 | 空会话仍调用翻译，工具失败后声称完成 |

严格成功为 0/2，但这次 0/2 是可信的 Base 模型能力结果：模型加载、ReAct 循环、
工具回放、参数校验、失败反馈、指标汇总和轨迹写出全部正常，Benchmark 自身异常数为 0。

### 8.2 已修复问题的验收证据

第一，非法参数不再被离线环境放行。模型把 `search_arxiv_papers` 的 `query`
参数传给 `get_recently_submitted_cs_papers` 时，环境返回：

```text
工具参数错误: got an unexpected keyword argument 'query'
```

说明基于真实 Python 函数签名的参数校验已经生效。需要注意，`AR` 是合法的 arXiv
计算机分类（Hardware Architecture）；该动作报错的直接原因是多传了 `query`，
而选择 `AR` 是否符合当前任务应由参数准确率评价。

第二，空会话翻译不再产生虚假 PENDING 任务。`infeasible_no_session` 调用
`translate_arxiv_pdf(ref=null)` 时，环境明确返回没有可指代论文，不再出现
`paper_id=None` 的翻译任务。

第三，终止状态和任务成功已经分开。空会话任务虽然最终输出 `FINISH`，终端仍显示
`FAIL(WRONG_RESULT)`；报告同时给出“正常结束率 50%”和“严格成功率 0%”，
不再将模型声称完成误报为 PASS。

### 8.3 Bad Case 1：标题指代下载任务

任务是“下载标题包含 `Image Restoration` 的论文”。setup 已经把 5 篇 CV 论文放入
当前会话，标准路径只有：

```text
download_arxiv_pdf(ref="Image Restoration") → FINISH
```

模型实际轨迹为：

```text
recent-search（混入 query）
→ keyword-search
→ download(ref=null)
→ cache-status(ref=null)
→ 重复 cache-status(ref=null)
→ FORCE_STOP
```

逐步分析：

1. 模型没有利用 setup 已提供的会话状态，反而重新检索；
2. 第一次检索混淆两个搜索工具的参数边界；
3. 收到参数错误后能改用正确的关键词工具，说明具备局部纠错能力；
4. 搜索成功后错误地把 `ref=null` 理解成“自动使用搜索结果”，但它实际表示
   `last_active_paper_id`，搜索列表本身不会自动产生当前活跃论文；
5. 在没有 `ref`、`paper_id` 和 active paper 的情况下反复查询缓存，无法创造指代对象；
6. 相同失败动作重复，最终撞到迭代上限。

该轨迹暴露出六类能力缺口：已有状态利用、工具选择、参数归属、`ref=null` 语义、
失败恢复和重复动作抑制。它适合直接构造成 SFT 正例与困难负例，而不是马上用于 GRPO。

### 8.4 Bad Case 2：空会话不可行任务

任务是“把刚才那篇论文翻译一下”，但这是没有检索列表、下载记录和 active paper 的
全新会话。正确行为应当是不调用任何工具，向用户说明缺少上下文并请求标题或 arXiv ID，
然后结束。

模型实际调用 `translate_arxiv_pdf(ref=null)`。环境已经明确反馈指代对象不存在，
模型仍输出“任务已完成”并 `FINISH`。因此这条轨迹的指标是：正常结束为真、工具序列
不准确、参数分为 0、工具执行失败 1 次、严格成功为假。

该轨迹反映的不是模型不会结束，而是它没有检查前置状态，并且不能区分“工具调用已经
提交”和“工具执行明确失败”。建议在 SFT 数据中加入成对样本：有会话时执行翻译，
无会话时澄清或拒绝执行。

### 8.5 报告指标解释

“正常结束率 50%”来自两个任务中只有 `infeasible_no_session` 输出了 `FINISH`；
它不是业务成功率。“严格成功率 0%”才表示两条任务都没有同时满足正常结束、工具序列、
参数、指代、解析和工具执行正确。

“平均工具失败 2.5”来自第一条轨迹 4 次失败（非法搜索参数、下载无指代、两次缓存无指代）
和第二条轨迹 1 次失败，合计 5 次除以 2 个样本。

“Benchmark 异常 0”只表示没有模型加载失败、未捕获 Python 异常或结果写出失败；工具错误
已经被捕获并写入 Observation，所以它与“平均工具失败 2.5”不矛盾。

离线回放中的参数检查、内存状态读取和失败返回通常不足 1ms，而当前计时以整数毫秒记录，
因此工具时间显示为 0ms，不代表工具没有执行。

### 8.6 当前报告仍有两个边界

第一，“指代解析准确率 100%”目前不能理解为模型真的解析正确。这两个任务没有声明
`expected_paper`，指标实现会把不适用项默认设为 1.0，以免严格成功被无故扣分；但报告
又把默认值当作真实样本平均，造成 100% 的展示假象。后续应加入 `ref_applicable` 或只对
声明了目标论文的任务计算该指标，不适用时显示 `n/a`。

对于标题指代压力任务，还应声明固定锚点论文 ID。这样模型即使用序号、完整 ID 或另一段
等价标题子串，只要最终解析到同一篇论文就能得分；反之，使用过短子串匹配到错误论文会被
准确识别。

第二，`search_arxiv_papers` 与 recent-search 的搜索后处理尚未完全统一。当前
`BaseAgent` 主要只为 recent-search 保存会话论文并展示前三篇标题，关键词搜索可能只返回
“成功获取 N 条记录”。这不是本次任务失败的根因，因为标准路径根本不需要再次搜索；但它会
影响将来的“关键词搜索 → 按序号下载”链路。完整基线前应让两种搜索工具共用相同的会话写入
和结果格式化逻辑。

### 8.7 对难度分档的限制

当前每个任务只有 1 次采样，`0/1` 被机械地分入 floor，不能据此断言该任务一定不适合
GRPO。要判断组内奖励是否存在方差，至少应重复采样 3～4 次：稳定 `0/4` 的任务优先做
SFT，约 `1/4～3/4` 的任务才可能进入 GRPO 候选，稳定 `4/4` 的任务更适合作为评测上限。

从现有轨迹判断，这两条任务的失败均属于基础工具语义和状态判断不足，应优先通过 SFT
解决。只有 SFT 后同一任务出现成功与失败并存，才适合继续用 GRPO 优化策略稳定性。

### 8.8 当前阶段结论与下一步

当前状态可以固定记录为：

```text
本地模型接入：通过
离线快照回放：通过
非法参数校验：通过
空指代 fail-fast：通过
严格成功判定：通过
轨迹保存：通过
Base 模型任务表现：0/2
```

相同两题已经完成冒烟目的，不需要继续重复单次运行。完整 Base 基线前，先统一两个搜索工具
的会话处理，并修正指代指标的 N/A 展示与目标论文声明；随后选择 6～10 条不同类别任务，
每条重复 2～3 次做小规模基线。确认任务、快照、状态转移和指标都可信后，再冻结环境并运行
完整任务集。

面试中可将这轮工作总结为：

> 我通过真实 Base 模型的两轮冒烟测试，不只验证了本地推理和离线工具链路，还发现并修复
> 了测试替身保真度与成功口径问题。v2 中非法参数和空指代均能 fail-fast，FINISH 也不再
> 等同于成功。进一步分析轨迹后，我又识别出关键词搜索的会话后处理不一致，以及 N/A 指标
> 被展示成 100% 的问题。在冻结 Base—SFT—RL 对比环境前，我会先消除这些评测偏差，避免
> 将框架缺陷或统计假象误判为模型能力。

## 9. 冻结评测环境前的第二轮修复

根据 v2 轨迹，继续完成两项只修框架、不改变 Base 模型能力的改造。

第一，统一 recent-search 和 keyword-search 的会话语义。两种工具返回的论文列表现在都会
写入当前 session 的 `last_papers`，并用相同格式向模型展示论文数量和前三篇标题。这样
`search_arxiv_papers → download_arxiv_pdf(ref=1)` 与 recent-search 的多步链路具有一致
状态转移，避免同样是搜索工具却只有一种能被后续指代。

第二，为参数和指代指标增加 applicability 标记。任务未声明 `expected_tool_args` 或
`expected_paper` 时，内部中性分仍保持 1.0，避免严格成功被无故扣分；但聚合报告只对真正
声明标准答案的样本计算准确率，没有分母时显示 `n/a`，不再把“不适用”伪装成 100%。
CSV/JSON 明细同时输出 `arg_applicable` 和 `ref_applicable`，便于追溯每条分数的分母。

逐任务报告新增“严格成功率”列，避免只看正常结束率或工具准确率判断任务是否成功。
相关搜索状态、N/A 聚合、严格成功和多轮环境测试共 83 条，全部通过。

下一步使用不依赖 v1 切分的代表性任务集合做小规模 Base 基线，覆盖单步检索、关键词检索、
可选参数、ID 指代、跨步状态、多跳链路、负向约束和不可行请求。建议每条重复 3 次，共 24
条轨迹；该规模足以继续发现环境问题，也能初步观察同一任务的采样方差，但仍不作为最终
Base 统计结果。

## 10. 24 条 Base 小基线：结果、Bad Case 与环境修复

### 10.1 实验结果

本轮使用 8 个代表性任务、每题重复 3 次，共得到 24 条有效轨迹，Benchmark 未发生未捕获
异常。主要指标如下：

| 指标 | 结果 |
|---|---:|
| 正常结束率（FINISH） | 88% |
| 严格成功率 | 21%（5/24） |
| `pass^1 / pass^2 / pass^3` | 21% / 17% / 12% |
| 工具调用准确率 | 21% |
| 参数准确率 | 42% |
| 指代准确率 | `n/a` |
| 平均工具失败 | 0.5 |
| 撞迭代上限 | 3 |
| 声称完成但结果错误 | 16 |

`ref_score` 已正确显示为 `n/a`，证明上一轮 applicability 修复生效。88% 的 FINISH 与 21%
的严格成功之间存在 67 个百分点差距，说明当前 Base 模型的主要问题不是输出格式，而是
错误地判断“任务已经完成”：24 条轨迹没有解析失败，却有 16 条在工具序列或结果错误后仍
声称完成。

### 10.2 难度分档

| 任务 | 严格成功 | 暂定档位 | 结论 |
|---|---:|---|---|
| `opt_force_dl` | 3/3 | ceiling | 直接单步下载能力稳定 |
| `constraint_search_no_file` | 2/3 | middle | 暂时存在组内方差，修复环境后复核 |
| 其余 6 项 | 0/3 | floor | 先做 SFT，不直接进入 GRPO |

其余 6 项是 `search_AI_1d_3`、`search_kw_agentic_rl`、`ref_ctrl_id_download`、
`state_dl_then_cache`、`multi_cv3_dl1` 和 `infeasible_unsupported_action`。当前分档只是小样本
诊断结果，不能直接作为最终训练集划分。

### 10.3 逐类 Bad Case

1. **完成边界错误**：`search_AI_1d_3` 在已经完成检索后继续下载、翻译或查缓存；模型不懂
   “只检索”任务应立刻 FINISH。
2. **搜索工具混淆**：`search_kw_agentic_rl` 把 `query` 传给 recent-search，或遗漏关键词
   搜索要求的 `days=30`，还会在错误后重复换工具。
3. **忽略 setup 状态**：`ref_ctrl_id_download` 本可直接按 arXiv ID 下载，却先做无关搜索；
   搜索结果随后覆盖会话列表，使下载指向错误论文。
4. **状态顺序错误**：`state_dl_then_cache` 三次都先查 `ref=null` 的缓存，没有执行必要的
   “下载 → 查缓存”状态链。
5. **多步任务被回退数据污染**：`multi_cv3_dl1` 使用空关键词搜索后得到无关论文，再下载
   该回退论文或直接结束。
6. **不可行请求处理错误**：`infeasible_unsupported_action` 面对不支持的邮件发送请求，
   调用缓存查询并用 `paper_id="未找到"` 填参，而不是不调用工具并解释能力边界。

这些失败对应五类优先 SFT 能力：完成后停止、工具选择与参数约束、利用既有会话状态、
多步状态顺序、不可行请求的澄清或拒绝。当前 floor 任务没有组内成功样本，直接做 GRPO 会
缺少有效奖励方差。

### 10.4 新发现的离线回放问题

本轮还发现一个不能归因于模型的环境保真度问题。`MockArxivEnv` 为未知关键词提供带
`offline_fallback` 标记的确定性论文子集，目的是让回放可复现；但 `BaseAgent` 过去忽略
该标记，向模型返回“成功获取”，并可能把无关论文写入会话。空字符串 `query=""` 也能
触发该回退，而真实关键词工具会拒绝空查询。

这会产生两种污染：一是错误动作得到虚假成功 Observation，二是无关论文覆盖
`last_papers`，导致后续 ref 错误。因此本轮结果可用于发现问题和设计 SFT 模式，但涉及
关键词回退的失败轨迹不能直接作为干净训练数据。

已进行两项修复：

1. replay 参数校验对空白关键词 fail-fast，返回 `工具参数错误: query 不能为空`；
2. `BaseAgent` 检测到 `offline_fallback` 后返回明确工具失败，不把回退论文保存到 session，
   并提示检查完整的 `query` 和 `days`。

保留环境底层的确定性回退能力，便于直接测试与诊断；只是在面向 Agent 的交互层不再把它
包装成业务成功。新增测试分别覆盖空关键词拒绝，以及回退结果不污染会话。

### 10.5 验证状态与下一步

本地已通过 `py_compile` 语法检查。当前 Windows 解释器缺少项目依赖，无法在本机完成完整
单测；发现的其他测试报错属于本机依赖缺失和已有切分文件未纳入新增关键词任务，不是本次
两处改动造成。应在服务器的 `agentic-arxiv` 环境运行完整测试后再复跑 Benchmark。

下一步必须使用完全相同的 8 题 × 3 次配置，改用新前缀
`qwen25_15b_base_pilot_v2`。只有保持模型、seed、任务和采样次数不变，才能判断指标变化
来自环境修复。重点比较：

- 空查询和快照未命中是否都成为明确工具失败；
- 无关回退论文是否不再进入后续下载与指代；
- 原本稳定通过的 `opt_force_dl` 是否仍为 3/3；
- `constraint_search_no_file` 的 middle 档是否仍成立；
- FINISH 与严格成功率的差距是否缩小。

v2 通过后再冻结 Benchmark 环境，随后从失败轨迹构造 SFT 数据。SFT 后应先复跑相同评测，
将仍有成功/失败混合的 middle 任务用于 GRPO；稳定 floor 继续补 SFT，稳定 ceiling 留作回归
测试。这样才能形成可信的“Base → Bad Case → SFT → 再评测 → GRPO”证据链。

## 11. pilot_v2 复测：环境修复验收与随机种子隔离

### 11.1 v1/v2 对比

`qwen25_15b_base_pilot_v2` 仍为 8 题 × 3 次，共 24 条有效轨迹、0 个 Benchmark 异常。

| 指标 | pilot_v1 | pilot_v2 | 变化 |
|---|---:|---:|---:|
| 严格成功 | 5/24（21%） | 4/24（17%） | -1 条 |
| 正常结束率 | 88% | 62% | -26 pp |
| 工具准确率 | 21% | 17% | -4 pp |
| 参数准确率 | 42% | 49% | +7 pp |
| 平均工具失败 | 0.5 | 1.5 | +1.0 |
| 撞迭代上限 | 3 | 9 | +6 |
| 平均迭代数 | 3.2 | 3.7 | +0.5 |
| 平均 Token | 4883.4 | 5183.1 | +299.7 |

成功样本为 `opt_force_dl` 的 3/3 与 `constraint_search_no_file` 的 1/3。前者连续两轮保持
3/3，是可信 ceiling；后者从 2/3 变为 1/3，但仍属于 middle。其余六项仍是 0/3 floor。

严格成功下降不表示模型权重退化。本轮修改了环境 Observation：过去未知关键词会得到一批
被包装成成功的无关论文，现在明确返回工具失败。模型面对真实错误反馈后重复尝试，因而工具
失败数和 FORCE_STOP 增加。这正是修复所期望暴露出的 Base 能力，而不是回归。

### 11.2 环境修复已经生效

轨迹给出了直接证据：

- `search_arxiv_papers(query="")` 返回 `工具参数错误: query 不能为空`；
- 快照未命中的关键词返回“回退子集不代表查询匹配结果”，不再显示为成功搜索；
- `multi_cv3_dl1` 不再下载由空关键词产生的无关回退论文；
- `ref_ctrl_id_download` 的错误关键词检索不再覆盖 setup 中的论文列表。

因此 pilot_v2 可以作为“离线回退污染已修复”的验收轮。它仍然显示模型不会从错误中恢复：
例如关键词检索缺少 `days=30` 后，模型没有按提示补齐参数，而是在 recent-search 与多个错误
query 之间循环。

### 11.3 进一步发现：全局调用序号污染随机种子

本地 Transformers 客户端过去每次生成使用：

```text
seed = base_seed + 全局模型调用序号
```

这意味着，即使两次运行都传入 `--seed 42`，只要前面的某个任务因为 Observation 改变而
多走一轮，后面所有任务都会拿到不同随机数流。pilot_v2 的早期任务迭代次数明显增加，因而
后续样本无法与 pilot_v1 做严格逐轨迹配对；`constraint_search_no_file` 从 2/3 变为 1/3
不能只归因于环境修改。

已将随机流改为由 `task_id + trial` 稳定派生，并在每条轨迹开始前重置调用计数：

```text
stream_seed = base_seed + stable_hash(task_id, trial)
第 n 轮 ReAct 使用 stream_seed + n
```

stream key 不含 session prefix，也不依赖任务运行顺序；不同 Agent 模式对同一 task/trial 使用
相同起始随机流，便于公平比较。这样后续 Base、SFT、RL 即使迭代长度不同，也不会改变其他
任务的采样随机性。

### 11.4 当前结论与下一步

pilot_v1 用于发现虚假成功，pilot_v2 用于验证回退 fail-fast；两者都属于评测链路调试数据，
不作为最终 Base 数字。同步随机流修复并通过服务器测试后，应运行两份配置完全相同、仅
prefix/output 不同的 `pilot_v3a` 与 `pilot_v3b`：

1. 两次运行的工具序列、参数、终止状态、Token 数应逐 task/trial 一致；
2. 时间指标允许轻微波动；
3. 若行为字段一致，即可冻结 Benchmark 代码与快照；
4. 将 v3 作为正式 Base 小基线，然后开始构建 SFT 正例和困难负例。

现阶段数据优先级不变：完成后停止、关键词工具及完整参数、直接利用 ID/setup、状态链顺序、
不可行请求不调用工具。`constraint_search_no_file` 可保留为未来 GRPO 候选，但必须在冻结后的
v3 中重新确认其组内方差。

## 12. pilot_v3 双跑差异：离线运行时状态泄漏

### 12.1 表现与定位

pilot_v3a/v3b 的语义比较只报告一处差异：

```text
state_dl_then_cache trial=2: tokens 7210 != 7176
```

逐步比较发现，两轮的前两步动作相同，但下载 Observation 不同：第一次为
`existed=False`，第二次为 `existed=True`。原因是离线下载桩使用真实磁盘上的 PDF 是否存在
判断缓存命中；v3a 写出的占位文件被 v3b 继承。随后缓存状态 Observation 还带有每次不同的
`downloaded_at/updated_at` 墙上时钟。虽然随机种子已经按 task/trial 隔离，但模型输入发生了
变化，因此第三步是否附加 `paper_id` 不同，最终相差 34 Token。

这证明差异不来自 4090 的 CUDA 随机性，而是测试替身仍依赖跨运行可变状态。

### 12.2 修复

离线环境现在按以下规则处理：

1. `existed` 只由当前 task/trial 内的 `(session_id, paper_id)` 下载记录决定，不再读取磁盘
   文件是否已存在；同一轨迹第一次下载为 false，重复下载为 true；
2. mock PDF 大小固定为离线占位内容的 39 bytes，不受旧文件或真实 PDF 大小影响；
3. mock `PdfAsset` 的 `downloaded_at` 与 `updated_at` 固定为确定性时间，避免墙上时钟进入
   cache-status Observation；
4. 每个离线 task/trial 开始前重置 MemoryStore、离线环境运行时下载集合、mock 翻译序号和
   内存事件，消除跨任务及跨试验状态泄漏；
5. 增加“磁盘已有文件仍视为当前 trial 首次下载”和“trial reset 清空状态”的回归测试。

### 12.3 下一步

pilot_v3a/v3b 已完成故障定位，不需要作为正式 Base 结果。同步本轮修复并通过服务器测试后，
再次双跑 `pilot_v4a/v4b`。两次应使用相同 prefix 和 seed，仅输出目录不同。若
`traces.jsonl` 哈希一致且语义比较一致，即可冻结评测环境并把 v4 记录为正式 Base 小基线。

## 13. pilot_v4 双跑通过：Benchmark 正式冻结

服务器上的相关单元测试已全部通过。使用相同模型、任务集、快照、seed 和 session prefix，
仅改变输出目录，连续运行 `pilot_v4a/v4b` 后得到：

```text
TRACES_IDENTICAL
SEMANTIC_RESULTS_IDENTICAL
```

冻结轮仍使用 session prefix `qwen25_15b_base_pilot_v3`；v4a/v4b 是输出目录/实验轮次标签。
prefix 只是会话标识，只要双跑保持一致并在 manifest 中如实记录，就不影响结果有效性。

正式 Base 小基线结果：

| 指标 | 数值 |
|---|---:|
| 有效轨迹 / Benchmark 异常 | 24 / 0 |
| 正常结束率 | 67% |
| 严格成功率 | 21%（5/24） |
| `pass^1 / pass^2 / pass^3` | 21% / 17% / 12% |
| 工具调用准确率 | 21% |
| 参数准确率 | 47% |
| 平均解析失败 / 工具失败 | 0 / 1.6 |
| 平均迭代 / Token | 3.6 / 5089.9 |

任务分层为：`opt_force_dl` 3/3，是稳定 ceiling；`constraint_search_no_file` 2/3，是
middle；其余六项均为 0/3 floor。正常结束率比严格成功率高 46 个百分点，说明 Base 模型
经常会输出 FINISH，但并没有完成正确的工具序列。

参数准确率 47% 又显著高于工具/严格成功率 21%，说明不少轨迹能填对部分参数，却因为选错
工具、增加多余动作或没有及时停止而失败。与此同时解析失败为 0，证明 ReAct/JSON 格式不是
首要瓶颈；后续数据应重点训练决策、状态和停止边界，而不是继续堆格式样本。

逐任务能力解释：

- `search_AI_1d_3`：经常已正确 recent-search，随后又做 keyword-search；缺少完成后停止。
- `search_kw_agentic_rl`：遗漏 `days` 或改用 recent-search；缺少工具边界和完整参数约束。
- `opt_force_dl`：能利用 setup 论文列表，正确执行 `download(ref=1, force=true)`。
- `ref_ctrl_id_download`：把明确 arXiv ID 当成搜索词，而不是直接传给下载工具。
- `state_dl_then_cache`：不稳定掌握“先下载、再查缓存”的前置状态；多做搜索也会严格失败。
- `multi_cv3_dl1`：检索后不理解结果已经进入 session、可直接用 `ref=1`，反复二次搜索。
- `constraint_search_no_file`：2/3 能正确设置 `save_to_file=false` 并及时停止，存在可优化方差。
- `infeasible_unsupported_action`：面对不支持的动作仍调用无关缓存工具，而非解释能力边界。

这说明以下可复现性问题均已消除：

- 关键词快照回退被包装成虚假成功；
- 空关键词与非法工具参数未 fail-fast；
- 搜索结果污染后续 session 指代；
- 全局模型调用次数改变后续任务的随机种子；
- 占位 PDF 的磁盘残留改变 `existed`；
- 墙上时钟进入缓存 Observation；
- MemoryStore、mock 翻译序号等状态跨 task/trial 泄漏。

因此 pilot_v4 可以作为正式 Base 小基线。后续 SFT 和 RL 评测必须保持任务定义、快照、指标、
生成 seed 与运行参数不变；如确需修改评测环境，应升级版本并重新运行 Base，而不能直接与
v4 数字比较。

下一阶段进入数据构建。先从 v4 的 floor Bad Case 构造 SFT 正例与困难负例，优先覆盖：

1. 搜索任务完成后立即停止，不擅自下载或翻译；
2. recent-search 与 keyword-search 的选择及完整参数；
3. 直接利用 setup 中已有的序号、标题或 arXiv ID；
4. `下载 → 查询缓存` 等有序状态链；
5. 不支持的请求不调用无关工具，明确说明能力边界。

`opt_force_dl` 继续作为 ceiling 回归样本；`constraint_search_no_file` 是否进入 GRPO 候选，
以冻结后的 v4 实际成功分布为准。构建数据前还需保存代码提交、模型与快照哈希、运行命令及
v4a 原始产物，形成可追溯的 Base manifest。

## 14. 冻结当前 62 条任务的 v2 切分

### 14.1 为什么不直接沿用旧 v1

`data/splits/v1.json` 是早期 59 条任务的历史实验协议。当前 `expanded` 已增加 3 条
`keyword_search` 任务，共 62 条。如果直接覆盖 v1，同一文件名就会同时指代两套不同任务，
旧实验也无法重算。因此 v1 保持不可变，新增 `data/splits/v2_62.json`。

v2 不复制 v1 中的 `rates`。任务成功率取决于模型、提示词、工具语义、快照和评分器；旧 rates
是在旧条件下测得的，不能代表当前 Qwen2.5-1.5B Base。当前完整 Base 跑完前，v2 故意不生成
`rl_train`，避免用过时难度分层选择 GRPO 数据。

### 14.2 为什么把 pilot 8 条设为 dev

pilot 8 条是从 62 条中按能力维度挑出的诊断子集，并非另一套任务：

```text
当前 expanded 62
├── train 36：允许据此构建训练数据
├── dev 8：pilot 中已反复查看轨迹，用于调试/Bad Case 分析
├── iid_test 14：同模板换参数的盲测
└── ood_test 4：未见模板或更长组合链的盲测
```

因为我们已阅读 pilot 轨迹，并据此确定“停止边界、关键词工具、直接 ID、状态链、不可行请求”
等优化方向，它们已经参与了人工决策，不能再声称是未见测试数据。将其固定为 dev，既保留了
快速迭代价值，也避免把针对这 8 条的改进误报成测试集泛化。

关键词检索族也按模板做了三路覆盖：`search_kw_llm` 在 train，
`search_kw_agentic_rl` 在 dev，`search_kw_rag` 在 iid_test。这能检验模型学到的是
“选择 keyword 工具并补全参数”的规则，而不是背下关键词 `agentic rl`。

### 14.3 v2 的防泄漏校验

`tests/test_splits.py` 增加以下不可变条件：

1. 62 个任务必须恰好出现一次，不能遗漏、重复或跨集合；
2. pilot 8 条必须与 dev 完全一致；
3. 每个 iid_test 任务的模板必须在 train 中出现；
4. OOD 模板只能存在于 ood_test，不能泄漏到 train/dev/iid；
5. keyword_search 族必须分别覆盖 train/dev/iid；
6. 完整 Base 前 v2 不得带旧 rates；Base 冻结后 rates 必须只覆盖 train；
7. `load_split(显式路径:dev)` 必须能正确读取 pilot 集。

完整 Base 可以评测所有 62 条以记录基准，但构建 SFT 数据和调参时只能分析 train/dev 轨迹。
iid/ood 应保持盲测：训练完成前只保留原始产物，避免根据其具体 Bad Case 反向修改训练数据。

## 15. Qwen2.5-1.5B 完整 Base（v2_62）

### 15.1 完整性与整体结果

四个 split 均按 `repeat=3` 完成：train 108、dev 24、iid 42、ood 12，共 186 条轨迹，
Benchmark 异常为 0。整体加权结果如下：

| 指标 | 结果 |
|---|---:|
| 正常结束率 | 59.1% |
| 严格成功率 | 18.8%（35/186） |
| 工具序列准确率 | 26.3% |
| 参数准确率 | 44.6% |
| 假完成率 | 22.6% |
| 平均迭代数 | 3.93 |
| 平均 Token | 5691.3 |

分集合的严格成功率为：train 14.8%、dev 20.8%、iid 33.3%、ood 0%。OOD 平均迭代数
5.3、平均 Token 7628.2，均高于整体，说明模型遇到未见组合链时不只是答错，还会产生更长的
无效探索。IID/OOD 只记录聚合基线，不使用其逐任务 Bad Case 反向修改训练数据。

### 15.2 train 难度谱与 RL 候选

36 条 train 按严格成功率分为：floor 27、middle 6、ceiling 3。6 条 middle 为：

- `chain_cv5_cache_dl_tr_cache`：1/3；
- `multi_cr5_cache1`：1/3；
- `search_CL_30d_10`：1/3；
- `search_CL_7d_5`：1/3；
- `search_LG_3d_10`：2/3；
- `search_RO_3d_8`：1/3。

这些成功率已写入 `data/splits/v2_62.json`，并附带模型、seed、repeat、session prefix、产物路径
和 train summary 的 SHA256。`rl_train` 不重复落盘，而由 train 与 rates 动态计算为上述 6 条。

但这不意味着现在应立即 GRPO。27 条 floor 占 train 的 75%，说明 Base 的主要问题仍是基础
工具决策、状态链和停止边界。应先构建 SFT 数据把部分 floor 推到 middle，再重新评测和划分
RL 训练集；否则大量 prompt 的组内奖励会全部为零，GRPO 没有有效梯度。

## 16. SFT 数据生成的防泄漏改造

原 `scripts/generate_sft_data.py --task_set expanded` 会直接读取全部 62 条 `EXPANDED_SPECS`，
其中包含 train/dev/iid/ood。若据此生成并训练，IID 的原始实例和 OOD 的长链模板都会进入
SFT，训练后 Benchmark 的提升无法解释为泛化。

生成入口现在执行以下约束：

1. `expanded` 必须显式传入版本化的 `PATH:train`；裸 `train` 会落到历史 v1，因此也拒绝；
2. dev、iid_test、ood_test 均不能作为正式 SFT 来源；
3. 当前合法入口为 `data/splits/v2_62.json:train`，应恰好选择 36 个 source task；
4. 每个训练样本记录 `source_task_id`、`source_split` 和 `trajectory_step`，可以反查数据血缘；
5. setup、depends_on 或专家工具步骤一旦失败，整次生成立即失败，不能把失败 Observation 后的
   `FINISH` 写成“专家答案”；
6. 确定性专家轨迹在写出前还要经过与 Benchmark 相同的严格成功判定；
7. `--use_llm` 产生的轨迹只有严格成功时才进入 SFT，失败轨迹只记录 REJECT。

因此 dev 只用于发现“直接 ID、状态链、停止边界”等能力缺陷；可以从 train 模板生成新的参数
和措辞来覆盖这些能力，但不能复制 dev 原题。IID/OOD 在训练完成前继续保持任务级盲测。

### 16.1 首次生成暴露的搜索会话状态断裂

首次用 v2 train 生成时，`chain_ai5_dl2_tr2` 的搜索成功返回论文，但紧接着
`download_arxiv_pdf(ref=1)` 报“未找到论文”。原因不是快照缺数据，而是搜索工具的公开 schema
不接收 `session_id`：生成器直接调用 env 后，没有像 BaseAgent 那样把搜索结果通过
`SideEffectManager` 写入当前 session，所以下一步无法解析序号 `1`。

修复后，专家工具执行分为两个层次：Action 只保留模型应学习的公开参数；`session_id` 作为框架
状态单独传给执行层。搜索返回后显式写入 `last_papers`，下载/缓存等需要 session 的工具仅在执行
副本中注入 session_id，训练标签中不出现机器生成的会话标识。翻译也走
`LocalSideEffectManager` 的确定性 mock，不启动真实翻译线程。每个 source task 开始前还会显式
重置 MemoryStore，与 Benchmark 的 trial 隔离规则一致。

新增回归用例用一条“搜索 → 下载 → FINISH”轨迹验证：搜索结果必须成为下一步可解析的会话
状态，且任何 assistant Action 都不得包含 `session_id`。这类测试验证的是 Agent 的状态转移，
而不只是单个工具函数是否返回成功。

## 17. 从 85 条 seed 开始的第一阶段扩增

通过严格专家校验的 85 条样本来自 v2 的 36 个 train source task。第一阶段不直接创造新的
工具参数组合，而是做确定性的语义保持语言扩增：每个 seed 使用 6 种任务请求包装和 2 种
Thought 表达，共得到 `85 × 6 × 2 = 1020` 行。Action、工具参数、历史 Observation、步骤顺序
均保持不变。

这样设计的原因是，1.5B 模型容易记住固定指令措辞。先改变表面语言，可以训练它在不同表达
下仍选择相同工具和参数；但这些样本共享原来的 36 种任务语义，因此不能把“1020 行”解释成
“1020 个独立任务”。正式报告必须同时写行数和 semantic task 数量。

`scripts/augment_sft_data.py` 对每条输出记录以下审计字段：

- `source_task_id`、`source_split`、`trajectory_step`：继承原始数据血缘；
- `parent_sample_sha256`：指向 85 条 seed 中的父样本；
- `sample_sha256`：扩增样本的内容指纹，用于拒绝重复；
- `augmentation.kind/task_variant/thought_variant`：记录使用的变体规则。

脚本执行前会 fail-fast 检查：seed 必须恰好覆盖 v2 的全部 train task，不能出现 dev/IID/OOD
source id；prompt 必须有唯一的“当前任务”边界；assistant 必须有唯一 Action 边界；任务正文
也不能与任一留出任务原文相同。输出后生成 manifest，记录输入/输出 SHA256、行数、唯一指纹数、
source task 数和 heldout overlap。

`tests/test_augment_sft_data.py` 回归验证三类不变量：只替换 prompt 的当前任务区域；扩增前后
Action 文本逐字节相同；一旦 source id 属于 heldout 就拒绝生成。第一阶段产物建议命名为
`data/sft/sft_v1_linguistic.jsonl`，保留 `sft_v0_seed.jsonl` 原件，不做覆盖。

这批 1020 行可用于 QLoRA 数据管线冒烟和“有/无语言扩增”的消融实验，但还不是最终数据集。
下一阶段应从 train 模板生成新的参数组合和状态轨迹，并再次通过离线专家执行和严格成功判定；
只有那一层才真正增加任务语义覆盖度。

## 18. 参数化语义扩增（parametric v1）

语言扩增通过后，新增 `scripts/generate_parametric_sft_data.py`。它从 26 个 v2 train 父任务
派生 65 个新任务，预计生成 159 条逐决策专家样本。覆盖 search、keyword_search、optional、
state、constraint、infeasible、两步 composite 和 long_chain 八类能力。

本阶段改变的是真实任务参数，包括论文类别、检索天数、返回数量、论文序号、翻译线程数、
force/keep_dual 选项以及同一长链中的引用位置。每条派生任务都进入
`MockArxivEnv(replay)` 实际执行；只有完整轨迹通过 Benchmark 的严格成功判定，才写入数据。
这与直接字符串替换不同：参数必须能被快照和会话状态正确支持，setup、Action 和任务描述也
必须互相一致。

为保护测试结论，生成器设置了两层边界：

1. 血缘边界：每个 `parent_task_id` 必须属于 v2 train，不能来自 dev/IID/OOD；派生文本和 id
   不能与现有 62 条 Benchmark 完全相同；
2. 拓扑边界：派生任务的正式 steps 和 setup 工具序列必须与 train 父任务逐项相同，只允许
   改参数。v2 OOD 留出的 `(composite, 3)` 与 `(composite, 4)` 不会被创造出来。

产物每行记录 `derived_task_id`、`parent_task_id`、`generation_parameters`、`dataset_stage` 和
`sample_sha256`。manifest 进一步固定 split、snapshot、output 三个 SHA256，以及父任务分布、
类别分布、链长分布和完整派生任务标准答案。

新增 `tests/test_parametric_sft_data.py`，验证 65 个派生任务 id 唯一、父任务全部来自 train、
工具拓扑保持不变、不复用 Benchmark 原文，以及所有派生 composite 都保持两步。这样能明确
回答面试中的关键问题：参数扩增提升的是“同一策略在新参数下的执行能力”，而不是通过训练
OOD 工具链来制造虚假的泛化提升。

## 19. 参数化 seed 的语言扩增与最终 SFT 混合

65 个参数化任务严格执行后得到 159 条逐决策 seed。为了避免原始 36 个任务的 1020 条语言
数据在训练中压过参数化任务，对这 159 条也使用相同的 6 种任务包装和 2 种 Thought 表达，
得到 `159 × 12 = 1908` 条。扩增前由 `augment_parametric_sft_data.py` 独立审计 seed manifest、
输入 SHA256、派生任务 id、train 父级、generation parameters、任务正文、trajectory_step 和
messages 指纹，不能把普通 seed 的宽松假设直接套到参数数据上。

最终混合不是按两个文件各取 50%，而是按语义任务密度判断：原始侧
`1020 / 36 = 28.33` 行/任务，参数侧 `1908 / 65 = 29.35` 行/任务，两者几乎一致。因此保留
两侧全部数据，形成 2928 行、101 个语义任务实例的训练集，比强行 1:1 下采样更合理。

`build_sft_train_mix.py` 在合并前验证两个来源文件与各自 manifest 的 SHA256 和冻结行数，
重新计算每条 messages 指纹，拒绝来源内或来源间重复，添加 `mixture_source`，然后以 seed=42
确定性打乱。输出 manifest 记录两个来源、样本量、语义任务数、rows/task 和所有输入哈希。

这一设计需要在面试中表述为“任务粒度近似均衡”，不能说成“两个数据集行数均衡”：最终
两侧行数约为 1:1.87，但这是因为参数侧有 65 个新任务，原始侧只有 36 个任务；每个任务实际
获得的训练监督量接近一致。

## 20. 单张 RTX 4090 的 4-bit QLoRA SFT

原 `rl/train_sft.py` 使用 `AutoModelForCausalLM.from_pretrained(model)` 直接加载全参数模型，
没有量化配置、LoRA adapter、可训练参数检查或数据 manifest 校验。即使 1.5B 全参可能勉强
运行，这也不是项目计划中可迁移到更大模型的 QLoRA 链路。

改造后的默认方案为：NF4 4-bit 基座、double quant、BF16 compute、LoRA `r=16/alpha=32/
dropout=0.05`，目标层覆盖 Qwen2.5 attention 的 q/k/v/o projection 与 MLP 的
gate/up/down projection。优化器使用 `paged_adamw_8bit`，micro batch 为 1、梯度累积 8，
并启用非 reentrant gradient checkpointing。学习率从全参常用的 2e-5 调整为 adapter 训练
更常见的 1e-4。

QLoRA 的加载方式按锁定版本实现。TRL 0.29.1 的 `SFTTrainer` 构造函数只有 `peft_config`，
没有新版文档中的 `quantization_config` 参数，因此不能照搬 main 分支示例。训练入口先用
`AutoModelForCausalLM` 和 `BitsAndBytesConfig` 手动加载 4-bit 基座，调用一次
`prepare_model_for_kbit_training(use_gradient_checkpointing=False)` 完成冻结与 LayerNorm 准备，
再用 `get_peft_model` 包装 LoRA。已经包装好的 PeftModel 传给 Trainer 时不再传 `peft_config`；
gradient checkpointing 和 input grads 由 TRL 按 `SFTConfig` 打开，避免重复包装或重复准备。

训练入口新增三组 fail-fast：

1. 数据：`sft_v3_train_mix.jsonl` 必须有 `qlora_sft_train_mix` manifest，文件 SHA256、行数、
   唯一指纹数必须一致；
2. 运行环境：加载模型前检查 CUDA、BF16、关键依赖版本、GPU 名称与总/空闲显存；
3. 模型：Trainer 构造后要求 `is_loaded_in_4bit=True`，所有可训练参数名必须属于 `lora_`，且
   可训练参数不能为零。否则拒绝把全参或错误冻结的训练误报成 QLoRA。

`--inspect_only` 只读取 tokenizer 和数据，执行 manifest 与 token 长度体检，不加载模型。
冒烟阶段使用 `--max_steps 30 --no-verify`，并记录 loss、可训练参数比例和峰值显存到
`final/training_manifest.json`。QLoRA 的 `final` 默认只保存 adapter 和 tokenizer；基座仍由
`base_model` 路径引用，这也是它比完整模型 checkpoint 小得多的原因。

首次冒烟命令在参数解析阶段因 `--no-verify` 失败：当时 SFT 的 `--verify` 仍是单向
`store_true`，默认虽为 False，却不会自动生成反向选项。训练没有开始，也没有产生 checkpoint。
短期可直接删除 `--no-verify`；代码随后改为 `argparse.BooleanOptionalAction`，使
`--verify`、`--no-verify` 与省略参数三种写法都具有明确语义，并与 GRPO/OPD 入口保持一致。

第二次冒烟已通过数据审计、token 长度检查、4090/BF16 检查和 4-bit 模型加载，但在构造
`SFTConfig` 时出现 `unexpected keyword argument 'use_cache'`。原因是 `use_cache` 控制模型
前向时是否保存 KV cache，属于 `model.config`；锁定的 TRL 0.29.1 并未把它定义成
`SFTConfig` 字段。QLoRA 分支在模型加载后本来就执行了 `policy.config.use_cache=False`，所以
从训练参数中删除它不会改变训练语义，反而明确了“模型配置”和“Trainer 配置”的职责边界。

训练入口同时增加 dataclass 字段过滤：构造配置前读取当前 `SFTConfig` 的真实字段集合，过滤
跨版本不存在的参数并显式告警。这样版本差异不会在加载约 3 GiB 基座之后才以 `TypeError`
中断；核心 QLoRA 模型侧设置仍由独立检查保证，不能依赖过滤器静默补救。该次异常发生在
`SFTTrainer` 初始化和第一个 optimizer step 之前，因此没有训练 checkpoint，也无需清理模型
输出后再重跑。

## 21. SFT 结果审计与语义指代评分

一轮正式 QLoRA SFT 已完成。最终 adapter 的 train loss 为约 0.0647，峰值显存约 5.73 GiB，
可训练参数 18,464,768。固定 train 切分三次评测中，严格成功率从 Base 的约 14.8% 提升到
约 63.9%，正常结束率从约 60.2% 提升到 100%，工具序列准确率从约 22.2% 提升到约
88.9%。这说明 SFT 已经显著学会 ReAct 格式、工具选择和链路执行，但“正常 FINISH”本身仍
不能作为业务成功证据。

逐轨迹审计发现原严格指标存在一类假阴性：任务标准答案要求 `ref=1` 或标题子串时，模型可能
使用 `ref=null`（指向 setup 留下的 last-active 论文）或另一个等价序号；工具最终返回的
`paper_id` 完全正确，但原始参数字符串不同，仍被判错。这种误判若直接进入 GRPO，会把正确
轨迹作为负样本，奖励方向与业务目标相反。

修复方案不是放松全部参数，而是增加“离线语义 oracle”：对每个任务在冻结快照中依次执行
setup 和标准 steps，生成与 `expected_tools` 等长的 `expected_paper_ids`。搜索步骤记为 null，
下载、翻译、缓存步骤记录实际解析到的论文 ID。Benchmark 和 GRPO 奖励仍严格检查工具顺序、
非 ref 参数、工具异常和终止类型；只有 `ref` 从字符串相等改为“对应步骤最终解析到同一篇
论文”。多论文长链逐步比对各自 target，不使用单一全局 paper id。

同时把 `AgenticArxivMultiTurnEnv` 的工具签名和状态语义与生产工具对齐：recent search 接受
`save_to_file/output_path`，download 接受 `force`，translate 接受 `force/service/threads/
keep_dual/paper_id`，cache 接受 `paper_id`，并在每条 rollout 内维护 downloaded/translated
状态。这样合法的 Benchmark 标准动作不会在 GRPO 环境里因为 mock 签名缺参数而失败，缓存
Observation 也能真实反映前序动作。

该改动后的正确性边界是：同一篇论文的不同引用形式可等价；解析到不同论文仍为失败；省略
`force/threads/keep_dual` 等业务参数仍为失败；多调、少调、乱序、工具报错或假完成仍为失败。
在进入 RL 前必须先通过语义 oracle 的 62 任务 fail-fast、相关单元测试，并用同一 SFT adapter
重跑 train/dev r3，依据修正后的成功率重新划分 floor/middle/ceiling。

## 22. 语义重评分结果与 GRPO 探针集

现有轨迹重评分后，train 的严格成功率从 63.89% 提升到 77.78%，参数准确率从 72.45%
提升到 82.18%，指代准确率为 85.51%。变化精确来自 5 个任务、每个 3 次，共 15 条轨迹：
`opt_force_tr`、`opt_keep_dual`、`opt_threads` 使用 null 指代 setup 中的 last-active 论文，
`ref_ctrl_word_marionette` 和 `ref_stress_uncertainty_aware` 使用不同 ref 形式但解析到同一论文。
dev 严格成功率保持 37.5%，说明新规则没有通过整体放宽参数制造虚假泛化提升。

修正后 train 仍有 8 个稳定失败任务：两条长链分别遗漏 keep_dual 或最后一次下载，三条
infeasible 任务仍错误调用工具，三个 ref 压力任务调用失败或没有正确维持状态。dev 还有 5 个
稳定失败，继续只用于开发诊断，不并入训练。当前确定性 r3 中 train 形成 28 个 ceiling、8 个
floor、0 个 middle；这不等于 GRPO 一定无梯度，因为 GRPO 会以 temperature>0 对同一 prompt
采样多条轨迹，实际判据应是组内 reward 标准差，而不是贪心解码的 0%/100%。

因此新增 `data/splits/v3_grpo_probe.json:rl_probe`，只包含上述 8 个 v2 train 失败任务，用于
10 步随机多采样探针。它不是正式 RL 训练集：只有探针日志中出现非零 `reward_std`、且
`frac_reward_zero_std < 1` 的任务/批次才有相对优势和有效梯度。dev/IID/OOD 继续隔离。

`train_grpo.py` 同时改为单卡 4090 的 QLoRA 续训入口：读取 SFT adapter_config 中的基座路径，
以 4-bit NF4 + double quant + BF16 加载冻结基座，再用 `PeftModel.from_pretrained(...,
is_trainable=True)` 恢复现有 SFT LoRA 权重。训练前验证只有 `lora_` 参数可训练并检查 4-bit
标记；新增 max_steps、paged 8-bit optimizer、gradient checkpointing、adapter 输出和训练
manifest。max_turns 还会按“最长工具链 + FINISH”做 fail-fast，防止轮数预算让标准解法天然
不可达。

## 23. 单卡 QLoRA GRPO 冒烟、可观测性修复与最终奖励方差审计

### 23.1 4-bit SFT Adapter 续训链路已经跑通

GRPO 从 `outputs/sft_qlora_v1_e1/final` 读取 SFT adapter，通过其
`adapter_config.json` 找到 Qwen2.5-1.5B-Instruct 基座，再以 NF4 4-bit、double quant 和
BF16 compute 加载冻结基座，并用 `PeftModel.from_pretrained(..., is_trainable=True)` 恢复
18,464,768 个可训练 LoRA 参数。首轮 10 step 探针完成，无 OOM、NaN 或 CUDA 异常，峰值
PyTorch allocated memory 为约 5.42 GiB，证明当前单张 4090 的瓶颈不是模型加载或反向显存。

量化模型日志中的 `model_parameters` 不能解释成模型结构从约 15.4 亿参数缩成约 9 亿参数。
bitsandbytes 会以打包形式存储 4-bit 权重，部分 `numel()` 统计看到的是底层打包存储；模型逻辑
参数量没有改变。训练日志中约 907,081,216 的量化基座统计与 manifest 中约 925,545,984 的
总统计之差恰好是 18,464,768 个 LoRA 参数。相对于原始约 15.4 亿参数，真实可训练比例约为
1.2%，而不是日志按打包存储分母计算出的约 2%。

### 23.2 首轮探针暴露的可观测性缺口

首轮日志只有 TRL 的总 `reward`、`reward_std` 和 `frac_reward_zero_std`。10 步中平均
`frac_reward_zero_std` 为 62.5%，即只有约 37.5% 的 prompt 组产生组内奖励差异。但全局
`reward_std > 0` 不能证明 GRPO 一定有梯度：它可能只表示不同任务的平均奖励不同；GRPO 的
advantage 在同一个 prompt 的多条 completion 内计算，真正需要的是逐 prompt 组内方差。

进一步检查发现，当使用 `--report_to none` 时，`train_grpo.py` 只有在外部日志 backend 非空
时才执行 `tracker.bind(trainer)`。奖励函数虽然持续调用 `tracker.record()`，五项奖励分量却只
积压在 `_pending`，既不出现在控制台，也不会落盘；长训练还会造成不必要的内存增长。

修复包括：

1. 无论是否启用 TensorBoard/wandb，都把 tracker 接入 TRL `_metrics["train"]`；
2. 按 `task_id` 汇总同一 prompt 的 generations，记录 mean、std 和 zero_std；
3. 对 completion 数与 task_id 数做 fail-fast，防止错位奖励静默进入训练；
4. 单独累计每个任务的原始奖励和每组标准差，在训练结束或保护器退出前写出
   `reward_probe_summary.json`；
5. 修正启动提示：`report_to=none` 只表示不发送外部曲线，控制台仍会显示奖励分量；
6. 新增组内有效/零方差区分、JSON 持久化和输入长度错位回归测试。

### 23.3 奖励分量说明了为什么 SFT 后仍需要 RL

接入分量日志后的 G=2 探针中，format 始终为 1、parse error rate 始终为 0，process 平均约
0.848；但 tool 平均约 0.485、argument 平均约 -0.297、outcome 平均约 0.313，工具调用失败率
约 50.8%，正常 FINISH 率约 79.7%。这说明 SFT 已解决输出格式和基本 ReAct 结构，剩余矛盾
集中在工具语义、参数填写、会话状态和最终任务结果，属于序列级、可验证结果级问题，具备使用
Agentic RL 的合理性。

当前 8 步都处于 `RewardCalculator(curriculum_steps=30)` 的早期阶段。完整基础权重为
`format/tool/argument/process/outcome = 1/3/2/1/3`，tool、argument、outcome 在前 30 步乘以
`1/3`，因此实际日志权重为 `1/1/0.6667/1/1`。所以 total reward 看起来在 0.4～0.6 并不等于
业务结果已经较好；跨过课程边界时，即使策略不变，总奖励分布也可能变化。正式实验必须同时
记录五项原始分量、当前权重和严格成功率，不能只比较 total reward。

### 23.4 `clipped_ratio` 不是 512 token 不够

将 `max_completion_length` 从 2048 降到 512 后，实际 completion 最大只有约 115 tokens，
但 `clipped_ratio` 仍在 0～0.5 之间。逐步比对发现它与 `1 - rollout/finished` 完全一致：
`finished=0.75` 时 clipped 为 0.25，`finished=0.5` 时 clipped 为 0.5。这里的 clipped 主要表示
轨迹在 `max_turns=5` 内没有主动 FINISH，而不是撞到 512 token 上限。当前应保留 512，并把
“未学会结束/发生循环”作为 Bad Case；盲目增加 token 上限或 max_turns 只会延长无效探索。

### 23.5 为什么从 G=2 改为 G=4

逐任务 G=2 审计每个任务只有两条 completion，整体仅约 31.25% 的任务组具有非零方差，其中
`ref_stress_state_across` 一度被观察为 100% 零方差。对于成功概率约 0.5 的二元结果任务，
G=2 抽到一好一坏的概率只有 50%；G=4 出现至少一种不同结果的概率提升到 87.5%。4090 的
G=2 峰值显存只有约 5.42 GiB，因此增加采样数比继续扩大普通 batch 更能直接改善 GRPO 的
组内排序信号。

冻结 SFT 策略（`lr=0`）的 G=4 初步审计把整体有效组比例提升到约 62.5%，并使
`ref_stress_state_across` 出现明显方差。这证明 G=2 的零方差结论存在小样本假阴性。随后用
G=4、每任务 8 组、每组 4 条，共 32 条 rollout/任务进行最终复核；`lr=0` 保证统计反映固定
SFT 策略与奖励环境，而不是边训练边改变采样分布。

### 23.6 G=4 最终审计结果

最终审计共有 8 个任务、64 个 prompt 采样组、256 条 rollout，其中 35 个组具有非零组内
方差，整体 informative group fraction 为 `35/64 = 54.69%`。

| 任务 | rollout | 组数 | 平均奖励 | 奖励范围 | 平均组内 std | 有效组比例 | 当前判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `chain_ro5_dl_three` | 32 | 8 | 0.370 | -0.107～0.446 | 0.127 | 75.0% | 强候选：多步链存在稳定排序信号 |
| `infeasible_no_session` | 32 | 8 | 0.495 | 0.050～1.000 | 0.332 | 75.0% | 强候选：拒绝/误调用呈明显二元差异 |
| `infeasible_unknown_id` | 32 | 8 | 0.614 | 0.050～1.000 | 0.332 | 75.0% | 强候选：有较大结果方差 |
| `ref_stress_state_across` | 32 | 8 | 0.413 | 0.125～0.500 | 0.121 | 75.0% | 强候选：G=2 曾是假阴性 |
| `ref_ctrl_word_handover` | 32 | 8 | 0.441 | -0.111～0.500 | 0.086 | 50.0% | 中等候选：保留状态指代多样性 |
| `infeasible_zero_index` | 32 | 8 | 0.881 | 0.029～1.000 | 0.163 | 37.5% | 边界候选：均值偏高，接近 ceiling |
| `ref_stress_image_restoration` | 32 | 8 | 0.460 | 0.125～0.500 | 0.053 | 37.5% | 边界候选：有方差但信号较弱 |
| `chain_lg10_dl_tr_last` | 32 | 8 | 0.545 | 0.456～0.551 | 0.006 | 12.5% | 弱候选：近乎恒定，优先重构或剔除 |

最终结果没有简单复用第一次 G=4 的四组统计。随着每任务组数从 4 增加到 8，整体有效比例从
初步观察的 62.5% 收敛到约 54.7%，`ref_stress_state_across` 也从初步的 100% 回落到 75%。
这说明“某任务有/没有方差”本身也是随机估计，必须同时报告 generations、group_count 和
冻结策略条件，不能根据单次两样本结果做硬筛选。

### 23.7 当前训练集决策与下一步

`v3_grpo_probe.json` 的 8 个任务全部来自 v2 train 的稳定失败项；dev、IID 和 OOD 没有用于
挑选、改题或调奖励，因此当前数据链路没有测试集泄漏。GRPO 是在线算法，这里的“RL 数据”
不是预生成模型答案，而是冻结的任务 prompt、离线环境状态、可验证标准和奖励函数；completion
在训练时由当前策略现场采样。

当前建议把 4 个 75% 有效任务作为核心，`ref_ctrl_word_handover` 作为中等难度补充；
`infeasible_zero_index` 与 `ref_stress_image_restoration` 可低比例保留做边界覆盖；
`chain_lg10_dl_tr_last` 只有 1/8 组有效且平均组内 std 约 0.006，直接反复训练的算力收益最低，
应优先检查是否可以通过 train 血缘内的参数化变体增加决策分叉，否则从正式 RL 集中剔除。

上述审计后已新增 `data/splits/v4_grpo_train.json:rl_train`。切分以
`informative_group_fraction >= 0.375` 为纳入阈值，保留 7 个任务；同时满足
`informative_group_fraction < 0.25` 和 `mean_group_std < 0.02` 的
`chain_lg10_dl_tr_last` 被记录到 `excluded_low_variance`，而不是从历史 probe 中删除。v4 内嵌
全部 8 个任务的审计统计、来源模型、G、温度、冻结学习率和防泄漏策略，使筛选原因可追溯。

`train_grpo.py` 已新增 `--reward_curriculum_steps` 并写入 training manifest。默认值仍为 30，
用于复现旧探针；正式 SFT 后主实验必须显式传 0，从第 0 步启用完整的
`1/3/2/1/3` 权重。Canary 也改为把当前 global step 传入奖励计算，避免训练跨过课程边界后，
Canary 仍错误使用 step=0 的早期权重。下一步先用 v4、G=4、课程 0 做 20 步训练冒烟，确认
新权重下仍有组内方差、KL 稳定且 adapter/manifest 正常保存，再决定正式训练步数。

面试中可将这一阶段概括为：**先用 SFT 后真实 Bad Case 构造无泄漏 RL 候选，再通过冻结策略的
逐 prompt 多采样审计验证奖励是否有组内区分度；资源允许时将 G 从 2 提到 4，使有效采样组从
约三成提升到五成以上，并据此冻结正式 GRPO 数据，而不是看到全局 reward_std 非零就直接开训。**

## 24. 原始 GRPO rollout 审计：从奖励统计追到模型原文

之前的 `reward_probe_summary.json` 只能回答“奖励是否有组内方差”，不能回答“同组回答
具体错在哪里”。Benchmark 中 `avg_parse_failures=0` 也不能代替这一审计：Benchmark 用的是评测温度与
确定性生成，GRPO 训练 rollout 用 `temperature=1.0` 对同一 prompt 随机采样 4 条，两者不是
同一个生成分布。而且没有在当时保存的模型原文无法事后从奖励均值还原，因此旧的 60 步训练
需要做一次短采样复跑，不需要再完整训 60 步。

本次增加了三层审计链路：

1. `make_multiturn_rollout_func` 在每一轮解码后立即保存 `raw_assistant_turns`。这是模型自己生成的
   文本；环境后续追加的 `Observation:` 只进入 completion token 流供下一轮使用，不会被误判为
   模型幻觉。rollout 把 policy/environment token mask 返回给 TRL 以隔离策略 loss；
   `trajectory_results` 另存解析后的 history、token 计数、是否被截断以及是否跑满 `max_turns`。
2. 奖励函数在同一个调用中把“实际用来计分的轨迹”、五个奖励分量和 total reward 交给
   `RolloutAuditWriter`。因此 JSONL 里的文本、轨迹和分数具有同一条样本血缘，不是用另一个
   解析脚本重放推测。
3. 审计器以 TRL 的 prompt-major 顺序每 `num_generations` 条切成一组，验证组内 `task_id`
   一致后计算 `group_reward_mean/std`。若一组混入不同任务会 fail-fast，避免把跨 prompt 的
   奖励差当成 GRPO 组内方差。

每条 JSONL 记录含 `task_id/group_index/generation_index`、`raw_assistant_turns`、
`trajectory`、`reward_breakdown`、`reward`、`group_reward_std` 和 `active_anomalies`。异常规则包括：
缺少或无法解析 Action、缺失 Thought、`<think>` 标签不平衡/单轮多块、单轮多个 Action、模型
自行编造 Observation、未知工具、工具执行失败、未 FINISH、截断和达到轮数上限。另外
`scripts/audit_grpo_rollouts.py` 会汇总格式异常率、零方差组比例和“组内 4 条原文完全相同”
比例，并打印低奖励异常原文。

训练入口的 `--save_rollout_traces` 默认把记录写到 `output_dir/rollout_traces.jsonl`；
`--rollout_trace_max_samples N` 只限制原文文件大小，汇总计数仍覆盖全部生成。JSONL 每批
立即 flush，并且训练异常退出时也会写入 `.summary.json`，便于排查 OOM 或半途中断。
训练入口同时显式提供 `--seed`（默认 42）并写入 manifest；对比 SFT 与 GRPO 策略时应固定
任务切分、快照、G、温度、生成长度和种子，只替换策略 adapter。

面试中可将这一阶段概括为：**奖励曲线是聚合现象，不是错误根因。我在奖励函数的计分边界
持久化模型原始输出、环境执行轨迹与分量奖励，用同 prompt 组号回溯零方差和低奖励原因，并将
模型自主生成与环境 Observation 严格分离，避免审计本身产生误判。**

### 24.1 原始轨迹发现 setup 缺失与奖励倒置

首次原始审计共采集 32 条 / 8 组 rollout：格式异常率为 0，组内原文完全重复率为 0，
但 16 条出现工具执行失败。其中 reference/setup 任务占绝大多数：
`ref_stress_image_restoration` 7/8、`ref_stress_state_across` 4/4、
`ref_ctrl_word_handover` 3/4。轨迹显示它们都在直接调用论文工具时收到
“未找到论文；请先搜索”。

根因不是模型忘了搜索。`TaskSpec.setup` 定义的是任务开始前已存在的会话状态：
BenchmarkRunner 会先静默执行 setup 搜索，不把它计入模型工具链，然后要求模型直接使用
标题子串下载/查缓存。GRPO 旧 rollout 只调用了空环境 `reset()`，没有执行 setup；
但评分端仍按“setup 已存在”的标准答案评分。因此形成矛盾：

- 模型按标准工具链直接下载：环境必然失败，但 tool/argument 分量较高；
- 模型为了让环境可用而额外搜索：工具执行可以成功，但因多调一次工具与标准链不一致而被扣分。

实测在 `ref_stress_image_restoration` 上出现
`bad_mean=0.280 > clean_mean=-0.225`，在 `ref_ctrl_word_handover` 上出现
`bad_mean=0.350 > clean_mean=-0.225`，证明这不只是日志噪声，而是会使 GRPO 更偏好失败轨迹的
奖励倒置。

修复分两层：

1. GRPO prompt 以不暴露给模型的精确文本映射恢复 `task_id`；每条独立环境 reset 后先执行该任务
   setup，setup 动作只记在审计元数据中，不进入 policy history、工具序列、reward 或 loss。
   prompt 无法唯一映射回任务、setup 含未知工具或执行失败时全部 fail-fast。
2. `_outcome_score` 增加 `tool_exec_failures > 0 -> -1.0` 硬约束。`FINISH` 只表示模型决定停止，
   不能将已失败的环境转移包装成正的 outcome。

这个发现会使旧 v4 审计中三个 setup 任务的奖励分布失效，也意味着先前 20/60 步
GRPO checkpoint 只能作为调试证据，不能作为正式试验结论。修复后必须从冻结 SFT adapter
重跑奖励审计，再重新冻结 RL 任务切分；修复前后的 total reward 不能直接比较。

`chain_ro5_dl_three` 的未完成轨迹则是真实策略 Bad Case：标准路径为
“搜索→下载 1→下载 2→下载 3→FINISH”，模型却插入两次不必要的缓存查询，在第 5 轮
仍只下载到第 2 篇。它应由正常的 tool/process/outcome 奖励处理，不应通过单纯增加 `max_turns`
让冗余路径蒙混过关。

### 24.2 setup 修复后的 8 组验证

修复后从冻结 SFT adapter 重跑 `v3_grpo_probe.json:rl_probe`，得到 32 条 / 8 组 rollout。
总异常样本从修复前的 `17/32 = 53.125%` 降至 `3/32 = 9.375%`，格式异常仍为 0，
组内原文完全重复仍为 0。关键的因果验证是：

- `ref_ctrl_word_handover`、`ref_stress_image_restoration`、
  `ref_stress_state_across` 三类 setup 任务共 12 条 rollout，异常从修复前的
  `14/16` 降为 `0/12`；
- 不再出现“标题指代任务全部要求先搜索”，证明 setup 已进入每条独立会话环境；
- 剩余 3 条 `tool_error` 全来自 `infeasible_no_session` 2 条和
  `infeasible_unknown_id` 1 条，表示模型在应拒绝/要求用户补充信息时仍幻觉调用工具，
  这是真实策略 Bad Case，也是可用的 GRPO 二元排序信号。

但本轮只有每任务 1 组，`zero_reward_std_group_fraction=5/8=62.5%`，不能据此判定
某一任务应保留还是剔除。它只完成了环境修复验收，正式筛选必须再用冻结策略采集
每任务约 8 组（G=4 时每任务 32 条 rollout）。

## 25. setup-aware v5 GRPO：从中间带训练到采样窗口饱和

### 25.1 冻结 SFT 策略重新筛选 v5

setup 修复后，以 `outputs/sft_qlora_v1_e1/final` 为冻结策略、`G=4`、每任务 8 组重新审计
8 个候选任务。修复前 v4 的三个 reference 任务统计作废，不再用于正式选择。新的 64 组中，
去掉 `ref_ctrl_word_handover` 的 8 个恒定满分组后，剩余 7 个任务共有 56 组，其中 34 组
具有非零组内方差，`informative_group_fraction=34/56=60.7%`。因此建立
`data/splits/v5_grpo_train.json:rl_train`。所有入选任务仍只来自 v2 train；dev、IID、OOD 未参与
选题、奖励修改或超参数选择。

这里的 256 是 rollout 样本数，不是组数。一次 prompt 生成 4 个候选组成一组，所以
`256 / 4 = 64` 组；去掉一个任务的 32 条、即 8 组后，得到 224 条、56 组。

### 25.2 30 步 smoke 验收

30 optimizer step 的 smoke 使用 `batch_size=4`、`grad_accum=2`、`G=4`，共生成 240 条
rollout，即 60 个 prompt 组。训练损失为 `0.011773`，峰值显存约 9.02 GiB；格式异常为 0，
异常样本 `17/240=7.08%`，60 组中 30 组有非零奖励方差。两个 setup-dependent reference
任务均未再出现环境预置缺失。由此确认单卡 4090 的 QLoRA GRPO 加载、环境、奖励、反向传播、
adapter 保存和审计链路全部可用，v5 从 `ready_for_smoke` 升格为
`validated_for_training`。

### 25.3 正式请求120步，实际在第47步提前停止

正式运行仍从原始 SFT adapter 干净启动，而不是从 smoke 或旧 GRPO checkpoint 继续。请求
`max_steps=120`，实际在 global step 47 触发奖励方差保护：连续 5 个日志窗口的组内奖励方差
为 0、reward 为 1.0，Trainer 正常提前停止并保存最终 adapter。实际统计为：

- runtime `515.10s`，train loss `0.0002868`；
- 376 条 rollout、94 个 prompt 组；
- 累计 30 个 informative group，比例 `30/94=31.9%`；
- 15 条异常，全部为工具执行错误，异常率 `3.99%`；
- 无格式错误、未完成或达到轮数上限；
- 峰值显存约 8.75 GiB，说明停止与显存无关。

不可行任务的表现明显改善，但仍保留学习空间：`infeasible_no_session` 平均奖励 0.766、有效组
比例 57.1%，`infeasible_unknown_id` 平均奖励 0.854、有效组比例 35.7%。
`infeasible_zero_index` 平均奖励 0.971、有效组仅 7.1%，已经接近 ceiling。reference 两个压力
任务没有工具异常，说明 setup 修复在正式训练中保持稳定。

### 25.4 为什么不能把“连续5步满分”直接写成“7个任务全部收敛”

保护器观察的是最近连续日志窗口，不是对全部 7 个任务做一次穷举评测。随机 sampler 可能连续
抽到已经学会的容易任务；而累计94组仍有30组非零方差，逐任务统计也显示
`infeasible_no_session` 等任务并未全部成为 ceiling。因此更严谨的结论是：**第47步时当前采样
窗口饱和，继续按相同分布采样的边际收益下降**。是否真正学到并泛化，必须用冻结解码配置比较
SFT 与 GRPO 在同一 `rl_train` 和 dev 上的严格成功率，而不能用训练 reward 自证。

为避免实验记录歧义，训练 manifest 后续新增 `requested_max_steps`、`actual_steps`、
`stopped_early`、`stop_reason` 和 reward variance guard 状态；保护器提示也由“第47步起连续5步”
改成“截至第47步已连续5次”，避免把触发步误认为连续区间起点。正式长跑还新增周期
checkpoint 与恢复参数；默认关闭，因而不改变历史实验。

下一步先固定快照、seed、repeat、生成参数，用 SFT 与本次 GRPO final 分别跑训练集诊断和 dev。
只有当训练集改善且 dev 不退化时，才把该 checkpoint 视为有效 RL 产物。若还要继续 RL，应以
当前 checkpoint 冻结重扫 v2 train，重新选择新的 middle band；不能对已经逐渐变成 ceiling 的
旧7任务使用 `--allow_zero_variance` 强行堆到120步。

### 25.5 同协议 SFT vs GRPO：有效，但提升集中在少数能力

使用相同离线快照、regex agent、`repeat=3`、评测 seed 45，对冻结 SFT 与第47步 GRPO final
分别评测 v5 `rl_train` 和 v2 dev。训练集诊断共21条：严格成功率从 `1/21=4.76%` 提升到
`7/21=33.33%`，tool accuracy 从42.86%提升到71.43%，argument accuracy 从23.41%提升到
51.98%，平均工具失败从0.6降到0.4，平均 token 从3918.9降到3487.5。completion rate 保持
100%，false-finish rate 仍为14.29%，因此不是依靠更多假完成换取成功率。

6个新增严格成功全部来自两个任务：

- `infeasible_no_session` 从3次都错误调用翻译工具，变成3次都明确判断无法完成并直接 FINISH；
- `infeasible_zero_index` 从3次错误下载默认论文，变成3次都拒绝非法第0篇。

这两类轨迹的 Thought 都是“该任务无法通过现有工具完成或参数无效”，不是空输出或无理由
终止。因此没有观察到用裸 FINISH 欺骗 outcome 的 reward hacking。另一方面，
`infeasible_unknown_id` 仍会把 `2999.99999` 当作 ref 调下载；`chain_ro5_dl_three` 仍少下载一篇
并假完成；两个 reference stress 任务仍分别使用错误的 `query` 参数和空 `ref`。这说明训练 reward
的高均值没有自动转化为贪心解码下的全面严格成功，必须保留独立 Benchmark。

dev 共24条：严格成功率从 `10/24=41.67%` 到 `12/24=50%`，tool accuracy 保持87.5%，
argument accuracy 从59.72%小幅升至62.5%，工具失败从0.3降至0.2，没有观察到 dev 退化。
新增的2个成功都来自未进入 RL 的 `search_kw_agentic_rl`：SFT 三次中仅一次补全 `days=30`，
GRPO 三次全部补全。这是参数遵循能力的正迁移证据，但只涉及一个 dev 任务，不能表述成广泛
泛化或统计显著提升。

最终判断为：**v5 GRPO 是有效但较窄的策略改进**。它证明了可验证奖励能纠正部分“不可行请求
仍强行调用工具”的 SFT Bad Case，并在一个未训练关键词搜索任务上出现正迁移；同时没有解决
长链完整性、未知 ID、标题子串 ref 等剩余问题。应冻结该 checkpoint 作为 v5 结果，不再在已
饱和的7任务上堆步数。若继续 RL，应以当前策略冻结重扫 v2 train，选择新的 middle band；
dev、IID、OOD继续排除在选题和调参之外。

### 25.6 全量 train 重扫暴露“同文本、不同隐藏状态”的任务身份冲突

用 v5 checkpoint 冻结重扫 v2 的36个 train 任务时，启动前 fail-fast：

```text
两个不同任务生成了相同 prompt，GRPO 无法安全映射 setup:
infeasible_no_session, ref_ctrl_null_translate
```

这两个任务的用户文本都是“把刚才那篇论文翻译一下”，但语义标准刻意相反：

- `infeasible_no_session` 没有 setup，会话为空，正确行为是不调用工具并说明无法指代；
- `ref_ctrl_null_translate` 预先通过 setup 写入最近论文，正确行为是调用翻译工具并使用会话中的
  last-active paper。

因此它们是用于验证状态敏感性的正负对照组，不能删除任一任务，也不能通过在用户文字中加入
task id 来作弊。旧实现只用可见 prompt 文本反查 task id，在8任务候选中没有同时遇到这对文本，
扩大到36个 train 后才暴露歧义。

修复是在 GRPO 的结构化 prompt message 中加入内部 `_task_id` 元数据。custom rollout 首先读取
该字段选择正确的 per-rollout setup，随后在调用 tokenizer/chat template 前显式删除它。因此：

1. 环境能够区分相同文本对应的空会话和有状态会话；
2. 模型收到的 token 与 Benchmark 可见 prompt 保持一致，看不到 task id；
3. 文本映射只保留为“可见 prompt 唯一”任务的兼容 fallback；重复 prompt 不再被压成错误的一对一
   字典；
4. `Dataset.from_list` 后立即校验嵌套 `_task_id` 是否完整保留，版本不兼容时在生成前 fail-fast；
5. 回归测试同时验证相同 content 保留不同隐藏 id、各自 setup 正确执行，以及 tokenizer 从未收到
   `_task_id`。

这个问题体现了 Agent 任务中“观察文本相同但隐状态不同”的部分可观测性。训练样本身份不能只由
自然语言字符串决定；环境元数据和模型可见 token 必须分层传递，否则会把状态条件相反的任务错误
合并，造成 setup 与奖励标签冲突。
