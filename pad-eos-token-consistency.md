# PAD/EOS 与训推 token 一致性问题：复现证据、影响和修复

记录日期：2026-10-22。本文整理已经确认的两个实现问题，区分原始日志观测、代码复现和尚未完成的因果验证。第二个问题包括图像预处理不一致，以及异常回退造成的文本历史
token 重编码。

| 问题                              | 已确认的行为                                                                                               | 修复原则                                                                                           |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 多轮 concat 改写 PAD              | 实际采样的是 PAD，后续导出的训练序列将它换成 EOS，却继承 PAD 的 behavior logprob 和 `loss_mask=1`          | 原样保留已采样 token；额外补入的 EOS 单独作为 prompt，`loss_mask=0`；继承监督前校验完整 token 前缀 |
| 多模态训推 token/图像预处理不一致 | v51 同一请求有 2820 / 2660 个图像 token；fine64 另有图像 token 数相同、实际文本前缀却少 11 个 token 的请求 | 对齐像素上限、解码、resize 和处理设备；保留原始文本 token IDs，验证图像 grid 和 `pixel_values`     |

两个问题可以独立出现。**v51 已确认触发 PAD→EOS 问题**；不能因为该样本的 native prompt 长度与 dump 不等，就将它排除。两者相差的 160 个
token，恰好由第二个问题解释。

可以先核对以下三个具体位置；完整证据、复现命令与修复说明见后文：

| 问题                            | 实验                            | 轨迹                                 | 发生位置           | 直接观测                                              |
| ------------------------------- | ------------------------------- | ------------------------------------ | ------------------ | ----------------------------------------------------- |
| PAD→EOS，且图像 token 不同      | `1015-1618-…-whitebox64-min-v7` | `rollout/51/1053.jsonl:10`，sample 9 | 第 41/97 次生成    | native 采样 PAD；dump 对应 `gen` 位置变成 EOS         |
| PAD→EOS，图像 token 相同        | 同上                            | `rollout/61/1264.jsonl:4`，sample 3  | 第 17/58 次生成    | 输入/输出长度相同，仍改写 PAD                         |
| 文本历史重编码，图像 token 相同 | `1021-0123-…-fine64-nostd-v2`   | `rollout/0/22.jsonl:2`，sample 1     | 第 11 次生成的输入 | dump 前缀 112703，native 112692，恰等于重编码后的长度 |

## 1. 实验与编号约定

原始问题实验：

```text
1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7

日志根目录：
/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7

不可变代码快照：
/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/source/code
```

修复部署实验：

```text
1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1

日志根目录：
/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1

不可变代码快照：
/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1/source/code

发布记录中的 source_revision：
94ff6cb61e6186c80ed91a32d022b161e016a6cfbf00c6e3b2f1073ebe656fa9
```

补充复现文本历史重编码的实验：

```text
1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2

日志根目录：
/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2
```

**修复版本约定**：第 3.4、4 节记录的是 `imgpadfix-v1` 不可变快照中实际部署并验证的修复。2026-10-22 整理时，当前仓库 HEAD 为
`5fc73e476`，图像修复已采用另一种实现：在 AReaL 侧准备共同图像表示，保留 SGLang 的像素策略。两种方案见第 3.6 节，不能把快照中的 SGLang
补丁当作当前工作区仍存在的文件。

本文的 `v51`、`v61` 指 `rollout/51/`、`rollout/61/` 的版本目录，不能直接等同于第 51、61 次 optimizer
update。JSONL 行号和“第几次生成”从 1 开始；`sample_idx`、token 位置和 segment 下标从 0
开始。多轮生成次数指该轨迹内的模型调用次数。

## 2. 问题一：PAD 被换成 EOS，behavior logprob 沿用 PAD

### 2.1 触发路径

当前模型的特殊 token 为：

```text
PAD = 248044 = <|endoftext|>
EOS = 248046 = <|im_end|>
```

这里的 PAD 是模型实际采样出的停止 token，不是 batch 对齐时补入的 padding。

旧快照中存在以下调用链：

1. `areal/api/io_struct.py:108` 的
   `ModelResponse.output_tokens_without_stop`：正常停止时，去掉输出末尾连续的 EOS/PAD。
1. `areal/experimental/openai/client.py:711` 的
   `_concat_prompt_token_ids_with_parent`：下一轮拼接使用
   `parent.input_tokens + parent.output_tokens_without_stop`，再补一个 EOS。
1. `areal/experimental/openai/types.py:183` 的 `to_tensor_dict`：按父序列长度继承原来的
   `logprobs`、`loss_mask`、`versions`、`turn_ids`，没有检查子 prompt 的 token 前缀是否与父序列一致。

旧逻辑的关键部分相当于：

```python
parent_tokens = parent.input_tokens + parent.output_tokens_without_stop
parent_tokens += [EOS]

# 后续按原父序列长度继承监督，未核对 token 身份。
logprobs = parent_logprobs + child_prompt_zeros + child_output_logprobs
loss_mask = parent_loss_mask + child_prompt_zeros + child_output_ones
```

例如父轮采样 `[A, PAD]`，对应 logprob 为 `[lp_A, lp_PAD]`、mask 为 `[1, 1]`。下一轮 prompt 中变成
`[A, EOS]`，EOS 位置却仍然使用 `lp_PAD` 和 `mask=1`。父轮被合并进最终 concat 轨迹后，这个错误位置会进入导出的监督序列。

若原输出以 `[EOS, PAD]` 等多个停止 token 结尾，旧逻辑还会压缩长度，使本来属于停止 token 的监督落到下一段 context
上。最小复现覆盖了这种情况。

### 2.2 真实复现 A：v51，第 41/97 次生成

| 定位项                             | 值                                                                                                                                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rollout 文件                       | [rollout/51/1053.jsonl:10](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/rollout/51/1053.jsonl:10)                             |
| `sample_idx` / session             | `9` / `1053:9-0`                                                                                                                                                                                     |
| Arena task                         | `online-2610170301-yq4dvs:sxj-whitebox-ready-b-20261012-cn-orig-l080-9000s:0`                                                                                                                        |
| 模型生成                           | 第 **41/97 次**；`gen_turn_index=40`；`segment_index=81`                                                                                                                                             |
| 原生日志                           | [native/sglang-slurmd-55-1.log.20:78928](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/native/sglang-slurmd-55-1.log.20:78928) |
| Native RID                         | `b858095395464b709045771af786852f`                                                                                                                                                                   |
| 原生完成时间                       | `2026-10-17 03:15:51.612`，北京时间                                                                                                                                                                  |
| 实际停止 token / logprob           | `248044` / `-8.956513404846191`                                                                                                                                                                      |
| Dump 中对应停止 token              | `248046`，位于 `gen` 段，对应 `loss_mask=1`                                                                                                                                                          |
| Dump 中的位置                      | `95208`，从 0 开始                                                                                                                                                                                   |
| 该轮输出长度                       | native 和 dump 均为 `1594`                                                                                                                                                                           |
| 原始 reward / dump 中归一化 reward | `0.8529000282` / `-1.3723421097`                                                                                                                                                                     |

原生输出与 dump 的末尾分别为：

```text
Native: ... hți = n<|endoftext|>
Dump:   ... hți = n<|im_end|>
```

证据链：

- Native `output_token_logprobs` 直接记录最后一个 token 为 PAD，以及该 PAD 的 logprob。
- 对 native 完整输出仅做旧逻辑规定的停止后缀替换后，解码文本与 dump 的整个生成片段匹配，SHA256 为
  `fe02647bcbc7cd426c953958c3f90f699e82f37c235201bd056ed25771043be0`。
- 独立 Arena 请求回执确认第 41 次调用的输入/输出为 `93615/1594`，时间与 native RID 对应；并非只根据相似文本定位。
- native 输入长度为 `93455`，比框架/dump 的 `93615` 少 `160`，由本轨迹图像 token 的 `2820−2660` 完整解释，见第 3
  节。
- 父序列长度为 `95209`，下一轮 prompt 长度为 `95421`，满足旧 `to_tensor_dict` 继承父监督的长度条件。

详细证据：[native_pad_v51_evidence.json](analysis/outputs/reward-audit-1015-v7/native_pad_v51_evidence.json)。

### 2.3 真实复现 B：v61，第 17/58 次生成，图像 token 数一致

| 定位项                             | 值                                                                                                                                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rollout 文件                       | [rollout/61/1264.jsonl:4](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/rollout/61/1264.jsonl:4)                               |
| `sample_idx` / session             | `3` / `1264:3-0`                                                                                                                                                                                     |
| Arena task                         | `online-2610170934-ewvu2z:sxj-whitebox-ready-b-20261012-cn-orig-l019-9000s:0`                                                                                                                        |
| 模型生成                           | 第 **17/58 次**                                                                                                                                                                                      |
| 原生日志                           | [native/sglang-slurmd-50-0.log.13:90261](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/native/sglang-slurmd-50-0.log.13:90261) |
| Native RID                         | `447de2fb43074246acfaddadf71b36a9`                                                                                                                                                                   |
| 原生完成时间                       | `2026-10-17 09:42:05.864`，北京时间                                                                                                                                                                  |
| 输入长度 / 输出长度                | native 与 dump 均为 `53072/6474`                                                                                                                                                                     |
| 实际停止 token / logprob           | `248044` / `-14.28192138671875`                                                                                                                                                                      |
| Dump 中 token / mask / 位置        | `248046` / `1` / `59545`                                                                                                                                                                             |
| 原始 reward / dump 中归一化 reward | `0.9670000076` / `+0.3354902565`                                                                                                                                                                     |

该轮的图像 token 数两侧均为 `[770, 744, 770, 770]`，没有上述图像 token 数差异，仍然发生了 PAD→EOS
改写。替换停止后缀后的完整生成片段匹配 SHA256 为
`2a904420c4cb0d33b1fe310b2e6c3c1f83c46f890d7787443e0633b7165f2511`。

PAD 出现在尚未结束的工具参数中；下一轮收到工具参数错误后继续生成。轨迹最终有较高的正 reward，因此此问题也可能污染成功轨迹，不能仅在失败样本中查找。

详细证据：[native_pad_positive_evidence.json](analysis/outputs/reward-audit-1015-v7/native_pad_positive_evidence.json)。

**证据边界**：dump 保存解码后的生成片段、长度和 reward，没有保存原始 logprob 数组。`gen` 标签由 dump 导出代码按 `loss_mask=1`
生成，见 [workflow_executor.py](areal/infra/workflow_executor.py)。PAD logprob 被继承到 EOS
位置这一点，由实验快照的代码路径和真实 tensor 导出最小复现共同确认。我们没有从 dump 直接观测到该位置的最终 advantage、重算 EOS 概率、过滤结果或实际
optimizer batch 归属。

### 2.4 覆盖范围与最小复现

审计匹配到 **12 次非末轮的 PAD→EOS 改写**：其中 3 次同时有图像 token 数差异，9 次没有。已有 native 日志最早独立确认到
v51，但不代表实验第一次触发必定在 v51。

Native 轮转日志仅覆盖北京时间 `2026-10-17 00:57` 至 `2026-10-18 04:24`；不能把 12 次当作整个训练的精确发生次数。统计见
[native_pad_summary.json](analysis/outputs/reward-audit-1015-v7/native_pad_summary.json)。全部
12 条轨迹列于附录 A。

最小复现脚本：[concat_pad_eos_repro.py](analysis/outputs/reward-audit-1015-v7/concat_pad_eos_repro.py)。它使用真实的
concat、父子 cache 关联、停止 token 去除和 tensor 导出实现，仅 tokenizer/template 输入为构造数据。

从仓库根目录，在有 PyTorch 且可读旧快照的 Python 环境中执行：

```bash
OLD_SOURCE=/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1015-1618-8n-256k-bs16-gs12-token-mean-whitebox64-min-v7/source/code
python analysis/outputs/reward-audit-1015-v7/concat_pad_eos_repro.py \
  --extract-runtime --source-root "$OLD_SOURCE" \
  --output /tmp/areal-pad-eos-old-snapshot.json
```

这里必须使用旧实验快照；该脚本断言的是旧 bug，不能拿修复后的工作区替代。`--extract-runtime`
从源文件提取相关定义并执行，避免加载无关依赖，不改动实验代码。

单 PAD 用例的关键输出：

```text
parent_actual_tokens:    [10, 11, 248044]
child_training_tokens:  [10, 11, 248046, 20, 21, 30, 248046]
position=2:
  actual_sampled_action = 248044
  training_action       = 248046
  inherited_logprob     = -3.0
  loss_mask             = 1
```

快照复现结果保存在
[concat_pad_eos_repro_snapshot.json](analysis/outputs/reward-audit-1015-v7/concat_pad_eos_repro_snapshot.json)。2026-10-21
记录后，2026-10-22 又对旧快照运行了一次，四个用例均通过旧行为断言；本次结果见
[PAD 复查结果](analysis/outputs/reward-audit-1015-v7/concat-pad-doc-recheck-1022.json)。

### 2.5 为什么是训练正确性问题

PPO/importance sampling 的分子与分母必须对应同一个 action。改写后计算的却相当于：

```text
exp(log π_train(EOS | h_train) − log π_behavior(PAD | h_rollout))
```

即使暂不考虑图像预处理差异，这也不是 EOS 或 PAD 的正确 importance ratio。它还改变了后续 token 的历史上下文；多个停止 token
被压缩时，则进一步造成监督位置错配。

影响可能包括错误的 EOS 梯度、importance ratio 异常以及后续序列的错误监督。梯度方向仍取决于 advantage、clipping
和过滤等因素，不能概括成每次触发都会强化提前 EOS。v61 的正归一化 reward 说明存在正向信用被错误分配的可能，不等于已测得该 token 的正 advantage。

### 2.6 具体修复

修复代码：[client.py](areal/experimental/openai/client.py) 的
`_concat_prompt_token_ids_with_parent`、`_prepare_prompt`，以及
[types.py](areal/experimental/openai/types.py) 的 `to_tensor_dict`。

1. 完整保留已采样 token。父前缀改为 `input_tokens + output_tokens`，保留 PAD 和所有重复停止 token，不再用
   `output_tokens_without_stop` 构造训练前缀。
1. 需要关闭轮次时追加 EOS，不替换原 token。若父输出为空或末尾不是 EOS，再追加一个合成 EOS。PAD 结尾因此成为 `[..., PAD, EOS]`。PAD
   保留原 logprob 和 mask；新增 EOS 属于 child prompt，其
   `logprob=0`、`loss_mask=0`、`version=-1`、`turn_id=-1`、token reward 为 0。
1. 按规范消息模板确定拼接边界。通过父消息模板的 EOS 边界定位新 context，不再数原始采样前缀中的 EOS，避免保留重复停止 token
   后误截下一条消息。父、子模板使用相同 tools 和 `chat_template_kwargs`，tools iterator 先转成 list。
1. 继承监督前校验完整父前缀。要求 child `input_tokens[:parent_len]` 与父导出的 `input_ids` 逐 token
   一致；替换、截短等情况直接报错。等长合法前缀使用 `>=` 路径正常继承。
1. 其他字段同步对齐。保留父 `versions`、`turn_ids`、`token_rewards`；多模态 token type IDs 对父输出和新增 EOS 补
   0，图像 tensor 保持关联。

修复后的原则是：**已采样 token 不改身份，未采样的模板 token 不参与 policy loss。**

当前仓库对应提交为
`9d7b5b607`（`fix(experimental): preserve sampled prefixes in concat prompts`）；其修改保持上述
token 保真与监督前缀校验原则。

## 3. 问题二：图像预处理与 token 序列在训推两侧不一致

### 3.1 在哪个轨迹上复现

使用第 2.2 节同一实验、同一轨迹和同一 native 请求：

```text
rollout/51/1053.jsonl，第 10 行，sample_idx=9，session=1053:9-0
第 41/97 次生成
RID=b858095395464b709045771af786852f
```

该请求包含三张原始输入图片。来自 Arena 原始线程的映射记录见
[vision_v51_original_input_images.json](analysis/outputs/reward-audit-1015-v7/vision_v51_original_input_images.json)，对应线程
item 为 52、56、60。复现使用这些原图，而不是任务后续生成的 render。

以下 grid 均为 `[T, H_grid, W_grid]`，图像尺寸按“宽×高”列出。模型 `patch_size=16`、`merge_size=2`，因此图像
token 数为 `T × H_grid × W_grid / 4`。

| 图片 | 原始尺寸   | 框架/HF grid | HF token 数 | 原生 SGLang grid | SGLang token 数 |
| ---- | ---------- | ------------ | ----------- | ---------------- | --------------- |
| 1    | `1200×720` | `[1,44,76]`  | 836         | `[1,44,76]`      | 836             |
| 2    | `900×1300` | `[1,82,56]`  | **1148**    | `[1,76,52]`      | **988**         |
| 3    | `1200×720` | `[1,44,76]`  | 836         | `[1,44,76]`      | 836             |
| 合计 |            |              | **2820**    |                  | **2660**        |

第二张原图 SHA256：

```text
c041091bbbac8f3251bceea9ab8670ebbfda76d299b59daea6eb0232315bd78c
```

本地原图：[c041…78c.png](analysis/outputs/reward-audit-1015-v7/vision_inputs/c041091bbbac8f3251bceea9ab8670ebbfda76d299b59daea6eb0232315bd78c.png)。

对该原图，HF 分支 resize 到 `896×1312`，旧 SGLang 分支 resize 到 `832×1216`，分别得到 1148 和 988 个图像
token。160 的差异来自实际图像分辨率与 grid，不是日志计数方式不同。请求长度也满足：

```text
93615 − 93455 = 2820 − 2660 = 160
```

原始长度与 token 数证据见
[native_pad_v51_evidence.json](analysis/outputs/reward-audit-1015-v7/native_pad_v51_evidence.json)；原图尺寸计算复现见
[vision_actual_image_resize.json](analysis/outputs/reward-audit-1015-v7/vision_actual_image_resize.json)。

### 3.2 根因：像素上限、额外 resize 和处理设备不同

实际 checkpoint 为 `/storage/openpsi/models/Qwen__Qwen3.8-27B`，其 `config.json` 的
`model_type` 为 **`qwen3_5`**。修复按实际 model type 分发，不能只根据模型目录中的 “Qwen3.8” 命名判断。

模型 HF `preprocessor_config.json` 中配置：

```text
size.longest_edge = 16777216
size.shortest_edge = 65536
patch_size = 16
merge_size = 2
```

这里 `longest_edge` 在该处理器中承担的是像素面积上限，不能按字面理解为允许 16777216 像素的单边长度。

实验对应的 SGLang 为 `0.5.18.dev10-g85b539146`。其 `qwen_vl.py` 在默认开启的
`SGLANG_VLM_QWEN_LIMIT_PIXEL` 分支中，对 `qwen3_5` / `qwen3_5_moe` 将该上限除以 16：

```text
16777216 / 16 = 1048576 pixels
1048576 / (16 × 2)^2 = 1024 visual tokens
```

框架侧用未修改上限的 HF processor 生成训练 prompt、`image_grid_thw` 和 `pixel_values`；native SGLang
重新处理原图时使用了缩小的上限。训练下游直接消费框架保存的图像 tensor，没有再 resize 成 SGLang 的低分辨率版本。

此外，旧 SGLang 在 HF processor 前还执行一次独立的 smart resize，并可能走 Tensor/GPU 图像处理路径。即使两个请求的图像 token
数相同，也不能据此证明 `pixel_values` 相同：resize 次数、插值实现和设备都需要对齐。

还有一个直接相关的 token 保真风险：旧路径中 Tensor 图像缺少 `.height` 属性，会使 `resolve_image_token_counts`
失败，原生代码随后回退到 `decode+retokenize`。这样文本相同也不保证 token IDs 保持不变，尤其是拼入历史采样 token
时。修复验证同时要求该回退不发生、非图像 token IDs 原样保留；不能只比较图像 token 总数。

### 3.3 用原图复现像素上限差异

脚本：[reproduce_vision_resize_v51.py](analysis/outputs/reward-audit-1015-v7/reproduce_vision_resize_v51.py)。从仓库根目录执行：

```bash
SGLANG_SOURCE=/storage/openpsi/users/public/codes/sglang-python-0.5.18.dev10-g85b539146
MODEL_PATH=/storage/openpsi/models/Qwen__Qwen3.8-27B
IMAGE_PATH=analysis/outputs/reward-audit-1015-v7/vision_inputs/c041091bbbac8f3251bceea9ab8670ebbfda76d299b59daea6eb0232315bd78c.png

python analysis/outputs/reward-audit-1015-v7/reproduce_vision_resize_v51.py \
  "$SGLANG_SOURCE" "$MODEL_PATH/preprocessor_config.json" \
  /tmp/areal-vision-v51-resize.json --image "$IMAGE_PATH"
```

脚本读取原图、校验源码分支，执行该版本 HF/SGLang 的实际尺寸计算函数，得到 `1148 vs 988`；2026-10-22 再次运行通过，见
[原图尺寸复查结果](analysis/outputs/reward-audit-1015-v7/vision-resize-doc-recheck-1022.json)。该脚本本身不执行完整
processor 或 vision encoder。历史 JSON 中“当时无法运行容器”的说明仅描述早期尺寸复现阶段；后续已补做真实容器中的完整 processor
对比，见第 4 节。

### 3.4 已部署到 imgpadfix-v1 的具体修复

补丁入口位于修复实验快照：[sglang_qwen35_vl.py](/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1/source/code/areal/engine/sglang_qwen35_vl.py)，由
[sglang_model_patches.py](/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1/source/code/areal/engine/sglang_model_patches.py)
在 SGLang 启动前根据 `model_type` 安装。

1. 对 `qwen3_5` / `qwen3_5_moe` 取消额外的 1M 像素限制，保留模型配置的上限；其他 model type 的像素策略保持各自分支。
1. 对这两类模型跳过 HF 前额外的 smart resize，将原图交给 HF processor 处理一次。
1. 在该安装路径中设置 `gpu_image_decode=False`，让 PNG/JPEG 都经原始 PIL/CPU 解码路径进入处理器。
1. 修改实际 `base_processor.py` 的设备选择分支，让这两类模型的 HF fast processor 在 CPU 上运行，与框架侧一致。该固定
   SGLang 版本没有 `_fast_image_processor_device` hook，仅添加同名方法不会生效。
1. 使用现有的保留原始 input IDs 的图像占位符展开路径，并验证展开结果等于框架 token 序列，不经过失败后的文本重编码回退。
1. 补丁对原始源码和补丁结果进行 SHA256 校验，两个文件先全部验证再写入，带文件锁、单文件原子替换和重复安装幂等性。未知源码拒绝修改；不是不受版本约束的字符串替换。

经过验证的原始/补丁文件摘要：

| 文件                | 原始 SHA256                                                        | 修复后 SHA256                                                      |
| ------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `qwen_vl.py`        | `285fd0e0eef80d8c54db71724ef7cc369ae9379023437d3e2015deb65df69e17` | `04b9c53b6153c4c50e93997b298e5e90f8c27b7eadc917a216c84dad029cd1ec` |
| `base_processor.py` | `2396ed6de130e17048eac64e97a2c665b7a33e8cdb42427b7ba5dd1e8be1ef61` | `eca3fb0ae1022ad3d2daeee2eb9710967d560e245b305322687397219bfbddf4` |

### 3.5 范围与训练影响

历史 dump 共 18238 条轨迹，其中 **3845 条，约 21.1%** 出现超过旧 native 上限的图像；约 18.8% 的生成 token
位于这类上下文下。已有 native 完成日志中的 1543332 个图像实例，最大为 1024 token，没有超过 1024
的实例。相应统计范围不同，不能将两组数量直接相除。详见
[vision_followup_summary.json](analysis/outputs/reward-audit-1015-v7/vision_followup_summary.json)。

这会让 behavior 侧使用的图像 grid/像素与训练侧不一致，改变视觉上下文和相应位置编码，并可能影响策略更新的方差、过滤率以及训练与部署表现。

它与 PAD 问题的数学性质不同：**不同策略采用不同预处理，并不自动使 importance sampling 无效**，前提仍是对同一个已采样 action
正确计算各自概率。PAD 问题直接破坏 action 身份；图像问题首先是预处理与上下文不一致，另有 decode/retokenize 带来的 token
序列风险。不能把两者混为一种错误。

### 3.6 当前仓库的修复实现：共同图像表示，保留 SGLang 像素策略

提交 `5fc73e476`（`fix: align Qwen-VL image inputs with SGLang`）改为在 AReaL 输入端处理，不修改 SGLang
源文件：

1. [prepare_sglang_qwen_images](areal/utils/qwen_vl_image.py:42) 按 SGLang 0.5.18
   的有效像素预算和尺寸规则，将原图 resize 一次并转为 RGB。
1. 用同一张处理后图片生成训练侧 HF tensor，同时把它无损编码成 PNG 发送给 SGLang。图像尺寸已经对齐，后端额外 resize 不再改变像素；PNG 也避开
   nvJPEG 解码分支。
1. [compact_image_tokens](areal/utils/sglang_image.py:13) 在传输时将每张图的展开 token
   连续段压成一个占位符，保留所有文本 token；训练仍保留完整展开序列。legacy/v2 适配器和 resubmission 使用相同约定。
1. Chat Completions、Responses 和多轮缓存/concat 同时保留共同的 wire image 与对应训练 tensor。

默认 `SGLANG_VLM_QWEN_LIMIT_PIXEL=true` 时，该方案让训练跟随旧 SGLang 的约 1M 像素预算；第 3.4 节方案则让 SGLang
跟随模型/HF 的 16M 配置。因此，同一张 v51 竖图在两种一致方案中可以分别使用 988 或 1148 个视觉
token。验收标准是各自方案内训推一致，不能要求两种方案都输出 2820。

该实现针对 OpenAI/Arena 入口、SGLang 后端和 `Qwen3VLProcessor`。proxy/server 需要相同的
`SGLANG_VLM_QWEN_LIMIT_PIXEL` 设置，并使用未被原地改写的 HF processor、无服务端 image-size
override。实现范围与验证记录见 [当前图像修复说明](docs/plans/2026-10-21-qwen-vlm-image-alignment.md)。第 4 节的
92 项测试与线上证据属于历史部署快照，不应移用为本提交的上线证明。

### 3.7 补充实证：fine64 的图像 token 相同，文本历史却被重编码

**具体轨迹**：`1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2`，
[rollout/0/22.jsonl:2](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2/rollout/0/22.jsonl:2)，`sample_idx=1`，session
`22:1-0`，第 **11 次生成**的输入；Arena task 为
`online-2610210136-lqqpmj:fine-rl64-clean-20261018-m208:0`。

| 观测                                           | 值                                                                                                                                                                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 匹配的原生 Finish                              | [sglang-slurmd-56-0.log.20:6185](/storage/openpsi/experiments/logs/admin/sxj-qwen38-game-rl/1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2/native/sglang-slurmd-56-0.log.20:6185)，`2026-10-21 02:11:56.826` |
| Native RID                                     | `6c4c6c31ed8242d4b5f176c9e0dd3110`                                                                                                                                                                                   |
| dump 记录的输入长度                            | **112703**                                                                                                                                                                                                           |
| 拼接 dump 前缀文本，再 canonical encode 的长度 | **112692**                                                                                                                                                                                                           |
| native 实际 `prompt_tokens`                    | **112692**                                                                                                                                                                                                           |
| dump / native 视觉 token 数                    | **1404 / 1404**，图像长度校正为 0                                                                                                                                                                                    |
| dump 前缀中的 U+FFFD 替代字符                  | 0                                                                                                                                                                                                                    |

先用完整生成文本的 SHA256 将 native 输出与 dump 生成片段匹配，再比较该轮输入。实际少的 **11 个 token** 不能用图像 token
数解释，却与历史文本重编码的长度变化完全吻合。证据见
[具体轨迹匹配结果](analysis/outputs/imgpadfix-audit-1022/vision-tokenization-consumption-match.json)。原始完整前缀
IDs 未保存，因此这不是逐 ID 的原始前缀还原。

源码消费链为：

```text
HF processor 先得到 canonical input_ids
  → SGLANG_MM_AVOID_RETOKENIZE 尝试按图片尺寸展开原始输入 IDs
  → resolve_image_token_counts 遇到 Tensor.height 异常
  → 保留 HF canonical input_ids，记录 falling back to decode+retokenize
  → Qwen 最终化没有恢复原始 IDs
  → TokenizerManager 实际消费 mm_inputs.input_ids
```

这是两个不同的分支：PIL 成功时会覆盖 canonical IDs，保留原始文本 token；Tensor 异常时才沿用重编码结果。旧 v7 的 v51 原图是
PNG，对应原生日志文件中没有此回退警告；其输入 `93455 = 93615 − 160` 符合保留原始 token 的分支。因此，不能把 v51
的图像分辨率差异当作它也发生了历史重编码的证据。

在 fine64 baseline 单个服务抽取的 100 条初始 Finish 时间窗内，有 99 条 `.height` 回退警告；85 条 Finish 保留了可关联的
processor Perf，85/85 均可按同 PID 的同步处理顺序关联警告。修复实验抽取的 100 条 Finish 均有
Perf，该窗口没有回退警告。这是指定窗口的证据，不能当作全实验发生率；警告本身没有 RID，上述第 11 次生成也缺少可对应的
Perf，不能断言已将该条请求与某条警告一一配对。

另用真实源码分支进行 CPU 控制复现：构造与 v51 文本相同、但多 19 个 token 的非 canonical 前缀。PIL 分支返回且被 manager 消费的长度为
**93455**；Tensor 分支实际触发 `.height`，返回且被消费的长度为 **93436**。这验证了回退结果确实进入后续请求，而不是只有警告。19-token
前缀是构造用例，不是恢复出的 v51 原始 IDs。源码摘要、消费结果、窗口统计和解码分支条件见
[consumption 证据](analysis/outputs/imgpadfix-audit-1022/vision-tokenization-consumption-results.json)。

只读复现脚本（需要能读取原日志及源码的权限、PyTorch/Pillow/tokenizers，以及目录中已有的 fixture/probe JSON）：

```bash
sudo -n /opt/conda/bin/python analysis/outputs/imgpadfix-audit-1022/vision-tokenization-consumption-probe.py
sudo -n /opt/conda/bin/python analysis/outputs/imgpadfix-audit-1022/vision-tokenization-consumption-match.py
```

以上命令使用本次验证过的 Python 环境，`sudo` 用于读取原日志；已有读取权限时可直接运行。前一个脚本执行相关 AST 源码分支，后一个脚本只扫描 baseline
的 `rollout/0` 直到找到首条长度变化匹配。它们仅向 stdout 输出 JSON，不启动模型或 GPU
服务。这两项是原日志仍保留时的复现命令，后续轮转限制见下一段。

**03:26 后补充**：通过该 m208 任务的真实 `full.jsonl` artifact 已恢复前两张传输图片，均为 JPEG。重建的 data URL
SHA256 与前 11 次模型调用回执的 `sourceSha256` 逐项相等，字节数、MIME、尺寸和顺序相等，且均未再压缩/省略/旋转。图像 grid
`[1,36,60]`、`[1,48,72]` 与 native 记录相同。因此，MIME 现已补证；`.height` 警告本身没有 RID
的限制仍然成立。证据：[图片恢复](analysis/outputs/imgpadfix-audit-1022/m208-original-image-recovery.json)、[传输身份与历史验证](analysis/outputs/imgpadfix-audit-1022/transport-recovery-validation.json)。注意这里
`sourceSha256` 是 data URL 字符串的摘要，与图片 bytes 的 SHA256 分开记录。

同时发现原生日志已轮转：本次复核时同名 `.log.20` 的最早保留时间为 2026-10-21 02:59:51，晚于目标
02:11:56。上表文件名/行号是历史定位，当前同名文件第 6185 行已不能当作原证据。先前已白名单保存
[该次实际输出 IDs、logprob、元数据与原始行 hash](analysis/outputs/imgpadfix-audit-1022/vision-m208-gen11-native-evidence.json)；新取得的独立
Arena 第 11 次回执再次确认输入/输出长度为 `112703/28511`。完整原始输入 IDs 仍未恢复。

保存的 28511 个实际输出 IDs 可完整解码为 dump 中该生成段；canonical encode 后为 28510 个 token，实际有 **8
处分词替换**。据此定义了同文本、同实际图片的历史保留/重编码对照前缀 `141357/141356`，见
[m208 对照输入](analysis/outputs/imgpadfix-audit-1022/m208-history-replay-fixture.json)。它以
canonical 的第 11 次输入为共同基底，保留该轮真实采样动作；更早历史中缺失的 11 个 token 没有补造，因此不是恢复出的完整原请求，也尚未用于模型概率或
reward 因果测量。

### 3.8 补充控制：token 数相同，视觉 embedding 仍会变化

同一 fine64 baseline 的 `rollout/0/0.jsonl:2`、sample 1，对应任务
`online-2610210136-xjghdw:fine-rl64-clean-20261018-cn-orig-l015:0`。从真实任务 artifact 恢复的
anchor JPEG 为 `648×432`、151722 bytes，SHA256 为
`5c7227e6f42c3650916fab5610e206d68849824f926dc364a8f2d92d4aca57d8`；传输身份见
[l015 图片恢复记录](analysis/outputs/imgpadfix-audit-1022/l015-original-image-recovery.json)。

固定初始 checkpoint 的全部 333 个视觉 tensor、BF16 权重和同一份 PIL 解码结果，仅比较“旧 SGLang Tensor 额外
resize→HF”与“原图直接→HF”。两边均为 grid `[1,28,40]`、280 个视觉 token；但完整 HF CPU vision encoder 输出的相对
L2 差为 **14.61%**，RMSE 为 **0.137862**。相对 L2 以原图直接→HF 的输出范数为分母。

因此修复验收还需要比较 `pixel_values`，不能只看图像 token 数。这是单张真实输入的 CPU 控制实验，未运行历史 nvJPEG、生产 GPU
后端或语言模型；14.61% 不能换算成 EOS 概率或 reward 损失。结果与复现代码见
[视觉 encoder 控制结果](analysis/outputs/imgpadfix-audit-1022/vision-cpu-encoder-results.json)、[CPU 脚本](analysis/outputs/imgpadfix-audit-1022/vision_cpu_encoder_probe.py)。

## 4. 修复后的验证与上线证据

### 4.1 定向测试：92 项通过

发布时的 [JUnit 记录](analysis/outputs/reward-audit-1015-v7/fix-tests.xml) 显示
`tests=92`、`failures=0`、`errors=0`、`skipped=0`。

| 测试文件，相对于修复实验的不可变代码快照                  | 用例数 |
| --------------------------------------------------------- | ------ |
| `tests/experimental/openai/test_concat_sampled_tokens.py` | 12     |
| `tests/experimental/openai/test_concat_prompt.py`         | 8      |
| `tests/test_interaction_token_rewards.py`                 | 10     |
| `tests/test_sglang_qwen35_vl_patch.py`                    | 4      |
| `tests/test_sglang_model_patches.py`                      | 21     |
| `tests/test_sglang_multimodal_roundtrip.py`               | 9      |
| `tests/experimental/openai/test_vlm_trajectory.py`        | 28     |

复查历史测试应使用第 1 节的修复实验快照，不能假定当前工作区始终保留相同文件。该快照中的
[PAD 回归测试](/storage/openpsi/experiments/admin/sxj-qwen38-game-rl/1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1/source/code/tests/experimental/openai/test_concat_sampled_tokens.py)
覆盖多轮 PAD、重复停止 token、EOS、length/abort/空输出、父前缀错误、等长前缀和 tools generator，并核对
token、logprob、mask、version、turn ID、token reward 对齐。

多模态 roundtrip 测试覆盖横图、竖图、非 32 整数倍尺寸、低于最小/高于最大像素、多图，以及尾部包含历史生成 token 的情况。要求展开后 `input_ids`
一致，grid 和 `pixel_values` 使用 `rtol=0, atol=0` 比较。参考 HF processor 与 SGLang processor
分别实例化，避免共享 processor 被 SGLang 原地修改后掩盖 bug；另有测试显式在默认限像素开关开启时验证补丁和设备分支。

### 4.2 实际 actor/rollout 容器交叉验证

使用 v51 恢复出的三张原图，在真实的两类运行环境中比较：

| 环境    | PyTorch        | Transformers | torchvision    |
| ------- | -------------- | ------------ | -------------- |
| Actor   | `2.9.1+cu129`  | `5.3.0`      | `0.24.1+cu129` |
| Rollout | `2.11.0+cu130` | `5.12.1`     | `0.26.0+cu130` |

修复后结果：

```text
grid = [[1,44,76], [1,82,56], [1,44,76]]
image_tokens = [836,1148,836]，合计 2820
pixel_values.shape = [11280,1536]
prompt_ids_exact = true
cross_actor_pixels_exact = true
```

证据：[fix-actor-preprocessor.json](analysis/outputs/reward-audit-1015-v7/fix-actor-preprocessor.json)、[fix-rollout-preprocessor.json](analysis/outputs/reward-audit-1015-v7/fix-rollout-preprocessor.json)。这些是指定输入上的精确预处理比较，不代表已经证明整个
GPU vision encoder、模型 logits 或 RL 更新在两个后端数值完全相同。

### 4.3 真实服务上的生效检查

修复部署到 `1021-0227-8n-256k-bs16-gs12-fine64-imgpadfix-v1` 后，对全部 **16 个 SGLang
服务**核对了安装文件的补丁摘要，并观察到真实 Arena 生成请求。

例如 `native/sglang-slurmd-0-0.log` 中 RID：

```text
b3fc9de6355e4401855abc97388e4781
```

该请求的两张图 grid 为 `[1,24,36]` 和 `[1,66,72]`，图像 token 数为 **216 和 1188**。1188 超过旧上限
1024，确认新服务确实走到了修复后的图像处理路径。

证据：[fix-live-vision-evidence.json](analysis/outputs/reward-audit-1015-v7/fix-live-vision-evidence.json)、[fix-launch-validation.json](analysis/outputs/reward-audit-1015-v7/fix-launch-validation.json)。后者是启动验证时的记录，其中
`first_optimizer_step_observed=false` 仅表示当时尚未观察到第一步更新，不代表实验后来没有训练。

## 5. 与 reward 崩溃的关系及结论边界

这两个问题均已有实现级证据和复现，修复目标是恢复 action、监督与视觉输入的一致性。但目前的证据**不能证明它们是 reward
崩溃的唯一根因，也不能保证修复后分数立即提升**。

- PAD 问题在覆盖到的日志中确认了 12 次，缺少逐 token 的最终训练过滤和 optimizer 更新记录，不能仅据此量化它对整体 reward 的贡献。
- 图像上限不一致从 v0 就已存在，不是仅在 reward 崩溃时才出现；曝光率和失败率的相关性不能替代因果实验。
- `imgpadfix-v1` 部署的图像修复采用“native 跟随 HF/模型配置”的方案，相对旧 rollout 提高了部分图片的分辨率，并选择 CPU
  处理路径，因此也会改变实际输入分布和预处理耗时。若需要低分辨率，应在训推两侧统一设置，而非仅给 native 限制。
- 修复实验与后续对比的 `1021-0123-8n-256k-bs16-gs12-token-mean-fine64-nostd-v2` 还存在 reasoning
  prefill/parser 等源码差异，不是严格只修改这两个问题的 A/B。该对比不能单独估计本次修复的收益。

后续因果验证应固定 checkpoint、题目与采样设置，分别比较一致的低/高像素策略，并单独隔离 parser/prefill 变化；同时记录实际采样 action、训练
action、behavior logprob、mask 以及过滤结果。

## 附录 A：已确认的 12 次 PAD→EOS 改写

以下文件都位于原问题实验的 `rollout/<version>/` 下，生成轮次从 1 开始。`sample_idx` 与 session 中的编号不保证相等，按原 dump
分别列出。

| 版本 | 文件:行         | sample_idx | session     | 生成轮次/总次数 | PAD logprob | 原始 reward |
| ---- | --------------- | ---------- | ----------- | --------------- | ----------- | ----------- |
| v51  | `1053.jsonl:10` | 9          | `1053:9-0`  | 41/97           | -8.956513   | 0.852900    |
| v59  | `1199.jsonl:7`  | 6          | `1199:10-0` | 2/121           | -12.800145  | 0.000000    |
| v60  | `1238.jsonl:7`  | 6          | `1238:7-0`  | 64/107          | -11.982524  | 0.000000    |
| v61  | `1264.jsonl:4`  | 3          | `1264:3-0`  | 17/58           | -14.281921  | 0.967000    |
| v87  | `1754.jsonl:1`  | 0          | `1754:0-0`  | 15/16           | -10.693049  | 0.000000    |
| v88  | `1773.jsonl:8`  | 7          | `1773:7-0`  | 8/20            | -10.293958  | 0.000000    |
| v88  | `1759.jsonl:1`  | 0          | `1759:0-0`  | 17/20           | -10.486783  | 0.000000    |
| v89  | `1793.jsonl:12` | 11         | `1793:11-0` | 15/22           | -10.637104  | 0.000000    |
| v93  | `1853.jsonl:4`  | 3          | `1853:3-0`  | 18/23           | -10.056206  | 0.000000    |
| v93  | `1853.jsonl:5`  | 4          | `1853:4-0`  | 14/16           | -9.051925   | 0.000000    |
| v94  | `1879.jsonl:2`  | 1          | `1879:1-0`  | 8/9             | -9.284281   | 0.000000    |
| v95  | `1894.jsonl:3`  | 2          | `1894:2-0`  | 11/19           | -10.667962  | 0.000000    |

完整 native 文件、行号、RID、时间、token 位置及匹配摘要见
[native_pad_rewrites.json](analysis/outputs/reward-audit-1015-v7/native_pad_rewrites.json)；与图像
token 差异的逐条对照见
[vision_pad_pair_comparison.json](analysis/outputs/reward-audit-1015-v7/vision_pad_pair_comparison.json)。
