用一个具体的例子走一遍完整流程：

### 设定

```
total_num_envs = 64
group_size = 8
rollout_epoch = 16
micro_batch_size = 32
num_action_chunks = 8 (chunk size)
```

### 第 1 步：环境初始化

64 个环境被分成 **8 组**（`64 / group_size=8`），每组 8 个环境**共享同一个初始状态**：

```
组 0: env_0 ~ env_7    → 都从同一个初始状态开始
组 1: env_8 ~ env_15   → 都从另一个初始状态开始
...
组 7: env_56 ~ env_63  → 都从另一个初始状态开始
```

共享初始状态的目的是让 GRPO 能做**组内比较**——同一个起点、不同的动作、不同的结果，才能比出哪个动作更好。

### 第 2 步：Rollout 收集数据

一个 rollout epoch 的流程：

```
1. 64个环境各自发送当前观测给 rollout worker
2. Rollout worker 用 VLA 模型推理，生成 64 组动作（每组 8 步，即一个 chunk）
3. 64个环境各自执行 8 步动作，返回奖励
4. 得到 64 条数据，每条包含：观测、动作(56维)、奖励(8步)、logprobs 等
```

重复 `rollout_epoch=16` 次：

```
第 1 轮: 64 条数据  (环境走了第 1~8 步)
第 2 轮: 64 条数据  (环境走了第 9~16 步)
...
第 16 轮: 64 条数据 (环境走了第 121~128 步)

总共: 64 × 16 = 1024 条数据
```

注意：16 轮 rollout 是**连续的**——不是 reset 16 次，而是每个环境连续走 `16 × 8 = 128` 步，形成一段轨迹。

### 第 3 步：计算 Advantage

GRPO 对每个组（8 个环境）做组内比较：

```
组 0 的 8 个环境各自完成一段轨迹，累积奖励分别是：
  env_0: 0, env_1: 0, env_2: 5.0, env_3: 0, env_4: 0, env_5: 5.0, env_6: 0, env_7: 0

  mean = 1.25, std = 2.31
  env_2 的 advantage = (5.0 - 1.25) / 2.31 = +1.62  (强化)
  env_0 的 advantage = (0 - 1.25) / 2.31 = -0.54    (抑制)
```

完整流程是：

#### 1：预处理（`preprocess_embodied_advantages_inputs`）

原始 rewards shape 是 `[16, 64, 8]`（16 个 chunk × 64 个环境 × 8 步/chunk）。reshape 成 `[128, 64]`（128 步 × 64 个环境），变成一条平坦的时间线。

#### 2：累加成总分（`calculate_scores`）

```134:152:rlinf/algorithms/utils.py
def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])
    # ...
```

从最后一步往前累加所有奖励，遇到 done（回合结束）就清零重新算。最终 `scores` 的 shape 是 `[8, 8]`（8 组 × 每组 8 个环境），每个值是那个环境整段轨迹的**总回报**。

用你的例子：

```
环境 0 的 128 步奖励: [0,0,0,...,5.0,0,...,0]  → scores[0] = 5.0
环境 1 的 128 步奖励: [0,0,0,...,0,...,0]       → scores[1] = 0
```

#### 3：GRPO 组内比较（`compute_grpo_advantages`）

```89:121:rlinf/algorithms/advantages.py
def compute_grpo_advantages(
    rewards: torch.Tensor,    # 这里的 rewards 其实是上一步算出的 scores
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    grouped_rewards = rewards   # [8, 8]  (8组 × group_size=8)
    grouped_reward_mean = grouped_rewards.mean(dim=1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=1, keepdim=True)

    advantages = (grouped_rewards - grouped_reward_mean) / (grouped_reward_std + 1e-6)

    # 广播回 [128, 64] 的 shape，同一个环境的所有步共享同一个 advantage 值
    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask
```

#### 总结流程

```
rewards [16, 64, 8]
  │
  ├─ reshape → [128, 64]      (展平成 128 步 × 64 环境)
  │
  ├─ calculate_scores          (每个环境的 128 步奖励累加成 1 个总分)
  │   → scores [8, 8]          (8 组 × 8 个环境/组)
  │
  ├─ compute_grpo_advantages   (组内比较: 标准化)
  │   → advantages [8, 8]      (每个环境一个 advantage 值)
  │
  └─ 广播回 [128, 64]          (同一个环境的所有 128 步共享同一个 advantage)
      → 用于 policy loss 计算
```

**128 步的奖励会被累加成一个总分**，然后再做 GRPO 的组内比较。每个环境最终只有一个 advantage 值（好/差），这个值被广播到它的所有 128 步上——"你整段轨迹的表现比组内平均好，所以你这 128 步的所有动作都应该被强化"。

---

### 第 4 步：训练更新

1024 条数据送入 actor 做梯度更新，但 GPU 一次放不下 1024 条，所以按 `micro_batch_size=32` 切分：

```
1024 条数据 ÷ 32 = 32 个 micro-batch

micro-batch 1: 数据 0~31   → forward → backward → 梯度累积
micro-batch 2: 数据 32~63  → forward → backward → 梯度累积
...
micro-batch 32: 数据 992~1023 → forward → backward → 梯度累积

全部累积完毕 → optimizer.step() → 参数更新一次
```

### 总结图

```
total_num_envs = 64
├── 分成 64/group_size = 8 组（GRPO 组内比较用）
├── 每组 group_size=8 个环境共享初始状态
│
rollout_epoch = 16
├── 每轮：64个环境各走 1 chunk (8步)
├── 16轮连续走完，每个环境走 128 步
├── 总数据量 = 64 × 16 = 1024 条
│
micro_batch_size = 32
├── 1024 条数据切成 32 个 micro-batch
├── 逐个做 forward/backward，梯度累积
└── 最后统一更新一次参数

这就是一个 training step。
```

**总体来说，一个**`step`**包括**

  ① 同步权重 (actor → rollout)

  ② 收集数据 (env ↔ rollout, 16轮 × 64环境)

  ③ 计算 advantage (GRPO 组内比较)

  ④ 训练更新 (forward/backward/optimizer.step)

  ⑤ 可选: 评估 / 保存 checkpoint

### 哪个影响什么


| 参数                 | 影响            | 调大            | 调小           |
| ------------------ | ------------- | ------------- | ------------ |
| `total_num_envs`   | 并行环境数         | 数据多、显存高       | 数据少、显存低      |
| `group_size`       | GRPO 组内比较的样本数 | 比较更充分、但独立状态更少 | 比较粗糙、但覆盖更多状态 |
| `rollout_epoch`    | 每个环境走多长       | 轨迹更长、数据更多     | 轨迹更短、训练更快    |
| `micro_batch_size` | GPU 单次处理量     | 更快但更占显存       | 更省显存但更慢      |


### Loss 计算

`train/loss = policy_loss - entropy_bonus × entropy_loss + kl_beta × kl_loss`

- **policy_loss**：PPO 风格的 clipped ratio × advantage，让好动作的概率增大、差动作的概率减小
- **entropy_loss**：策略熵，鼓励探索（前面有负号，所以熵越大 loss 越小）
- **kl_loss**：限制策略偏离参考策略

`train/entropy_loss` 跟policy loss不一样。

RL 训练中衡量**策略本身的不确定性/随机性**：

`entropy = -Σ P(token) × log P(token)` 

这个值越大，说明策略在多个 token 之间概率分布越均匀（越随机、越有探索性）。越小说明策略越确定（集中在少数几个 token 上）。

RL时，并没有每个token如何更新的“正确答案”，只有反馈信号。

### **Cross Entropy（SFT）的"正确答案"**

告诉模型**具体该输出哪个 token**：

```
输入: "pick up the red mug"
正确答案: token #4521 (某个具体的动作值)
模型预测: P(token #4521) = 0.3
loss = -log(0.3) = 1.2
```

是逐 token 的精确监督——"这个位置就应该输出这个值"。

### **Reward（RL）的"正确答案"**

只告诉模型**结果好不好**，不告诉你具体该怎么做：

```
模型输出了 128 步动作 → 任务成功了 → reward = 5.0
模型输出了 128 步动作 → 任务失败了 → reward = 0
```

reward 只是一个标量评分，不包含"每一步应该输出哪个 token"的信息。模型需要通过大量尝试（GRPO 的组内比较），自己摸索出哪些动作序列能拿到高分。

