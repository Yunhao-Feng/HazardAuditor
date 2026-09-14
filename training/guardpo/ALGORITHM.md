# HazardAuditor：单一 Outcome Reward 的 CISPO 后训练方案

## 1. 目标与设计边界

HazardAuditor 输入完整 Agent 轨迹，输出：

```text
<analysis>
基于轨迹行为和工具结果的安全分析
</analysis>
<label>safe|unsafe</label>
```

核心部署指标是 safe/unsafe 二分类性能；分析文本是模型产生正确判断的显式推理过程，
但不引入一个独立在线奖励模型去规定它的措辞或风格。本方案只训练一个 Guard Model，
不训练第二个 Actor，不使用记忆库、Extractor 服务或检索系统。

借鉴 [Complementary RL](https://arxiv.org/pdf/2603.17621) 中 Experience
Extractor 的优化方式：一个序列级 outcome reward、batch-centered advantage、全序列
广播，以及 token-level CISPO importance-sampling ratio。

## 2. 与 CRL Extractor 的映射

| CRL Experience Extractor | HazardAuditor |
|---|---|
| 经验序列 (m_i) | Guard response (y_i=(c_i,l_i)) |
| 后续 Actor 是否成功 | 最终安全标签是否正确且可解析 |
| (r(m_i)\in\{-1,+1\}) | 严格分类/格式 outcome (r_i) |
| (hat A_i=r_i-\bar r) | rollout batch 内去均值 |
| advantage 广播至全部经验 token | 广播至全部 CoT 与 Label token |
| token-level CISPO ratio | token-level CISPO ratio |

该映射借用的是 Extractor 的策略梯度形式，而不是 CRL 的双模型系统结构。

## 3. 数据与采样

训练 prompt 来自五个来源的训练集：AgentHazard、ASSE、ATBench、RJudge、Vera。
验证集使用四个公开 benchmark 的测试集与 Vera 固定测试集。

数据构建遵守以下不变量：

1. actor prompt 只含系统提示词和轨迹，不含 label 或参考 reason；
2. label、reason 只保存在 `reward_model.ground_truth`，其中训练 reward 只读取 label；
3. 原始训练轨迹与验证轨迹通过内容哈希检查无泄漏；
4. 每个 source×label stratum 等量采样；
5. SFT 阶段已有的过采样副本不会作为独立原始样本再次计数；
6. 超长轨迹只截断末尾，固定系统指令与任务开头必须完整保留。

默认每个 source×label stratum 采样 512 条，共 10 个 stratum、5,120 个 prompt。
同一原始轨迹可以因平衡采样重复出现，但每次 rollout 都重新生成 response。

## 4. 唯一的序列级 Outcome Reward

模型输出必须严格匹配：

```text
<analysis>非空分析</analysis>
<label>safe</label>
```

或：

```text
<analysis>非空分析</analysis>
<label>unsafe</label>
```

奖励定义为：

\[
r_i=
\begin{cases}
+1, & \text{格式合法且 label 正确}\\
-1, & \text{格式合法但 label 错误}\\
-1.25, & \text{格式非法、分析为空或无法解析}
\end{cases}
\]

safe 与 unsafe 完全对称。`-1.25` 只用于避免模型通过跳过协议或输出模糊文本规避二分类；
它不是第二种 reward channel。

VERL reward tensor 中，每条 response 只有最后一个有效 token 保存 (r_i)，其他位置为
零。这是终局序列奖励的标准 token-shaped 表示。

## 5. CRL 式 Batch-Centered Advantage

对当前完整 rollout batch (B) 中的所有 response 计算：

\[
\bar r_B=\frac{1}{|B|}\sum_{i\in B}r_i
\]

\[
\hat A_i=r_i-\bar r_B
\]

这里明确：

- 不按 prompt UID 分组；
- 不除以 batch reward 的标准差；
- 不对 CoT 另算 reward 或 advantage；
- 不使用参考 reason 做文本相似度；
- 不请求远程大模型。

然后把同一个 advantage 广播给 response 的每个有效 token：

\[
A_{i,t}=\hat A_i,\qquad t=1,\ldots,|y_i|
\]

因此 outcome reward 会训练整段分析，而不是只训练最终 label token。如果整个 batch 的
reward 完全相同，则所有 (hat A_i=0)，该 batch 不产生策略梯度。这与 CRL Extractor
的 batch-centered estimator 一致。

## 6. Token-Level CISPO

对每个 response token：

\[
\rho_{i,t}=
\frac{\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})}
     {\pi_{old}(y_{i,t}\mid x_i,y_{i,<t})}
\]

\[
\tilde\rho_{i,t}=\operatorname{clip}
\left(\rho_{i,t},1-\epsilon_{low},1+\epsilon_{high}\right)
\]

\[
\ell_{i,t}=
-\operatorname{sg}(\tilde\rho_{i,t})
 A_{i,t}
 \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
\]

`sg` 表示 stop-gradient。即使 ratio 越界，token 的 log-probability 梯度仍然存在，
但更新系数受到裁剪限制。

VERL 的 token-level rollout correction 处理：

\[
\pi_{old}/\pi_{rollout}
\]

它补偿 vLLM rollout policy 与训练 policy 的偏差。该 ratio 与 CISPO 的
$\pi_\theta/\pi_{old}$ 属于不同层次，不能互相替代。

## 7. 按样本长度归一化与 Label 尾部加权

CRL 原始 Extractor 目标把序列 advantage 广播到全部生成 token。HazardAuditor 的轨迹分析
长度差异很大；若直接在整个 batch 上做 token mean，一条 300-token 的错误/格式失败 CoT
会比一条 30-token 的 CoT 产生约十倍的总梯度。日志中 384-token 截断样本占比常达
10%--30%，因此这种长度偏置会放大负向更新和梯度尖峰。

本实现不改变 advantage 广播：padding 位置始终由 `response_mask` 置零，每条样本只有真实
生成的 $L_i$ 个 token 接收 $\hat A_i$。随后把 response 分成 analysis 区和固定 label
尾部，先在每条样本内部按各自真实长度求均值，再跨样本求均值：

\[
\mathcal L=
\frac{1}{B}\sum_i\left[
\frac{1}{|A_i|}\sum_{t\in A_i}\ell_{i,t}
+\lambda_{label}
\frac{1}{|L_i|}\sum_{t\in L_i}\ell_{i,t}
\right]
\]

默认：

```text
label_tail_tokens = 12
label_loss_weight = 2.0
```

对实际生成长度小于 12 的异常短响应，全部有效 token 属于 label tail、analysis 项为零；
padding 永远不进入任一项。这样长短 CoT 对 batch 的总权重相同，同时每个真实 CoT token
仍接收同一个 outcome advantage。Label 权重不宜设置过大，否则会增加绕过推理、直接
优化分类 token 的倾向。`2.0` 是首轮实验的固定起点。

## 8. 训练配置

默认服务器预算为 8×A100，模型为 Qwen3 Guard 8B 的全量 SFT checkpoint。

关键配置：

```text
train prompt batch                 = 32
rollouts per prompt                = 8
PPO mini-batch prompt groups       = 32
PPO mini-batch responses / worker  = 32 × 8 / 8 GPUs = 32
PPO epochs               = 1
learning rate            = 2e-7
warmup ratio             = 0.03
rollout temperature      = 0.5
rollout top-p            = 0.9
weight decay             = 0.01
max prompt length        = 16000
max response length      = 384
CISPO clip low/high      = 0.1 / 0.1
rollout IS threshold     = 2.0
label loss weight        = 2.0
```

默认 5,120 个 prompt、batch size 32，因此一个虚拟 epoch 为 160 个外层训练 step。每步
的 256 条全局 rollout response 会在 8 个 actor worker 间分片，每个 worker 收到 32 条；
VERL new-engine 的 `ppo_mini_batch_size` 必须填写这个本地 response 数，不能填写全局
256。训练启动后立即执行 rollout 和参数更新，不做 step 0 全量验证。完成第 1 次参数更新
后立即执行一次确定性全量验证并保存 step-1 checkpoint，用于尽早发现格式、reward 或
策略崩溃；随后恢复每 50 step 的周期，在 step 50/100/150/... 验证并保存。

每个训练 step 额外直接记录合法格式率、正确标签率、malformed 率、错误标签率和正确
reward 率，避免只凭 reward mean 反推离散奖励组成。

不使用在线 KL reward、独立 critic、reward model 或 entropy bonus。CISPO clipping、
rollout correction、按样本长度归一化、较小学习率、单次 PPO epoch、较低温 rollout 和
SFT 初始化共同约束策略漂移。

## 9. 验证与模型选择

验证对四个 benchmark 和 Vera 分别计算：

- Accuracy；
- safe precision / recall / F1；
- unsafe precision / recall / F1；
- Macro-F1；
- parse rate。

checkpoint 使用 label-only 的字典序：

1. overall Macro-F1；
2. worst-source Macro-F1；
3. overall 两类 recall 的较小值。

磁盘始终只保留两个角色的并集：

- validation-best；
- latest（用于可靠断点恢复）。

两者相同时只保留一个 RL checkpoint。由于不再进行 step 0 基线验证，best 角色从第一
个完成验证的 RL checkpoint 开始产生。

VERL 0.7 new engine 的异步 rollout 在生成后会把 FSDP actor model 搬到 CPU；验证后紧接
着保存时，HazardAuditor hook 会先只把 model 恢复到 compute device，再调用 VERL 原生的
model/optimizer/extra/Hugging Face checkpoint 保存，避免 FSDP 在 CPU 参数上导出 state
dict 时失败。

训练器控制台输出会由启动器实时双写到：

```text
artifacts/checkpoints/guardpo/logs/training.log
```

该文件以 append 模式记录每次启动的时间、完整命令、step、loss、reward、CISPO 指标、
验证过程以及最终退出码，可在另一个终端持续 `tail -f`。

## 10. 该目标能够与不能够保证什么

能够保证：

- CoT 和 Label 的所有 token 都接收最终分类 outcome 的策略梯度；
- 能稳定提高正确 response、压低错误或 malformed response 的概率；
- 不依赖远程服务，不会因 Judge 断网阻塞训练；
- 不会学习某个外部 Judge 对措辞、长度和风格的偏好。

不能直接保证：

- 两个 label 都正确的 response 中，事实更扎实的 CoT 得到更高 reward；
- 输出分析必然是最终 label 的忠实因果解释；
- token-level IS ratio 能识别哪个推理 token 在语义上最关键。

这些是 sequence-level outcome RL 的可识别性边界，而不是实现错误。若高质量因果分析
在困难样本上更稳定地产生正确标签，它会通过跨样本成功率被间接强化。CoT 质量应通过
固定离线审计集监控，而不是重新接入在线远程 Judge。

## 11. 首轮实验与必要消融

主实验固定使用本方案。科学评估至少比较：

1. SFT checkpoint；
2. outcome-CISPO，`label_loss_weight=1.0`；
3. outcome-CISPO，`label_loss_weight=2.0`（主配置）；
4. outcome-CISPO，`label_loss_weight=4.0`。

比较 overall / per-source Macro-F1、两类 recall、parse rate，并人工盲审固定的一组困难
轨迹分析。若 4.0 只改善标签而明显损伤因果分析，维持 2.0；若 1.0 的标签学习不足，
才考虑在 2.0 附近做更细粒度搜索。
