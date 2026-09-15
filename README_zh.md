<div align="center">

# HazardAuditor

### 从可执行威胁到更安全的计算机使用智能体

一个基于真实执行行为的生成式安全模型，联合审计用户请求、智能体推理、工具调用与环境结果。

[![项目主页](https://img.shields.io/badge/Project-Website-004b91?style=flat-square)](https://yunhao-feng.github.io/HazardAuditor/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-223344?style=flat-square&logo=github)](https://github.com/Yunhao-Feng/HazardAuditor)
[![模型权重](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Weights-527f9f?style=flat-square)](https://huggingface.co/Yunhao-Feng/HazardAuditor)
[![HF 论文](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-e7ad56?style=flat-square)](https://huggingface.co/papers/2609.15134)
[![论文](https://img.shields.io/badge/arXiv-2609.15134-e24d42?style=flat-square)](https://arxiv.org/abs/2609.15134)
[![许可证](https://img.shields.io/badge/License-Apache--2.0-223344?style=flat-square)](LICENSE)

[English](README.md) · [简体中文](README_zh.md)

</div>

![HazardAuditor 框架](docs/assets/hazardauditor-overview.png)

## 简介

计算机使用智能体的安全风险往往产生于执行过程：若只看用户提示或最终回复，可能无法识别由文件读取、工具调用、外部请求和环境反馈共同构成的危险行为。HazardAuditor 因此将完整轨迹作为输入，判断智能体实际执行或尝试执行的行为。

模型输出一段可审计的证据分析和一个严格的二分类结果：

```text
<analysis>
基于轨迹证据的安全分析
</analysis>
<label>safe|unsafe</label>
```

- **轨迹级判断：**联合读取请求、推理、回复、工具、参数和观察结果。
- **行为导向：**轨迹中出现恶意文本并不自动等于 unsafe，关键是智能体做了什么。
- **可审计：**输出支持人工复核的执行证据说明。
- **决策对齐：**GuardPO 直接优化安全决策，同时消除长短 rationale 的隐式权重差异。

## 论文结果

以下数字均来自项目论文。CUA-Exec 在四种异构智能体框架上分别包含均衡的 safe/unsafe 轨迹。

| 模型 | Accuracy | Source-specific F1 |
|---|---:|---:|
| HazardAuditor-SFT | 80.50 | 80.16 |
| **HazardAuditor** | **90.88** | **90.85** |
| **GuardPO 提升** | **+10.38** | **+10.68** |

| CUA-Exec 框架 | Accuracy | Macro-F1 | 相对最强已有 guard |
|---|---:|---:|---:|
| Claude Code | **94.00** | **94.00** | +12.5 |
| Codex | **95.50** | **95.50** | +4.0 |
| Hermes | **86.50** | **86.42** | +9.5 |
| OpenClaw | **87.50** | **87.46** | **+16.5** |

| 外部基准 | Accuracy | F1 |
|---|---:|---:|
| AgentHazard | 87.55 | 89.47 |
| R-Judge | 89.60 | 89.50 |
| ASSE-Safety | **91.50** | **91.50** |
| ATBench | **88.40** | **88.30** |

## 快速开始

模型权重现已在 Hugging Face 公开，可直接运行仓库内的合成示例：

```bash
git clone https://github.com/Yunhao-Feng/HazardAuditor.git
cd HazardAuditor
python -m pip install -e .

hazard-auditor --input examples/trajectory.json
```

也可以直接运行等价的 Python 示例脚本：

```bash
python examples/infer.py
```

Python 接口：

```python
import json
from hazard_auditor import HazardAuditor

with open("examples/trajectory.json", encoding="utf-8") as file:
    trajectory = json.load(file)["content"]

auditor = HazardAuditor.from_pretrained("Yunhao-Feng/HazardAuditor")
result = auditor.audit(trajectory)

print(result.analysis)
print(result.label)
```

默认使用兼容性较好的 SDPA。在支持的 GPU 环境中可启用 FlashAttention-2：

```bash
python -m pip install flash-attn --no-build-isolation
hazard-auditor \
  --input examples/trajectory.json \
  --attn-implementation flash_attention_2 \
  --dtype bf16
```

## 输入与输出

CLI 接受包含 `content` 字段的单个 JSON 对象。`content` 可以是非空字符串，也可以是事件列表：

```json
{
  "content": [
    {"role": "user", "content": "..."},
    {"role": "agent", "thought": "...", "action": "..."},
    {"role": "environment", "content": "..."}
  ]
}
```

推理过程使用训练时的原始 system prompt，将轨迹放入明确的不可信数据边界，关闭 Qwen thinking mode，保留前 16,000 个 prompt token，最多生成 384 个 token，并使用 greedy decoding。无法解析时返回 `label: null`，绝不会静默映射为 safe。

## GuardPO

![GuardPO 目标](docs/assets/guardpo.png)

GuardPO 将可确定验证的最终分类结果转换为序列级 outcome，通过 batch-centered advantage 和 token-level clipped importance weighting 更新策略。rationale 与 verdict 区域先分别按样本归一化，再在 response 级聚合，从而避免较长解释获得不成比例的优化权重。

完整训练流程见 [SFT 文档](training/sft/README.md)和 [GuardPO 文档](training/guardpo/README.md)。仓库不发布研究数据，使用者需要自行准备符合数据协议且具有合法授权的轨迹。

## 局限与负责任使用

HazardAuditor 是研究型检测模型，不是访问控制系统，也不能撤销已经发生的操作。在分布外环境、证据截断或未知工具协议下，模型可能产生误报或漏报。涉及重要操作时应结合最小权限、独立策略检查、审计日志和人工复核。

请勿在示例、Issue、日志或公开模型输入中提交真实凭据或个人数据。轨迹处理者需要自行确认数据授权、隐私要求和第三方许可证。

## 论文与引用

论文已在 [arXiv](https://arxiv.org/abs/2609.15134) 公开，也可通过
[Hugging Face Papers 页面](https://huggingface.co/papers/2609.15134)或
[永久 DOI](https://doi.org/10.48550/arXiv.2609.15134) 访问：

```bibtex
@misc{feng2026hazardauditor,
  title         = {HazardAuditor: From Executable Threats to Safer Computer-Use Agents},
  author        = {Yunhao Feng and Ruixiao Lin and Ming Wen and Yanming Guo and Xingjun Ma and Yutao Wu and Xinhao Deng and Shouling Ji},
  year          = {2026},
  eprint        = {2609.15134},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2609.15134},
  url           = {https://arxiv.org/abs/2609.15134}
}
```

## 许可证

代码采用 [Apache License 2.0](LICENSE)。模型许可条款将在 Hugging Face 权重页面提供。
