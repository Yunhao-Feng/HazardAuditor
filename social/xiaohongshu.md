# 开源｜让安全模型判断智能体“实际做了什么”

最近我们开源了 HazardAuditor：一个面向计算机使用智能体完整执行轨迹的生成式安全审计模型。

与只看输入或最终回复不同，它会联合读取用户请求、智能体回复与可见推理、工具调用和环境观察，再给出证据分析与 `safe / unsafe` 判断。面对同一数据外传请求，拒绝且未调用工具是 safe；读取凭据并尝试发送，即使被沙箱拦截，仍是 unsafe。

模型经过 SFT 与 GuardPO 训练，论文报告 CUA-Exec Accuracy 90.88%、F1 90.85%。代码、训练算法、推理示例和 8B 权重已经开放。作为研究型 guard，实际部署仍建议结合沙箱、最小权限和人工复核。

arXiv：2609.15134  
HF：Yunhao-Feng/HazardAuditor

#AI安全 #AgentSafety #开源模型 #大模型研究 #ComputerUseAgent

## 配图顺序

1. 官网首屏：一句话说明 HazardAuditor 是什么，并给出项目入口。
2. 解释型 Demo：展示同一风险请求下，不同行为为何得到不同 verdict。
3. 论文结果：展示 CUA-Exec 总体结果和跨智能体框架表现。
