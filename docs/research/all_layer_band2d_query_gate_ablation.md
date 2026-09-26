# 全层二维边界带：query 通道调制消融

日期：2026-09-25

## 对照与假设

上一轮作业 15521 去掉 MLP 的 log 概率输入，最佳 AP 60.9385、AP75 66.5184、小目标 AP 59.8073（实际日志 epoch 77）。相对保留该输入的 15509，AP 小幅增加，AP75 与小目标 AP 未同步改善。

本轮检验：让 query 调节候选视觉特征的通道强度，是否比直接把 query 拼接给评分 MLP 更有利于定位。两种融合都能够学习特征交互，本轮不预设通道调制必然更好。

## 唯一融合改动

新增 `query_fusion`，默认 `concat` 保留旧结构；本轮设为 `channel_gate`，保持 `use_log_prior=False`。

每层独立计算：

```text
gate = 1 + tanh(query_projection(query))
modulated_feature = candidate_feature * gate
delta = score_MLP(concat(modulated_feature, edge_embedding, relative_offset))
corrected_logits = original_logits + delta
```

每个 query 生成一组 64 维通道权重，共用于其四条边和全部候选；各候选特征仍独立保留。MLP 不再额外拼接 query，输入由 137 维变为 73 维（64+8+1）。没有添加候选间交互或额外损失。

gate 模式的 query 投影权重、偏置初始化为零，因此初始 gate=1。评分头末层继续零初始化，初始 logits 与原检测器一致。默认 concat 模式的初始化及权重形状保持兼容。

## 训练设置

继承 15521 的全量配置：80 epoch、8 GPU、batch 64、seed 0，仅加载 `weight/hgnetv2/PPHGNetV2_B4_stage1.pth`；encoder、decoder 和六个修正头重新初始化，联合训练。

沿用 `5×7` 边界带、法向 ±8 输入像素、depthwise `3×3` + pointwise `1×1`、原生辅助层/DN 损失、增强、多尺度、EMA、梯度检查点和阶段切换策略。本轮不同时修正此前的阶段回载 epoch 元数据偏移。

对照 15521 的最佳 AP 及对应 AP75、小目标 AP，同时比较第一阶段和第二阶段走势。融合方式与输入维度发生变化，随机种子相同不意味着所有训练随机数逐项相同；单次实验只作为方案筛选依据。

## 入口

- 配置：`configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_band2d_query_gate_full.yml`
- 脚本：`slurm_files/run_train_dfine_uavswarm_all_layer_band2d_query_gate_full.slurm`
- 输出：`/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_band2d_query_gate_full`
- 日志：`logs/dfine_uavswarm_all_layer_band2d_query_gate_full/`

本轮与上一轮移除 log 概率输入的实验分开归档。

## 验证与作业

21 项定向测试通过：新门控单元测试 6 项、旧概率头/no-prior 回归 14 项、六层原生损失联合训练集成 1 项。覆盖初始恒等门控、真实乘法调制、零初始化残差、默认 concat 兼容及门控/视觉/检测器梯度。

实际 YAML 模型在 CPU 构建成功，六个头均启用 `channel_gate`、关闭 log 概率输入、MLP 输入 73 维，门控投影为零初始化。合并配置确认与 15521 的训练设置一致；脚本通过 `bash -n`，代码通过 `git diff --check`。

已提交 Slurm 作业 **15526**。

启动确认：作业在 `gpu12` 运行，已完成 epoch 0 首个训练批次，完整原生损失为 **44.8458**，首批峰值 allocated 显存 **14,477 MiB**；截至该批次未出现运行错误。

## 完整训练结果

15526 完成 80 轮，状态 `COMPLETED`、退出码 `0:0`，耗时 7:02:39。以下指标均取各自 AP 最佳轮：

| 实验 | 最佳 epoch | AP | AP50 | AP75 | 小目标 AP |
| --- | --- | --- | --- | --- | --- |
| 15521，query 直接拼接 | 77 | 60.9385 | 94.9262 | 66.5184 | 59.8073 |
| 15526，query 通道调制 | 79 | 61.0662 | 94.4596 | 67.5865 | 59.7663 |
| 差值（百分点） | — | +0.1277 | -0.4665 | +1.0681 | -0.0410 |

第二阶段最后 8 轮平均 AP 为 60.7477→60.7646（+0.0169），平均 AP75 为 66.3277→66.8922（+0.5645），平均小目标 AP 为 59.6181→59.4907（-0.1274）。第一阶段末 12 轮平均 AP 几乎持平，平均 AP75 提高约 0.6062 个百分点。

两版都在 epoch 74 评估后回载第一阶段权重；15521 还在 epoch 78 后回载，而 15526 没有。末轮比较受到该差异影响，但门控版在 epoch 74–78 已连续取得更高 AP75，因此改善信号并非只出现在末轮。

结论：本次单种子消融显示较持续的严格 IoU 指标改善信号，但总体 AP 基本持平，小目标尚未获益；不能认定已解决候选特征利用问题。

最佳文件为输出目录的 `best_stg2.pth`，六层 MLP 实际输入均为 73 维，六层门控投影均从零发生参数更新。文件内部 `last_epoch=76` 对应日志 epoch 79，仍存在阶段回载引起的计数偏移。
