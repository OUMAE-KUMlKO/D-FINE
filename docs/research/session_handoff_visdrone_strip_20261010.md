# D-FINE 边界特征实验会话交接 2026年10月10日

本会话从 GT 引导的 strip 定位实验推进到预测引导、逐层分布修正、二维边界采样带和 query 通道调制，并在 UAVSwarm、VisDrone 上进行了对照与诊断。当前主要问题是：G5 在 UAVSwarm 上有正向结果，但在 VisDrone 上尚未优于原版 D-FINE；自适应带宽、只保留前两层修正和延长训练均未改变这一结论。

**用户最新决定：采用已经保存的最佳 checkpoint 作为本轮结果，暂停继续增加训练长度。** 下一步候选视觉特征消融仅处于建议阶段，尚未批准实施。本次仅新增交接文档，不修改代码、不提交 commit、不启动训练。

## 1 工作环境与约定

- 仓库：`/home/zzy/projects/occ_study/D-FINE_mine/D-FINE`。
- 分支：`localization/gt-guided-strip-refinement`。
- Python：`/home/zzy/.conda/envs/dfine/bin/python`；训练在 Slurm 集群运行。
- VisDrone 数据位于 `/home/zzy/datasets/visdrone`；历史 UAVSwarm 数据使用 `/cluster/home/zzy/dataset` 下的数据，复现时以对应冻结配置为准。
- 下文 AP 等指标均为百分数形式，差值为百分点；epoch 使用日志中的 **0-based** 编号，例如 epoch 79 表示第 80 轮。
- 常规结果为本地 COCO 格式验证集、`maxDets=100` 的评估，不直接等同于 VisDrone 官方评测结果。
- 比较模型时，AP75、APsmall 等取各自最佳 AP 的同一个 checkpoint，不能拼接各指标独立最大值。
- 用户要求轻量执行、最小修改和最小充分验证；不自动提交代码，获得批准后使用中文 commit 信息，并写入对应实验结果。
- **用户不希望引入更高分辨率的 stride 特征。** 后续方案应遵守这一约束。

## 2 研究路线与模型名称

### 2.1 从 GT 引导到当前 G5

| 阶段 | 主要做法与已经得到的认识 | 记录 |
| --- | --- | --- |
| GT 引导 | 在 GT 边界位置提取 strip 特征，证明准确边界信息有较强定位收益；推理使用 GT，属于 oracle 参考 | [早期方案与结果](prediction_guided_strip_refinement_plan.md)、[几何复评](g5_15526_geometry_20260929.md) |
| P1 单预测框 | 用预测框边界替换 GT 边界；评估必须使用正确的 B0 最佳权重，不能把阶段不同造成的差异当作提升 | [早期方案与结果](prediction_guided_strip_refinement_plan.md) |
| P2 分布候选 | 扩大到边界分布候选；四边均有距 GT ≤1 输入像素候选的比例由 23.68% 增至 95.74%，但覆盖充分没有自动转化为明显 AP 收益 | [早期方案与结果](prediction_guided_strip_refinement_plan.md) |
| 概率融合到 query | 用上一层概率融合 strip 特征，再通过本层 query 和原回归头更新；曾测试冻结检测器、最后两层适配器，收益很小 | [早期方案与结果](prediction_guided_strip_refinement_plan.md) |
| P3 与逐层直接修正 | 用户明确目标是每层利用视觉特征修正预测分布，而不只是给候选概率增加直接监督；随后推进全层联合训练 | [概率修正方案](strip_candidate_probability_calibration_plan.md)、[全层训练](all_layer_decoder_strip_training.md) |
| 二维采样带 | 从窄 strip 改为边界对齐的二维采样带，加 `3×3` 局部编码；进行短程续训与全量训练 | [二维采样带](all_layer_boundary_band_2d.md) |
| MLP 输入消融 | 去掉直接输入修正 MLP 的原始 log 概率，再用 query 通道调制替代 query 与候选特征直接拼接 | [去掉 log 概率](all_layer_band2d_mlp_log_prior_ablation.md)、[query 调制](all_layer_band2d_query_gate_ablation.md) |
| 规范对照 | 统一训练设置，在两个数据集比较 G0/G5；后来重点补充主对照的种子，没有完成原定全部 G0–G5 矩阵 | [规范实验方案](uavswarm_visdrone_controlled_experiments_plan.md)、[历史 G5 汇总](g5_experiment_history_20260929.md) |
| VisDrone 改进 | 尺寸和密度诊断 → 自适应带宽 → 逐层框质量诊断 → 仅前两层 → 统一续训至 120 轮 | 本文第 4–7 节 |

早期几个可核查的 AP：B0 为 **59.8366**，P2 最佳 **59.8749**，冻结检测器的最后两层 decoder strip 最佳 **59.8597**。这些微小差异不能与后来的全量 G5、统一 batch 对照混为一谈。

### 2.2 规范实验的 G0–G5 定义

| 组别 | 每候选采样 | 编码 | 修正 MLP 的 log 概率输入 | query 融合 |
| --- | --- | --- | --- | --- |
| G0 | 无新增边界分支 | 原版 D-FINE | 不适用 | 原版结构 |
| G1 | `5×3`，法向 ±1 输入像素 | depthwise `3×1` | 有 | 拼接 |
| G2 | `5×7`，法向 ±8 输入像素 | depthwise `3×1` | 有 | 拼接 |
| G3 | 同 G2 | depthwise `3×3` | 有 | 拼接 |
| G4 | 同 G2 | depthwise `3×3` | 无 | 拼接 |
| G5 | 同 G2 | depthwise `3×3` | 无 | 通道调制 |

G1–G5 保留 pointwise `1×1`、64 维候选特征和原 logits 残差。**G0 是原版 D-FINE 模型结构**；规范实验共用训练基础设施，并不表示用 GT 分支或关闭开关后的 G5 冒充原版。早期 GT-G1 与此表中的预测引导 G1 含义不同，阅读旧记录时必须区分。

### 2.3 当前 G5 的真实前向路径

每个 decoder 层保留原有回归路径，在候选边界位置从 stride 8 的 encoder 特征上采样。每条候选边得到固定的 `5×7` 网格，因此不同预测框长度不影响输出张量尺寸：网格沿边长伸缩，点数固定。

局部编码为 depthwise `3×3` 加 pointwise 卷积。编码后沿切向做带有效性掩码的平均，**法向 7 个位置仍保留，展平后通过 `edge_fusion` 线性融合**；不能把当前实现解释为“把法向所有位置平均掉，完全丢失内外侧信息”。

简化公式为：

```text
F_k = 二维边界带局部编码与融合
F'_k = F_k × (1 + tanh(query_projection(query)))
Δlogit_k = MLP(F'_k, 边的 embedding, 候选相对位置偏移)
修正后 logits = 原 logits + 有效候选的 Δlogit
修正后分布与框继续传入后续 decoder 层
```

G5 的 MLP 不直接接收原始 log 概率，但原 logits 仍通过残差保留。目前 MLP 对各候选分别计算修正，尚无新增的显式相邻候选特征比较模块；原有 softmax 仍然在候选之间归一化，不能说候选完全没有耦合。

关键实现：`src/zoo/dfine/all_layer_strip.py`、`src/zoo/dfine/strip_probability_head.py`、`src/zoo/dfine/pred_strip_head.py` 中的 `encode_distribution_candidates`。

## 3 UAVSwarm 与原固定带宽 G5 的历史结果

### 3.1 不同实验口径要分开

| 对照口径 | G0 或旧 B0 AP | G5 AP | 差值 | 含义 |
| --- | ---: | ---: | ---: | --- |
| 历史 B0 对 15526 | 59.8366 | 61.0662 | +1.2296 | batch 分别 128/64，属于探索结果 |
| UAVSwarm 规范 seed 0/1/2 平均 | 60.6382 | 60.8226 | +0.1843 | 三个配对均为正，统一设置后收益较小 |
| VisDrone 三种子缓存复评平均 | 28.1975 | 27.9060 | −0.2915 | 与训练日志汇总存在细微数值差异 |

UAVSwarm 规范对照的平均 AP75 为 **66.2090 → 66.6080**，按未舍入数值计算平均提升 **0.3991** 个百分点。原计划增加 seed 3，后续用户决定不再跑完剩余项；未完成配对不计入上述三组平均。

VisDrone 原训练日志的三种子平均差为 **ΔAP −0.2930、ΔAP75 −0.4293**；引用时不要把训练日志的差值与缓存复评的均值混用。相关来源为 [历史 G5 汇总](g5_experiment_history_20260929.md) 和 [VisDrone 初次诊断](visdrone_g0_g5_diagnostics_20261006.md)。

15526 与后来固定带宽的 G5 属于同一模型设计，但训练设置和权重不同。其最佳 AP75 为 **67.5865**、APsmall 为 **59.7663**。旧版本 checkpoint 内部 epoch 计数存在偏移，内部 epoch 76 对应日志 epoch 79，查历史最佳轮应以相应记录为准。

### 3.2 15526 四边误差复评

| 模型 | 四边平均绝对误差，输入像素 | 平均匹配 IoU |
| --- | ---: | ---: |
| 历史 B0 | 1.702827 | 0.796307 |
| G5 15526 | 1.599195 | 0.801779 |
| GT-G1 oracle 历史参考 | 1.058836 | 0.886424 |

15526 相对历史 B0 的四边误差降低 **6.09%**，支持定位确有改善；但 GT 引导仍有明显优势。GT-G1 使用 GT 推理，且表中包含不同训练设置，不能作为公平的常规检测排名。详细口径与权重位置见 [15526 几何复评](g5_15526_geometry_20260929.md)。

## 4 VisDrone 尺寸和密度诊断

已有 checkpoint 的纯几何诊断中，G0/G5 四边 MAE 为 **2.638583 → 2.647100 输入像素，增加 0.323%**。因此当前证据不支持“VisDrone 整体定位已经改善，只被分类掩盖”的说法；各类、各尺寸存在不同方向的变化。

| 数据分布，验证集输入坐标 | UAVSwarm | VisDrone |
| --- | ---: | ---: |
| 每图平均目标数 | 7.15 | 70.73 |
| 目标短边中位数，像素 | 15.76 | 9.88 |
| 短边不足 8 像素的比例 | 8.51% | 39.27% |

VisDrone 在统一输入尺度下目标更小、更密集。这支持研究采样范围的合理性，但不能仅凭数据分布就确认退化由邻近目标污染造成。

| 按原图面积分组 | G5 相对 G0 的 MAE 变化 | AP 差值 |
| --- | ---: | ---: |
| small | +0.57%，变差 | −0.180 |
| medium | −0.20%，改善 | −0.412 |
| large | −4.59%，改善 | −1.249 |

此前会话中提到的 **+0.49% / −1.18% / −5.90%** 使用的是**输入面积分组**，与上表的原图面积分组不同，不应混用。类别也不能代替尺寸：truck、bus、van 的退化与 bicycle 的改善，不能直接推出“大目标退化、小目标改善”。

按每图目标数量分组，≤30、61–100、>100 目标的平均 ΔAP 分别为 **+0.075、−0.280、−0.469**；后两组在 3/3 个 seed 中下降。稀疏组略有正向结果，密集图退化更明显；但局部邻居数量的分析没有稳定单调规律，**尚未证明采样带混入邻居就是因果机制**。

来源：[初次诊断](visdrone_g0_g5_diagnostics_20261006.md)、[尺寸与密度细分](visdrone_size_density_diagnostics_20261006.md)。MAE 的相对百分比变化与 AP 的百分点差值是不同量。

## 5 自适应采样带和仅前两层实验

### 5.1 自适应带宽

在不增加更高分辨率特征的约束下，将固定法向半带宽 8 输入像素替换为：

```text
半带宽 = clamp(0.25 × 修正前预测框在该法向的尺寸, 2, 8)
```

横、纵方向分别计算，采样仍为 `5×7`，推理不使用 GT。以下均为 **VisDrone、seed 0、batch 32、8 GPU、80 轮**，初始全量训练仅加载 backbone 预训练权重。四组最佳均在日志 epoch 79。

| 模型 | AP | AP50 | AP75 | APsmall | APmedium | APlarge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 原版 D-FINE | 28.1196 | 47.8534 | 27.9655 | 18.4971 | 39.9493 | 56.5424 |
| 原固定带宽 G5 | 27.9700 | 47.5302 | 27.7728 | 18.5352 | 39.5223 | 55.4103 |
| 自适应六层 G5 | 27.9668 | 47.3316 | 27.8112 | 18.5472 | 39.7013 | 57.0859 |
| 自适应前两层 G5 | 27.7790 | 47.1061 | 27.6664 | 18.2046 | 39.7006 | 56.2904 |

自适应六层相对固定 G5 的 AP 为 **−0.0031**，基本持平；相对 G0 为 **−0.1528**。APlarge 的提升不能解释为普遍的大目标收益：其中只有 5 个大 people 实例的类别 AP 变化，贡献了该净提升约 76%。

自适应六层经历作业 16350 的节点故障和 16351 的 signal 53 中断，最后由 **16352 完成**；保存结果覆盖 epoch 0–79。结果与分类别解释见 [自适应带宽结果](visdrone_adaptive_band_results_20261009.md)。

### 5.2 每层修正前后的框质量

诊断作业 **16447** 完成。固定带宽和自适应六层模型分别用最佳 EMA，在全部 548 张验证图、640 输入、FP32 下追踪；GT 只用于统计。共 38,759 个 GT、38,725 个匹配对，少量未匹配来自拥挤图中 300 query 的数量限制。

主表对每个模型固定最终几何 Hungarian 配对后，逐层比较同一个 query 与 GT；另做层前、层后匹配的敏感性检查。以下 IoU 差为百分点，MAE 差为输入像素；MAE 负值表示改善。

| decoder 层 | 固定 G5 ΔIoU | 自适应 G5 ΔIoU | 自适应 G5 Δ四边 MAE |
| --- | ---: | ---: | ---: |
| 1 | +0.469949 | +0.534070 | −0.062532 |
| 2 | +0.090692 | +0.110022 | −0.007476 |
| 3 | −0.000839 | +0.004158 | +0.000544 |
| 4 | +0.008907 | +0.010651 | −0.000275 |
| 5 | −0.001115 | −0.010857 | +0.001447 |
| 6 | −0.004664 | −0.004175 | +0.000587 |

前两层在三种配对口径下均有正向定位收益；后两层修正很小，平均轻微退化，中间层部分结论依赖匹配口径。**完整模型内一次修正的局部收益，不等价于删除其他层后重新训练的最终 AP 收益。** 后续前两层实验正是检验这一推断，而不是已经证实它。

详见 [逐层诊断结果](visdrone_layer_effect_results_20261009.md)；原始追踪位于 `output/visdrone_layer_effect_20261009/`，包含 `fixed.json`、`adaptive.json` 及对应 traces。

### 5.3 只保留前两层

新增 `DFINEAllLayerStrip.num_refine_layers`：默认 `None` 保持六个修正头，设为 `2` 时只创建前两个修正头。**原 D-FINE 的六层 decoder 仍全部保留**，后四层不加 strip 修正，原回归、辅助监督和梯度路径继续工作。

配置为 `configs/controlled/visdrone_g5_adaptive_band_first2.yml`，作业 **16449 完成，耗时约 6 小时 58 分钟**，最佳 AP 为 **27.7790**；比自适应六层低 **0.1878**，APsmall 低 **0.3426**，未支持“仅前两层更好”。实现时 15 项相关局部测试通过。

80 轮各组都在末轮达到最佳，前两层版本末 8 轮 AP 仍增加约 0.3673，因此当时有理由检查训练预算。原配置 MultiStepLR 的 milestone 为 500，80 轮内没有衰减；`stop_epoch=72` 后仅训练 8 轮最终增强阶段。

## 6 统一续训至 120 轮

### 6.1 已执行的设置

三组分别从原 epoch 79 的 `last.pth` 完整恢复模型、optimizer、EMA、scaler 和 warmup，保持 seed 0、batch 32、8 GPU；按 G0 → 自适应六层 → 自适应前两层串行运行。

- epoch 80–99：backbone LR 为 `1.25e-5`，其余为 `2.5e-4`。
- epoch 100–119：各组学习率乘 `0.1`。
- `stop_epoch=72` 不变；设置 `stage2_rollback: false`，避免验证下降时回退到旧第一阶段权重。
- 设置 `resume_lr_schedule: true`，从恢复轮次和 `base_lrs` 对齐新调度，避免旧计数偏移或重复恢复时重复衰减。
- 输出使用新目录，原 80 轮结果没有覆盖；三组采用同一冻结源码快照。

根目录为 `output/visdrone_continue120_20261009/`，`source/` 保存实际执行源码，`jobs.json` 保存任务信息，`logs/` 保存 Slurm 日志。详见 [续训实施记录](visdrone_continue120_plan.md)。本轮同时改变训练长度、后期衰减和回退策略，不能拆分各项贡献。

### 6.2 最终状态与本轮采用结果

| 模型 | 作业 | 状态 | 原 80 轮 AP | 续训最佳 AP | 最佳 epoch | 最后完成 epoch | 最后 AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 16707 | COMPLETED，0:0 | 28.1196 | **28.6227** | 96 | 119 | 27.9632 |
| 自适应六层 | 16708 | COMPLETED，0:0 | 27.9668 | **28.4190** | 96 | 119 | 27.7994 |
| 自适应前两层 | 16709 | FAILED，1:0 | 27.7790 | **28.1545** | 94 | 98 | 28.1201 |

上述三个作业均已退出，不能把第三项写为正常完成 120 轮。最后一次核查中，前两项分别用时约 1:52:17、6:16:42；第三项用时 2:07:37，于 2026-10-10 07:27:22 退出。

| 最佳 checkpoint | AP50 | AP75 | APsmall | APmedium | APlarge | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0，epoch 96 | 48.7670 | 28.4398 | 19.2467 | 40.3924 | 59.6246 | 43.1308 |
| 自适应六层，epoch 96 | 48.2831 | 28.2323 | 19.2515 | 39.9514 | 59.7370 | 42.8436 |
| 自适应前两层，epoch 94 | 48.0637 | 28.0721 | 18.9299 | 39.9320 | 58.2401 | 42.7782 |

续训最佳相对原 80 轮分别增加 **0.5031、0.4522、0.3755 AP**，说明额外训练最初有用；但六层相对 G0 仍为 **−0.2037 AP**，前两层为 **−0.4682 AP**，模型排序未变。这些只是 seed 0 的续训结果，不能当作三种子验证。

### 6.3 收敛判断与失败原因

G0 和六层从最佳到末轮分别下降 **0.6595、0.6196 AP**，同时训练 loss 继续降低约 **7.16%、8.15%**。这符合后期泛化退化或优化目标与验证指标不一致的现象，不能继续仅用“尚未收敛”解释。AP 在衰减前已出现平台；衰减后下降并不能证明衰减是原因，因为没有不衰减的同预算对照。

前两层在 epoch 98 仅比自身最佳下降 **0.0343 AP**，随后在 **epoch 99** 训练中失败，尚未到 epoch 100 的学习率衰减。不要把它与另外两组的长期下降混为一谈，也不能归因于尚未执行的衰减。

首个错误出现在 matcher 计算 GIoU 时，`src/zoo/dfine/box_ops.py` 的断言：

```python
assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
```

这里 `boxes1` 来自预测框；能够确定存在不满足检查的预测坐标，可能是坐标顺序异常或 NaN 等非有限值。没有保存出错张量，具体根因未确定。后续 Gloo barrier 超时属于失败后的清理问题，不能当作首因，也没有证据将其写为 OOM 或 GT 数据错误。

错误日志：`output/visdrone_continue120_20261009/logs/continue120_visdrone_g5_adaptive_band_first2_s0_16709.err`。尚未修复、未重启；`last.pth` 保留 epoch 98 的完整状态，但用户已经决定暂停延长训练。

## 7 必须保留的结果与权重

### 7.1 本轮最佳权重

以下路径均相对于仓库根目录，使用 `best_stg2.pth`，不要用末轮 `last.pth` 替代最佳结果：

```text
output/visdrone_continue120_20261009/results/visdrone/seed0/g0/best_stg2.pth
output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth
output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band_first2/best_stg2.pth
```

已经核查 checkpoint 元数据，三份 `last_epoch` 分别为 **96、96、94**。推理复评应沿用原 EMA 权重选择口径。

### 7.2 结果数据与旧权重目录

- [续训结果完整 JSON](../../output/visdrone_continue120_20261009/analysis/continuation_results_20261010.json)：原 epoch 79、最佳、末轮、曲线及 checkpoint 元数据。
- [续训曲线](../../output/visdrone_continue120_20261009/analysis/continuation_results_20261010.png)。
- 原 G0、固定 G5：`output/controlled_20260926_212304/results/visdrone/{g0,g5}/`。
- 原自适应六层：`output/controlled_visdrone_adaptive_band_20261006_seed0/results/visdrone/seed0/g5_adaptive_band/`。
- 原自适应前两层：`output/visdrone_g5_adaptive_band_first2_20261009/results/visdrone/seed0/g5_adaptive_band_first2/`。
- [自适应结果 JSON](results/visdrone_adaptive_band_results_20261009.json)、[逐层诊断 JSON](results/visdrone_layer_effect_results_20261009.json)。

部分旧方案文档仍保留提交当时的 RUNNING 或 Dependency 状态，属于历史快照；当前结果以本交接文档及对应结果 JSON 为准。实际复现实验应优先使用各输出目录的冻结源码和配置。

## 8 代码与提交状态

近期已经存在的中文提交：

| commit | 内容 |
| --- | --- |
| `cc32ae1` | 完善多种子对照与定位评估并归档 G5 实验结果 |
| `aa5f482` | 补充 VisDrone 尺寸与密度诊断并记录 G5 性能退化 |
| `92ecc8c` | 增加自适应边界采样带并记录 VisDrone AP 27.9668 |
| `b345389` | 新增逐层框质量诊断并记录前两层定位收益 |

写本交接文档时，**暂存区为前两层实现，未暂存部分为续训调度与提交工具**。不要为了整理文档而覆盖或重新分配这些更改。

已暂存的 6 个文件：

```text
configs/controlled/visdrone_g5_adaptive_band_first2.yml
docs/research/visdrone_first_two_strip_plan_20261009.md
src/zoo/dfine/all_layer_strip.py
tests/test_controlled_submission.py
tests/test_first_two_layer_strip.py
tools/submit_controlled_experiments.py
```

未暂存的已跟踪文件：

```text
src/solver/det_solver.py
tests/test_det_solver_stage_sync.py
```

续训工作中尚未跟踪的文件：

```text
configs/controlled/include/continue120.yml
configs/controlled/visdrone_g0_continue120.yml
configs/controlled/visdrone_g5_adaptive_band_continue120.yml
configs/controlled/visdrone_g5_adaptive_band_first2_continue120.yml
docs/research/visdrone_continue120_plan.md
slurm_files/run_resume_dfine_controlled.slurm
tests/test_continuation_submission.py
tests/test_resume_schedule.py
tools/submit_visdrone_continuation.py
```

本交接文档同样新增但未暂存、未提交。若后续获得提交授权，可以把前两层实验、续训实现与结果分别整理为两个阶段性 commit，保留中文说明及真实结果。

续训实现完成时，调度和阶段同步共 14 项测试、提交和配置 3 项测试通过，合计 **17 项**；还做过 Slurm 语法、CLI 预览和 diff 格式检查。这些属于当时的验证记录，本次文档任务没有重复执行训练或代码测试。

局部测试环境可沿用 `PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp`。同一进程多次对照 YAML 配置时使用 `load_config(path, {})`，避免旧加载函数共享默认字典的影响；本轮没有扩大修改该加载器。

## 9 下一步建议与授权边界

### 9.1 已经决定的事项

1. 本轮使用第 7 节的最佳 checkpoint；暂停继续增加训练长度，不自动恢复 16709。
2. 不引入更高分辨率 stride 特征。
3. 暂时不能宣称自适应带宽解决 VisDrone 退化，也不能宣称仅前两层修正优于六层。
4. 保留 UAVSwarm 的正向证据，同时区分历史 +1.2296 AP 与规范三种子 +0.1843 AP。

### 9.2 待用户确认的下一步

优先用已有六层 G5 checkpoint 检查**模型是否有效利用不同候选的视觉信息**，先做推理诊断，再决定是否修改结构。建议在 UAVSwarm 和 VisDrone 各选明确的 checkpoint，对每个模型比较以下三种推理方式：

| 方式 | 改变什么 | 想回答的问题 |
| --- | --- | --- |
| 正常 | 保留候选各自的 `F_k` | 提供该 checkpoint 自身基准 |
| 候选视觉均值 | 同一 query、同一条边的有效候选特征取均值，再复制到各候选 | 候选之间的视觉差异是否重要 |
| 候选视觉置乱 | 只在同一 query、同一条边内置乱有效候选的 `F_k` | 视觉内容与候选位置的正确对应是否重要 |

均值和置乱都应保留候选位置、原 logits、边标识、query 调制和有效性掩码；排除无效候选，置乱固定随机源。报告相对同一模型正常推理的 AP、AP75 和固定配对四边误差变化，不直接拿两个数据集的绝对 AP 比较敏感性。配对口径应事先明确，避免修正后重新匹配掩盖变化。

如果 UAVSwarm 明显依赖候选视觉差异，而 VisDrone 不敏感，会支持继续检查 VisDrone 的候选特征可区分性或使用不足；如果两者都敏感，说明特征确实在起作用，还需研究哪些目标、修正幅度或分类置信度抵消收益。推理扰动存在分布变化，这些结果用于定位问题，不能单独证明新结构有效。

可能的后续结构是：在候选维度 K 上增加轻量的一维局部比较，再预测 logit 修正；继续保留六层 decoder、stride 8、二维采样带和 query 调制。**这是未实施、未批准的新假设，不能当作下一步必然有效的方案。** 也不应把“再次移除 MLP 的 log 概率输入”作为新实验，因为 G5 已经完成这一消融。

若之后再训练，应先对 epoch 99 的非法预测框做有限范围复现，定位首个非有限值或非法坐标；当前没有证据支持直接用 clamp 掩盖问题。现有逐层 trace 工具对六个修正头和最终输出有相应假设，若用于前两层模型，需先适配，不能无检查复用。

本次交接请求没有授权上述新诊断、结构改动或训练提交；新对话应先依据用户的新指令确定执行范围。

## 10 新对话可直接使用的开场说明

```text
请先阅读 docs/research/session_handoff_visdrone_strip_20261010.md，沿用其中的实验结果、最佳 checkpoint 和工作区状态。
当前采用最佳 checkpoint，暂停增加训练长度，不要自动恢复失败作业。
请继续讨论并具体化“已有 checkpoint 的候选视觉特征均值/置乱消融”方案，解释怎样区分 UAVSwarm 与 VisDrone 的差异。
不要引入更高分辨率 stride 特征，也不要改动现有暂存内容或自动 commit；新的诊断、代码修改和训练按我后续批准的范围执行。
```
