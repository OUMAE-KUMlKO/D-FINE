# 全层边界二维采样带实验

日期：2026-09-24

本轮先检验边界附近二维特征编码是否有帮助；MLP 输入保持原样，后续再单独讨论调整。

## 1. 本轮改动

每层 decoder 仍对每条边的全部 33 个候选位置分别提取特征、预测 logits 残差，然后重新积分得到框并传给下一层。六层均参与，各层使用独立参数。

| 项目 | 上一版全层 strip | 本轮二维带 |
| --- | --- | --- |
| 沿边界采样 | 沿当前预测框对应边长均匀取 5 点 | 保持 5 点 |
| 法向采样 | ±1 输入像素内取 3 点 | ±8 输入像素内均匀取 7 点 |
| 每候选采样形状 | `C×5×3` | `C×5×7` |
| 空间编码 | depthwise `3×1` + pointwise `1×1` | depthwise `3×3` + pointwise `1×1` |
| 空间汇聚 | 沿边界池化后拼接 3 列，线性映射至 64 维 | 沿边界池化后拼接 7 列，线性映射至 64 维 |

四条边均按“框内→候选边界→框外”排列法向采样列。变长边界通过固定点数采样得到统一尺寸；`3×3` 指采样网格上的卷积核，而非原图中的连续 `3×3` 像素块。

使用原 encoder 的第 0 层特征，步长为 8；±8 输入像素约覆盖法向 ±1 个特征网格。该半宽是本轮固定实验设置，尚未证明最优。每个候选都有自己的采样带，并非只在最终预测框边界采样。

本轮同时调整法向范围、法向点数和局部卷积，检验这套局部编码方案的整体效果，不能据此把收益单独归因于 `3×3` 卷积。

## 2. 保持不变的部分

- 候选位置、六层概率残差反馈、原候选有效性规则以及无效候选回退逻辑。
- MLP 输入仍为：候选特征 64 维、query 投影 64 维、边标识 8 维、相对边界偏移 1 维、原 log 概率 1 维，共 138 维。
- 所有候选独立编码，原 logits 加上修正量后执行完整 softmax；不按原概率预先混合候选特征。
- 初始化时修正头末层为零；旧配置默认仍使用 3 列和 `3×1`，保留旧权重形状。

## 3. 训练与比较

本轮提交一组 12 epoch 实验，加载强 B0，冻结原检测器，只训练六个新头。保持全局 batch 96、8 GPU、固定 640 输入、seed 0、学习率 0.00025、关闭 EMA、梯度检查点和原 B0 固定匹配的 `5×L1 + 2×GIoU` 监督。

初始权重：`/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923/b0/best_stg2.pth`。二维带分支重新初始化。

数据：`/cluster/home/zzy/dataset/UAVSwarm-dataset-master/train`，标注为同数据集下的 `annotations/train_half.json` / `annotations/val_half.json`。

主要对照（均以百分数报告）：

| 对照 | 最佳 AP | 对应 AP75 |
| --- | --- | --- |
| 强 B0 | 59.8366 | 65.9820 |
| 上一版全层 strip，12 epoch，作业 15444 | 59.8632 | 66.1064 |

训练结束后同时检查最佳轮与末轮的 AP、AP75、小目标 AP 及走势，判断是否出现超过前几轮微小波动的收益。单次种子结果不能证明统计显著性。

## 4. 入口与记录

- 配置：`configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_all_layer_band2d_12ep.yml`
- 脚本：`slurm_files/run_train_dfine_uavswarm_all_layer_band2d_12ep.slurm`
- 输出：`/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_band2d_12ep`
- 日志：`logs/dfine_uavswarm_all_layer_band2d_12ep/`

35 项定向 CPU 测试通过：二维带头 5 项、概率头及共享候选采样回归 22 项、全层新旧集成 8 项。验证了四边采样方向、变长边界的固定尺寸、零初始化保持 B0、六层梯度和冻结检测器；脚本通过 `bash -n`，本轮代码通过 `git diff --check`。

已通过 `sbatch` 提交作业 **15492**。本轮改动未 commit，保留原暂存区。

启动确认：作业在 `gpu12` 运行，已完成 epoch 0 首个训练批次，loss 为 **0.1825**，日志记录峰值 allocated 显存 **14,797 MiB**；截至该批次未出现运行错误。完整验证结果待训练产生。

## 5. 短程结果与全量训练

作业 15492 已完成 12 轮。最佳结果在 epoch 1：AP **59.9030**、AP75 **66.2136**、小目标 AP **58.7158**；末轮 epoch 11 为 AP **59.7035**、AP75 **65.5371**、小目标 AP **58.5663**。最佳 AP 仅比强 B0 高 **0.0664 个百分点**，随后回落，尚不足以支持显著改善定位的结论。

用户要求改为从头全量训练。本轮保留同一套 `5×7`、法向 ±8 输入像素、`3×3` 编码和原 MLP 输入，切换到 `DFINEAllLayerStrip`：

- 仅加载 `weight/hgnetv2/PPHGNetV2_B4_stage1.pth`，encoder、decoder、六个新头重新初始化；脚本不传 `--tuning` / `--resume`。
- 按原完整训练配置联合优化可训练的 backbone 层、encoder、decoder 和六个边界头，使用原 `DFINECriterion` 的分类、框、分布、辅助层及 DN 损失。
- 80 epoch、8 GPU、全局 batch 64、seed 0；保留原数据增强、多尺度、EMA、学习率与第 72 轮阶段切换，启用边界头梯度检查点。
- batch 64 与上一版全层全量实验一致；原 B0 的 batch 为 128，后续对比时需要考虑这一差异。

为避免重现作业 15445 在阶段切换时的 `EOFError`，`src/solver/det_solver.py` 在读取 `best_stg1.pth` 前增加分布式 barrier，等待 rank 0 完成前一轮 checkpoint 写入。单进程不执行 barrier。

全量训练入口：

- 配置：`configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_band2d_full.yml`
- 脚本：`slurm_files/run_train_dfine_uavswarm_all_layer_band2d_full.slurm`
- 输出：`/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_band2d_full`
- 日志：`logs/dfine_uavswarm_all_layer_band2d_full/`

全量入口验证：4 项二维带集成测试通过（含联合训练的六层卷积及检测器梯度），2 项阶段切换同步测试通过；合并配置确认仅 backbone 预训练、80 轮、batch 64、EMA 和第 72 轮切换，Slurm 脚本语法检查通过。

已提交全量训练作业 **15509**。本轮未 commit。

启动确认：作业在 `gpu16` 运行，已完成 epoch 0 的首个训练批次，完整原生损失为 **44.7494**，首批峰值 allocated 显存为 **14,693 MiB**；截至该批次未出现运行错误。该损失包含分类、辅助层和 DN 等多项，与短程实验的框损失数值不可直接比较。

## 6. 全量训练结果

作业 15509 已完整完成 epoch 0–79，Slurm 状态 `COMPLETED`、退出码 `0:0`，耗时 7:03:18。

| 结果 | AP | AP75 | 小目标 AP |
| --- | --- | --- | --- |
| 强 B0 最佳，epoch 75 | 59.8366 | 65.9820 | 58.6336 |
| 二维带全量最佳，epoch 76 | 60.7504 | 66.7188 | 59.8611 |
| 二维带全量末轮，epoch 79 | 60.3299 | 66.6149 | 59.6012 |

最佳 AP、AP75、小目标 AP 分别比强 B0 高 0.9138、0.7368、1.2275 个百分点；第二阶段全部 8 轮 AP 在 60.1506–60.7504。由于 batch 64 与 B0 的 128 不同，不能把收益全部归因于二维带。

在相同的第一阶段轮数下，二维带最佳 AP 为 58.6600，低于旧 strip 全量实验的 59.6901；旧 strip 未完成第二阶段，因此此次结果不能证明 `3×3` 二维带优于旧 strip。

日志在 epoch 77 验证后记录 `Refresh EMA at epoch 77 with decay 0.9998`，对应原训练流程回载 `best_stg1.pth`。随后 loss 回升和 AP 回落受到该回载影响，不能直接判定为连续训练过拟合。

最佳权重为输出目录中的 `best_stg2.pth`，含 EMA。该文件内部 `last_epoch=68`，与日志最佳 epoch 76 不同：epoch 72 回载了第一阶段最佳 epoch 63，内部计数随后递增，未恢复当前外层 epoch。后续续训需处理计数偏移；本次指标取实际完整验证日志。

本轮 MLP 输入始终未改变，仍为 138 维，包含原始 log 概率。
