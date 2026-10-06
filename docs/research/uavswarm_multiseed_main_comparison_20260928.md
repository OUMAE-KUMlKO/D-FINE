# UAVSwarm G0 / G5 补充三个种子

日期：2026-09-28。用户决定暂不继续 VisDrone，先补充 UAVSwarm G0/G5 的三个 seed，并串行训练。

## 范围

- 保留已完成的 seed=0 主对照（G0：15540，G5：15541）。
- 新增 seed=1、2、3，每个种子运行 G0、G5，共 **6 次全量训练**；完成后主对照共4个种子、8次训练。
- seed=0 的 G3 仍暂停，保留 `output/controlled_20260926_212304/results/uavswarm/g3/paused_epoch048.pth`；G4（15547）继续保持 hold，本批不恢复。
- 已取消尚未启动的 VisDrone G1–G4（15548–15551），保留已完成的 VisDrone G0/G5 结果。

## 不变的训练条件

G0 为 D-FINE-L 原结构，G5 为二维边界采样带、3×3编码、MLP无log概率输入、query通道调制的完整方案。均使用 UAVSwarm 既有划分、8 GPU、全局batch64、80 epoch、640基础输入，以及相同优化器、增强、多尺度、EMA、DN、损失和阶段回载策略。每次从 HGNet B4 backbone 预训练开始，不加载检测器历史权重。

`--seed`、`controlled_seed` 和 `PYTHONHASHSEED` 同步设置。公共模型初始化、worker增强随机源和 DistributedSampler 使用对应seed；同seed的G0/G5配对，采样器seed在所有rank上相同，worker generator保留rank偏移。seed0的既有行为保持不变。

## 提交与顺序

```bash
/home/zzy/.conda/envs/dfine/bin/python tools/submit_controlled_experiments.py \
  --datasets uavswarm --groups g0 g5 --seeds 1 2 3 --submit
```

顺序：seed1 G0 → seed1 G5 → seed2 G0 → seed2 G5 → seed3 G0 → seed3 G5。使用 `afterok`，前一项成功后才启动下一项；失败则取消后续依赖任务。

新批次保存独立代码/配置/预训练权重快照，不依赖 VisDrone 数据。`jobs.json` 记录实际seed、作业ID、日志及输出路径；结果写入 `results/uavswarm/seed<种子>/<组别>/`。

## 汇总口径

每个seed按验证AP选择最佳checkpoint，其余指标取同一checkpoint。报告seed0–3的AP、AP50、AP75、APsmall原值、均值和样本标准差，以及每个seed内G5−G0的配对差值；末轮和阶段曲线作为辅助，不选最好seed代替全部结果。

## 本次执行

提交前通过10项局部测试及Slurm脚本语法检查，验证三种seed的传递、组内配对、跨rank采样一致性和六项串行依赖。

2026-09-28 15:03 提交。批次目录：`output/controlled_20260928_150346`。

| 顺序 | 作业 ID | seed | 组别 | 成功依赖 |
| --- | --- | --- | --- | --- |
| 1 | 16003 | 1 | G0 | 无 |
| 2 | 16004 | 1 | G5 | 16003 |
| 3 | 16005 | 2 | G0 | 16004 |
| 4 | 16006 | 2 | G5 | 16005 |
| 5 | 16007 | 3 | G0 | 16006 |
| 6 | 16008 | 3 | G5 | 16007 |

提交后已确认16003在gpu20运行，其余五项等待Dependency；15547仍为JobHeldUser。日志位于本批`logs/`，例如`controlled_uavswarm_g0_s1_16003.out`。

## 2026-09-29 范围终止

用户决定不再运行剩余任务。16008于完成6/80轮后主动停止，15547取消，当前无运行或排队任务。seed0 G3继续保持停止；VisDrone不再继续。最终保留seed0、1、2三个完整G0/G5配对用于统计，不将seed3 G0单独混入均值。完整G5历史汇总见`g5_experiment_history_20260929.md`。
