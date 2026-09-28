# UAVSwarm / VisDrone 受控实验执行记录

日期：2026-09-26。对应 `uavswarm_visdrone_controlled_experiments_plan.md`，每组 `seed=0`，每个数据集 G0–G5 各一次，共 12 次全量训练。

## 配置与数据

- 入口：`configs/controlled/{uavswarm,visdrone}_g{0..5}.yml`，组别含义与方案一致。
- 全部使用 8 GPU、80 epoch、640 基础输入和原有多尺度/增强/损失/EMA/阶段回载策略，仅加载本地 HGNet B4 backbone 预训练。
- UAVSwarm 全局 batch **64**；VisDrone 所有组统一为 **32**，验证 batch 同训练 batch。每个数据集内部保持相同学习率和其他训练设置。
- VisDrone 显存预检使用 RTX 3090 24 GiB、800×800 随机输入与最密集图片的真实标注（最多 902 个有效目标）。G3 单卡 batch 8 OOM；单卡 batch 4 时 G3、G5 均完成两步 AMP 前向、原生损失、反向及优化器更新，峰值分别约 15118、15190 MiB。此检查用于确定显存预算，不产生精度结论。
- VisDrone 图片仍读取 `/home/zzy/datasets/visdrone/VisDrone2019-DET-{train,val}/images`。`tools/dataset/prepare_visdrone_controlled.py` 生成 `output/visdrone_controlled/instances_{train,val}.json`，原数据不改动。
- 共 7019 张图片路径有效，原 TXT 与源 JSON 的有效类别/边框逐项匹配。类别 0 保留为 pedestrian；原 TXT 的忽略区域和 others 转为各类别的 crowd 框，训练时排除、验证时使用 COCO crowd 匹配规则。删除 1 个退化有效框、2 个退化忽略区域。训练集 343204 个有效框、验证集 38759 个有效框。
- 使用本地 COCO 验证口径，`maxDets=100`，不等同于 VisDrone 官方评测；所有组共用同一份处理后的标注。

## 公平性与权重记录

- `controlled_seed=0` 隔离模型构建 RNG；已实测两个数据集 G0/G5 的 1173 个公共参数张量完全相同。各组数据 loader 使用独立固定 generator，DDP 包装后保留，避免新增分支影响数据采样/增强随机流。
- 修复阶段回载后 `last_epoch` 偏移；最佳 checkpoint 使用真实训练 epoch。
- 每轮验证后、可能的回载前写入 `last.pth`，保留末轮实际评估的权重；阶段回载行为不变。最佳 AP 与其余指标取同一 checkpoint。

## 串行提交

```bash
/home/zzy/.conda/envs/dfine/bin/python tools/submit_controlled_experiments.py --submit
```

不加 `--submit` 仅打印顺序。提交时新建 `output/controlled_<时间>/`，将代码、配置、backbone 权重和处理后的 VisDrone 标注复制为该批次快照，避免后续工作区修改影响排队任务。各任务独立初始化，不加载前一任务的权重。

顺序：UAVSwarm G0 → G5 → VisDrone G0 → G5 → UAVSwarm G1 → G2 → G3 → G4 → VisDrone G1 → G2 → G3 → G4。

使用 `afterok:<前一任务ID>`：前一任务成功后下一项自动进入资源调度；若失败，依赖任务自动取消。实际开始时间仍取决于集群资源。

批次目录包含 `jobs.json`（任务 ID、依赖、日志与输出路径）、`logs/`、`results/<数据集>/<组别>/`、`source/` 和显存预检记录。训练启动状态见下方提交记录。

## 本次提交

2026-09-26 21:23 提交，批次目录：

`/home/zzy/projects/occ_study/D-FINE_mine/D-FINE/output/controlled_20260926_212304`

| 顺序 | 作业 ID | 数据集 / 组别 | 成功依赖 |
| --- | --- | --- | --- |
| 1 | 15540 | UAVSwarm G0 | 无 |
| 2 | 15541 | UAVSwarm G5 | 15540 |
| 3 | 15542 | VisDrone G0 | 15541 |
| 4 | 15543 | VisDrone G5 | 15542 |
| 5 | 15544 | UAVSwarm G1 | 15543 |
| 6 | 15545 | UAVSwarm G2 | 15544 |
| 7 | 15546 | UAVSwarm G3 | 15545 |
| 8 | 15547 | UAVSwarm G4 | 15546 |
| 9 | 15548 | VisDrone G1 | 15547 |
| 10 | 15549 | VisDrone G2 | 15548 |
| 11 | 15550 | VisDrone G3 | 15549 |
| 12 | 15551 | VisDrone G4 | 15550 |

提交后已确认 15540 在 gpu16 进入 `Epoch [0/80]`，首个 batch loss 为 44.7827、峰值显存 5520 MiB。其他 11 项为 `PENDING (Dependency)`；`scontrol` 确认使用 `afterok` 和 `KillOInInvalidDependent=Yes`。

启动前验证：18 项相关局部测试通过、Slurm shell 语法检查通过、12 份配置的组内公共设置一致、G0/G5 公共初始化一致、两个数据集真实样本读取正常，以及上述 GPU 显存预检通过。代码未提交 commit，整批训练使用提交时的独立快照。
