# 原版 D-FINE 关闭 DDF：UAVSwarm 与 VisDrone 三种子对照

用户于 2026-10-20 批准补齐原版模型关闭蒸馏的实验，并直接启动训练。实验不含 G5、末端框修正或稀疏 strip-query。

## 设置与实施计划

- 六组：UAVSwarm、VisDrone 各 seed 0、1、2，均为完整 80 轮。按 UAVSwarm seed0、VisDrone seed0、UAVSwarm seed1、VisDrone seed1、UAVSwarm seed2、VisDrone seed2 提交。
- 每个 seed 的两组无相互依赖，可由资源情况决定是否同时运行；seed1 两组均依赖 seed0 两组结束，seed2 两组均依赖 seed1 两组结束，使用 `afterany`。失败或取消会释放下一对，不重试或恢复原作业。
- 原版 `DFINE` 六层 decoder，HGNetv2 B4 主干预训练，检测器从头训练，无检测 checkpoint 初始化。保留原增强、学习率、EMA、AMP、第72轮阶段切换及回载规则。
- 8 GPU，总 batch 为 UAVSwarm 64、VisDrone 32；每作业32 CPU、128 GiB内存、24小时时限。
- 只将 `loss_ddf` 从1.5改为0；FGL=0.15、VFL=1、bbox=5、GIoU=2、辅助层和DN监督保留。不删除 `local` 损失分组。
- 冻结运行时来自 `output/controlled_visdrone_20260929_seed12/source`，与 UAVSwarm 多seed基线的运行时代码一致。相对seed0旧运行时，只有 `src/core/yaml_config.py` 和 `src/misc/dist_utils.py` 的分布式采样器种子补丁。每组同时设置 `seed`、`controlled_seed`；seed0行为仍与旧受控基线一致。
- VisDrone验证使用正值标注ID版本，与G5关闭DDF一致；历史原版的验证ID含0，跨历史结果比较仍需注明这个极小的口径差异。

实施顺序：

1. 新增两份 `configs/dfine/ablation/*_baseline_ddf_off.yml`，明确 `model: DFINE`。
2. 新增 `tools/check_baseline_ddf_off.py` 和通用 `slurm_files/run_train_baseline_ddf_off.slurm`；检查合并配置相对原版仅有允许差异，检查实际criterion权重和数据路径。
3. 冻结上述受控源码并加入本次配置及入口，记录来源；对六组逐一执行配置检查及脚本语法检查。
4. 提交六组作业，记录ID、依赖与日志路径，确认调度状态及首组启动情况。不修改现有暂存内容，不自动commit。

## 结果口径

每组按总AP选择最佳轮，AP50/AP75/APsmall取同一轮；完整运行后补充末8轮均值。三个种子完整结束后再汇总均值与标准差，不混入中断组，也不按每个指标独立挑选峰值。

优先与同数据集、同seed、同80轮协议的原版DDF开启比较，再与G5关闭DDF比较。已有G5种子和训练规则并非全部严格配对，不将缺失的G5 seed2或中断运行当作完整三种子结果。

具体而言，近期UAVSwarm G5关闭DDF运行时未采用受控模型/采样器种子机制，solver也保留较旧的内部epoch回载逻辑；本轮优先与原版受控baseline匹配，因此与该G5结果不是仅结构不同的严格配对。VisDrone G5关闭DDF的17479沿用了受控seed0运行时，比较更直接。

## 提交记录

2026-10-20 15:32已提交六组；六份展开配置及实际criterion权重检查通过，启动脚本语法检查通过。

实验根目录：`output/dfine_baseline_ddf_off_20261020_153047/`。冻结源码在`source/`，协议与来源在`manifest/provenance.json`，提交命令及完整日志路径在`jobs.json`。

| 作业 | 数据集 | seed | 依赖 |
| --- | --- | ---: | --- |
| 17698 | UAVSwarm | 0 | 无 |
| 17699 | VisDrone | 0 | 无 |
| 17700 | UAVSwarm | 1 | 17698、17699均结束 |
| 17701 | VisDrone | 1 | 17698、17699均结束 |
| 17702 | UAVSwarm | 2 | 17700、17701均结束 |
| 17703 | VisDrone | 2 | 17700、17701均结束 |

首次调度检查：17698已在gpu4运行，17699因`AssocGrpGRES`等待账户GPU配额，后四组等待上述依赖。这不是六组同时运行的承诺；实际启动还受账户配额与集群调度限制。

逐轮指标保存到`results/<dataset>/seed<seed>/log.txt`，训练控制台为同目录`train.log`，Slurm日志为`logs/<dataset>_seed<seed>_<jobid>.out`与`.err`。

## 完成结果与解释（2026-10-21）

六组均完成80轮，Slurm状态均为`COMPLETED`、退出码`0:0`。逐轮日志中的训练损失有限，所有DDF损失贡献均为0。以下单位均为百分数，epoch为日志中从0开始的编号；各指标取总AP最佳的同一轮。

| 数据集 | seed／作业 | 最佳epoch | AP | AP50 | AP75 | APsmall | 末8轮平均AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UAVSwarm | 0／17698 | 77 | 60.7859 | 95.1318 | 66.6291 | 59.7929 | 60.5820 |
| UAVSwarm | 1／17700 | 75 | 60.8857 | 95.1339 | 66.3558 | 59.8397 | 60.6851 |
| UAVSwarm | 2／17702 | 76 | 61.3438 | 95.2855 | 67.4545 | 60.0920 | 60.9077 |
| VisDrone | 0／17699 | 79 | 28.5011 | 48.0961 | 28.3012 | 18.7299 | 28.3518 |
| VisDrone | 1／17701 | 79 | 27.8754 | 47.4210 | 27.9302 | 18.1271 | 27.6792 |
| VisDrone | 2／17703 | 79 | 28.3406 | 48.1324 | 28.2612 | 18.7443 | 28.1586 |

六组指标日志：

- UAVSwarm：[seed0](../../output/dfine_baseline_ddf_off_20261020_153047/results/uavswarm/seed0/log.txt)、[seed1](../../output/dfine_baseline_ddf_off_20261020_153047/results/uavswarm/seed1/log.txt)、[seed2](../../output/dfine_baseline_ddf_off_20261020_153047/results/uavswarm/seed2/log.txt)。
- VisDrone：[seed0](../../output/dfine_baseline_ddf_off_20261020_153047/results/visdrone/seed0/log.txt)、[seed1](../../output/dfine_baseline_ddf_off_20261020_153047/results/visdrone/seed1/log.txt)、[seed2](../../output/dfine_baseline_ddf_off_20261020_153047/results/visdrone/seed2/log.txt)。

三种子比较中的开启DDF结果来自：seed0为`output/controlled_20260926_212304/results/<dataset>/g0/log.txt`；UAVSwarm seed1、2为`output/controlled_20260928_150346/results/uavswarm/seed<seed>/g0/log.txt`；VisDrone seed1、2为`output/controlled_visdrone_20260929_seed12/results/visdrone/seed<seed>/g0/log.txt`。完整路径、12项指标、最佳轮及末8轮均值见[机器可读汇总](../history/results/dfine_baseline_ddf_off_summary_20261021.json)。

| 数据集 | 开启DDF平均AP ± 样本标准差 | 关闭DDF平均AP ± 样本标准差 | AP均值变化 | 末8轮平均AP的均值变化 |
| --- | ---: | ---: | ---: | ---: |
| UAVSwarm | 60.6382 ± 0.2927 | 61.0052 ± 0.2975 | +0.3669 | +0.2908 |
| VisDrone | 28.1975 ± 0.1274 | 28.2390 ± 0.3250 | +0.0415 | +0.0817 |

UAVSwarm关闭DDF的seed0、1、2分别变化−0.0465、+0.1050、+1.0422 AP，平均收益主要来自seed2，尚不能表述为每个seed稳定改善。VisDrone分别变化+0.3815、−0.2530、−0.0039 AP，平均AP75变化−0.0682；当前没有稳定的总体提升证据。VisDrone历史开启DDF组使用含0的标注ID，关闭组使用正值ID，尚未全部统一复评。

双方均关闭DDF时，与已有G5的比较为：

| 数据集／seed | 原版AP | G5 AP | G5 AP增量 | G5 AP75增量 |
| --- | ---: | ---: | ---: | ---: |
| UAVSwarm／0 | 60.7859 | 61.2316 | +0.4457 | +0.9696 |
| UAVSwarm／1 | 60.8857 | 61.0894 | +0.2037 | +1.2159 |
| VisDrone／0 | 28.5011 | 28.5186 | +0.0175 | +0.0088 |

VisDrone中，此前G5关闭DDF相对原版开启DDF约+0.3990 AP，原版自身关闭DDF已获得+0.3815 AP，因此不能将前者整体归为G5结构收益。G5的APsmall仍高0.3390点，但APmedium低约0.2051点，总AP基本抵消，且只有一个G5 seed。UAVSwarm的G5 AP75优势值得保留为线索，但因前述随机数控制及阶段回载差异，尚非严格配对的结构证据，且缺少G5 seed2。

本轮原版关闭DDF的UAVSwarm seed2达到61.3438 AP，已超过所选G5关闭DDF模型的61.2316；后者是用户指定归档模型，不能继续称为当前所有实验中的最高单次AP。本节仅汇总已完成作业，不自动提交补充训练。
