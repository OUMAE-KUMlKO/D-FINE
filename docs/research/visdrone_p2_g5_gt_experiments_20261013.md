# VisDrone P2预测引导与六层G5 GT引导实验

用户已批准修改代码并提交两组训练。本轮不自动提交Git，不修改已有暂存内容，不恢复旧失败作业，不增加stride更小的特征。

## 实验定义

### P2：预测分布引导框残差

复用既有 `DFINEPredRefiner`，与已完成GT候选oracle使用相同冻结G0：`output/visdrone_continue120_20261009/results/visdrone/seed0/g0/best_stg2.pth` 的EMA权重。预测分布在有效候选上归一化、池化编码后的条带特征，再与query及框编码一起预测中心/尺度残差。训练与推理均使用预测分布；前向不读取GT，全部有效普通query均修正。损失沿用固定baseline匹配的L1+GIoU。

沿用GT候选oracle的12epoch、seed0、640输入、stride8、total batch96、8GPU、AMP、无EMA、无SyncBN、AdamW lr0.00025、epoch10衰减0.1、warmup50。新增修正头重新初始化，不从GT训练头继续。

### G5-GT：六层候选视觉门控oracle

结构基于刚完成诊断的六层自适应带宽G5，保留原候选集合、5×7二维条带、query通道调制、逐候选logit残差、积分解码和修正框反馈。每层使用修正前预测框与修正前分类分数做2/5/2 Hungarian匹配；对匹配query，每条边由GT选择最近有效候选，只保留该候选的视觉特征，其余候选视觉特征置零。未匹配query、去噪query保持原G5路径。GT不直接写入输出概率、候选坐标或输出框。

这是GT候选位置提示的oracle；门控模式本身携带GT信息，不能将结果单独解释为视觉特征的辨别能力。GT用于前向指导的配对不等于原criterion的监督配对；后者维持原G5行为。逐层反馈及联合训练会改变分类和最终query分工，因此仅比较完整检测指标，不跨模型认定同query对应同一目标。

从 `output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth` 的EMA epoch96严格加载。无新增可训练参数，主干和六层G5保持原联合训练范围。固定12epoch短程适应、seed0、640、total batch32、8GPU、AMP、无SyncBN、无EMA；AdamW主学习率0.000025，backbone非norm参数0.00000125，epoch10衰减0.1、warmup50。原LQE、损失权重和安全回退保持；固定尺寸数据处理与oracle对照对齐。该短程oracle不是重新完整训练的严格性能上界，也不能将其与静态checkpoint的差异全部归因于GT。

## 实施计划与验证

- [x] P2独立VisDrone配置、GT无关性/冻结检测器局部测试、真实batch12的两步AMP预检。
- [x] G5-GT独立模型、GT候选视觉门控、普通G5默认路径不变；nearest有效候选、空GT/未匹配/DN、梯度、严格加载测试。
- [x] 评估器区分“前向需要GT”和“冻结baseline几何统计”；联合G5使用正常COCO AP并明确标注oracle。
- [x] G5-GT配置与真实batch4两步AMP预检，验证全部六层修正头及原检测器梯度、有限框、无GT缺失静默退化。
- [x] 独立冻结源码及结果目录；明确主机内存；预检成功后提交两组训练，记录job id和状态，不自动重试。
- [x] P2同checkpoint补充只保留baseline GT匹配query修正的评估，以分离“修正哪些query”的GT辅助；普通全query AP作为主结果。

## 状态

2026-10-13：实现中。既有9个候选诊断文件保留暂存状态；本轮新文件与代码修改不暂存、不commit。

P2配置、预检和双口径评估局部测试通过；GPU预检17131已提交，运行根目录 `output/visdrone_p2_prediction_20261013/`。G5-GT核心20项相关测试、评估器5项相关测试通过，另经局部只读审查未发现阻塞问题。

P2预检17131通过：两次更新均有限，检测器参数和缓冲不变，带GT/无GT前向完全一致；GPU峰值6.37GiB。正式训练17132已提交，8GPU、64GB主机内存、12epoch；训练后自动执行同checkpoint的两种推理范围评估。

G5-GT实际配置严格加载及optimizer分组检查通过，新增预检单测2项通过；GPU预检17133已提交，运行根目录 `output/visdrone_g5_gt_candidate_20261013/`。

G5-GT预检17133通过：EMA epoch96共1263个key严格加载；空GT与普通G5一致、真实GT会改变预测；两次有效AMP更新后backbone/encoder/decoder及六个refiner均实际更新，全部可训练参数均具有有限梯度，输出框有限。峰值显存约8.21GiB。最初两次AMP梯度溢出由GradScaler跳过并降低scale，随后两步有效；没有恢复失败作业。正式训练17134已提交。

## 提交结果

| 实验 | GPU预检 | 正式训练 | 轮数 | GPU/总batch | 主机内存 | 提交后状态 |
|---|---:|---:|---:|---|---|---|
| P2预测引导 | 17131，COMPLETED 0:0 | 17132 | 12 | 8 / 96 | 64GiB | PENDING，Priority |
| 六层G5 GT候选视觉门控 | 17133，COMPLETED 0:0 | 17134 | 12 | 8 / 32 | 64GiB | PENDING，Priority |

两组时限均2小时，无自动requeue；这只是作业资源上限，训练轮数固定12。运行目录中的 `jobs.json` 保存作业号、提交命令和观察状态，`experiment_selection.json` 保存checkpoint及设置，`preflight.json` 保存实际GPU验证。队列状态会随调度变化，以上为本轮提交后的状态。

本轮新增15项局部测试均通过，直接相关既有回归通过，Slurm语法与未暂存diff空白检查通过。关键生产代码仅涉及G5头的可选GT视觉mask、新GT G5包装器/注册和评估器oracle能力区分；P2直接复用已有模型。当前新增/修改内容均未暂存、未commit，原9个诊断文件的暂存状态保留。

## 2026-10-14方案变更

用户批准将G5-GT改为与原G5对齐的完整训练。尚未运行的12轮微调作业17134已取消；旧源码、预检和配置保留作历史记录，不修改为新方案。P2作业17132不变。新方案从HGNetv2主干预训练初始化，按80+40轮训练，详见 `visdrone_g5_gt_full_training_20261014.md`。

完整训练最终替代作业：单卡预检17141已通过，正式80+40轮作业17143已提交（PENDING / Priority）。首次预检17139因增强后crowd元数据长度过期失败，修复并验证后使用新快照；17140因依赖失败取消，从未开始。完整协议及当前状态见[完整训练记录](visdrone_g5_gt_full_training_20261014.md)。P2作业17132不变。
