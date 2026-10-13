# VisDrone 六层 G5 门控与参考框对照执行计划

用户已批准修改代码并执行 E1–E3；E0 复用现有最佳结果，不重训。LQE 不改。已有暂存内容保留，本阶段不自动提交。

## 实验定义

| 组别 | 修正量门控 | 下一层参考框 | 训练 |
| --- | --- | --- | --- |
| E0 | 关闭 | 本层修正后 | 复用原六层自适应带宽 G5 |
| E1 | 开启 | 本层修正后 | 新训 |
| E2 | 关闭 | 本层修正前 | 新训 |
| E3 | 开启 | 本层修正前 | 新训 |

共同保持六层、stride8、5×7、3×3、query channel_gate、自适应带宽、原损失与 LQE。所有层输出框、损失、LQE 和跨层累加 logits 都使用修正后的分布。参考框选项只影响下一层 reference，包括位置编码和 attention 采样；本层修正前框仍包含之前层的累积修正，不是独立 G0。

## 实现任务

- [x] 在 `StripProbabilityHead` 增加默认关闭的 `adaptive_correction_gate`。每 query、每边预测一个 `g=1+tanh(MLP(context))`，形状 `[N,4,1]`，乘在最终 delta 上再加 prior。context 使用该边有效候选的编码特征均值、已有 query 投影、edge embedding、有效候选原 delta 的均值与平均绝对离差。末层权重/bias 为0，初始g严格等于1。门控模块初始化隔离 CPU RNG，保持旧参数初始化及后续随机状态不被新增参数消耗改变。门控关闭时 state_dict 和旧前向不变，无效候选沿用原屏蔽。
- [x] `DFINEAllLayerStrip` 透传上述开关，并增加默认 `next_reference_source='corrected'`，允许 `'pre_correction'`。decoder 保存本层 strip 修正前的解码框；最终输出和分布递推仍使用修正后值，仅下一层 detach reference 根据选项选择。原默认调用及原版 D-FINE 保持行为不变；不扩展到无关 refiner。
- [x] 先写有意义的局部测试，再实现：默认兼容、g=1对齐及共享参数初始化、边独立缩放与无效候选、门控梯度/训练可用；参考框和位置编码路径切换、修正后输出/分布/LQE不变、原默认与无refiner行为；至少一次真实小模型训练前向/反向验证。
- [x] 新建 E1/E2/E3 各80轮和continue120配置，共六份，复用现有 G5/续训配置。新增专用提交入口与Slurm脚本，冻结源码、记录配置/初始化/结果路径/作业ID，复用现有训练及续训入口，禁止提交E0及覆盖旧输出。
- [x] 保持E0的训练配方：只加载同一backbone预训练，从epoch0开始80轮，再从本组epoch79 last.pth完整恢复到epoch119；总batch32、8GPU、seed0，第二段epoch100衰减、禁用stage2回退。未改变初始化来源或增加总轮数。原E0曾有节点中断，不能承诺逐步随机状态完全复现。
- [x] 必要局部测试、配置差异和Slurm语法验证通过后，做最小实际模型冒烟验证并提交E1–E3。训练作业失败不自动重试或恢复；不恢复旧失败作业。

## 验证与报告

E0：`output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth`，epoch96 EMA。历史训练评估AP28.418829，正GT编号复评AP28.419801。训练评估继续沿用原标注与口径；最终比较时统一正编号复评，避免混用。关注总AP/AP50/AP75和bicycle、truck、people召回，以及awning-tricycle、van的AP。

已验证：新增模型测试9/9、直接受影响的既有测试41/41、配置与提交链测试3/3通过；六份配置除实验开关和名称/输出路径外，与对应E0配置一致。Python与Slurm语法检查通过。

真实GPU预检要求每组完成两次有效AMP优化更新；初始scale溢出允许正常降scale并跳过更新，最多尝试12批；非有限loss或持续溢出失败。原delta零初始化，因此第一步门控梯度可以为0。预检不保存checkpoint，不影响正式训练初始化。

GPU预检作业16817在gpu6完成（COMPLETED，退出码0:0，68秒）。三组均在5批内完成两次有效更新；其余3批由AMP正常降scale并跳过。峰值allocated显存约9.13 GiB/卡（单卡batch4）；E1/E3第二次有效更新中六层门控末层均获得有限非零梯度。预检JSON位于 `output/visdrone_gate_reference_20261010/gate_preflight/`。

## 已提交的正式训练

| 组别 | 作业ID | 启动条件 | 输出子目录 |
| --- | --- | --- | --- |
| E1 | 16818 | 调度到资源后启动 | `g5_adaptive_band_gate` |
| E2 | 16819 | E1成功结束 | `g5_adaptive_band_pre_reference` |
| E3 | 16820 | E2成功结束 | `g5_adaptive_band_gate_pre_reference` |

每个作业申请8GPU，依次完成80轮和计划内40轮续训；设置no-requeue与afterok依赖。输出根目录：`output/visdrone_gate_reference_20261010/results/visdrone/seed0/`；每组分别保存 `epoch80/` 和 `continue120/`。源码已冻结在同一运行根目录的 `source/`，完整作业、日志路径见 `jobs.json`，实验定义见 `experiment_selection.json`。

提交后确认：E1在gpu6为RUNNING，已完成epoch0第0批，训练loss48.9670，未见训练异常；E2/E3为PENDING（Dependency）。已有暂存文件未改动，新增改动未暂存、未commit。

最终选取各组两阶段中表现最佳的EMA权重，再按统一正编号口径比较。当前尚无新实验AP结果，不据预检loss推断最终性能。
