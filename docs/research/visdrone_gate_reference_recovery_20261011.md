# E1–E3 继续训练记录

当前状态：已修正过严的中间框检查，八卡预检16997通过；E1作业16998从原epoch91完整断点恢复，已完成epoch92训练及验证（AP28.374172），保存完整checkpoint并进入epoch93。E2作业16999、E3作业17000等待前序成功后从头训练。以下保留早先判断及失败记录，最终规则见“16990失败后的保护规则修正”。

用户已明确批准：优先从E1中断前完整checkpoint继续；若无法继续则从头训练。E2/E3此前未启动，仍需训练。E0复用，不重训；六层G5、LQE、stride与原训练配方不变。已暂存内容不改，不自动commit。

## 执行顺序

- [x] 以新增matching-query合法性保护读取E1 `continue120/last.pth`（epoch91），验证完整状态恢复及实际AMP更新能力；八卡训练首批仍触发保护，未完成参数更新，失败现场已保存。
- [x] 根据八卡预检失败，选择fresh：使用原backbone预训练重新训练E1的80+40轮。E2/E3从头训练80+40。新训练沿用batch32、8GPU、seed0及epoch100学习率衰减。
- [x] 冻结最终代码到新目录，保留旧epoch89最佳checkpoint及旧运行结果。此次为fresh重训，结果单独存放；恢复模式工具另支持历史最佳归档及最终AP比较。
- [x] 提交E1→E2→E3的afterok链，确认实际启动及首批日志。不设置失败自动循环重试。

## 执行结果

单GPU预检16988：E1 epoch91完整状态加载成功，EMA验证前向通过，但训练前向在第2层本层strip修正前触发非法框保护；E2/E3从头训练均完成两次有效AMP更新。单卡训练BN与原8GPU SyncBatchNorm不等价，不据此直接否定E1断点。

八卡预检16989：使用实际DDP及89个SyncBatchNorm模块、原total batch32、原workers和epoch92分布式采样；所有rank恢复完整状态并通过EMA前向，但首个训练batch的rank2/rank7在第2层本层strip修正前触发非法框保护。所有rank同步退出，未完成有效参数更新。结果目录：`output/visdrone_gate_reference_resume8_preflight_20261011/E1_resume/`，失败现场位于对应rank目录下的 `prediction_failures/epoch92_step0_rank*.pth`。

因此选择fresh重新训练E1，而非仅因单卡BN预检失败而放弃恢复；E2/E3同样从头训练。旧epoch91完整断点、epoch89最佳与旧实验目录只读保留。本结论仅说明当前断点在既定配方及保护下无法继续首批更新，不代表已确定原NaN的全部原因。

正式运行目录：`output/visdrone_gate_reference_recovery_20261011/`。`source/`为冻结源码，`experiment_selection.json`记录fresh原因及八卡预检证据，`jobs.json`记录依赖及输出位置；预检汇总亦复制为 `resume8_preflight_summary.json`。

| 实验 | 作业 | 方式 | 依赖 |
| --- | --- | --- | --- |
| E1 自适应门控 | 16990 | fresh，80+40轮 | 无 |
| E2 修正前框作为下一层参考 | 16991 | fresh，80+40轮 | afterok:16990 |
| E3 门控+修正前参考 | 16992 | fresh，80+40轮 | afterok:16991 |

提交后E1在gpu14通过模型加载和原生FLOPs统计，进入 `Epoch: [0/80] [0/202]`，首批loss为48.9670，日志未见预测合法性错误。E2/E3等待依赖。这里只确认训练已经启动，不代表完成120轮或已经证明长期稳定。

## 必要验证

- 恢复提交及原提交工具共8项单测通过；新增分布式预检工具5项CPU测试通过。
- Slurm语法、预检CLI及修改差异空白检查通过。
- 真实八卡预检验证了完整状态恢复、实际SyncBatchNorm转换、所有rank同步退出及失败rank现场保存；没有对原checkpoint写入。
- 未commit，未修改既有暂存内容。后续长训练是否稳定仍需实际运行确认。

## 16990失败后的保护规则修正

16990运行9分1秒，在epoch0 step183的rank0、第6层G5修正前触发了过严的几何检查；16991/16992依赖取消。现场仅1/1200个matching候选高度为负（-0.0001597404），无NaN/Inf。保存logits做FP32解码后仍为负（-0.0001605749），不是仅由最终半精度舍入造成。现场位于该运行的 `epoch80/prediction_failures/epoch0_step183_rank0.pth`。

此前把中间prior宽高非正作为致命错误不符合原模型语义：真实strip head已对该候选跳过修正并保留原分布，native cxcywh→xyxy在损失/后处理处将负宽高截为零。因此16989失败也不能证明epoch91完整断点不可续训。

本次用户已批准修改后训练。采用最小修正：仅非有限prior仍同步停止；有限非正尺寸prior记录计数并沿用原路径。G5新产生的非有限/非正尺寸修正仍整query回退到原有限分布。原始输出框、累计logits、LQE及下一层参考保持一致，不额外钳制raw box，不引入虚构分布或更高分辨率特征。该保护保证不接受非法G5修正，并不宣称所有原生raw中间框都严格正尺寸。

- [x] 回归测试先复现过早停止，再验证有限非正prior、修正回退、非有限停止及真实strip+loss+backward。模型保护11项通过；真实strip测试的E1/E2/E3三个分支均确认native输出一致、损失及梯度有限；参考框与训练引擎接入测试通过。
- [x] 原8GPU SyncBatchNorm、batch32下重新验证epoch91完整状态的EMA前向及有效AMP更新。16997的8个rank全部通过EMA前向及两次有效AMP更新，loss/gradient有限；实际89个SyncBatchNorm模块。
- [x] 预检通过后提交从epoch92恢复E1到119、E2/E3 fresh 80+40的依赖链；冻结新目录、保留旧最佳；已验证E1完整首轮训练及验证，后续训练仍在进行。

新增 `layer_N_prior_invalid_geometry_queries` 分开记录有限非正prior；`fallback_queries` 表示最终选择原分布的次数，也可能包含原本已非正的prior，不能把该计数全部解释为G5新产生的几何错误。旧的head输入跳过/解码回退计数保留。

修复后八卡预检：16997，源码与输出目录 `output/visdrone_gate_reference_prior_guard_preflight_20261011/`。复用原epoch91完整断点，仅作有界内存更新，不写训练checkpoint。

预检首批确实再次遇到旧16989失败时的rank2/rank7非正prior：两rank在第2–6层各计到1个候选；本次均完成参数更新而未被提前终止。两次更新的全局梯度范数分别为48.469772和43.061470，均有限（随后按原clip规则处理）。因此此次通过包含目标异常分支的实际执行。

采用resume模式恢复E1：epoch92–119，原完整model/optimizer/EMA/scaler/scheduler/warmup状态恢复，epoch100学习率衰减；归档原epoch89最佳，并在E1完成后比较历史最佳与本续训段最佳。E2/E3按原80+40轮从头训练。正式目录：`output/visdrone_gate_reference_prior_guard_recovery_20261011/`。

| 实验 | 本次作业 | 方式 | 依赖 |
| --- | --- | --- | --- |
| E1 自适应门控 | 16998 | resume，epoch92–119 | 无 |
| E2 修正前参考 | 16999 | fresh，80+40轮 | afterok:16998 |
| E3 门控+修正前参考 | 17000 | fresh，80+40轮 | afterok:16999 |

正式续训前完整EMA验证AP为28.351558（0.2835155789510674），与原epoch91记录一致。首轮epoch92的202个batch均完成，平均loss30.230595，验证AP28.374172、AP50 48.298416、AP75 28.123080；已进入epoch93。新 `continue120/last.pth` 已核实last_epoch=92，含model/optimizer/EMA/scaler/scheduler/warmup完整状态。该结果仅确认首轮恢复成功，尚未完成epoch119。原epoch89最佳已单独保存在新运行的 `results/visdrone/seed0/g5_adaptive_band_gate/archive/pre_resume_best_stg2.pth`。

epoch92各层有限非正prior计数（全8rank累计，非独立目标数）：90、10976、10139、8335、8615、8552。对应训练仍正常完成，说明该分支实际经过验证；不能把这些计数解释为G5新产生错误的数量。
