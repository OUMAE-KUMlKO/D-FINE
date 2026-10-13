# VisDrone 预测合法性保护与数值异常诊断

用户已批准补齐最终修正分布的合法性检查、回退记录及短诊断。六层G5、门控、E1–E3参考框设置、LQE及训练预算保持既定定义；不增加修正幅度限制，不自动重启失败训练，不修改已暂存文件或commit。旧运行源码与checkpoint只读保留。

## 已知证据

E1作业16818完成epoch0–91，在epoch92输出NaN及负高度预测框，触发GIoU断言；后续NCCL超时是连带结果。E2/3依赖取消、未启动。epoch91完整状态可读，epoch89为当前最佳（训练口径AP28.355244）。`source/NaN.pth`仅含模型；模型参数有限，无效anchor的预期+inf与mask对应。原失败批次输入/RNG未保存，不能保证精确复现。

## 实现与验证计划

- [x] 模型：先新增局部回归测试，覆盖正常路径不变、实际使用精度转换溢出、非正宽高、整query四边回退、原分布异常停止、两种参考来源与LQE/累计一致、有效梯度。再在distribution_refiner路径接受修正的位置检查有限logits与实际解码框。只检查matching queries，DN保持原路径；无refiner不改变行为；不增加checkpoint状态。
- [x] 记录：decoder按层统计最终回退，strip wrapper按层记录原有head拒绝/输入无效计数；每forward重置，checkpoint重计算不重复累加。新增PredictionSafetyError携带层、阶段、DN数量及必要张量，异常query编号保留完整序号。
- [x] 训练现场：先测试错误现场保存和同步停止；训练前向捕获上述异常，在进入criterion前同步失败标志。失败rank保存输入、targets、模型和错误张量，所有rank停止，避免单rank报错后其余rank继续进入损失同步。成功batch记录按层计数。仅启用保护的G5路径增加这些行为。
- [x] 短诊断：新脚本只读取NaN模型权重与旧配置，单GPU、batch4、epoch92增强、最多16批/180秒前向，不建optimizer、不更新权重；首异常批保存输入，比较AMP和FP32（恢复buffers/RNG后同输入重放）。现场明确记录新模型源码与旧配置来源。无异常只报告该小样本未复现。
- [x] 运行局部测试、有界GPU诊断及同批原源码精度对照，记录结果及未确定部分。训练恢复取决于诊断，不因短测试通过宣称长期训练已修复。

关键位置：`src/zoo/dfine/dfine_decoder.py`、`all_layer_strip.py`、新增`prediction_safety.py`；`src/solver/det_engine.py`与现场保存辅助文件；专用诊断工具/Slurm及新局部测试。

## 执行结果

运行目录：`output/visdrone_prediction_safety_20261011/`。本节记录首次诊断结束时的状态，当时训练未恢复，旧源码和checkpoint未改动；后续修正与恢复见文末。

1. GPU短诊断16986（gpu14，COMPLETED，16秒）：首批4张图、640输入，在第2层（零基layer_index=1）本层strip修正前发现负高度。AMP及FP32均触发。实际检测候选共1200个，其中407个非法；另有410个DN查询非法，含填充项。首批输入、RNG、buffers和张量已保存为 `diagnostic/event_batch.pth`。原始检查曾包含DN，随后收窄保护范围为matching queries，保持native DN原路径，防止把不参与损失的padding误判为致命异常。
2. 原冻结源码同批完整前向16987：从原NaN权重分别运行AMP/FP32各一次，只前向，不更新参数。每次恢复相同的批前buffers/RNG。两者最早的实际候选负高度框均位于第2层本层修正前；最终AMP有413/1200个负高度框，FP32有410/1200个。两次均未出现NaN/Inf。后续核对发现native cxcywh→xyxy会将负宽高截为零，因此这仅复现了raw框负高度，不能宣称复现GIoU断言或原批次NaN。详见 `replay_original.json`。

这里的“本层修正前”包含前层G5的累计影响，不能据此把责任单独归给原生D-FINE，也不能排除门控影响。直接换FP32仍存在非法框；仅在本层strip输出上回退无法修复已无效的prior。当前策略是保存现场并停止，尚不具备确认长期稳定后恢复训练的证据。

后续恢复预检确认了一个诊断限制：上述16986/16987使用单GPU训练模式BN，而原训练为8GPU SyncBatchNorm。因此同批AMP/FP32对照仍有效，但这些单卡结果不能证明原8卡统计方式下必然出现相同非法框，也不能单独判定epoch91完整断点不可续训。用户随后批准优先恢复、不可恢复则从头训练，执行记录见 `visdrone_gate_reference_recovery_20261011.md`。

训练异常快照保存前向前torch RNG、输入及失败前向后的模型/buffers，标记为诊断文件而非完整续训checkpoint；BN等buffer若更新，不能据此承诺严格逐步重放。专用同批诊断另保存了前向前buffers以进行公平精度对照。

## 最小充分验证

- 初版模型新增与直接影响测试30项通过；DN范围修正后的新增及直接相关测试17项通过，覆盖无效DN不触发、native DN逐值一致、matching异常仍捕获。
- 训练现场/成功前向/引擎统计测试通过；本地真实双进程Gloo验证rank1失败时两rank均停止，只有rank1保存现场。共4项通过，DN修正后重跑引擎接入测试通过。最初沙箱禁止回环通信，获准后在本机回环完成双进程测试，无GPU训练。
- 诊断工具4项CPU测试、CLI及Slurm语法检查通过。
- 保存的GPU张量用最终matching-only检查在CPU复核，恰好报告407个非法matching查询；410个DN排除，完整query编号保留（最小213，DN前缀180）。无需重复GPU扫描。
- `git diff --check`与修改文件Python语法检查通过。未改原暂存内容，未commit；未恢复E1，也未重提E2/E3。

相关单测：`tests/test_prediction_safety.py`、`tests/test_prediction_safety_diagnostics.py`、`tests/test_prediction_safety_diagnostic.py`。最后一个文件测试独立诊断工具，复数文件测试训练引擎现场保存及集体停止。

## 后续修正

16990 fresh训练在epoch0 step183被有限负高度prior检查提前中止，证明初版把中间框正尺寸作为致命条件过严。修复后：finite非正prior保持native路径并按层计数，真实strip head沿用原有跳过策略；NaN/Inf prior仍保存现场、同步停止；非法G5修正仍整query回退原有限分布，不额外改变raw框或累计logits。DN、LQE、六层及参考框实验设置不变。

原epoch91完整断点在修正后的八卡预检16997中通过全部rank的EMA前向和两次有效AMP更新，已提交E1续训16998、E2 fresh16999、E3 fresh17000。原断点此前触发几何检查并不代表状态损坏。完整执行记录见 `visdrone_gate_reference_recovery_20261011.md`。
