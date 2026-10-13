# VisDrone 最佳 checkpoint 的修正强度对照

> 当前会话采用 subagent-driven-development 分工执行；沿用 writing-plans、test-driven-development 和 verification-before-completion。仅验证，不训练、不自动提交。

**目标：** 区分新增修正是否整体有害或过强。用户已批准在相同最佳 EMA 上比较 alpha=0、0.5、1。

**实现：** 在每层 strip 评分 MLP 的输出处乘 alpha，实际前向为 `原logits + alpha * 有效候选修正量`。模型后续正常递推。alpha=0关闭整个新增修正；它不等价于视觉特征置零，也不等价于另行训练的G0。

**技术：** 推理 hook、PyTorch、现有COCO及固定配对几何指标、单GPU Slurm。

## 共同约束

- 最佳权重固定为 `output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth`，EMA，epoch96。
- 新跑alpha=0、0.5；alpha=1使用作业16808的正常结果（AP28.418829），复用 `output/visdrone_feature_ablation_20261010/results/summary.json` 和 `predictions.pth`，不重复全量正常推理。
- 只缩放新增delta，不缩放原logits、不替换候选视觉特征。六层、stride8、query调制、有效性掩码及原分布残差保留。
- 单GPU、FP32、batch4、同一548图验证集。第一批恢复alpha=1后与缓存正常输出核对；按image_id取缓存，并核对图像尺寸、GT和checkpoint元数据。
- 所有模式复用作业16808的正常query–GT配对，不重新匹配；原COCO maxDets=100。GT仅用于指标。
- 不修改先前未提交内容的实验定义，不覆盖旧输出，不恢复16709，不启动训练或其他alpha。

## 任务1：修正强度hook

文件：`tools/analysis/strip_correction_scale.py`、`tests/test_strip_correction_scale.py`。

- [x] 先写测试观察失败；实现 `strip_correction_scale(model, alpha)` 上下文，仅hook各 `head.score_head` 输出并乘alpha；alpha=1原样返回，退出移除hook，拒绝训练/GT模型及非有限或负alpha。
- [x] 用小型真模型和非零评分头验证alpha=1完全一致、alpha=0等于同一模型原生decoder路径、首层alpha=0.5仅将新增delta减半并保留prior，正常/异常退出后恢复。

## 任务2：复用正常缓存的验证入口

文件：`tools/analysis/diagnose_correction_scale.py`、必要局部测试。

- [x] 加载现有正常summary/cache，校验模型、全验证和精度；复用每图normal输出与固定配对，核对当前样本ID、尺寸和GT。
- [x] 仅跑0与0.5两次推理；第一批额外检查恢复后的alpha=1与缓存一致。
- [x] 复用已有COCO与LayerEffectMetrics，输出全部常规12项指标、相对alpha=1差值、固定配对几何结果与最终预测缓存。
- [x] 局部验证缓存对齐、强度hook及完整合成样本评估流程。

## 任务3：运行及汇总

文件：`slurm_files/run_diagnose_visdrone_correction_scale.slurm`。

- [x] 局部测试、Slurm语法、独立只读检查通过后冻结源码。
- [x] 输出根目录 `output/visdrone_correction_scale_20261010/`，单GPU完成两种强度的完整验证。
- [x] 报告真实结果，判断是否支持简单降低修正强度；不由此自动增加训练或结构实验。

## 实施与作业记录

- 修正强度hook及缓存对齐测试先因模块缺失失败，实现后通过；加入真实COCO合成样本的“特征诊断生成缓存→强度诊断复用缓存”全流程测试。最终10项局部测试通过。
- 已验证alpha=0与同权重原生decoder路径一致，alpha=1与普通推理完全一致，alpha=0.5只减半新增delta；六层hook正常/异常退出均清理。
- CLI及Slurm语法检查通过，独立只读检查未发现必须修复的问题。
- 作业：**16809**，单GPU；输出根目录 `output/visdrone_correction_scale_20261010/`。
- 冻结源码：`source/`；日志：`logs/scales_16809.out`、`logs/scales_16809.err`。
- 结果：`results/summary.json`、`results/predictions.pth`；实际设置与任务信息见 `experiment_selection.json`、`jobs.json`。
- 本轮未更新模型权重、未提交commit。上一项特征消融结果及冻结源码保留。

局部验证命令：

```bash
PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp /home/zzy/.conda/envs/dfine/bin/python -m unittest test_strip_correction_scale test_correction_scale_diagnostic test_candidate_feature_diagnostic -v
```

## 完整验证结果

作业16809在gpu14完成，**COMPLETED，ExitCode=0:0**，耗时2分37秒；实际推理与汇总144.0秒。新跑alpha=0、0.5，复用16808的alpha=1结果。首批恢复alpha=1与缓存输出核验通过（atol/rtol均为1e-6）。全部548图、640×640输入、38,759个有效GT、38,725个固定query–GT配对，所有模式无非法配对。

| alpha | AP | 相对alpha=1的ΔAP | AP75 | ΔAP75 | 四边MAE（输入像素） | ΔMAE | 匹配IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0：关闭新增修正 | 28.1541 | −0.2647 | 27.8124 | −0.4198 | 2.503480 | +0.042075 | 0.683832 |
| 0.5：修正减半 | 28.3303 | −0.0886 | 28.1891 | −0.0431 | 2.472209 | +0.010804 | 0.687023 |
| 1：原强度，复用 | 28.4188 | 0 | 28.2322 | 0 | 2.461405 | 0 | 0.688215 |

AP/AP75为百分数，差值为百分点。APsmall也呈0→0.5→1逐步提高：19.0098→19.2062→19.2514；AR100为42.3264→42.6530→42.8436。

**结论：这两个减弱设置都未改善结果，不支持通过统一减弱或关闭修正解决当前VisDrone退化。** 原强度在三组中AP、AP75、四边MAE和IoU均最好。此前“整体修正过强”的优先猜测未得到支持；单checkpoint、两个减弱取值的结果，不能排除按目标选择性控制修正的可能性，也不证明训练门控会有效。

与已经保存的G0最佳AP28.6227对照，当前G5关闭修正后的AP低约0.4686个百分点，开启修正恢复约0.2647个百分点，仍低约0.2039个百分点。这是同权重推理干预与独立模型结果的描述，不能把alpha=0称为原版G0，也不能据此证明联合训练必然损坏了原检测器。后续若继续排查，应比较G0/G5的漏检、误检与定位错误构成，而非直接增加修正强度实验；本轮没有自动开展该诊断。

精简归档：[visdrone_correction_scale_20261010.json](results/visdrone_correction_scale_20261010.json)。完整指标、尺寸/类别几何分组和新预测缓存保存在本批次的 `results/summary.json`、`results/predictions.pth`，正常预测及配对仍引用作业16808缓存。
