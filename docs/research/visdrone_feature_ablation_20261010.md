# VisDrone 已有 checkpoint 的视觉特征消融实施计划

> 执行方式：当前会话按 subagent-driven-development 分工，局部验证后启动实验；不自动提交本轮新增代码。

**目标：** 补齐 VisDrone 对候选视觉特征的依赖证据。UAVSwarm 已有正向依赖结论，旧随机化定义未找到；用户确认本次按随机噪声定义，不声称与旧实现完全等价。

**实现：** 独立推理诊断工具在六个 strip 头的 `edge_fusion` 输出处临时挂钩，替换已编码候选视觉向量，随后沿用 query 通道调制和原 logits 残差。正常/置零/标准正态噪声三种模式使用同一最佳 EMA；只评估，不训练或改变 checkpoint。

**技术：** PyTorch 推理 hook、现有 YAML/EMA 加载、COCO evaluator、固定几何 Hungarian 配对、Slurm 单 GPU。

## 共同约束

- 现有工作已提交：361dee8（前两层结果）、fb5db9a（续训与结果）；在当前实验分支继续工作。
- 权重固定为 `output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth`，EMA，epoch96，历史 AP=28.4190。
- 保留 stride8、自适应采样带、六层 decoder、候选位置、有效性掩码、query 调制和原 logits 残差；仅替换候选视觉向量。后层状态随前层预测自然递推。
- `normal` 原样输出，`zero` 全零，`noise` 独立 N(0,1)。独立随机生成器，基础种子0，每批 `seed + batch_index * 1009`，层内再偏移；固定验证顺序和 batch，记录设置，不宣称与旧 UAVSwarm 随机化完全一致。
- 当前 checkpoint 不重训，不延长训练，不恢复16709；新输出使用独立目录，不覆盖已有实验。
- AP/AP75 按原 COCO maxDets=100；几何匹配仅由正常输出计算一次，扰动模式沿用相同 query–GT 配对。GT 只用于指标。

## 任务1：候选视觉干预

文件：`tools/analysis/candidate_feature_ablation.py`、`tests/test_candidate_feature_ablation.py`。

- [x] 先写局部测试并观察失败：正常完全等价；置零只清空评分输入中的视觉部分；噪声种子复现且不消耗全局 RNG；正常退出和异常退出都移除 hook；拒绝训练模式。
- [x] 实现 `candidate_feature_ablation(model, mode, seed=0)` 上下文管理器，模式 `normal/zero/noise`。仅对 `model.refiner.heads[*].edge_fusion` 注册 forward hook；替换输出而不原地修改共享特征，保留其余输入。每层独立随机源，层种子为 `seed + 1000003 * layer_index`。
- [x] 运行 `PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/zzy/.conda/envs/dfine/bin/python -m unittest test_candidate_feature_ablation -v`。

## 任务2：完整验证与结果汇总

文件：`tools/analysis/diagnose_candidate_features.py`，必要局部测试。

- [x] 复用 `load_ema_model`，严格载入最佳 EMA，关闭预训练下载；单一验证 loader，FP32。
- [x] 每批依次正常、置零、噪声推理；各自独立 COCO evaluator。正常最终框做纯几何 Hungarian 匹配后，复用 `LayerEffectMetrics` 计算正常→扰动的四边MAE、IoU及原有短边/类别分组；不重新匹配。
- [x] 输出 JSON 包含 checkpoint/epoch/模式/噪声/精度/数据集/数量、各模式常规12项 COCO 指标、相对正常的百分点差、固定配对几何结果。保存轻量最终预测缓存供复核，不保存中间大张量。
- [x] 第一批检查 normal hook 与普通推理完全一致，退出干预后恢复普通结果；拒绝非有限输出。全验证正常 AP 与历史28.4190比较，若明显不一致则标记并先核对口径。

## 任务3：冻结源码并运行

文件：`slurm_files/run_diagnose_visdrone_features.slurm`。

- [x] 局部测试、CLI和Slurm语法检查；独立只读检查本次新增内容。
- [x] 使用现有 freeze_source 保存 src/configs/数据配置，再复制本轮诊断脚本及依赖到新 `output/visdrone_feature_ablation_20261010/source/`。
- [x] 单 GPU、FP32、batch4、4 workers、4 threads，启动一次包含三模式的完整验证，记录 job ID、源码和日志路径。
- [x] 汇总实际结果，明确依赖证据与相对G0收益是不同问题。未获批准不自动启动后续训练或更多扰动种子。

## 实施验证

特征干预测试先因缺少模块失败，实现后5项通过；固定几何配对与非法预测测试先因缺少模块失败，实现后2项通过。追加一项完整合成单图验证，真实执行小模型、三种干预、后处理、COCO评估、固定配对及缓存写出；最终合计8项通过。Slurm脚本通过 `bash -n`，CLI帮助可正常运行。

置零测试确认首层评分输入中的视觉维被清空、非视觉维及原分布输入保持原样，六层均受干预；随机噪声可按种子重现，不消耗全局随机源。首批正常/恢复检查在实际完整验证中也会执行。

```bash
PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp /home/zzy/.conda/envs/dfine/bin/python -m unittest test_candidate_feature_ablation test_candidate_feature_diagnostic -v
```

## 作业记录

- Job ID：**16808**，单GPU，启动节点 gpu6。
- 批次：`output/visdrone_feature_ablation_20261010/`，任务与配置分别见 `jobs.json`、`experiment_selection.json`。
- 冻结源码：上述批次的 `source/`，包含实际执行的诊断脚本。
- 日志：`logs/features_16808.out`、`logs/features_16808.err`。
- 输出：`results/summary.json`、`results/predictions.pth`。
- 已确认载入全部548张验证图，真实首批推理完成，正常模式与移除干预后的普通推理逐元素完全一致。
- 独立只读检查通过；本轮新增代码不自动提交，不恢复旧失败训练。

实际执行入口（在冻结源码目录运行，路径参数见 experiment_selection.json）：

```bash
python tools/analysis/diagnose_candidate_features.py \
  --config configs/controlled/visdrone_g5_adaptive_band_continue120.yml \
  --checkpoint /home/zzy/projects/occ_study/D-FINE_mine/D-FINE/output/visdrone_continue120_20261009/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth \
  --output-file /home/zzy/projects/occ_study/D-FINE_mine/D-FINE/output/visdrone_feature_ablation_20261010/results/summary.json \
  --prediction-cache /home/zzy/projects/occ_study/D-FINE_mine/D-FINE/output/visdrone_feature_ablation_20261010/results/predictions.pth \
  --batch-size 4 --workers 4 --threads 4 --device cuda --seed 0 \
  --expected-ap 28.4190 --ap-tolerance 0.02
```

## 完整验证结果

作业16808已 **COMPLETED，ExitCode=0:0**，耗时3分42秒。实际推理与汇总耗时210.5秒，覆盖全部548张验证图、640×640输入、38,759个有效GT；固定正常输出的38,725个query–GT配对，各模式无非法配对。少量GT未匹配来自300 query数量限制。以下AP、AP75为百分数，差值为百分点；四边MAE为输入像素。

| 模式 | AP | ΔAP | AP75 | ΔAP75 | 四边MAE | ΔMAE | 匹配IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 正常 | 28.4188 | 0 | 28.2322 | 0 | 2.461405 | 0 | 0.688215 |
| 特征置零 | 28.0539 | −0.3649 | 27.6973 | −0.5349 | 2.522334 | +0.060930 | 0.682407 |
| 随机噪声 | 26.1703 | −2.2486 | 25.2354 | −2.9968 | 2.653929 | +0.192525 | 0.665582 |

正常AP与历史28.4190相差−0.000171个百分点，参考核验通过。置零和噪声分别有56.66%、65.08%的固定配对IoU下降；按GT输入短边分组的MAE变化也均为退化：

| 输入短边 | 配对数 | 置零ΔMAE | 噪声ΔMAE |
| --- | ---: | ---: | ---: |
| <8像素 | 15,861 | +0.068966 | +0.136726 |
| 8至<16像素 | 11,418 | +0.042915 | +0.163594 |
| 16至<32像素 | 8,050 | +0.041627 | +0.215605 |
| ≥32像素 | 3,396 | +0.129720 | +0.495691 |

**结论：当前VisDrone最佳六层checkpoint确实依赖候选视觉特征。** 清除特征后AP、AP75与固定配对定位均变差，因此不应再把“完全没有使用候选视觉信息”作为当前退化的主要解释。

这只说明该已训练模型对这一路输入的依赖，不等价于G5优于G0。原比较中G5仍比G0低0.2037 AP；UAVSwarm旧实验的具体干预定义未找到，不能用本次掉点幅度直接比较两个数据集的依赖强度。标准正态噪声会改变真实特征的取值分布，其掉点也不能直接解释为视觉分支的贡献大小。本次只使用一个checkpoint及一个噪声种子，没有训练新模型。

精简可归档结果：[visdrone_feature_ablation_20261010.json](results/visdrone_feature_ablation_20261010.json)。完整结果与最终预测缓存分别位于批次的 `results/summary.json`、`results/predictions.pth`，包含其余COCO指标、原图面积与类别几何分组。
