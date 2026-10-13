# VisDrone GT最近候选oracle实验

用户已批准补充第二种oracle。复用历史UAVSwarm作业15433的 `DFINEGTCandidateRefiner`，GT逐边从原33个候选中选最近有效位置，保持预测框切向跨度及学习式残差修正。冻结原检测器和分类分数，仅训练新增refiner。结果用于分析候选机制的定位潜力，不视为无GT检测性能或严格上界。

## 配方与对照

- 原模型实现不变，640输入、现有stride8、hidden_dim64、沿边5点、内/边/外3点；不接入六层G5或新增高分辨率特征。
- 沿用第一种VisDrone oracle的G0最佳EMA：`output/visdrone_continue120_20261009/results/visdrone/seed0/g0/best_stg2.pth`。全新初始化候选修正头，不继承第一种oracle的头或优化器。
- 12轮、seed0、8GPU、总batch96、AMP；AdamW lr0.00025、weight_decay0.000125，warmup50、milestone10/gamma0.1；无EMA和SyncBN。
- 使用第一种oracle同一训练标注、同一正ID验证标注副本。现有基线复评AP28.6235，第一种oracle最佳epoch11 AP36.0279/AP75 39.2931；基线结果路径为 `output/visdrone_gt_strip_oracle_20261013/results/baseline/oracle_eval.json`，不重复训练基线。
- 比较AP/AP50/AP75/尺寸指标、固定配对IoU/边误差及有效修正比例。两种oracle差异同时涉及候选离散位置、预测切向跨度与有效采样，不能全部归因为选择器。

## 执行计划

- [x] 新增VisDrone候选配置 `configs/dfine/gt_refinement/dfine_hgnetv2_l_visdrone_gt_candidate_oracle.yml`，对比解析后配方，确保只适配类别、数据和输出路径。
- [x] 最小扩展 `tools/analysis/check_visdrone_gt_oracle.py` 以显式选择候选模型；保持原边界模型默认行为。运行现有候选局部测试，新增必要的模型选择验证。
- [x] 新增 `slurm_files/run_visdrone_gt_candidate_oracle.slurm`；冻结本次源码与同一验证标注至独立输出目录。运行阶段曾要求既有暂存内容不动、不提交commit；当前用户已授权本阶段整理与中文commit。
- [x] 单GPU batch12短预检：初始预测与原检测器一致、分类分数固定、只有refiner更新、两次有效AMP更新和显存检查。
- [x] 预检成功后执行12轮训练，已完成epoch0–11；预检/训练失败不自动重试。

## 执行记录（历史排队状态）

运行目录：`output/visdrone_gt_candidate_oracle_20261013/`；冻结源码为 `source/`，正式训练输出为 `results/oracle/`。配方、标注来源、基线及第一种结果位置记于 `experiment_selection.json`，独立加载并核对的配置保存于 `resolved_config.json`。

必要验证：历史候选模型架构/优化器/scheduler不变；本次与第一种VisDrone的完整train/val dataloader及训练配方相同；训练标注逐字节一致，正ID验证标注直接复制自第一种。现有候选头、候选模型、配置及预检工具局部测试共16项通过；两份Slurm脚本语法检查通过，训练入口局部核对通过。

已提交预检17103和正式训练17104，后者依赖 `afterok:17103` 并设置无效依赖时取消。训练入口进一步核对报告成功、候选模型类型、两次有效更新、batch12及相同checkpoint/配置。提交命令详见 `preflight_job.json`、`jobs.json`。

截至2026-10-13本次提交检查：17103为 `PENDING (Priority)`，17104为 `PENDING (Dependency)`；尚无GPU预检结果，正式训练尚未开始。失败不自动重试。

后续按用户批准，将预检17103的主机内存原地更新为32768MB（32GB），并在 `run_preflight.slurm` 显式设置 `--mem=32G`。原因是分区默认内存为整节点，原请求约1TB限制了共享节点调度。作业号及17104的成功依赖保持不变。

## 最终结果（2026-10-13）

17103预检通过：batch12、两次有效AMP更新，初始框误差0、分类分数和冻结张量不变、仅refiner有梯度；峰值allocated/reserved显存1.5850/2.0078 GiB。17104已完成epoch0–11，最佳epoch11。

最佳AP/AP50/AP75为36.9038/51.4178/41.2042，AP小/中/大为28.0329/48.3326/60.8490；相对同口径基线提升8.2803 AP，相对条带oracle提升0.8760 AP。固定38725对的IoU由0.675676增至0.767173，四边平均误差由2.792260降至2.104482输入像素；全部38725对有效修正，无效采样0，未匹配GT34个。

最佳checkpoint：`output/visdrone_gt_candidate_oracle_20261013/results/oracle/best_stg1.pth`（epoch11，另有`checkpoint0011.pth`）；指标来源为同目录`log.txt`，预检来源为运行目录`preflight.json`。完整路径、逐边误差、尺寸与计数见[阶段汇总](visdrone_stage_results_20261013.md)及其JSON。上述排队描述保留为早期历史记录。
