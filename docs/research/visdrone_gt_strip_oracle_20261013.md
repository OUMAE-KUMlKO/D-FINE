# VisDrone GT边界条带oracle实验

用户选择复现UAVSwarm第一种oracle：GT四边条带采样，冻结检测器，仅训练新增框修正头。E1–E3暂停进一步分析或修改。运行阶段曾要求不提交commit、不改既有暂存内容；当前用户已授权本阶段整理与中文commit。

## 已确认的结构与配方

复用强基线UAVSwarm G1（15347）的 `DFINEGTRefiner` / `GTStripHead`，不使用GT最近候选模型，也不接入六层G5改动。GT仅决定匹配和条带采样位置，不将GT坐标/误差直接输入修正头，不用GT框替换输出。

- HGNetv2-B4、原Encoder/Decoder/分类头全部冻结并保持eval；只训练全新refiner，分类分数固定，保留全部300 queries。未匹配或采样无效的候选保留原框。
- 最高分辨率原Encoder输出stride8，hidden_dim64，沿边5点、内/边/外3点、kernel3、offset1输入像素、max_log_scale2；模型结构不变。
- 固定640×640，12 epochs（0–11），8GPU、总batch96、seed0、AMP；不使用EMA或SyncBN。
- AdamW lr0.00025、betas[0.9,0.999]、weight_decay0.000125；warmup50，MultiStepLR milestone10/gamma0.1，沿用原scheduler语义。
- GTRefinementCriterion：L1框损失×5、GIoU×2，不训练分类，不新增分布损失。

冻结基线使用VisDrone原D-FINE G0最佳EMA（epoch96）：`output/visdrone_continue120_20261009/results/visdrone/seed0/g0/best_stg2.pth`。只加载检测器权重，不继承其optimizer或训练轮数。类别数和数据路径适配为VisDrone 10类。

## 数据与对照

沿用已准备的VisDrone训练/验证集和ignore/crowd约定。仅在新运行冻结目录内生成验证标注副本，将annotation ID加1，消除ID0与COCO“未匹配”标记冲突；图像ID、框、类别、面积及原标注不变。基线和oracle统一使用该副本。

先用 `feature_mode=baseline` 复评冻结检测器：GT仅用于固定配对几何统计，不改变原预测或分数。再从新初始化head训练12轮oracle。报告oracle AP/AP50/AP75、尺寸指标、固定配对IoU/边误差、有效修正比例与未匹配GT；不能作为不使用GT的检测性能或严格性能上界。UAVSwarm旧结果使用旧数据口径，比较时重点看各数据集相对自己基线的变化。

## 执行清单

- [x] 核对UAVSwarm真实运行配方及模型；新增仅覆盖数据/类别/输出路径的配置，保持batch96。
- [x] 对照解析后的配置，确认模型、结构、优化器、scheduler及增广均与UAVSwarm原配方一致；现有GT模型17项测试、新增预检工具2项测试通过，Slurm语法检查通过。
- [x] 冻结新运行源码及修正后的验证标注副本，记录配方与来源。
- [x] 单GPU batch12预检初始等价、只训练refiner、冻结张量不变、两次有效AMP更新及显存。更新仅在内存，不保存训练checkpoint。
- [x] 提交依赖预检成功的同口径基线复评及12轮训练作业。
- [x] 确认GPU预检通过、正式训练实际启动和首轮指标；已完成全部12轮。

关键文件：`configs/dfine/gt_refinement/dfine_hgnetv2_l_visdrone_gt_stage1.yml`、`slurm_files/run_visdrone_gt_strip_oracle.slurm`、`tools/analysis/check_visdrone_gt_oracle.py`。完整结果见文末及[阶段汇总](visdrone_stage_results_20261013.md)。

## 提交记录（历史排队状态）

运行目录：`output/visdrone_gt_strip_oracle_20261013/`。冻结源码位于 `source/`，配方及验证标注转换记录见 `experiment_selection.json`。预检作业17077已提交，当前因 `AssocGrpGRES`（账户组GPU资源配额）排队，尚无GPU预检结果。

正式作业采用afterok依赖：仅预检成功才执行基线复评和12轮oracle训练；入口还读取预检JSON核对成功、两次有效更新、batch12及相同基线。依赖失败则取消后续作业，不自动重试，不占用或终止其他实验。

正式作业号17079，当前为 `PENDING (Dependency)`；预检17077为 `PENDING (AssocGrpGRES)`。提交命令和结果路径记录在 `jobs.json`，正式结果将写入 `results/baseline/` 与 `results/oracle/`。截至本次记录，GPU预检及正式训练均尚未开始。

## 最终结果（2026-10-13）

17077预检通过：batch12、两次有效AMP更新，初始框误差0、分类分数和冻结张量不变、仅refiner有梯度；峰值allocated/reserved显存1.3995/1.5645 GiB。17079已完成同口径基线复评和epoch0–11训练，最佳epoch11。

最佳AP/AP50/AP75为36.0279/50.9625/39.2931，AP小/中/大为26.7919/48.0231/61.5060；相对同口径基线AP28.6235提升7.4043 AP。固定38725对的IoU由0.675676增至0.749658，四边平均误差由2.792260降至2.292977输入像素；有效修正37260对，采样无效1465对，未匹配GT34个。

最佳checkpoint：`output/visdrone_gt_strip_oracle_20261013/results/oracle/best_stg1.pth`（epoch11，另有`checkpoint0011.pth`）；指标来源为同目录`log.txt`。基线来源：`results/baseline/oracle_eval.json`；预检来源：运行目录`preflight.json`。完整路径、逐边误差、尺寸与计数见[阶段汇总](visdrone_stage_results_20261013.md)及其JSON。上述排队描述保留为早期历史记录。
