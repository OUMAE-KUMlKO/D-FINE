# 五个指定模型的独立归档与复评

本目录是五个用户指定模型的Git索引、结构说明及归档工具。完整权重和训练目录保存在集群归档目录中，不加入Git。

## 分支、提交与目录

- 当前归档分支：`archive/selected-models-20261021`。
- 整理阶段提交：`13d83be6c7cc9cdfee20e372afb6742a5629436b`，中文说明为“整理稀疏边界特征实验并归档原版D-FINE关闭蒸馏三种子结果”。
- 整理前研究分支：`research/visdrone-sparse-strip-query`。
- 五模型验收标签：`selected-models-20261021-v1`，固定本次五模型实际复评通过的版本。
- 代码仓库：`/home/zzy/projects/occ_study/D-FINE_mine/D-FINE`。
- 完整归档根目录：`/cluster/home/zzy/D-FINE/desired_results/`。
- 归档根目录的`VERSION.json`记录最终归档提交号；`README.md`、每个模型的`README.md`及`manifest.json`均注明本分支和标签。

切换到本分支用`git switch archive/selected-models-20261021`。后续研究开发应在研究分支继续；需要回到已验收的归档工具版本时用标签定位。

**五个模型分别使用各自归档目录下的`source/`。** 本分支当前顶层`src/`包含研究代码历史，不作为五个旧checkpoint共同的复评实现。每份冻结源码独立保存，后续研究分支修改不会被归档评估脚本导入。

## 选定模型和准确指标

以下指标来自各自原训练日志总AP最高的同一轮，单位为百分数。不是跨seed挑选的新对照表，也不表示每项都是该方法目前的最高单次结果。

| 归档目录名 | 数据集与方法 | 原作业 | seed | AP | 最佳权重 |
| --- | --- | ---: | ---: | ---: | --- |
| `visdrone_dfine_ddf_on_seed0_ap28.1196` | VisDrone原版，开启DDF | 15542 | 0 | 28.119640 | `training/best_stg2.pth`的EMA |
| `visdrone_dfine_ddf_off_seed1_ap27.8754` | VisDrone原版，关闭DDF | 17701 | 1 | 27.875368 | `training/best_stg2.pth`的EMA |
| `visdrone_g5_ddf_off_seed0_ap28.5186` | VisDrone六层G5，关闭DDF | 17479 | 0 | 28.518592 | `training/best_stg2.pth`的EMA |
| `visdrone_prediction_strip_seed0_ap28.9968` | VisDrone预测分布strip框修正 | 17132 | 0 | 28.996845 | `training/best_stg1.pth`的model |
| `uavswarm_g5_ddf_off_seed0_ap61.2316` | UAVSwarm六层G5，关闭DDF | 17361 | 0 | 61.231608 | `training/best_stg2.pth`的EMA |

第二个模型按用户明确指定保留seed1的27.8754，未替换成seed0的28.5011。UAVSwarm所选G5也不再是全部实验最高单次AP；原版关闭DDF seed2的61.3438属于另一实验，本次不纳入这五个目录。

## 五个模型的原始文件在哪里

所有以下相对路径均相对本代码仓库。归档后不依赖原实验output目录中的源码、标注或checkpoint。

| 模型 | 原完整训练目录 | 原冻结源码 |
| --- | --- | --- |
| VisDrone原版 | `output/controlled_20260926_212304/results/visdrone/g0/` | `output/controlled_20260926_212304/source/` |
| VisDrone原版关闭DDF | `output/dfine_baseline_ddf_off_20261020_153047/results/visdrone/seed1/` | `output/dfine_baseline_ddf_off_20261020_153047/source/` |
| VisDrone G5关闭DDF | `output/visdrone_g5_ddf_off_20261019_103206/results/g5_ddf_off/` | `output/visdrone_g5_ddf_off_20261019_103206/source/` |
| VisDrone预测分布strip | `output/visdrone_p2_prediction_20261013/results/p2/` | `output/visdrone_p2_prediction_20261013/source/` |
| UAVSwarm G5关闭DDF | `output/uavswarm_g5_ddf_ablation_20261016_145524/results/g5_ddf_off/` | `output/uavswarm_g5_ddf_ablation_20261016_145524/source/` |

每个原训练目录的`log.txt`保留在归档的`training/log.txt`，其余所有checkpoint、summary、样本图和eval产物也完整复制。外部Slurm日志补充到各归档`logs/`，原作业/源码来源记录补充到`metadata/`。

## 结构差别与实现位置

### 原版D-FINE及原版关闭DDF

HGNetv2-B4主干、HybridEncoder、六层原生D-FINE decoder，使用stride 8/16/32特征、300个query和每边33个分布bin。关闭DDF不改变推理结构：仅训练时DDF权重从1.5改为0，保留FGL=0.15及分类、L1、GIoU、辅助层和DN监督。

各自`source/src/zoo/dfine/dfine.py`、`dfine_decoder.py`与`dfine_criterion.py`保存模型和损失实现；训练配置位置详见各模型`manifest.json`的`config_rel`。

### 六层G5关闭DDF

类`DFINEAllLayerStrip`在六层decoder分别插入独立`StripProbabilityHead`。每条边的全部33候选都在stride8 encoder特征上读取5×7边界带，法向范围±8输入像素；64通道、DW3×3＋PW1×1编码，query生成通道门控调制候选特征。每候选MLP输出一个logit残差，加到原分布后重新解码，修正结果继续参与层间传递和LQE。`use_log_prior=False`，没有把视觉特征直接写回query。两个所选G5都是固定带宽版本，不包含后续稀疏采样或自适应修正量门控。

实现位于各自`source/src/zoo/dfine/all_layer_strip.py`、`strip_probability_head.py`及`dfine_decoder.py`。训练DDF=0，FGL=0.15。VisDrone为10类、UAVSwarm为1类，必须使用各自配置和权重。

### 预测分布加权strip框修正

类`DFINEPredRefiner`冻结完整原版检测器，只在第六层最终输出后执行一次`PredictionStripHead(sampling_mode=distribution)`。每边33候选，每候选沿边5点、法向内中外3列（±1输入像素）。先分别编码候选，再按原预测概率加权汇总；四边特征与query、框几何编码拼接，通过MLP预测中心偏移和宽高缩放。原分类分数保持，既不修改原边界分布，也不写回query。

实现位于归档`source/src/zoo/dfine/pred_refinement.py`、`pred_strip_head.py`和共用`gt_strip_head.py`。共用文件名含GT不表示这个模型推理用GT；本模型前向忽略targets，正常复评覆盖所有query，不使用matched-only oracle。

该实验从120轮预算的原版最佳checkpoint出发，冻结检测器再训练新增头12轮，只用L1、GIoU，不使用EMA。最终checkpoint包含完整检测器和修正头，归档配置将`DFINEPredRefiner.baseline_checkpoint`设为null，直接严格resume最终checkpoint，消除旧基线路径依赖。

## 归档内文件与运行方式

每个模型的目录结构：

```text
模型目录/
  README.md                  中文说明与本Git分支/标签
  manifest.json              原路径、模型身份、预期12指标与权重选择
  training/                 完整原训练目录（所有ckpt及日志）
  source/                   本实验自己的冻结源码与原始配置
  annotations/              实际train/val标注副本
  logs/                     外部训练控制台日志
  metadata/                 原运行协议、提交记录
  environment/              复评环境版本及pip依赖
  archive_eval.yml           独立展开评估配置
  eval.sh                    本目录独立运行入口
  summarize_eval.py          恢复12项COCO指标并与原日志比较
  evaluation/               新复评产物，latest指向最近一次
```

在有GPU的计算节点上执行：

```bash
cd /cluster/home/zzy/D-FINE/desired_results/visdrone_g5_ddf_off_seed0_ap28.5186
bash eval.sh
```

默认使用现有`/home/zzy/.conda/envs/dfine/bin/python`、8GPU和原验证总batch。可用`PYTHON_BIN=/其他环境/bin/python`切换环境，或`NPROC=1 bash eval.sh`单GPU验证（验证batch相应缩小）。可用`IMAGE_ROOT`指定同一验证集的图像目录；图像未复制到归档。

`CONFIG`和`CHECKPOINT`可覆盖配置与权重，`OUTPUT_DIR`可指定新的评估输出目录；默认输出带时间和作业号，不覆盖原训练日志。每个模型README也给出直接调用`source/train.py -c ... -r ... --test-only`的完整命令。

五个模型依次完整评估的集群入口为归档根目录的`run_evaluations.slurm`。提交时把`--output/--error`设置为根目录`logs/`中的新文件。

## 复现口径

- 原版VisDrone 28.1196保留历史验证标注ID（含0）；其他VisDrone模型保留各自正值ID标注。不能擅自统一标注后仍声称精确复现原数值。
- 前三项VisDrone及UAVSwarm G5使用EMA；预测分布strip使用普通model权重，关闭EMA。
- 验证输入640×640、原类别映射和后处理保持。正式验收采用8GPU原批次设置。
- UAVSwarm G5最佳外层日志epoch78，但checkpoint内部epoch70来自旧回载逻辑；以内存加载后的实际复评为验收依据。
- 不删除整个local损失分组，不改模型参数，不重新训练来补齐指标。
- `evaluation/latest/metrics.json`包含12指标、逐项差值和两位小数AP是否吻合；最终总表保存为本分支`results.json`及归档根目录同名文件。

## 实际复评验收（2026-10-21）

作业`17770`在`gpu6`上使用8张RTX 3090完成全部五模型评估，状态`COMPLETED`、退出码`0:0`、耗时4分1秒。**五个模型的12项COCO指标均与原训练最佳日志完全一致**，并非仅四舍五入后吻合。

| 模型 | 复评AP | 复评AP50 | 复评AP75 | 12指标最大绝对差值 |
| --- | ---: | ---: | ---: | ---: |
| VisDrone 原版 D-FINE（DDF开启） | 28.119640 | 47.853372 | 27.965542 | 0.000000 |
| VisDrone 原版 D-FINE（DDF关闭，seed1） | 27.875368 | 47.420996 | 27.930183 | 0.000000 |
| VisDrone 六层 G5＋关闭 DDF | 28.518592 | 48.081990 | 28.309960 | 0.000000 |
| VisDrone 预测分布加权 strip 特征修正输出框 | 28.996845 | 48.865192 | 28.834478 | 0.000000 |
| UAVSwarm 六层 G5＋关闭 DDF | 61.231608 | 94.762391 | 67.598768 | 0.000000 |

完整12指标、原日志位置与实际评估文件路径保存在[results.json](results.json)。归档根目录的`logs/evaluation_17770.out`、`logs/evaluation_17770.err`及`logs/evaluation_hardware_17770.txt`分别保存控制台、错误输出和硬件信息。每模型`evaluation/latest/`指向本次验收输出；`results.json`同时记录具体带时间的目录，后续再次评估不会改变这次记录。

UAVSwarm权重内部epoch70与最佳外层日志epoch78的差异已通过实际评估核实：该checkpoint确实重现61.23160763 AP，无需改选权重。

## 本分支工具文件

- `models.json`：五个用户选择、原路径与结构说明。
- `prepare_archives.py`：生成独立目录、复制完整训练产物、标注、源码和运行环境记录；已有目标目录时停止，避免覆盖。
- `configs/`：与归档中`archive_eval.yml`一致的五份展开配置。
- `eval.sh`、`summarize_eval.py`、`run_evaluations.slurm`：实际复制到归档的入口。
- `index.json`：创建归档后的完整索引。
- `results.json`：复评结束后的实测结果。

初次构建命令（目标必须尚未存在这五个模型目录）：

```bash
/home/zzy/.conda/envs/dfine/bin/python archives/selected_models/prepare_archives.py \
  --destination /cluster/home/zzy/D-FINE/desired_results
```

原训练文件不移动、不改写。归档目录保存独立副本，源码及标注不以软链接指向旧实验目录；仅图像数据和Python/CUDA环境作为外部运行条件。
