# 归档模型的 AI-TOD 尺度补评

用户已确认：在原 VisDrone / UAVSwarm 验证集上，为已有权重补充极小目标指标。

## 范围与方法

- D-FINE归档索引中的5个模型，以及 `desired_results_12floor` 中已复评的2个模型；RT-DETR归档中的R50vd/HGNetv2-L，VisDrone seed0与UAVSwarm seed0/1/2，共8组。
- 使用每个实验原冻结源码、最佳checkpoint、原model/EMA选择、640输入、后处理Top300、验证batch及类别映射。只新增独立评测包装器和结果目录，不修改归档训练文件。
- 一次推理保留COCO原分档，并增加AI-TOD分档：verytiny `[0,64]`、tiny `[64,256]`、small `[256,1024]`、medium `[1024,1e10]`，单位为原图像素面积；使用标注area，边界依官方实现为闭区间。
- IoU阈值为0.50:0.05:0.95。同时提取maxDets=100和1500；1500是每图每类别评估上限，模型仍最多输出300个框，不能称为AI-TOD数据集成绩。
- 原COCO12指标按标签和maxDets=100显式提取，避免扩展范围后库的summarize索引变化。与历史复评结果核对；12楼模型保留已知的历史训练/复评差异说明。
- 官方口径来源：https://github.com/jwwangchn/cocoapi-aitod/blob/master/aitodpycocotools/aitodpycocotools/cocoeval.py

## 执行顺序

1. 核对模型清单、配置、checkpoint及历史复评指标，生成本次独立清单。
2. 用局部测试验证尺度边界、无GT处理和不同maxDets下的指标提取；检查两种评测后端的参数接入。
3. 在GPU分区提交补评，沿用各模型原每卡batch，不指定单个节点；每组记录独立日志和JSON。
4. 汇总COCO复评差值以及APvt/APt/APs/APm，注明无有效GT或评测失败的组。

本轮不自动提交Git。

## 已提交的补评任务

- Slurm作业：**20024**。初始申请8 GPU、32 CPU、128 GB并等待资源；用户要求降低资源后，于2026-11-06原地改为**4 GPU、16 CPU、64 GB**，15:08:33由调度器分配到gpu14开始运行。没有指定单个节点，没有取消重提；Priority=1、Nice=0、原SubmitTime/EligibleTime/AccrueTime均保留（14:56:37）。
- 独立产物目录：`output/archived_aitod_sizes_20261106/`。
- `manifest.json`及`records/*.json`记录全部15组的冻结源码、配置、checkpoint、原权重类型、验证batch、随机种子和历史复评指标。
- `scripts/`保存本次三个Python脚本及Slurm脚本的独立副本。总日志为`slurm-20024.out/.err`；每组写入`<模型ID>/console.log`、`command.json`、`size_metrics.json`，失败时保留`failed.txt`。
- 任务随进度更新`results.json`、`results.csv`、`README.md`；COCO控制差值超过0.0001 AP点时标注`control_differs`，保留数值供分析。
- 已完成两项尺度统计局部测试，以及faster-coco-eval、pycocotools两种真实后端的合成评测，确认面积64的相邻分组、无GT为null、原12项COCO布局和两种maxDets提取均正常。完整GPU结果待任务运行。
- 15组均使用4个推理进程，每卡batch保持原值；清单的`archive_nproc`保留原6/8卡信息。总batch随进程数缩小；后续仍核对COCO指标以观察实际复评差值。
- 12楼两模型的控制值采用2026-10-21实际复评结果（UAVSwarm D-FINE 59.81865131、VisDrone G5 29.18299823），同时保留原训练指标；不会把已有复评差异误归因于本次尺度扩展。
