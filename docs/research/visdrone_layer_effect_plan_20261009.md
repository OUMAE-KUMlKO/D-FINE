# VisDrone 六层 strip 局部修正诊断

用户已批准检查每层修正前后的框质量。本阶段只增加诊断工具和结果，不修改训练模型，不训练、不提交 commit。

- [x] 用只读 forward hooks 记录六个 StripProbabilityHead 的输入框、输出框、有效修正 mask 和实际采样半宽；推理不传 GT。首批对照普通推理，确认输出一致、最后一层记录与模型输出一致，hooks 在异常时也清除。
- [x] 使用纯几何 Hungarian（class=0、bbox=5、GIoU=2）。主表用最终输出的一组 query–GT 配对固定六层；敏感性分析分别按各层修正前、修正后建立配对，但每个 before/after 比较内部绝不重新匹配。
- [x] 为各层统计前后 IoU、四边平均误差（输入像素）、改善/退化比例、跨越 IoU0.5/0.75 的数量、移动幅度与实际半宽。按原图面积（COCO闭区间）、输入短边、类别分组。原图面积与 target 行严格对齐。
- [x] 用局部测试验证同 query/GT 比较、非方形图像的坐标单位、分组边界、无效框处理、trace不改变正常推理；先验证新测试失败再实现。trace 4项、metrics 6项通过；CLI、Python编译与Slurm语法检查通过；独立检查无阻碍。
- [x] 对 VisDrone seed0 固定 G5 与自适应 G5 的 best_stg2 EMA 各评估一次完整548图；保存 CPU 框轨迹以便离线复核。首批验证后继续同一次进程，不重复全量推理。
- [x] 汇总每层及关键分组，区分局部即时效果与最终AP因果影响；归档 JSON 和中文报告。

实现分工：`tools/analysis/layer_strip_effect_metrics.py` 与 `tests/test_layer_strip_effect_metrics.py` 负责纯统计；`tools/analysis/all_layer_strip_trace.py` 和 `tests/test_all_layer_strip_trace.py` 负责只读采集；`tools/analysis/diagnose_all_layer_strip.py` 负责数据、匹配、缓存和CLI；Slurm脚本只运行上述两个已完成模型。

统计接口：`LayerEffectMetrics.update(before_boxes, after_boxes, targets, indices, image_size, valid=None, half_widths_px=None)`。预测为 `[B,Q,4]` 归一化cxcywh；targets包含归一化cxcywh `boxes`、`labels`、`area_original`；indices每图为(query_ids,gt_ids)；image_size为(H,W)；valid为[B,Q]；half_widths_px为[B,Q,2]，按左右/上下方向排列。`compute()` 返回 `invalid_pairs` 与 `groups`，每组包含样本数、before/after的mean_iou与mean_edge_mae_px、变化量、改善/退化/不变比例、阈值跨越计数和移动/带宽摘要。

轨迹接口：`trace_all_layer_strip(model, images)` 返回正常 `outputs` 和 `layers`（0起始索引字典）；每层包含 `boxes_before`、`boxes_after`、`valid`、`half_widths_px`。保留所有层的原始生产前向顺序，不绕过任一修正头。

作业：`16447`，脚本 `slurm_files/run_diagnose_visdrone_layer_effect.slurm`。日志目录 `logs/dfine_visdrone_layer_effect/`，结果和框轨迹位于 `output/visdrone_layer_effect_20261009/`。已确认训练模型四个相关源文件与自适应训练冻结快照完全一致。

已完成：作业16447为COMPLETED/0:0，两个模型各548图；报告见[逐层诊断结果](visdrone_layer_effect_results_20261009.md)。
