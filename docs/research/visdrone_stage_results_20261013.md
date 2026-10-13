# VisDrone阶段结果归档（2026-10-13）

E1–E3及两种GT oracle均已完成。用户已授权整理当前代码、实验结果并进行本阶段中文commit；早期记录中的“不提交commit”仅为当时运行阶段的限制。原始日志、checkpoint与冻结源码保留在各运行目录；精简机器可读结果见 [JSON](results/visdrone_stage_results_20261013.json)。

AP为百分制，epoch从0计数。Oracle三行使用相同正ID验证标注；E1–E3沿用各自原运行口径。

| 实验 | 最佳epoch | AP | AP50 | AP75 | AP小 | AP中 | AP大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 同口径复评 | 96 | 28.6235 | 48.7638 | 28.4438 | 19.2484 | 40.3924 | 59.6253 |
| GT边界条带oracle | 11 | 36.0279 | 50.9625 | 39.2931 | 26.7919 | 48.0231 | 61.5060 |
| GT最近候选oracle | 11 | 36.9038 | 51.4178 | 41.2042 | 28.0329 | 48.3326 | 60.8490 |
| 自适应门控 | 97 | 28.4188 | 48.3659 | 28.1377 | 18.9906 | 40.2222 | 55.9137 |
| 修正前参考 | 92 | 28.2469 | 48.0494 | 27.9668 | 18.8249 | 40.0448 | 57.9759 |
| 门控+修正前参考 | 93 | 28.6665 | 48.5340 | 28.3449 | 19.2127 | 40.6160 | 59.0891 |

两种oracle均完成epoch0–11，最佳均为epoch11；相对同口径G0，条带oracle提升7.4043 AP，候选oracle提升8.2803 AP，两者相差0.8760 AP。两者使用GT，不能作为无GT检测性能或严格上界，也不能把全部差异归于选择器。

固定基线Hungarian配对共38725对，GT38759个、未匹配GT34个、未匹配query125675个。全体固定配对的IoU / 四边平均绝对误差（640输入像素）为：G0 0.675676 / 2.792260，条带oracle 0.749658 / 2.292977，候选oracle 0.767173 / 2.104482。条带有效修正37260对（96.2169%），无效GT采样1465对；候选有效修正38725对（100%），无效采样0。三组非法基线框和非法指标配对均为0；逐边误差及尺寸分组见JSON。

E1续训完成epoch92–119，最终epoch97 AP28.418844，高于历史epoch89 AP28.355244；`resume_selection.json`已选出最终checkpoint。E2/E3分别完成80+40轮，最佳epoch92/93；三组均结束于epoch119。

两种oracle单GPU batch12预检均通过：初始框误差0、分类分数不变、冻结张量不变、仅refiner有梯度，均完成两次有效AMP更新。条带/候选峰值allocated显存分别1.3995/1.5850 GiB，reserved分别1.5645/2.0078 GiB；作业为17077→17079、17103→17104。

关键checkpoint和指标来源（相对仓库根目录）：

- G0 同口径复评：`output/visdrone_continue120_20261009/results/visdrone/seed0/g0/best_stg2.pth`；指标：`output/visdrone_gt_strip_oracle_20261013/results/baseline/oracle_eval.json`。
- GT边界条带oracle：`output/visdrone_gt_strip_oracle_20261013/results/oracle/best_stg1.pth`；指标：`output/visdrone_gt_strip_oracle_20261013/results/oracle/log.txt`。
- GT最近候选oracle：`output/visdrone_gt_candidate_oracle_20261013/results/oracle/best_stg1.pth`；指标：`output/visdrone_gt_candidate_oracle_20261013/results/oracle/log.txt`。
- 自适应门控：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_gate/best_overall_stg2.pth`；指标：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_gate/continue120/log.txt`。
- 修正前参考：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_pre_reference/continue120/best_stg2.pth`；指标：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_pre_reference/continue120/log.txt`。
- 门控+修正前参考：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_gate_pre_reference/continue120/best_stg2.pth`；指标：`output/visdrone_gate_reference_prior_guard_recovery_20261011/results/visdrone/seed0/g5_adaptive_band_gate_pre_reference/continue120/log.txt`。

必要验证：本阶段变更涉及17个测试模块、69项测试；沙箱内68项通过，1项双进程Gloo初始化受环境限制，沙箱外复核其所在4项测试模块全部通过。变更Slurm脚本语法与暂存差异空白检查通过。归档仅提取已有结果，未新增训练或诊断。
