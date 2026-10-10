# VisDrone 三组统一续训至120轮

用户批准增加epoch并安排学习率衰减。已先按用户限定提交暂存内容：92ecc8c（自适应带宽及AP结果）、b345389（逐层诊断）；未暂存的前两层版本保留，新续训改动暂不提交。

共同条件：VisDrone、seed0、batch32、8 GPU；G0、自适应六层G5、自适应前两层G5，各从对应epoch79的last.pth恢复模型、optimizer、EMA、scaler和warmup状态。三个任务按afterok串行，使用新输出目录和冻结源码，不覆盖原80轮结果。epoch80–99学习率为backbone=1.25e-5、其余=2.5e-4；epoch100–119同时乘0.1。stop_epoch保持72，增强仍处于最后阶段。关闭第二阶段验证下降时回退best_stg1的行为，不改变其他训练默认设置。

## 任务1：续训调度与连续优化

- [x] 新配置开关 `resume_lr_schedule: true`：DetSolver完成checkpoint恢复后，按YAML中的MultiStepLR（milestones=[100], gamma=0.1）重新设置调度。以checkpoint.last_epoch+1对齐下一轮实际训练编号，使用scheduler.base_lrs计算该轮应有LR，保留optimizer动量、EMA、scaler和已经完成的warmup；重复断点续训不得重复衰减。
- [x] 新开关 `stage2_rollback: false`：仅禁用验证未改善时的stage1回退；默认true保留历史行为。保留每轮last.pth和最佳权重选择。
- [x] TDD验证epoch79恢复、epoch99/100边界、已衰减checkpoint恢复、optimizer状态保留、开启/关闭回退、旧默认兼容。只修改必要solver路径与局部测试。

## 任务2：三组配置与串行提交

- [x] 增加共用continue120配置和G0/六层/前两层三个继承配置；保持原结构、数据、损失和batch参数。
- [x] 新提交入口复用原freeze_source，保存三个输入checkpoint的绝对路径；调用专用续训Slurm脚本，以afterok串行运行。脚本复制起点权重到新目录的best_stg2.pth作为初始最优，再用--resume载入原last.pth，不使用符号链接写权重。
- [x] 提交选择/依赖/路径局部测试与配置合并检查；独立检查核心调度边界。
- [x] 确认局部测试通过后提交一次三项训练，记录job ID、日志、输出及学习率配置；核对首项进入epoch80。

## 初始结果与输入

| 模型 | epoch79 AP | checkpoint |
| --- | ---: | --- |
| G0 | 28.1196 | output/controlled_20260926_212304/results/visdrone/g0/last.pth |
| 自适应六层 | 27.9668 | output/controlled_visdrone_adaptive_band_20261006_seed0/results/visdrone/seed0/g5_adaptive_band/last.pth |
| 自适应前两层 | 27.7790 | output/visdrone_g5_adaptive_band_first2_20261009/results/visdrone/seed0/g5_adaptive_band_first2/last.pth |

三份checkpoint均已只读确认last_epoch=79、scheduler.last_epoch=78、原milestones={500:1}、三组LR/base_lrs/initial_lr=[1.25e-5,2.5e-4,2.5e-4]，warmup last_step=16160已完成，EMA和scaler完整。新的调度应消除旧scheduler计数与实际epoch的偏差。

解释边界：本轮比较“增加训练时间＋统一衰减”的效果，不能拆分两者贡献。新阶段统一禁用回退，保持三组公平；本轮仍只有seed0。

## 实施验证

- 提交前原暂存内容的相关22项测试通过，两次中文commit均包含实验数值。
- solver新增7项调度测试，阶段同步/回退7项测试，共14项通过；提交/配置3项测试通过，合计17项续训相关测试。真实MultiStepLR验证80–99恒定、100降低、99/100重载不重复降低。
- 独立只读检查确认调度与回退开关满足要求，未发现阻碍；Slurm脚本语法、CLI预览及git diff --check通过。
- 新文件：四份continue120配置、tools/submit_visdrone_continuation.py、slurm_files/run_resume_dfine_controlled.slurm；核心改动在src/solver/det_solver.py。新输出中先复制起点为best_stg2.pth，保证即使续训没有改善也保留起点最优权重。
- 原YAML读取函数有共享默认字典；配置对照测试使用load_config(path,{})隔离读取。每个训练任务是独立进程，不受跨配置读取影响，本次没有扩大修改配置加载器。

## 已提交并核对启动

批次：`output/visdrone_continue120_20261009/`；源码快照在`source/`，日志在`logs/`，任务清单为`jobs.json`。

| 顺序 | 模型 | Job ID | 依赖 |
| --- | --- | --- | --- |
| 1 | G0 | 16707 | 无，gpu6已开始epoch80 |
| 2 | 自适应六层 | 16708 | afterok:16707 |
| 3 | 自适应前两层 | 16709 | afterok:16708 |

首项日志已确认完整恢复EMA、scaler、optimizer、scheduler；`Resume LR schedule at epoch 80: [1.25e-05, 0.00025, 0.00025]`。恢复起点重新评估AP=28.119639708569427，与原epoch79一致。已出现Epoch [80/120] [0/202]，首步loss=31.3688。后两项为Dependency等待，自动串行。

本次新增续训修改未提交commit；原80轮实验输出未覆盖。

## 最终结果与归档（2026-10-10）

以上启动状态及未提交说明为执行时的历史记录。用户现已授权将续训实现与真实结果归档提交；三项作业均已退出，采用各自已保存的最佳 EMA，不再延长训练、不恢复失败作业。

| 模型 | 作业状态 | 原80轮 AP | 最佳 AP | 最佳 epoch | 最后完成 epoch | 最后 AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0（16707） | COMPLETED | 28.1196 | 28.6227 | 96 | 119 | 27.9632 |
| 自适应六层（16708） | COMPLETED | 27.9668 | 28.4190 | 96 | 119 | 27.7994 |
| 自适应前两层（16709） | FAILED | 27.7790 | 28.1545 | 94 | 98 | 28.1201 |

| 模型的最佳 checkpoint | AP50 | AP75 | APsmall | APmedium | APlarge | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 48.7670 | 28.4398 | 19.2467 | 40.3924 | 59.6246 | 43.1308 |
| 自适应六层 | 48.2831 | 28.2323 | 19.2515 | 39.9514 | 59.7370 | 42.8436 |
| 自适应前两层 | 48.0637 | 28.0721 | 18.9299 | 39.9320 | 58.2401 | 42.7782 |

指标均为百分数、epoch 为 0-based，沿用本地 COCO 格式验证集 maxDets=100。每行取最佳 AP 的同一 checkpoint。六层和前两层相对 G0 仍为 −0.2037、−0.4682 AP，额外训练没有改变排序；仅 seed0，不视为多种子结论。

16709 在 epoch99 的 matcher/GIoU 预测框合法性断言处失败，未完成120轮，也尚未执行 epoch100 的学习率衰减。未保存出错张量，具体根因未确定；不把后续 Gloo 清理超时作为首因，不自动修复或重启。

最佳权重统一在 `output/visdrone_continue120_20261009/results/visdrone/seed0/{g0,g5_adaptive_band,g5_adaptive_band_first2}/best_stg2.pth`。完整原始汇总位于 `output/visdrone_continue120_20261009/analysis/continuation_results_20261010.json`；纳入版本管理的精简指标见 [续训结果摘要](results/visdrone_continue120_results_20261010.json)。
