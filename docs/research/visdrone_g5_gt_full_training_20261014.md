# VisDrone六层G5 GT引导完整训练

用户于2026-10-14批准将G5-GT改为完整训练。旧12轮微调作业17134在未开始时取消，旧快照不改；P2作业17132保持12轮及原设置。本轮不自动Git提交、不修改原暂存内容。

## 训练协议

与原VisDrone自适应六层G5对齐：仅加载HGNetv2 B4 stage1主干预训练权重，decoder及六个修正头重新初始化，不加载G0/G5检测器checkpoint。GT候选视觉门控定义保持不变，未匹配query和DN使用原G5路径，GT不直接替换输出框或概率。

- seed0，8GPU，总batch32；主机内存64GiB，作业时限48小时，无自动requeue。原G5完整120轮约19小时，仅作资源安排参考。
- 前80轮继承原G5全部优化器、增强、SyncBN、EMA配置。AMP通过原启动脚本启用；AdamW主lr2.5e-4/backbone1.25e-5，warmup500步，EMA decay0.9999/warmups1000。
- 原多尺度480–800、步长32，640重复4次；原增强保留。epoch72切换到最后阶段、停止多尺度并加载best_stg1完整状态，阶段回退策略与原80轮一致。
- 完成epoch79后，从last.pth完整恢复model/optimizer/EMA/scaler/scheduler/warmup，计划执行epoch80–119。绝对epoch100衰减0.1，续训阶段关闭stage2 rollback。这是预定两阶段流程，不是失败作业自动恢复。
- 模型层数、stride8特征、LQE、损失及安全回退维持已有实现。验证使用已纠正正annotation ID的同548张图；比较原G5时沿用相同标注口径复评记录。

## 实施与验证

- 完整80轮和continue120配置已经建立。展开配置与原冻结G5对照通过，差异仅GT模型包装、实验路径/名称、include元数据及正annotation ID验证标注。
- 8项GT模型测试与3项预检测试通过；Slurm入口语法检查通过。
- GPU预检17141完成，退出码0:0：真实增强训练batch4、800×800，两次有效AMP更新，backbone/encoder/decoder及六个strip head均实际更新；更新后GT引导影响预测，EMA更新与GT验证通过。
- 本次预检峰值显存9.29GiB；单批预检不代表所有密集图像的最大显存。
- 正式完整训练17143已提交，当前PENDING / Priority。8GPU、batch32、64GiB主机内存、48小时；依赖已经成功的17141，入口还校验预检JSON。P2作业17132不变，仍PENDING / Priority。

## 文件与运行记录

- 配置：`configs/controlled/visdrone_g5_gt_candidate.yml`、`visdrone_g5_gt_candidate_continue120.yml`。
- 入口：`slurm_files/run_visdrone_g5_gt_full.slurm`，复用已有80轮训练及40轮续训脚本。
- 预检：`tools/analysis/check_visdrone_g5_gt.py`，新增fresh初始化路径，旧EMA微调接口保留。
- 运行目录：`output/visdrone_g5_gt_full_20261014/`。最终冻结源码、标注及主干权重在`source_v2/`；预检结果`preflight_v2.json`；作业与协议分别记录于`jobs.json`、`experiment_selection.json`。
- 正式输出：`results/epoch80/`和`results/continue120/`；日志`logs/train_17143.out`、`logs/train_17143.err`。

## 首次预检发现并修复的问题

首次预检17139在真实增强训练批上失败：COCO loader在增强前已删除crowd框，剩余iscrowd全0；SanitizeBoundingBoxes删框只同步boxes/labels，iscrowd长度过期。依赖它的17140因此自动取消，从未启动。

修复仅在`src/zoo/dfine/gt_all_layer_strip.py`训练路径对长度不符、全部为零的iscrowd构造去字段的GT引导目标副本。原targets、decoder/criterion输入和数据增强不变，非零错位及评估路径仍严格校验。新增测试先复现错误，再验证修复及输入不变。首次快照`source/`与失败预检日志保留，修复后使用`source_v2/`通过17141，再提交17143；没有恢复失败训练或自动重试循环。
