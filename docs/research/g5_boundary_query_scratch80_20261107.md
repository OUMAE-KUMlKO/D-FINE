# G5 边界 Query 首版：80轮从头训练配对实验

## 目标与已认可方案

24轮继续训练中，新方案AP28.8798、G5对照AP28.9723，未验证出额外收益。
用户授权补充完整80轮实验，检验从第0轮加入前端、共同训练的效果。
沿用首版模型及监督，不同时改变采样布局、代表性定义或query构建方式。

两组为 `boundary_query` 与 `g5_control`，各80轮，seed0。
仅加载同一个HGNetv2-B4主干预训练文件：
`weight/hgnetv2/PPHGNetV2_B4_stage1.pth`。
encoder、decoder、六层G5 refiner均重新初始化；两组共有参数初值完全相同。
新增前端的输出头仍零初始化；不载入已训练G5的检测checkpoint。

## 训练协议

- 以归档G5的80轮训练配置和solver为基座。DDF=0、FGL=0.15，300 query。
- 总batch32、单节点8 GPU、seed0；隔离模型初始化随机数，使新增参数不改变训练数据/DN随机流。
- AdamW原分组：骨干对应参数1.25e-5，其余参数（含新增前端）2.5e-4；
  weight decay=1.25e-4，保留原encoder/decoder norm/bn免衰减规则。
- 500步LinearWarmup，原MultiStepLR milestones=[500]；完整80轮。
- EMA decay=0.9999、warmups=1000；AMP初始scale1024、criterion浮点输入转FP32，clip=0.1。
- 前72轮原增强和多尺度；epoch=72开始停止原policy中的三项强增强与多尺度，保留水平翻转。
- 保留归档solver的stage1最佳完整状态回载及stage2后续回载规则，包括optimizer、scaler、EMA与scheduler。
  外层epoch仍按0…79推进，保存回载前实际评估权重。
- 保留原EMA实现：solver修改的 `ema.decay` 不改变构造时闭包中的decay值；本轮不更改此历史行为。
- 两组使用相同标注、数据次序、原有参数初值和训练规则。每轮完整val548，保存逐轮指标和权重。

本轮恢复完整训练的原始学习率，前端不沿用24轮微调专用的1e-3；两组所有共有参数优化规则相同。
因初始化和训练协议均不同，主要比较本轮两组，而不以本轮与24轮结果之差独立归因于轮数。

## 实施步骤与文件

- [x] 配置：`configs/dfine/boundary_query_scratch80/{common,boundary_query,g5_control}.yml`。
- [x] 训练入口：`tools/train_boundary_query_scratch.py`，复用归档solver阶段逻辑并隔离初始化RNG。
- [x] 预检工具：`tools/preflight_boundary_query_scratch.py`；验证共有初值、原主干权重及真实数据AMP更新。
- [x] 冻结运行时：归档source + 首版模型 + 本轮入口，保存实际展开配置。
- [x] 提交单作业：依次运行GPU预检、boundary_query 80轮、g5_control 80轮。

提交脚本 `slurm_files/run_g5_boundary_query_scratch80.slurm` 申请一个完整8GPU节点，
不设nodelist/constraint/exclude，任意可用完整节点都可调度；总并行GPU数不超过8。
不自动提交commit。

## 提交与验证

运行目录：`output/g5_boundary_query_scratch80_20261107/`。
Slurm作业号：**20269**。提交时状态为 `PENDING`，原因 `Resources`；
`ReqNodeList`、`ExcNodeList` 均为空，申请1节点、8GPU。

训练入口4项、预检工具4项CPU局部测试通过。
冻结源码下的真实图片CPU初始化预检通过：1263个共有状态张量完全相同，
400个主干状态张量与同一个本地预训练文件完全相同，初始类别/框预测最大误差均为0。
两组模型构造均保留外部CPU随机状态。

GPU预检将在获得节点后执行，当前尚未运行GPU训练。
预检结果写入 `preflight.json`；日志为 `logs/20269.out`、`logs/20269.err`。
各组训练写入 `results/<group>/metrics.jsonl`、`status.json` 和 `stage_reload_events.jsonl`，
权重保留原 `best_stg1.pth`、`best_stg2.pth`、`last.pth`，另保存全程最佳 `best.pth`。
