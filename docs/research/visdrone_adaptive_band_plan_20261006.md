# VisDrone 自适应边界采样带实施记录

> 实施方式：按已批准方案执行局部 TDD，独立检查后提交 Slurm 训练。用户已授权先提交已有工作，再修改并启动训练。

目标：验证仅调整采样带法向半宽，能否改善 G5 在 VisDrone 上的小目标定位及检测 AP。

## 约束与接口

- 保留现有 stride8 边界特征、每层 decoder 分布修正、5×7 采样、3×3 编码、query 通道调制和无 log-prior 输入。
- 由本层修正前预测框计算输入像素宽高；左右边半宽 `clamp(0.25*w, 2, 8)`，上下边半宽 `clamp(0.25*h, 2, 8)`。
- 只改变候选附近的采样带，不改变分布支持点、候选中心位置、MLP 维度或损失；采样坐标沿用 detach。
- `StripProbabilityHead` 和全层模型增加 `bandwidth_mode='fixed'`、`bandwidth_ratio=0.25`、`min_offset_pixels=2.0`。`box_relative` 模式将现有 `offset_pixels` 作为上限；旧配置固定带宽完全保留。
- 新实验继承 `configs/controlled/visdrone_g5.yml`，仅启用上述参数。全量 80 epoch，seed=0，8 GPU，总 batch=32，仅加载 HGNetv2 backbone 预训练，不恢复检测 checkpoint。训练增强及多尺度策略沿用已有 G5，验证输入保持 640×640。
- 本轮先运行一个新模型，与已有同 seed G0/G5 对照；不额外重跑 baseline。训练使用独立输出目录及冻结代码快照。

## 实施步骤

- [x] 按原暂存内容与 VisDrone 诊断拆成两个中文提交：`cc32ae1`、`aa5f482`；已有局部测试 22 项通过。
- [x] 先添加 `tests/test_strip_adaptive_band.py`：非正方形输入下宽高轴映射、上下限、法向方向、候选中心不变、旧固定带宽保持一致，以及全层梯度/初始化等价；确认新增测试在旧实现上失败（4px 框应取半宽 2px，旧实现仍为 8px）。
- [x] 在 `src/zoo/dfine/pred_strip_head.py` 的 `_candidate_grid` 中按模式选择固定或逐框半宽；在 `strip_probability_head.py` 校验参数；在 `all_layer_strip.py` 两个模型包装器透传配置。
- [x] 新建 `configs/controlled/visdrone_g5_adaptive_band.yml`，验证与原 G5 合并配置只有带宽参数、实验名和输出目录不同；数据目录与 backbone 权重存在。
- [x] 为现有提交器增加显式选择 `g5_adaptive_band` 的入口，默认十二组保持不变；先复现漏提交的新测试失败，再通过全部 5 项提交器测试。
- [x] 完成局部单元测试、真实配置构建及训练入口检查；代码独立检查只覆盖本次改动。新增带宽测试 6 项、已有相关测试 45 项、提交器测试 5 项均通过。真实 YAML 构建得到 31,527,892 参数、6 个自适应头，沿用 stride8 特征；独立检查未发现阻塞问题。
- [x] 复用现有冻结代码与 Slurm 入口，保存配置、代码差异及作业清单，启动 seed0 全量训练并检查开始迭代。作业 `16350` 已在 `gpu12` 运行，日志出现 `Epoch: [0/80] [0/202]`，首批 loss=48.9670，显存峰值 8310 MiB；stderr 仅有原有 locale 警告。

## 验收与后续分析

启动前验证无 GT 参与带宽计算、无新增模型参数、原 G5 state_dict 可严格加载、六层均获得视觉分支梯度。训练后比较整体 AP、AP_S / AP_M / AP_L 与四边误差，不能将单独 MAE 下降等同检测性能提升。本次新改动先保留在工作区，训练快照固定其实际代码；不自动追加代码提交。

提交命令（通过实现验证后执行）：

```bash
/home/zzy/.conda/envs/dfine/bin/python tools/submit_controlled_experiments.py \
  --submit --datasets visdrone --groups g5_adaptive_band --seeds 0 \
  --run-root output/controlled_visdrone_adaptive_band_20261006_seed0
```

作业已提交：`16350`，2026-10-06 11:36:41 开始分配在 `gpu12` 上，8 GPU。代码快照位于 `output/controlled_visdrone_adaptive_band_20261006_seed0/source`，作业记录位于同级 `jobs.json`，基准提交 `aa5f482`，本轮源代码差异保存为 `working_changes.patch`，新增配置包含于快照。

- stdout：`output/controlled_visdrone_adaptive_band_20261006_seed0/logs/controlled_visdrone_g5_adaptive_band_s0_16350.out`
- stderr：`output/controlled_visdrone_adaptive_band_20261006_seed0/logs/controlled_visdrone_g5_adaptive_band_s0_16350.err`
- 训练输出：`output/controlled_visdrone_adaptive_band_20261006_seed0/results/visdrone/seed0/g5_adaptive_band`
