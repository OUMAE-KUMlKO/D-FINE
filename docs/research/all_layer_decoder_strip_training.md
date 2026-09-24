# 全层 Decoder Strip 概率修正：实现与训练记录

日期：2026-09-23

用户要求：每一层 decoder 均使用 strip conv 特征修正预测概率；启动一组沿用强 B0 的 12 轮实验，以及一组仅加载 backbone 预训练权重的全量训练。

## 1. 实际数据流

六层 decoder 的每一层均执行：

```text
本层 decoder query 更新
  → 原 bbox head 输出 + 上一层修正后的累计 logits
  → 得到本层临时分布和临时框
  → 按临时分布的全部候选位置提取 strip 特征
  → 本层独立 strip 头预测 logits 残差
  → 完整候选分布重新积分解码
  → 本层输出 / LQE / 下一层累计 logits 与注意力参考框
```

首层先通过原有 pre-bbox 和 bbox head 建立分布，再执行 strip 修正，因此索引 0–5 全部参与。每层有独立参数；不再局限于最后两层，也不在 decoder 执行完后单独修正最终框。

保留 P3 的逐候选编码和原 logits 先验，但将其放入每层迭代。各头末层零初始化，保留全部 bin 的概率质量；无效候选只屏蔽修正量。校准框无效时同时回退 logits，避免框与累计分布不一致。

新增头使用 `detach_inputs=False`，特征、query 和 logits 保留梯度；采样坐标与尺度仍 detach。累计分布更新不 detach；下一层注意力参考框沿用原 D-FINE 的 detach 规则。

原有训练用 DN query 保持原路径；strip 作用于每层全部常规检测 query。模型不把 targets 传入 strip 头。全量训练仍保留原有 DN 和辅助层监督，推理不使用 GT。

## 2. 两组已提交实验

| 项目 | 12 轮实验 | 全量实验 |
| --- | --- | --- |
| Slurm 作业 | 15444 | 15445 |
| 模型 | `DFINEAllLayerStripRefiner` | `DFINEAllLayerStrip` |
| 初始化 | 强 B0 `best_stg2.pth`，新 strip 头随机/零末层初始化 | 仅 HGNetV2 B4 预训练，encoder/decoder/strip 重新初始化 |
| 优化范围 | 冻结 B0，仅训练六个 strip 头 | 原配置允许训练的 backbone 层、encoder、decoder 和六个 strip 头 |
| Epoch | 12 | 80 |
| 总 batch / GPU | 96 / 8 | 64 / 8 |
| 损失 | 原 B0 固定匹配，`5×L1 + 2×GIoU` | 原 `DFINECriterion`，分类、框、FGL/DDF、辅助层与 DN 损失 |
| 分类分数 | 保留 B0 原值 | 随原检测器训练和 LQE 更新 |
| 数据处理 | 原短程实验的固定 640 输入 | 原 B0 的增强与多尺度日程，末阶段关闭增强 |
| EMA | 关闭 | 沿用原完整训练配置 |

全量训练 batch 从原 B0 的 128 调整为 64，为六层 strip 留出显存；两组均启用 strip 梯度检查点。因此这两组初始化、优化范围和预算不同，不能作为单一变量的严格消融。

强 B0 路径：`/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923/b0/best_stg2.pth`。

全量训练的唯一预训练文件：`weight/hgnetv2/PPHGNetV2_B4_stage1.pth`；脚本不传 detector `--tuning` 或 `--resume`。

数据仍为 `/cluster/home/zzy/dataset/UAVSwarm-dataset-master/train`，使用 `annotations/train_half.json` 和 `annotations/val_half.json`。

输出目录分别为：

- `/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_strip_12ep`
- `/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_strip_full`

## 3. 代码与运行入口

- `src/zoo/dfine/dfine_decoder.py`：新增训练/推理均可用的 `distribution_refiner`，每层更新后重新解码并沿原路径反馈。旧 `query_refiner` 保留，不允许两种 callback 同时启用。
- `src/zoo/dfine/all_layer_strip.py`：六个独立头、梯度检查点，以及冻结短程/联合全量两种模型封装。
- `src/zoo/dfine/strip_probability_head.py`：增加 `detach_inputs` 参数，默认值保留原 P3 行为。
- `configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_all_layer_strip_12ep.yml` 与 `configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_strip_full.yml`：两组训练配置。
- `slurm_files/run_train_dfine_uavswarm_all_layer_strip_12ep.slurm` 与 `slurm_files/run_train_dfine_uavswarm_all_layer_strip_full.slurm`：已提交的训练入口，均拒绝覆盖已有输出目录。

26 项相关 CPU 测试通过：概率头 9 项、全层 callback 6 项、旧 query callback 回归 6 项、模型集成 5 项。覆盖首层接入、六层反馈、DN 分拆、梯度跨层、原参数冻结与全量梯度、检查点输出/梯度一致以及两组配置。两份 Slurm 脚本通过语法检查。

本轮按用户授权直接提交两次正式训练，不另建 GPU 诊断作业，未自动 commit。

提交后的状态：`15444` 已在 `gpu16` 进入 epoch 0，首批 loss 为 0.1825，峰值 allocated 显存 7,092 MiB；错误日志仅有环境 locale 提示。`15445` 已排队，等待原因是 `AssocGrpGRES`（账户/关联组 GPU 配额），尚未进入模型加载与训练，不代表训练失败。
