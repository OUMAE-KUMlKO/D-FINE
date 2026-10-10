# VisDrone 仅前两层 strip 修正训练

用户已批准只保留前两层 strip 再训练一次。沿用最新自适应带宽 G5，完整六层 decoder 中仅第1、2层启用 strip，第3–6层仍训练原生分类、回归、辅助及DN损失。此前逐层诊断是设计依据，本次全量训练用于检验最终AP，不预设一定改善。

训练条件：VisDrone、seed0、80 epoch、总batch32、8 GPU；从backbone预训练开始，不载入旧检测器checkpoint。保留stride8、5×7二维采样带、3×3编码、query通道调制、无log-prior输入、自适应半宽clamp(0.25×预测框法向尺寸,2,8)。仅改变启用strip的层数。保留现有暂存内容，不提交commit。

- [x] 为 `DFINEAllLayerStrip` 添加 `num_refine_layers=None`，默认保持全部六层。配置2时只创建前两个独立strip头；后层回调原样返回原生分布，前两层修正仍传给后续decoder。拒绝非整数及超出1至decoder层数的值。
- [x] 在 `tests/test_first_two_layer_strip.py` 先验证缺少该参数导致失败，再验证：仅前两层调用、六层decoder和DN/辅助输出保留、后四层分布等于原生回归增量加上一层分布、前两层及后四层回归头均能获得梯度、默认全层兼容。
- [x] 新增 `configs/controlled/visdrone_g5_adaptive_band_first2.yml`，继承自适应配置，只覆盖 `DFINEAllLayerStrip.num_refine_layers: 2` 及输出命名；提交工具新增可显式选择的实验组，不改变默认实验列表。
- [x] 运行必要局部测试，核对合并配置、真实模型头数量及独立代码检查。
- [x] 使用原提交工具冻结本次源码，单独提交一项训练；沿用原训练脚本的节点排除名单，并确认本次未分配到此前异常的gpu20。确认作业及日志，记录job ID和路径。

启动命令（验证通过后执行一次）：

```bash
/home/zzy/.conda/envs/dfine/bin/python tools/submit_controlled_experiments.py --submit --datasets visdrone --groups g5_adaptive_band_first2 --seeds 0 --run-root output/visdrone_g5_adaptive_band_first2_20261009
```

必要测试命令：

```bash
PYTHONPATH=tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp /home/zzy/.conda/envs/dfine/bin/python -m unittest test_first_two_layer_strip test_all_layer_strip test_controlled_submission -v
```

验证与启动记录：

- 新模型测试先因缺少num_refine_layers参数失败；实现后4项新模型测试、5项原全层测试、6项提交测试，共15项通过。
- 已构建实际L模型并加载backbone预训练；确认2个strip头、6层decoder，31,217,902个可训练参数；合并配置为80epoch、batch32、seed0、无resume/tuning。
- 独立只读检查未发现阻碍；git diff --check通过。保留既有暂存内容，没有提交commit。
- 作业：16449，gpu14，8 GPU，已提交并进入RUNNING。Slurm实际排除列表为gpu1/gpu10/gpu18；本次使用gpu14。
- 源码快照：`output/visdrone_g5_adaptive_band_first2_20261009/source/`。
- 日志：`output/visdrone_g5_adaptive_band_first2_20261009/logs/controlled_visdrone_g5_adaptive_band_first2_s0_16449.out`及同名`.err`。
- 权重输出：`output/visdrone_g5_adaptive_band_first2_20261009/results/visdrone/seed0/g5_adaptive_band_first2/`。
- 已确认进入真实训练：日志出现Epoch [0/80] [0/202]，首步loss=48.9670，已完成前向和反向，未见启动异常。

## 最终结果与归档（2026-10-10）

以上 RUNNING 为提交时的历史记录。作业 16449 已完成 80 轮，耗时约 6 小时 58 分钟；最佳为日志 epoch 79（0-based）的 EMA，使用 `best_stg2.pth`。

| 模型 | AP | AP50 | AP75 | APsmall | APmedium | APlarge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 28.1196 | 47.8534 | 27.9655 | 18.4971 | 39.9493 | 56.5424 |
| 自适应六层 | 27.9668 | 47.3316 | 27.8112 | 18.5472 | 39.7013 | 57.0859 |
| 自适应前两层 | 27.7790 | 47.1061 | 27.6664 | 18.2046 | 39.7006 | 56.2904 |

指标为百分数，使用本地 COCO 格式验证集、maxDets=100；各行取同一个最佳 AP checkpoint，均为 seed0。前两层相对六层 AP 下降 0.1878 个百分点、APsmall 下降 0.3426 个百分点，未支持只保留前两层更好的假设。

用户于 2026-10-10 授权整理当前工作区并用中文提交，本阶段将实现与上述结果一同归档。后续续训属于另一阶段，不混入本表。
