# G5 Query 筛选实验索引

数据集为 VisDrone。各文件代表不同实验，不能混用训练范围或把 GT 验证成绩当作可部署成绩。

| 配置 | 入口 | 更新范围 | 验证用 GT |
|---|---|---|---|
| `selection.yml` | `tools/train_boundary_selection.py` | 首版筛选器与 G5 联合训练 | 否 |
| `frozen_target_probe.yml` | `tools/probe_selection_target.py` | 固定 G5，仅训练原代表性监督筛选器 | 否，另有 GT 诊断 |
| `frozen_quality_rank.yml` | `tools/probe_selection_target.py` | 固定 G5，仅训练质量与排序监督筛选器 | 否，另有 GT 诊断 |
| `gt_coverage_adapt12.yml` | `tools/train_gt_coverage_adaptation.py` | G5 最佳权重初始化，主要更新 decoder，12 轮 | 是 |
| `gt_coverage_full80.yml` | `tools/train_gt_coverage_full.py` | 主干预训练初始化，G5 全量训练，80 轮 | 是 |

`common.yml` 是公共训练配置；冻结探针和局部微调的入口会另外限制可训练参数。
GT 全量实验没有新增可学习筛选头，`selection_enabled: false` 是预期配置：选择由 GT 回调完成。

公共 GT 算法见 `tools/gt_query_selection.py`，包含保留原生 TopK 的候选池、受限贪心交换、GT 选择统计。
全量训练需在无梯度索引计算后，重新计算可微的候选分类/回归张量；旧微调的冻结张量不能直接用于全量训练。

## 结果与分析

- [80 轮 GT 实验中文结果](../../../docs/research/g5_gt_coverage_full80_20261109.md)
- [代码整理、旧方案不足和下一步设计](../../../docs/research/g5_query_selection_review_20261109.md)

从仓库根目录刷新结果记录（只读训练日志，不启动训练）：

```bash
python tools/summarize_gt_coverage_run.py
```

实际运行参数、启动命令和源码快照以
`output/g5_gt_coverage_full80_20261109/manifest.json`、`results/effective_config.yml` 和 `source/` 为准。
