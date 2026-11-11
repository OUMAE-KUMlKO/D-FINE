#!/usr/bin/env python3
"""从已有日志生成 GT 全量训练的中文结果记录，不加载模型或启动训练。"""
import argparse
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'output/g5_gt_coverage_full80_20261109'
DEFAULT_REPORT = ROOT / 'docs/research/g5_gt_coverage_full80_20261109.md'


def read_rows(path):
    # 训练仍在写日志时，只读取已经换行的完整记录。
    content = path.read_text()
    lines = content.splitlines()
    if content and not content.endswith('\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run = args.run.resolve()
    results = run / 'results'
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = read_rows(results / 'log.txt')
    best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
    last = rows[-1]
    status_path = results / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = status.get('status') == 'completed'
    state = '已完成' if completed else '阶段记录；尚未读取到完成标记'
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    labels = ['AP', 'AP50', 'AP75', 'AP小', 'AP中', 'AP大',
              'AR1', 'AR10', 'AR100', 'AR小', 'AR中', 'AR大']
    metrics = best['test_coco_eval_bbox']
    coverage = best['test_oracle_query_coverage']
    baseline_path = ROOT / 'output/visdrone_g5_no_distill_6gpu_20261019_232603/results/log.txt'
    baseline = max(read_rows(baseline_path), key=lambda row: row['test_coco_eval_bbox'][0])
    adapt_path = ROOT / 'output/g5_gt_coverage_adapt12_20261108/results/status.json'
    adapt = json.loads(adapt_path.read_text())
    summary = dict(记录时间=timestamp, 状态=state, 已评估轮数=len(rows),
                   计划轮数=manifest['epochs'], 最佳轮次从一计数=best['epoch'] + 1,
                   最后一轮从一计数=last['epoch'] + 1, 验证使用GT=True,
                   最佳指标百分制=dict(zip(labels, [100 * x for x in metrics])),
                   最后一轮AP=100 * last['test_coco_eval_bbox'][0],
                   最佳轮覆盖统计原始比例=coverage,
                   原G5最佳AP=100 * baseline['test_coco_eval_bbox'][0],
                   十二轮局部微调最佳AP=100 * adapt['best_ap'],
                   训练配置=manifest, 日志=str(results / 'log.txt'))
    lines = [
        '# G5＋GT 覆盖筛选：VisDrone 80 轮全量训练记录', '',
        f'记录时间：{timestamp}。状态：**{state}**。', '',
        f'已完成评估 {len(rows)}/{manifest["epochs"]} 轮；以下为记录时刻结果，轮次统一从 1 开始。',
        '本文件由 `tools/summarize_gt_coverage_run.py` 生成，训练结束后再次执行即可刷新。', '',
        '## 实验设置', '',
        '- 单组实验，VisDrone 训练集 6471 张、验证集 548 张、10 类；seed=0。',
        '- 仅加载 HGNetv2-B4 主干预训练；检测器其余模块重新初始化。',
        '- 主干可训练阶段、编码器、候选分类/回归头、decoder、G5 修正头共同更新；保留原生 stem/norm 冻结。',
        '- GPU 0、2、3、4、6、7；总 batch size 24；80 轮。',
        '- AdamW：主干学习率 1.25e-5，其余 2.5e-4；weight decay 1.25e-4；warmup 500 步。',
        '- EMA=0.9999、warmups=1000；AMP；梯度裁剪 0.1；DDF 权重 0，FGL 权重 0.15。',
        '- 沿用原 G5 增强、多尺度和阶段回载协议；内部 epoch=72 开始关闭强增强和多尺度。',
        '- 训练、验证均使用 GT；Top1200 候选池内最多交换 40 个原生 Top300，最终保留 300 个。',
        '- 每次按当前模型重新计算选择；GT 只决定索引，不将 GT 框或特征写入 query。', '',
        '## 当前结果', '',
        '| 实验 | 最佳 AP | 验证是否使用 GT |',
        '|---|---:|---|',
        f'| 原 G5，80 轮 | {summary["原G5最佳AP"]:.4f} | 否 |',
        f'| GT 筛选，12 轮局部微调 | {summary["十二轮局部微调最佳AP"]:.4f} | 是 |',
        f'| GT 筛选，80 轮全量训练（当前） | **{metrics[0]*100:.4f}** | 是 |', '',
        f'当前最佳出现在第 **{best["epoch"]+1}** 轮；比原 G5 高 **{(metrics[0]-baseline["test_coco_eval_bbox"][0])*100:.4f} AP**，'
        f'比 12 轮局部微调高 **{(metrics[0]-adapt["best_ap"])*100:.4f} AP**。', '',
        '| 最佳轮指标 | 数值（百分制） |', '|---|---:|',
        *[f'| {label} | {value*100:.4f} |' for label, value in zip(labels, metrics)], '',
        '## 候选覆盖率', '',
        '口径：候选预测类别正确、粗框 IoU≥0.5 时计为覆盖；对每个 GT 判断是否至少存在一个候选。',
        '这是多对多存在性覆盖率，不是最终检测召回率，也不是一对一匹配覆盖率。', '',
        '| 最佳轮候选集合 | 覆盖率 |', '|---|---:|',
        f'| 同一模型原生 Top300 | {coverage["native"]*100:.4f}% |',
        f'| GT 筛选后的 300 个 | {coverage["selected"]*100:.4f}% |',
        f'| 同一模型 Top1200 池 | {coverage["pool"]*100:.4f}% |', '',
        f'统计 {coverage["images"]} 张图、{coverage["gt"]} 个 GT；平均每图替换 {coverage["replaced_per_image"]:.4f} 个 query。',
        '原 G5 固定候选池对应覆盖率为 70.7500%；该数值仅约束当时的固定候选池。', '',
        '## 最近评估记录', '', '| 轮次 | AP | 筛选后覆盖率 |', '|---|---:|---:|',
        *[f'| {r["epoch"]+1} | {r["test_coco_eval_bbox"][0]*100:.4f} | '
          f'{r["test_oracle_query_coverage"]["selected"]*100:.4f}% |' for r in rows[-8:]], '',
        '## 结论边界', '',
        '- 当前结果支持 GT 覆盖筛选与完整联合训练存在明显收益，不能直接当作无 GT 筛选器的部署成绩。',
        '- 12 轮实验冻结了候选生成分支、主干、编码器及 G5 修正头，主要更新 decoder；',
        '  本次同时改变训练范围、轮数、学习率和增强，不能将差距全部归因于从头初始化。',
        '- 使用受限贪心交换，不保证全局最优；最多 40 次交换，也不构成筛选方法的理论 AP 上限。',
        '- 尚未对本次权重测量关闭 GT 筛选的 AP，不将原生 Top300 覆盖率冒充该 AP。', '',
        '## 可追溯文件', '',
        f'- 实验目录：`{run}`。',
        '- `manifest.json`：启动命令、GPU、初始化及源码快照路径。',
        '- `source/`：实际运行的冻结源码；本次仓库整理未修改该快照。',
        '- `preflight.json`：真实数据 AMP 更新、各主分支梯度与验证流程预检。',
        '- `results/effective_config.yml`：展开后的运行配置。',
        '- `results/log.txt`：逐轮 AP、覆盖率和优化器更新统计。',
        '- `results/best_stg1.pth`、`best_stg2.pth`、`last.pth`：原生 solver 权重文件；保留阶段语义。',
        '- `results/stage_reload_events.jsonl`：阶段回载记录。',
        '- `results/status.json`：训练完成后由训练入口写入。',
        f'- 原 G5 日志：`{baseline_path}`。',
        f'- 12 轮微调结果：`{adapt_path}`。', '',
        '代码入口与后续设计分析见同目录 `g5_query_selection_review_20261109.md`。', '',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(f'已更新中文记录：{args.output}；{state}，最佳 AP={metrics[0]*100:.4f}')


if __name__ == '__main__':
    main()
