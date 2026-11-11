#!/usr/bin/env python3
"""80-epoch selection-only scratch training using the native G5 solver/stage protocol."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver import TASKS


def float32_tree(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: float32_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float32_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(float32_tree(v) for v in value)
    return value


class Float32Criterion(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    @property
    def last_ddf_stats(self):
        return self.criterion.last_ddf_stats

    def forward(self, outputs, targets, **kwargs):
        return self.criterion(float32_tree(outputs), targets, **kwargs)


def build_config(path):
    cfg = YAMLConfig(path)
    # Paired common tensors start identically; model init does not consume data RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.yaml_cfg.get('seed', 0))
        _ = cfg.model
    cfg._criterion = Float32Criterion(cfg.criterion)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    dist_utils.setup_distributed(seed=0)
    cfg = build_config(args.config)
    if dist_utils.is_main_process():
        import yaml
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'effective_config.yml').write_text(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False))
    TASKS[cfg.yaml_cfg['task']](cfg).fit()
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
