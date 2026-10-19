"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def _load_stage_checkpoint(self, epoch, reason):
        """Keep the original reload, recording its actual state for the DDF study."""
        path = str(self.output_dir / "best_stg1.pth")
        before = self.last_epoch
        restored = self.load_resume_state(path)
        if not self.cfg.yaml_cfg.get("DFINECriterion", {}).get("log_ddf_diagnostics", False):
            return None
        event = {
            "outer_epoch": epoch,
            "reason": reason,
            "checkpoint": path,
            "internal_epoch_before": before,
            "internal_epoch_after": self.last_epoch,
            **restored,
        }
        if dist_utils.is_main_process():
            with (self.output_dir / "stage_reload_events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        return event

    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in (key for key in test_stats if key in ("coco_eval_bbox", "coco_eval_masks")):
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        trace_training = args.yaml_cfg.get("DFINECriterion", {}).get("log_ddf_diagnostics", False)
        optimizer_updates_total = 0
        for epoch in range(start_epoch, args.epochs):
            internal_epoch_before = self.last_epoch
            reload_events = []
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                reload_events.append(self._load_stage_checkpoint(epoch, "stage_transition"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            lr_before = [group["lr"] for group in self.optimizer.param_groups] if trace_training else None
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
                log_optimizer_updates=trace_training,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1
            internal_epoch_after_training = self.last_epoch
            optimizer_updates_total += train_stats.get("optimizer_updates", 0)

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # Oracle evaluations also contain metadata and geometry dictionaries;
            # only the COCO performance arrays participate in checkpoint selection.
            test_prefix = "OracleTest" if test_stats.get("oracle_evaluation", False) else "Test"
            for k in (key for key in test_stats if key in ("coco_eval_bbox", "coco_eval_masks")):
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"{test_prefix}/{k}_{i}", v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        reload_events.append(self._load_stage_checkpoint(epoch, "stage2_ema_refresh"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if trace_training:
                log_stats["training_trace"] = {
                    "outer_epoch": epoch,
                    "internal_epoch_before": internal_epoch_before,
                    "internal_epoch_after_training": internal_epoch_after_training,
                    "internal_epoch_after_validation": self.last_epoch,
                    "optimizer_updates_total": optimizer_updates_total,
                    "lr_before_training": lr_before,
                    "lr_after_validation": [group["lr"] for group in self.optimizer.param_groups],
                    "ema_updates_after_validation": self.ema.updates if self.ema else None,
                    "reload_events": reload_events,
                }

            if self.use_wandb:
                wandb_logs = {}
                metric_prefix = "oracle_metrics" if test_stats.get("oracle_evaluation", False) else "metrics"
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"{metric_prefix}/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )
            if test_stats.get("oracle_evaluation", False) and dist_utils.is_main_process():
                with (self.output_dir / "oracle_eval.json").open("w") as handle:
                    json.dump(test_stats, handle, indent=2)

        return
