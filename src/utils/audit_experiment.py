"""Check prepared trial coverage, budgets, trajectories and zero-LR save/reload parity.

This is read-only unless --report is supplied. Run on a CPU compute node.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.checkpoint_inventory import resolve_checkpoint
from src.utils.experiment import apply_split_index, file_digest
from src.utils.paths import manifests_dir, resolve_base_checkpoint, resolve_staging_path
from src.utils.prepare_experiment import plan_path


def compare_states(base: Path, trained: Path) -> dict:
    import torch
    # These are the trusted, explicitly configured upstream/project checkpoint files.
    before = torch.load(base, map_location="cpu", weights_only=False)["state_dict"]
    after = torch.load(trained, map_location="cpu", weights_only=False)["state_dict"]
    missing, added = sorted(before.keys() - after.keys()), sorted(after.keys() - before.keys())
    changed = [k for k in before.keys() & after.keys()
               if before[k].shape != after[k].shape or not torch.equal(before[k], after[k])]
    return {"equal": not (missing or added or changed), "missing": missing, "added": added, "changed": changed}


def null_monitor_parity(frame: pd.DataFrame) -> bool:
    columns = [c for c in frame if c.startswith("metric__")]
    if len(frame) != 2 or not columns:
        return False
    values = frame.sort_values("successful_updates")[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and np.allclose(values[0], values[1], rtol=1e-6, atol=1e-7))


def audit(config: Path, *, null=False) -> dict:
    from scripts.train_pipeline import _load_cfg, _resolve_grid
    from src.train.loop import descriptive_name
    cfg = _load_cfg(config_path=str(config))
    plan = json.loads(plan_path(str(cfg.run_name), str(cfg.track)).read_text(encoding="utf-8"))
    report = {"run": str(cfg.run_name), "track": str(cfg.track), "expected": len(plan["trials"]),
              "completed": 0, "diverged": 0, "pending": 0, "problems": [], "trials": [], "timing": []}
    timings = {}
    for index in range(int(cfg.corpus.n_splits)):
        current = apply_split_index(OmegaConf.create(OmegaConf.to_container(cfg)), index)
        for trial_index, trial in enumerate(_resolve_grid(current, single=False)):
            base, lr, frozen, query, accum, mode, min_rows, l2sp = trial
            name = descriptive_name(run_name=current.run_name, track=current.track, base_path=base,
                learning_rate=lr, seed=current.seed, use_lora=frozen, query_fraction=query,
                accumulate_grad_batches=accum, epoch_pass_mode=mode, min_train_rows=min_rows,
                l2sp_lambda=l2sp, adaptation_mode="frozen_backbone" if frozen else "full").removesuffix(".ckpt")
            path, prov = resolve_checkpoint(str(resolve_staging_path(current.checkpoint.trained_dir) /
                                               current.track / (name + ".ckpt")), current.track)
            row = {"trial": name, "status": "PENDING"}
            if path is None:
                report["pending"] += 1
                report["trials"].append(row)
                continue
            expected = plan["trials"][f"{current.run_name}/{trial_index}"]
            if prov.get("trial_identity", {}).get("sha256") != expected:
                report["problems"].append(f"{name}: checkpoint does not match the prepared identity")
            divergent = bool(prov.get("diverged", False))
            row.update(status="DIVERGED" if divergent else "OK", path=str(path),
                       successful_updates=prov.get("successful_updates"), adaptation=prov.get("adaptation"))
            report["diverged" if divergent else "completed"] += 1
            target = int(cfg.train.target_total_steps)
            if not divergent and prov.get("successful_updates") != target:
                report["problems"].append(f"{name}: incorrect successful-update budget")
            trajectory = manifests_dir() / "epochs" / current.track / (name + ".trajectory.csv")
            frame = pd.read_csv(trajectory) if trajectory.is_file() else pd.DataFrame()
            observed = frame.get("successful_updates", pd.Series(dtype=int)).tolist()
            if not divergent and observed != list(cfg.train.trajectory_steps):
                report["problems"].append(f"{name}: missing/duplicate trajectory measurements")
            if null:
                if lr != 0:
                    raise ValueError("--null requires a zero-learning-rate configuration")
                state = compare_states(resolve_base_checkpoint(base), path)
                row.update(null_state=state, null_monitor_equal=null_monitor_parity(frame))
                if not state["equal"] or not row["null_monitor_equal"]:
                    report["problems"].append(f"{name}: zero-LR state or monitor parity failed")
            history_path = trajectory.with_name(name + ".csv")
            if history_path.is_file():
                epochs = pd.read_csv(history_path)
                seconds = float(epochs.get("training_seconds", pd.Series(dtype=float)).sum())
                updates = int(prov.get("successful_updates", 0))
                if updates and seconds > 0:
                    key = (Path(base).stem, mode, bool(frozen))
                    timings.setdefault(key, []).append((seconds / updates,
                        float(frame.get("monitor_seconds", pd.Series(dtype=float)).max())))
            report["trials"].append(row)
    for (base, mode, frozen), values in timings.items():
        rate = max(v[0] for v in values)
        monitor = max(v[1] for v in values if math.isfinite(v[1])) if any(math.isfinite(v[1]) for v in values) else 0
        # Conservative extrapolation, not a measured guarantee; include five monitors/startup.
        minutes = math.ceil((5000 * rate + 5 * monitor + 600) * 1.3 / 60)
        report["timing"].append({"base": base, "mode": mode, "frozen": frozen,
            "seconds_per_update": rate, "max_monitor_seconds": monitor,
            "provisional_5000_update_walltime_minutes": minutes})
    report["passed"] = report["pending"] == 0 and not report["problems"] and (not null or report["diverged"] == 0)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.config, null=args.null)
    if args.report:
        from src.utils.atomic import write_json
        write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
