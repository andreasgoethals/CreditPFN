"""Materialize the second campaign's training configs from ``config/design.yaml``.

The reference recipe is written once; every experiment arm is the reference plus its
declared deviations. ``materialize`` writes one phase YAML per experiment, track and arm
group; ``undeclared_differences`` is what tests use to prove nothing else changed.

    python -m src.utils.design --write config/campaign
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "config/design.yaml"
BASES = {
    "pd": {"v2": "checkpoints/tabpfn-v2-classifier-v2_default.ckpt",
           "v2.6": "checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt",
           "v3": "checkpoints/tabpfn-v3-classifier-v3_default.ckpt",
           "tabicl-v2": "checkpoints/tabicl-classifier-v2-20260212.ckpt"},
    "lgd": {"v2": "checkpoints/tabpfn-v2-regressor-v2_default.ckpt",
            "v2.6": "checkpoints/tabpfn-v2.6-regressor-v2.6_default.ckpt",
            "v3": "checkpoints/tabpfn-v3-regressor-v3_default.ckpt",
            "tabicl-v2": "checkpoints/tabicl-regressor-v2-20260212.ckpt"},
}
#: Design keys an arm may change, and the config path each one controls.
DEVIATION_PATHS = {
    "learning_rate": "tunable.learning_rates",
    "l2sp_lambda": "tunable.l2sp_lambdas",
    "weight_decay": "optimizer.weight_decay",
    "optimizer": "optimizer.name",
    "seed": "seed",
    "successful_updates": "train.target_total_steps",
    "episode_rows": "train.max_rows_per_step",
    "query_fraction": "tunable.query_fractions",
    "pd_class_sampling": "train.context_sampling",
    "exposure": "tunable.epoch_pass_modes",
    "corpus": "corpus.min_train_rows",
    "trainable": "train.adaptation_mode",
    "models": "tunable.base_paths",
    "regression_objective": "train.regression_objective",
}
#: Paths a deviation legitimately changes as a consequence (bookkeeping, not science).
CONSEQUENCES = {
    "seed": ("experiment.training_seeds",),
    "successful_updates": ("train.trajectory_steps",),
    "trainable": ("tunable.frozen_backbone", "lora", "optimizer.l2sp_lambda", "tunable.l2sp_lambdas"),
}
LARGE_TABLE_ROWS = 10001


def load_design(path: Path = DESIGN) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def monitor_updates(budget: int) -> list[int]:
    """0, 50, 100, 150, 200, 250, then every 250 updates up to and including the budget."""
    points = [0, 50, 100, 150, 200, 250]
    points += list(range(500, budget + 1, 250))
    return sorted(set(p for p in points if p <= budget) | {budget})


def reference_config(track: str, design: dict | None = None) -> dict:
    design = design or load_design()
    ref = design["reference"]
    if track not in design["tasks"]:
        raise ValueError(f"Unknown task {track!r}")
    sampling = {"proportional": "stratified", "balanced": "balanced"}[ref["pd_class_sampling"]]
    budget = int(ref["successful_updates"])
    return {
        "track": track,
        "seed": int(ref["seed"]),
        "run_name": "cpt2_reference",
        "experiment": {"output_group": "reference", "fingerprint": True, "require_plan": True,
                       "training_seeds": [int(ref["seed"])]},
        "corpus": {"require_complete_registry": True, "min_train_rows": 0, "n_test_datasets": None,
                   "n_folds": int(design["partitions"]), "fold": 0, "split_seed": 1729,
                   "n_splits": int(design["partitions"]), "train_dataset_ids": None,
                   "test_dataset_ids": None},
        "optimizer": {"name": ref["optimizer"], "weight_decay": float(ref["weight_decay"]),
                      "l2sp_lambda": float(ref["l2sp_lambda"])},
        "scheduler": {"warmup_fraction": float(ref["warmup_fraction"]),
                      "min_lr_fraction": float(ref["min_lr_fraction"])},
        "train": {"context_sampling": "stratified" if track == "lgd" else sampling,
                  "adaptation_mode": "full", "regression_objective": "bar_nll",
                  "target_total_steps": budget, "epochs": 50, "max_epochs_for_step_budget": 8000,
                  "numerical_stopping_only": True, "monitor_seed": 31415,
                  "trajectory_steps": monitor_updates(budget),
                  "epoch_eval_subsample_samples": int(ref["monitor"]["rows_per_table"]),
                  "epoch_eval_n_estimators": 4, "epoch_eval_n_estimators_tabicl": 4,
                  "recovery_every_updates": 500, "dataloader_workers": 4, "prefetch_factor": 2,
                  "step_log_interval": 100, "grad_clip_norm": float(ref["grad_clip_norm"]),
                  "amp": ref["training_precision"] == "bf16", "max_rows_per_step": None},
        "tunable": {"learning_rates": [float(ref["learning_rate"])],
                    "l2sp_lambdas": [float(ref["l2sp_lambda"])],
                    "frozen_backbone": [False], "epoch_pass_modes": [ref["exposure"]],
                    "query_fractions": [float(ref["query_fraction"])],
                    "accumulate_grad_batches": [1], "adapter_families": [],
                    "classifier_base_paths" if track == "pd" else "regressor_base_paths":
                        [BASES[track][m] for m in design["models"]]},
        "evaluation": {"cache_controls": True, "results": {"save_predictions": True},
                       "tabicl_inference_options": {"use_amp": False, "use_fa3": False}},
        "checkpoint": {"trained_dir": "checkpoints/trained/reference"},
    }


def _set(cfg: dict, dotted: str, value):
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def apply_arm(cfg: dict, arm: dict, track: str) -> dict:
    cfg = copy.deepcopy(cfg)
    axis = "classifier_base_paths" if track == "pd" else "regressor_base_paths"
    for key, value in arm.items():
        if key not in DEVIATION_PATHS:
            raise ValueError(f"Undeclared deviation key {key!r}; add it to DEVIATION_PATHS")
        if key == "learning_rate":
            cfg["tunable"]["learning_rates"] = [float(value)]
        elif key == "l2sp_lambda":
            cfg["tunable"]["l2sp_lambdas"] = [float(value)]
            cfg["optimizer"]["l2sp_lambda"] = float(value)
        elif key == "weight_decay":
            cfg["optimizer"]["weight_decay"] = float(value)
        elif key == "optimizer":
            cfg["optimizer"]["name"] = str(value)
        elif key == "seed":
            cfg["seed"] = int(value)
            cfg["experiment"]["training_seeds"] = [int(value)]
        elif key == "successful_updates":
            cfg["train"]["target_total_steps"] = int(value)
            cfg["train"]["trajectory_steps"] = monitor_updates(int(value))
        elif key == "episode_rows":
            cfg["train"]["max_rows_per_step"] = int(value)
        elif key == "query_fraction":
            cfg["tunable"]["query_fractions"] = [float(value)]
        elif key == "pd_class_sampling":
            if track != "pd":
                raise ValueError("Class sampling is a PD-only deviation")
            cfg["train"]["context_sampling"] = {"proportional": "stratified", "balanced": "balanced"}[value]
        elif key == "exposure":
            cfg["tunable"]["epoch_pass_modes"] = [str(value)]
        elif key == "corpus":
            if value != "tables_above_10000_rows":
                raise ValueError(f"Unknown corpus policy {value!r}")
            cfg["corpus"]["min_train_rows"] = LARGE_TABLE_ROWS
        elif key == "trainable":
            if value == "frozen_backbone":
                cfg["tunable"]["frozen_backbone"] = [True]
            elif value == "diagscale":
                cfg["train"]["adaptation_mode"] = "diagscale"
            elif value == "lora_qv_r16":
                cfg["train"]["adaptation_mode"] = "lora"
                cfg["lora"] = {"enabled": False, "r": 16, "alpha": 32, "dropout": 0.0,
                               "target_modules": ["attention_qv"]}
                # L2-SP is undefined for zero-initialised factors (see RESEARCH_BRIEF).
                cfg["tunable"]["l2sp_lambdas"] = [0.0]
                cfg["optimizer"]["l2sp_lambda"] = 0.0
            else:
                raise ValueError(f"Unknown trainable scope {value!r}")
        elif key == "models":
            cfg["tunable"][axis] = [BASES[track][m] for m in value]
        elif key == "regression_objective":
            cfg["train"]["regression_objective"] = str(value)
    return cfg


def arm_configs(design: dict | None = None) -> dict[tuple[str, str, int], dict]:
    """{(experiment, track, arm index): config} for every training arm in the launch grid."""
    design = design or load_design()
    result = {}
    for name, spec in design["experiments"].items():
        if spec.get("kind") == "inference":
            continue
        for track in spec["tasks"]:
            for index, arm in enumerate(spec["arms"]):
                cfg = apply_arm(reference_config(track, design), arm, track)
                slug = spec["slug"]
                cfg["run_name"] = f"cpt2_{slug}_a{index}"
                cfg["experiment"]["output_group"] = slug
                cfg["checkpoint"]["trained_dir"] = f"checkpoints/trained/{slug}"
                result[(name, track, index)] = cfg
    return result


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, path + "."))
        else:
            out[path] = value
    return out


IDENTITY_PATHS = ("run_name", "experiment.output_group", "checkpoint.trained_dir")


def undeclared_differences(cfg: dict, arm: dict, track: str, design: dict | None = None) -> list[str]:
    """Config paths that differ from the reference but are not explained by the arm."""
    reference = flatten(reference_config(track, design))
    candidate = flatten(cfg)
    allowed = set(IDENTITY_PATHS)
    for key in arm:
        allowed.add(DEVIATION_PATHS[key])
        allowed.update(CONSEQUENCES.get(key, ()))
    if "l2sp_lambda" in arm:
        allowed.add("optimizer.l2sp_lambda")
    if "models" in arm:
        allowed.update({"tunable.classifier_base_paths", "tunable.regressor_base_paths"})
    differing = {p for p in set(reference) | set(candidate) if reference.get(p) != candidate.get(p)}
    return sorted(p for p in differing
                  if p not in allowed and not any(p.startswith(a + ".") for a in allowed))


def materialize(target: Path, design: dict | None = None) -> list[Path]:
    written = []
    for (name, track, index), cfg in arm_configs(design).items():
        path = target / cfg["experiment"]["output_group"] / f"{track}_a{index}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# GENERATED by src.utils.design from config/design.yaml ({name}, arm {index}). Do not edit.\n"
        path.write_text(header + OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="Folder to write generated phase configs into")
    args = parser.parse_args()
    if args.write:
        for path in materialize(args.write):
            print(path)
    else:
        for key, cfg in arm_configs().items():
            print(key, cfg["run_name"])


if __name__ == "__main__":
    main()
