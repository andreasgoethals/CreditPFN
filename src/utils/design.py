"""Materialize the second campaign's training configs from ``config/design.yaml``.

The reference recipe is written once; every experiment arm is the reference plus its
declared deviations. ``materialize`` writes one phase YAML per experiment, track and arm
group; ``undeclared_differences`` is what tests use to prove nothing else changed.

    python -m src.utils.design --write      # regenerates config/experiments/
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
    "single_source": "corpus.single_source",
    "grad_clip_norm": "train.grad_clip_norm",
}
#: Paths a deviation legitimately changes as a consequence (bookkeeping, not science).
CONSEQUENCES = {
    "seed": ("experiment.training_seeds",),
    "successful_updates": ("train.trajectory_steps",),
    "single_source": ("corpus.n_folds", "corpus.n_splits", "corpus.fold"),
    "corpus": ("corpus.train_size_policy", "corpus.train_source"),
    "trainable": ("tunable.frozen_backbone", "lora"),
}
LARGE_TABLE_ROWS = 10001
#: Generated configs: one folder per experiment, ``E<NN>_<slug>``; Experiment 0 is ``E00_pilots``.
CONFIG_ROOT = ROOT / "config/experiments"
PILOT_FOLDER = CONFIG_ROOT / "E00_pilots"


def load_design(path: Path = DESIGN) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def monitor_updates(budget: int) -> list[int]:
    """0, 50, 100, 150, 200, 250, every 250 updates to 2,500, then every 500 to the budget.

    A monitor point is a full inference pass over every monitored table with the GPU otherwise
    idle (about 45 s for TabPFN v2 in Experiment 0). The first study's curves were flat after
    2,500 updates, so the later part of a run is sampled at half the density."""
    points = [0, 50, 100, 150, 200, 250]
    points += list(range(500, min(budget, 2500) + 1, 250))
    points += list(range(3000, budget + 1, 500))
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
                  "adaptation_mode": "full", "regression_objective": str(ref["regression_objective"]),
                  "target_total_steps": budget, "epochs": 50, "max_epochs_for_step_budget": 8000,
                  "numerical_stopping_only": True, "monitor_seed": 31415,
                  "trajectory_steps": monitor_updates(budget),
                  "trajectory_rows": [int(v) for v in ref["monitor"]["processed_rows"]],
                  "epoch_eval_subsample_samples": int(ref["monitor"]["rows_per_table"]),
                  "epoch_eval_n_estimators": int(ref["monitor"]["preprocessing_views"]),
                  "epoch_eval_n_estimators_tabicl": int(ref["monitor"]["preprocessing_views"]),
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
            if value == "tables_above_10000_rows":
                cfg["corpus"]["min_train_rows"] = LARGE_TABLE_ROWS
            elif value in ("small_only", "large_matched"):
                cfg["corpus"]["train_size_policy"] = str(value)
            elif value == "domain_control":
                cfg["corpus"]["train_source"] = "domain_control"
            else:
                raise ValueError(f"Unknown corpus policy {value!r}")
        elif key == "single_source":
            # Train on one table (index into the id-sorted registry), evaluate on all others.
            cfg["corpus"]["single_source"] = int(value)
            cfg["corpus"]["n_folds"] = None
            cfg["corpus"]["n_splits"] = 1
            cfg["corpus"]["fold"] = 0
        elif key == "trainable":
            if value == "frozen_backbone":
                cfg["tunable"]["frozen_backbone"] = [True]
            elif value == "diagscale":
                cfg["train"]["adaptation_mode"] = "diagscale"
            elif value == "lora_qv_r16":
                cfg["train"]["adaptation_mode"] = "lora"
                cfg["lora"] = {"enabled": False, "r": 16, "alpha": 32, "dropout": 0.0,
                               "target_modules": ["attention_qv"]}
                # L2-SP keeps the reference lambda, applied to the effective update (alpha/r)BA.
            else:
                raise ValueError(f"Unknown trainable scope {value!r}")
        elif key == "models":
            cfg["tunable"][axis] = [BASES[track][m] for m in value]
        elif key == "regression_objective":
            cfg["train"]["regression_objective"] = str(value)
        elif key == "grad_clip_norm":
            cfg["train"]["grad_clip_norm"] = None if value is None else float(value)
    return cfg


def arm_configs(design: dict | None = None) -> dict[tuple[str, str, int], dict]:
    """{(experiment, track, arm index): config} for every training arm in the launch grid."""
    design = design or load_design()
    result = {}
    for name, spec in design["experiments"].items():
        if spec.get("kind", "training") in ("inference", "finetune"):
            continue
        slug = spec["slug"]
        for track in spec["tasks"]:
            for index, arm in enumerate(experiment_arms(name, track, design)):
                cfg = apply_arm(reference_config(track, design), arm, track)
                cfg["run_name"] = f"cpt2_{slug}_a{index}"
                cfg["experiment"]["output_group"] = slug
                cfg["checkpoint"]["trained_dir"] = f"checkpoints/trained/{slug}"
                result[(name, track, index)] = cfg
    return result


def experiment_arms(name: str, track: str, design: dict | None = None) -> list[dict]:
    """The declared arms of a training/transfer experiment (transfer arms are generated)."""
    design = design or load_design()
    spec = design["experiments"][name]
    if spec.get("kind", "training") == "transfer":
        return [{"single_source": i, "successful_updates": int(spec["successful_updates"]),
                 "learning_rate": float(spec["learning_rate"])}
                for i in range(int(design["tables"][track]))]
    return list(spec["arms"])


def experiment_by_slug(slug: str, design: dict | None = None) -> tuple[str, dict]:
    """(number, spec) of the experiment with this slug; code refers to slugs, never numbers."""
    design = design or load_design()
    matches = [(name, spec) for name, spec in design["experiments"].items() if spec["slug"] == slug]
    if len(matches) != 1:
        raise KeyError(f"No unique experiment with slug {slug!r}")
    return matches[0]


PILOT_DIR = "checkpoints/trained/experiment0"


def experiment_folder(name: str, design: dict | None = None) -> Path:
    """``config/experiments/E<NN>_<slug>`` for an experiment number such as ``E5``."""
    design = design or load_design()
    return CONFIG_ROOT / f"E{int(name[1:]):02d}_{design['experiments'][name]['slug']}"


def config_path(slug: str, track: str, arm: int = 0, design: dict | None = None) -> Path:
    """Generated config of one arm, addressed by the stable experiment slug."""
    name, _ = experiment_by_slug(slug, design)
    return experiment_folder(name, design) / f"{track}_a{int(arm)}.yaml"


def _pilot(cfg: dict, run_name: str, partition: int) -> dict:
    cfg["run_name"] = run_name
    cfg["experiment"]["output_group"] = "experiment0"
    cfg["checkpoint"]["trained_dir"] = PILOT_DIR
    # One partition only. split_index selects it; n_splits=1 keeps the plan to that partition.
    cfg["corpus"]["n_splits"] = 1
    cfg["corpus"]["fold"] = int(partition)
    return cfg


def pilot_configs(design: dict | None = None) -> dict[tuple[str, str], dict]:
    """{(pilot, track): config} for Experiment 0: the reference plus each pilot's declared arm."""
    design = design or load_design()
    spec = design["experiment0"]
    result = {}
    for name, pilot in spec["pilots"].items():
        for track in pilot["tasks"]:
            cfg = _pilot(apply_arm(reference_config(track, design), pilot["arm"], track),
                         f"cpt2e0_{name}", spec["partition"])
            if "single_source" in pilot["arm"]:
                cfg["corpus"]["fold"] = 0
            if "monitor_rows" in pilot:
                cfg["train"]["epoch_eval_subsample_samples"] = int(pilot["monitor_rows"])
            result[(name, track)] = cfg
    for variant in spec["recovery"]["variants"]:
        for track in design["tasks"]:
            for arm in ("uninterrupted", "resumed"):
                cfg = apply_arm(reference_config(track, design), recovery_arm(variant, design), track)
                cfg["train"]["deterministic"] = True   # exact-resume comparison needs strict kernels
                cfg["train"]["recovery_every_updates"] = 10
                cfg["train"]["trajectory_rows"] = []   # update milestones only: exact comparison
                result[(f"recovery_{variant}_{arm}", track)] = _pilot(
                    cfg, f"cpt2e0_recovery_{variant}_{arm}", spec["partition"])
    return result


#: Experiment-0 bookkeeping that is not a scientific deviation.
PILOT_BOOKKEEPING = ("corpus.n_splits", "corpus.fold", "train.epoch_eval_subsample_samples")
RECOVERY_BOOKKEEPING = ("train.deterministic", "train.recovery_every_updates", "train.trajectory_rows")


def recovery_arm(variant: str, design: dict | None = None) -> dict:
    design = design or load_design()
    recovery = design["experiment0"]["recovery"]
    return {"learning_rate": float(recovery["learning_rate"]),
            "successful_updates": int(recovery["successful_updates"]), **recovery["variants"][variant]}


def materialize_experiment0(target: Path | None = None, design: dict | None = None) -> list[Path]:
    """Write the Experiment 0 pilot configs into ``config/experiments/E00_pilots/``."""
    target = Path(target) if target is not None else PILOT_FOLDER
    written = []
    for (name, track), cfg in pilot_configs(design).items():
        path = target / f"{name}_{track}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# GENERATED by src.utils.design from config/design.yaml (experiment0 {name}). Do not edit.\n"
        path.write_text(header + OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
        written.append(path)
    return written


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


def materialize(target: Path | None = None, design: dict | None = None) -> list[Path]:
    """Write every campaign config into ``config/experiments/E<NN>_<slug>/<track>_a<i>.yaml``."""
    design = design or load_design()
    root = Path(target) if target is not None else CONFIG_ROOT
    written = []
    for (name, track, index), cfg in arm_configs(design).items():
        folder = root / experiment_folder(name, design).name
        path = folder / f"{track}_a{index}.yaml"
        folder.mkdir(parents=True, exist_ok=True)
        header = f"# GENERATED by src.utils.design from config/design.yaml ({name}, arm {index}). Do not edit.\n"
        path.write_text(header + OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Regenerate config/experiments/ (campaign arms and Experiment 0 pilots)")
    args = parser.parse_args()
    if args.write:
        for folder in sorted(CONFIG_ROOT.glob("E*_*")):
            for old in folder.glob("*.yaml"):
                old.unlink()
        for path in [*materialize(), *materialize_experiment0()]:
            print(path.relative_to(ROOT))
    else:
        for key, cfg in arm_configs().items():
            print(key, cfg["run_name"])


if __name__ == "__main__":
    main()
