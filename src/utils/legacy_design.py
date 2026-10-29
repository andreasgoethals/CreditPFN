"""Append-only TabPFN v2/v2.6 follow-ups; original experiment YAMLs stay frozen."""
from __future__ import annotations

from contextlib import contextmanager
import os
import re
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.paths import REPO_ROOT, outputs_dir


def validate_id(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9]{1,40}", identifier):
        raise ValueError("Extension ID must be 1–40 ASCII letters/digits")
    return identifier


def root(identifier):
    return outputs_dir() / "experiment0" / "extensions" / validate_id(identifier)


@contextmanager
def scope(identifier, experiment="experiment0"):
    validate_id(identifier)
    keys = ("CREDITPFN_LEGACY_ID", "CREDITPFN_EXPERIMENT", "CREDITPFN_LORA_WORKFLOW")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update(CREDITPFN_LEGACY_ID=identifier, CREDITPFN_EXPERIMENT=experiment)
    os.environ.pop("CREDITPFN_LORA_WORKFLOW", None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def limits():
    cfg = OmegaConf.load(REPO_ROOT / "config/experiment0/legacy.yaml")
    measured = OmegaConf.load(REPO_ROOT / "config/data.yaml").finetuning.max_rows_per_epoch
    inference = OmegaConf.load(REPO_ROOT / "config/eval.yaml").max_rows_per_model
    if list(cfg.versions) != ["v2", "v2.6"]:
        raise ValueError("Extension must explicitly name v2 and v2.6")
    for version in cfg.versions:
        if cfg.training_rows[version] != measured[version] or cfg.context_rows[version] != inference[version]:
            raise ValueError("Extension limits differ from the measured/native model caps")
    if not 0 < cfg.interrupt_after < cfg.target_updates or cfg.capacity_updates < 26:
        raise ValueError("Invalid extension correctness/capacity budgets")
    return cfg


def base_name(version, track):
    if version not in ("v2", "v2.6") or track not in ("pd", "lgd"):
        raise ValueError("Unknown extension model or task")
    kind = "classifier" if track == "pd" else "regressor"
    return f"tabpfn-{version}-{kind}-{version}_default.ckpt"


def configurations(identifier):
    """Materialize configs only in the owned manifest scope, never overwrite originals."""
    from src.train.config import load_train_config
    validate_id(identifier)
    caps = limits()
    configs = {}
    directory = root(identifier) / "configs" / "production"
    for version in caps.versions:
        tag = version.replace(".", "p")
        for experiment, track, rows in (
            (5, "pd", 8192), (5, "lgd", 8192), (6, "pd", 8192),
            (7, "pd", 2048), (7, "pd", int(caps.training_rows[version])),
            (8, "pd", int(caps.training_rows[version])), (8, "lgd", int(caps.training_rows[version])),
        ):
            source = (f"config/experiment7/pd_rows2048.yaml" if experiment == 7 else
                      f"config/experiment{experiment}/{track}.yaml")
            cfg = load_train_config(config_path=str(REPO_ROOT / source))
            name = f"e{experiment}_{track}_{tag}" + (f"_rows{rows}" if experiment == 7 else "")
            prefix = {5: "cpt_diagscale", 6: "cpt_corpus", 7: f"cpt_episode_rows{rows}", 8: "cpt_lora"}[experiment]
            cfg.run_name = f"{prefix}_legacy_{identifier}_{tag}"
            cfg.experiment.update(extension_id=identifier, legacy_gate=True, lora_gate=False, followup_gate=False)
            cfg.tunable["classifier_base_paths" if track == "pd" else "regressor_base_paths"] = ["checkpoints/" + base_name(version, track)]
            cfg.train.max_rows_per_step = rows
            cfg.checkpoint.trained_dir = f"checkpoints/trained/experiment{experiment}/extensions/{identifier}"
            if experiment == 7:
                cfg.experiment.reference_config = str(directory / f"e6_pd_{tag}.yaml")
            configs[name] = cfg
        cfg = OmegaConf.load(REPO_ROOT / "config/experiment4/context.yaml")
        cfg.run_name = f"context_legacy_{identifier}_{tag}"
        cfg.extension_id = identifier
        cfg.base_version = version
        cfg.families = ["tabpfn"]
        cfg.max_context_rows = int(caps.context_rows[version])
        cfg.context_rows = sorted(set([n for n in cfg.context_rows if n <= cfg.max_context_rows] + [cfg.max_context_rows]))
        configs[f"e4_{tag}"] = cfg
    return configs


def write_configs(identifier):
    paths = {}
    directory = root(identifier) / "configs" / "production"
    directory.mkdir(parents=True, exist_ok=True)
    for key, cfg in configurations(identifier).items():
        path = directory / f"{key}.yaml"
        if path.exists():
            if OmegaConf.load(path) != cfg:
                raise RuntimeError("Generated extension config changed; choose a fresh ID")
        else:
            with path.open("x", encoding="utf-8") as stream:
                OmegaConf.save(cfg, stream)
        paths[key] = path
    return paths
