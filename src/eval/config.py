"""Load evaluation settings alongside the matching training phase and partition."""
from omegaconf import OmegaConf

from src.train.config import load_train_config
from src.utils.experiment import apply_split_index


def load_eval_configs(eval_overrides: list[str], train_overrides: list[str],
                      config_path: str | None = None, split_index: int | None = None):
    eval_cfg = OmegaConf.load("config/eval.yaml")
    cli = OmegaConf.from_dotlist(eval_overrides)
    # Resolve an explicit base-path override before loading the training settings.
    base_path = OmegaConf.merge(eval_cfg, cli).train_cfg_path
    train_cfg = apply_split_index(load_train_config(
        train_overrides, config_path, base_path=str(base_path)), split_index)
    if "evaluation" in train_cfg:
        eval_cfg = OmegaConf.merge(eval_cfg, train_cfg.evaluation)
    # Explicit CLI choices take precedence over the phase's evaluation defaults.
    return OmegaConf.merge(eval_cfg, cli), train_cfg
