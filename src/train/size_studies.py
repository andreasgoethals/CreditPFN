"""Paired corpus-policy and episode-size designs; no changes to older campaigns.

Partition the complete registry first. Eligibility is a training-side intervention,
never a reason to redraw the evaluation population. Episode size varies on one
fixed large-table pool, with an explicitly shared 8,192-row control.
"""
from __future__ import annotations

from copy import deepcopy

from omegaconf import OmegaConf

REFERENCE_CONFIG = "config/experiment8/pd.yaml"
REFERENCE_ROWS = 8192
LARGE_MINIMUM = 10001


def study(cfg):
    return OmegaConf.select(cfg, "experiment.size_study", default=None)


def validate_config(cfg):
    """Fail before expensive work if a study silently changes its estimand."""
    kind = study(cfg)
    if kind is None:
        return
    if kind not in {"corpus_policy", "episode_size"}:
        raise ValueError("Unknown size study")
    from src.train.config import resolve_grid, training_members
    grid = resolve_grid(cfg, single=False)
    if str(cfg.track) != "pd" or int(cfg.corpus.n_folds) != 4:
        raise ValueError("Size studies require the four original PD dataset partitions")
    if cfg.corpus.train_dataset_ids or cfg.corpus.test_dataset_ids or cfg.corpus.n_test_datasets:
        raise ValueError("Size studies cannot override or redraw dataset partitions")
    if int(cfg.corpus.split_seed) != 1729 or not cfg.corpus.require_complete_registry:
        raise ValueError("Size studies require the complete original registry and partition seed")
    if any(t[2] or t[5] != "one_sample" or t[3] != .4 or t[4] != 1 for t in grid):
        raise ValueError("Size studies require full updates and one 60/40 episode per table visit")
    if cfg.train.context_sampling != "stratified" or cfg.train.get("adaptation_mode", "full") != "full":
        raise ValueError("Size studies require proportional sampling and full parameter updates")
    if any(training_members(cfg, "pd", family) != 2 for family in ("tabpfn", "tabicl")):
        raise ValueError("Size studies use two preprocessing views per update")
    if {t[6] for t in grid} != ({0, LARGE_MINIMUM} if kind == "corpus_policy" else {LARGE_MINIMUM}):
        raise ValueError("Unexpected training-corpus eligibility levels")
    cap = cfg.train.max_rows_per_step
    extension = OmegaConf.select(cfg, "experiment.extension_id")
    allowed = {REFERENCE_ROWS} if kind == "corpus_policy" else ({2048, 10000, 11000} if extension else {2048, 26000})
    if cap not in allowed:
        raise ValueError("Unexpected episode cap; the 8192-row reference belongs to experiment 6")
    data = OmegaConf.load("config/data.yaml").finetuning
    for base, *_ in grid:
        family = ("tabicl" if "tabicl-" in base else "v3" if "tabpfn-v3-" in base else
                  "v2.6" if extension and "tabpfn-v2.6-" in base else
                  "v2" if extension and "tabpfn-v2-" in base else None)
        if family is None:
            raise ValueError("Size studies require explicit TabPFN v3 or TabICLv2 weights")
        if cap > int(data.max_rows_per_epoch[family]):
            raise ValueError("Requested episode size exceeds a measured row cap")
        if extension and kind == "episode_size" and cap not in {2048, int(data.max_rows_per_epoch[family])}:
            raise ValueError("Legacy episode study uses 2048 or this model's measured ceiling")
        if data.max_cells_per_epoch.get(family, data.max_cells_per_epoch.get("default")):
            raise ValueError("A cell budget would silently shorten episodes in the size study")


def validate_split(cfg, split, minimum):
    if study(cfg) is None:
        return
    validate_config(cfg)
    if len(split.train) < 3:
        raise ValueError("Size study needs at least three eligible training tables in every partition")
    if {r.dataset_id for r in split.train} & {r.dataset_id for r in split.test}:
        raise ValueError("A held-out table entered the size-study training pool")
    if int(minimum) == LARGE_MINIMUM and any(r.n_rows < int(cfg.train.max_rows_per_step) for r in split.train):
        raise ValueError("The fixed large-table pool cannot fill every requested episode. "
                         "Reconsider the cap prospectively; never silently filter or repartition.")


def validate_runtime_cap(cfg, row_cap, cell_cap):
    if study(cfg) is not None and (int(row_cap) != int(cfg.train.max_rows_per_step) or cell_cap):
        raise ValueError("Runtime capacity settings would change the declared size-study episode cap")


def arm_inventory(cfg, split, minimum):
    """Record the policy's actual exposure, including small uncapped tables."""
    cap = int(cfg.train.max_rows_per_step)
    return {"min_train_rows": int(minimum), "episode_cap": cap,
            "training_tables": len(split.train), "training_rows": sum(r.n_rows for r in split.train),
            "tables": [{"id": r.dataset_id, "available_rows": r.n_rows,
                        "episode_rows": min(r.n_rows, cap),
                        "query_rows": max(1, round(min(r.n_rows, cap) * .4)),
                        "visit_share": 1 / len(split.train)} for r in split.train]}


def reference_signature(spec):
    """Only the episode cap and eligibility axis may differ from the anchor.

    The *resolved* dataset membership and bytes, base, code, environment, update
    budget, schedule, sampling and all remaining settings still have to match.
    """
    result = deepcopy(spec)
    result["config"]["train"]["max_rows_per_step"] = REFERENCE_ROWS
    result["config"]["corpus"]["min_train_rows"] = LARGE_MINIMUM
    return result


def evaluation_protocol(cfg, eval_cfg=None):
    """Freeze the common benchmark, excluding only where its files are written."""
    from src.utils.experiment import code_identity, environment_versions
    if eval_cfg is None:
        eval_cfg = OmegaConf.merge(OmegaConf.load("config/eval.yaml"), cfg.get("evaluation", {}))
    settings = OmegaConf.to_container(eval_cfg, resolve=True)
    settings.pop("train_cfg_path", None)
    settings.get("results", {}).pop("base_dir", None)
    return {"config": settings, "code_sha256": code_identity(stage="eval"),
            "versions": environment_versions(stage="eval")}


def match_reference(payload, reference):
    """Bind every new episode trial to exactly one equivalent large-table anchor."""
    from src.utils.experiment import digest_json
    if reference["config"]["experiment"].get("size_study") != "corpus_policy":
        raise ValueError("Shared control must be the experiment-6 corpus-policy study")
    if reference["config"]["train"]["max_rows_per_step"] != REFERENCE_ROWS:
        raise ValueError("Shared control must use 8192-row episodes")
    if payload["config"].get("evaluation") != reference["config"].get("evaluation"):
        raise ValueError("Shared control and episode study have different evaluation settings")
    if (not payload.get("evaluation_protocol")
            or payload["evaluation_protocol"] != reference.get("evaluation_protocol")):
        raise ValueError("Shared control and episode study have different benchmark protocols")
    lookup = {}
    for name, key in reference["trials"].items():
        spec = reference["identities"][key]
        if spec["trial"][6] != LARGE_MINIMUM:
            continue
        signature = digest_json(reference_signature(spec))
        if signature in lookup:
            raise ValueError("Ambiguous shared-control trial")
        lookup[signature] = name
    matches = {}
    for name, key in payload["trials"].items():
        signature = digest_json(reference_signature(payload["identities"][key]))
        if signature not in lookup:
            raise ValueError("Episode study differs from its shared control beyond episode size; "
                             "check data, partitions, recipe, code and environment")
        matches[name] = lookup[signature]
    if set(matches.values()) != set(lookup.values()):
        raise ValueError("Episode study and shared control do not cover the same partitions and bases")
    return matches
