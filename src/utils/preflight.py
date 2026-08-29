"""Everything that can be checked WITHOUT a GPU, before spending ~20M credits.

    python -m src.utils.preflight                 # all experiment configs
    python -m src.utils.preflight --config config/experiment1_pd.yaml

Exit code 0 = safe to submit. 1 = at least one FAIL.

This complements `scripts/cluster_report.py`, which measures the MACHINE (memory, throughput,
SLURM limits) and must run on the cluster. This checks the REPOSITORY: that the grid is what we
think it is, that nothing in it collides or is silently disabled, and that every file it will
reach for exists.

Every check here exists because something like it went wrong. In order of what it cost:

  * L2-SP silently off for half the grid, while the manifest recorded it as on (25-08-2026).
  * Two trials resolving to one filename, so one overwrites the other and a cell vanishes.
  * A config knob nothing reads, so setting it did nothing (`monitor_every`, 24-08-2026).
  * A sweep axis inherited from train.yaml that nobody meant to sweep, doubling the grid.
  * Row caps that OOM on the card they were sized for.
  * Trial packing that straddles two clusters and sends a 131 GB trial to an 80 GB card.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTS = ("experiment0_pd", "experiment0_lgd", "experiment1_pd",
               "experiment1_lgd", "experiment2_pd")

# MEASURED probe points, (rows, peak_GB_or_None, ran_ok). B200 183 GB, 2 training members.
# Sources: probe job 11524668 §9, plus the exp0 real-training run 11527923 that caught the v2
# OOM a LINEAR model had waved through.
#
# The earlier linear model (GB = rows/1k x slope x members) was WRONG and dangerous: TabPFN's
# row attention is O(rows^2), so peak memory is quadratic in the row cap, not linear. The linear
# model predicted v2 @ 14k = 111 GB ("fits, 61%") and it OOM'd at ~177 GB. So do not extrapolate
# a slope — anchor to measured points and refuse any cap at or beyond a known OOM.
PROBE_POINTS = {
    "v2":     [(10000, 79.2, True), (14000, None, False), (26000, None, False)],
    "v2.6":   [(10000, 109.4, True), (11000, 132.0, True), (26000, None, False)],
    "v3":     [(10000, 51.5, True), (26000, 131.5, True), (50000, None, False)],
    # tabicl's 50k failure is a cuDNN kernel-shape error, not OOM; treat it as a hard ceiling too.
    "tabicl": [(10000, 10.5, True), (26000, 27.0, True), (50000, None, False)],
}
CARD_GB = 183.0          # B200 usable

_PASS, _FAIL, _WARN = "ok  ", "FAIL", "warn"


def _resolved_roots() -> tuple["pathlib.Path | None", "dict[str, pathlib.Path]"]:
    """(checkpoints_dir, {track: processed_dir}) as the PIPELINE resolves them.

    Not repo-relative. On VSC these land on project storage via CREDITPFN_DATA_ROOT and
    data.yaml's `paths.data_source`; checking REPO/checkpoints reported seven failures on a
    perfectly staged cluster (25-08-2026).
    """
    sys.path.insert(0, str(REPO))
    ck = None
    proc: dict[str, pathlib.Path] = {}
    try:
        from omegaconf import OmegaConf

        from src.utils.paths import apply_data_source_from_cfg, processed_dir
        apply_data_source_from_cfg(OmegaConf.load(REPO / "config" / "data.yaml"))
        for track in ("pd", "lgd"):
            proc[track] = pathlib.Path(processed_dir(track))
    except Exception:
        for track in ("pd", "lgd"):
            proc[track] = REPO / "data/processed" / track
    try:
        from src.utils.stage_checkpoints import checkpoints_root
        ck = pathlib.Path(checkpoints_root())
    except Exception:
        ck = REPO / "checkpoints"
    return ck, proc


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))

    def ok(self, check: str, detail: str = "") -> None:
        self.add(_PASS, check, detail)

    def fail(self, check: str, detail: str = "") -> None:
        self.add(_FAIL, check, detail)

    def warn(self, check: str, detail: str = "") -> None:
        self.add(_WARN, check, detail)

    @property
    def n_fail(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == _FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == _WARN)

    def render(self, title: str) -> None:
        print(f"\n{'=' * 78}\n {title}\n{'=' * 78}")
        for status, check, detail in self.rows:
            print(f"  [{status}] {check}")
            if detail:
                for line in str(detail).split("\n"):
                    print(f"           {line}")


def _load(name: str):
    from omegaconf import OmegaConf
    base = OmegaConf.load(REPO / "config/train.yaml")
    path = REPO / (name if name.endswith(".yaml") else f"config/{name}.yaml")
    return OmegaConf.merge(base, OmegaConf.load(path)), path


def _grid(cfg) -> list[tuple]:
    """The cartesian product, in the same order train_pipeline._resolve_grid builds it."""
    t = cfg.tunable
    bases = (t.classifier_base_paths if cfg.track == "pd" else t.regressor_base_paths)
    return list(itertools.product(
        bases, t.learning_rates, t.frozen_backbone, t.query_fractions,
        t.accumulate_grad_batches, t.epoch_pass_modes, [cfg.corpus.get("min_train_rows", 0)],
        t.l2sp_lambdas,
    ))


def _base_key(path: str) -> str:
    s = str(path)
    for k in ("tabicl", "v2.6", "v3"):
        if k in s:
            return k
    return "v2"


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_grid(cfg, name: str, rep: Report) -> list[tuple]:
    grid = _grid(cfg)
    t = cfg.tunable
    axes = {
        "base": len(t.classifier_base_paths if cfg.track == "pd" else t.regressor_base_paths),
        "lr": len(t.learning_rates), "l2sp": len(t.l2sp_lambdas),
        "frozen": len(t.frozen_backbone), "pass": len(t.epoch_pass_modes),
        "qf": len(t.query_fractions), "accum": len(t.accumulate_grad_batches),
    }
    swept = {k: v for k, v in axes.items() if v > 1}
    splits = cfg.corpus.get("n_splits") or 1
    rep.ok(f"{name}: {len(grid)} trials x {splits} splits = {len(grid) * splits} cells",
           "swept: " + ", ".join(f"{k}={v}" for k, v in swept.items()))

    # An axis nobody meant to sweep, inherited from train.yaml, silently multiplies the grid.
    for axis in ("qf", "accum"):
        if axes[axis] > 1:
            rep.warn(f"{name}: '{axis}' has {axes[axis]} values",
                     "inherited from train.yaml? it multiplies the whole grid")
    return grid


def check_required_axes(cfg, name: str, rep: Report) -> None:
    """`learning_rates` / `l2sp_lambdas` / `frozen_backbone` are deliberately absent from
    train.yaml, so an experiment that forgets one would inherit nothing at all."""
    for key in ("learning_rates", "l2sp_lambdas", "frozen_backbone"):
        if key not in cfg.tunable:
            rep.fail(f"{name}: tunable.{key} missing", "must be set per experiment")
        elif cfg.tunable[key] is None:
            rep.warn(f"{name}: tunable.{key} is null", "legitimate, but confirm it is intended")


def check_name_collisions(cfg, name: str, grid: list[tuple], rep: Report) -> None:
    """Two trials mapping to one filename means one silently overwrites the other."""
    sys.path.insert(0, str(REPO))
    from src.train.loop import descriptive_name
    splits = cfg.corpus.get("n_splits") or 1
    seen: collections.Counter[str] = collections.Counter()
    for k in range(splits):
        run_name = f"{cfg.run_name}_s{k:02d}" if splits > 1 else cfg.run_name
        for b, lr, fz, qf, ac, pm, mtr, l2 in grid:
            seen[descriptive_name(
                run_name=run_name, track=cfg.track, base_path=b, learning_rate=lr,
                seed=cfg.seed, use_lora=bool(fz), query_fraction=qf,
                accumulate_grad_batches=ac, epoch_pass_mode=pm, min_train_rows=mtr,
                l2sp_lambda=l2)] += 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        rep.fail(f"{name}: {len(dupes)} filename collision(s)",
                 "\n".join(f"x{v}  {k}" for k, v in list(dupes.items())[:3]))
    else:
        rep.ok(f"{name}: {len(seen)} distinct checkpoint names, no collisions")


def check_checkpoints(cfg, name: str, grid: list[tuple], rep: Report,
                      ckpt_dir: "pathlib.Path | None" = None) -> None:
    """Resolve each base through the SAME roots the pipeline uses, not repo-relative."""
    wanted = sorted({str(t[0]) for t in grid})
    missing = []
    for rel in wanted:
        base = pathlib.Path(rel).name
        cands = [REPO / rel]
        if ckpt_dir is not None:
            cands.append(ckpt_dir / base)
        if not any(c.exists() for c in cands):
            missing.append(f"{base}   (looked in {', '.join(str(c.parent) for c in cands)})")
    if missing:
        rep.fail(f"{name}: {len(missing)} of {len(wanted)} checkpoint(s) missing",
                 "\n".join(missing) + "\nstage them: python -m src.utils.stage_checkpoints")
    else:
        rep.ok(f"{name}: all {len(wanted)} base checkpoints present",
               f"in {ckpt_dir}" if ckpt_dir else "")


def check_row_caps(rep: Report) -> None:
    """Every configured row cap must be at or below a MEASURED-safe row count for its base.

    Anchored to real probe points, never a linear slope: TabPFN's row attention is O(rows^2), so
    a slope model under-predicts and it already waved through v2 @ 14k (predicted 111 GB, OOM'd
    at ~177 GB on real data). The rule:
        cap <= largest measured-OK rows      -> OK  (measured safe)
        cap >= smallest measured-OOM rows    -> FAIL (at/beyond a known failure)
        in between                           -> WARN (untested; do not trust extrapolation)
    """
    from omegaconf import OmegaConf
    data = OmegaConf.load(REPO / "config/data.yaml")
    caps = data.finetuning.max_rows_per_epoch
    train = OmegaConf.load(REPO / "config/train.yaml")
    est = train.train.get("n_estimators_finetune", 2)
    members = int(est.get("default", 2) if hasattr(est, "get") else est)

    lines, worst = [], []
    for base, points in PROBE_POINTS.items():
        cap = caps.get(base)
        if cap is None or hasattr(cap, "get"):
            continue
        cap = int(cap)
        ok_rows = [r for r, _gb, ok in points if ok]
        oom_rows = [r for r, _gb, ok in points if not ok]
        max_ok = max(ok_rows) if ok_rows else 0
        min_oom = min(oom_rows) if oom_rows else None
        gb_at = {r: gb for r, gb, ok in points if ok and gb is not None}
        note = (f"measured OK to {max_ok}"
                + (f" ({gb_at[max_ok]:.0f} GB)" if max_ok in gb_at else "")
                + (f", OOM at {min_oom}" if min_oom else ""))
        lines.append(f"{base:7s} cap={cap:6d}  {note}")
        if min_oom is not None and cap >= min_oom:
            worst.append(base)
            rep.fail(f"row cap for {base} ({cap}) is at/above a MEASURED OOM ({min_oom})",
                     "\n".join(lines))
        elif cap > max_ok:
            rep.warn(f"row cap for {base} ({cap}) exceeds the largest measured-OK "
                     f"rows ({max_ok})",
                     "untested region; attention memory is quadratic so do not trust a linear "
                     "extrapolation — re-probe before trusting this cap")
    if not worst:
        rep.ok(f"row caps at/below measured-safe rows, {members} members", "\n".join(lines))


#: Official inference-context limits (max training/context rows a model accepts before raising).
#: TabPFN v2 hard-raises TabPFNValidationError above 10k; v2.6/v3/tabicl support far more. Above
#: these, eval fails ALL folds — untuned AND trained v2 failed on the big datasets at 50k
#: (exp0 eval j11532735, 28-08-2026).
_EVAL_CONTEXT_LIMIT = {"v2": 10000}


def check_eval_caps(rep: Report) -> None:
    """The eval's per-model context cap must not exceed the model's OFFICIAL inference limit."""
    from omegaconf import OmegaConf
    ev = OmegaConf.load(REPO / "config/eval.yaml")
    caps = None
    for node in (ev, ev.get("eval", {})):
        if hasattr(node, "get") and node.get("max_rows_per_model") is not None:
            caps = node.get("max_rows_per_model"); break
    if caps is None:
        rep.warn("eval max_rows_per_model not found", "cannot verify inference caps")
        return
    bad = []
    for base, limit in _EVAL_CONTEXT_LIMIT.items():
        cap = caps.get(base)
        if cap is not None and int(cap) > limit:
            bad.append(f"{base}: cap {int(cap)} > official limit {limit}")
    if bad:
        rep.fail("eval context cap exceeds a model's official limit",
                 "\n".join(bad) + "\nabove this, eval raises TabPFNValidationError on every fold")
    else:
        rep.ok("eval context caps respect each model's official inference limit",
               f"v2<={_EVAL_CONTEXT_LIMIT['v2']}, others larger by design")


def check_step_budget(cfg, name: str, rep: Report) -> None:
    """Equal optimizer steps is the whole point of `target_total_steps`; verify every cell
    lands on it rather than being clipped by `max_epochs_for_step_budget`."""
    budget = cfg.train.get("target_total_steps")
    if not budget:
        # A one-epoch control (experiment 0) has nothing to equalise — it exists to exercise
        # save+reload, not to train. Anything longer without a step budget IS a problem, because
        # an epoch buys a different number of updates in every cell.
        if int(cfg.train.get("epochs", 0) or 0) <= 1:
            rep.ok(f"{name}: one-epoch control, step budget not applicable")
        else:
            rep.warn(f"{name}: no target_total_steps and epochs={cfg.train.get('epochs')}",
                     "an epoch buys a different number of updates in every cell")
        return
    cap = cfg.train.get("max_epochs_for_step_budget")
    n_train = 13 if cfg.track == "pd" else 6
    # steps/epoch is sum(ceil(rows/cap)); its extremes are the corpus size (full_pass) and the
    # dataset count (accumulate). The accumulate arm is the one a low epoch cap can clip.
    worst_spe = n_train
    need_epochs = math.ceil(int(budget) / worst_spe)
    if cap and need_epochs > int(cap):
        rep.fail(f"{name}: accumulate needs {need_epochs} epochs for {budget} steps",
                 f"max_epochs_for_step_budget={cap} clips it to {int(cap) * worst_spe} steps")
    else:
        rep.ok(f"{name}: {budget} steps reachable in every cell",
               f"accumulate worst case {need_epochs} epochs (cap {cap})")


def check_l2sp_applies(rep: Report) -> None:
    """The 25-08-2026 bug: L2-SP was gated on the frozen-backbone flag, so it silently did
    nothing for every frozen TabPFN trial while the manifest still recorded the lambda."""
    src = (REPO / "src/train/loop.py").read_text(encoding="utf-8")
    m = re.search(r"l2sp_applicable\s*=\s*(.+)", src)
    if not m:
        rep.fail("L2-SP applicability guard not found", "src/train/loop.py")
        return
    expr = m.group(1).strip()
    if "use_lora" in expr:
        rep.fail("L2-SP is gated on `use_lora`", f"`{expr}` — use_lora IS the frozen-backbone "
                 "axis, so L2-SP would be off for every frozen trial")
    else:
        rep.ok("L2-SP applies in both adaptation modes", f"gated on `{expr}`")


def check_stale_knobs(rep: Report) -> None:
    """A knob nothing reads is a lie: setting it changes nothing, silently."""
    from omegaconf import OmegaConf
    code = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py"))
    )

    def leaves(node, prefix=""):
        if hasattr(node, "items"):
            for k, v in node.items():
                yield from leaves(v, f"{prefix}{k}.")
                if not hasattr(v, "items"):
                    yield f"{prefix}{k}", str(k)

    stale = []
    for cfg_file in sorted((REPO / "config").glob("*.yaml")):
        for full, leaf in leaves(OmegaConf.load(cfg_file)):
            if not re.search(rf'[\."\']{re.escape(leaf)}["\'\s\),\]:]|\.{re.escape(leaf)}\b',
                             code):
                stale.append(f"{cfg_file.name}: {full}")
    if stale:
        rep.fail(f"{len(stale)} config knob(s) no code reads", "\n".join(stale))
    else:
        rep.ok("every config knob is read somewhere in src/ or scripts/")


def check_slurm(rep: Report) -> None:
    import subprocess
    bad = []
    for sh in sorted((REPO / "scripts/slurm").glob("*")):
        if sh.suffix not in (".sh", ".slurm"):
            continue
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{sh.name}: {r.stderr.strip().splitlines()[:1]}")
    if bad:
        rep.fail(f"{len(bad)} SLURM script(s) fail `bash -n`", "\n".join(bad))
    else:
        rep.ok("every SLURM script parses")


def check_job_count(cfgs: list, rep: Report, trials_per_task: int = 2) -> None:
    """VSC rejects submissions past 500 QUEUED TASKS, and the failure mode is a silent gap.

    Each experiment is a SEPARATE `run_experiment.sh` call whose wave-submitter throttles itself
    below the ceiling, so what must fit under 500 is the LARGEST single experiment's task count,
    not the grand total across experiments (those never queue simultaneously). `trials_per_task`
    is auto-clamped in the launcher to a divisor of the per-base block, so model it the same way.
    """
    worst_name, worst_tasks = "", 0
    detail = []
    for cfg, name in cfgs:
        trials = len(_grid(cfg))
        splits = cfg.corpus.get("n_splits") or 1
        n_bases = len({t[0] for t in _grid(cfg)}) or 1
        block = max(1, trials // n_bases)
        tpt = next((d for d in range(trials_per_task, 0, -1) if block % d == 0), 1)
        tasks = math.ceil(trials / tpt) * splits
        if tasks > worst_tasks:
            worst_name, worst_tasks = name, tasks
        detail.append(f"{name:16s} {trials:3d} trials x {splits} splits "
                      f"-> {tasks:4d} tasks at {tpt}/task")
    detail.append(f"{'MAX (per submission)':20s} {worst_tasks:26d} tasks  [{worst_name}]")
    if worst_tasks > 500:
        rep.fail(f"{worst_name}: {worst_tasks} tasks exceeds the 500 submitted-job ceiling",
                 "\n".join(detail) + "\nsubmit fewer splits per wave, or raise TRIALS_PER_TASK")
    else:
        rep.ok(f"largest submission {worst_tasks} tasks, under the 500 ceiling",
               "\n".join(detail))


def check_packing_divides(cfgs: list, rep: Report, trials_per_task: int = 2) -> None:
    """A packed task must not straddle two model families: routing and the tabicl import
    preflight are both per-base, and a chunk spanning families would send a 131 GB TabPFN
    trial to whatever card the tabicl trial picked. Models the launcher's auto-clamp of
    `trials_per_task` to a divisor of the per-base block, so this verifies the invariant holds."""
    for cfg, name in cfgs:
        grid = _grid(cfg)
        n_bases = len({t[0] for t in grid}) or 1
        block = max(1, len(grid) // n_bases)
        tpt = next((d for d in range(trials_per_task, 0, -1) if block % d == 0), 1)
        per_task_families = collections.defaultdict(set)
        for i, t in enumerate(grid):
            per_task_families[i // tpt].add(_base_key(t[0]))
        bad = {k: v for k, v in per_task_families.items() if len(v) > 1}
        if bad:
            rep.fail(f"{name}: {len(bad)} task(s) straddle model families at {tpt}/task",
                     f"e.g. task {min(bad)} covers {sorted(bad[min(bad)])}")
        else:
            rep.ok(f"{name}: every packed task stays within one model family (at {tpt}/task)")


def check_train_eval_agree(name: str, rep: Report) -> None:
    """Train and eval must resolve the SAME run_name, split_seed and held-out datasets.

    They are two separate config-loading paths — `train_pipeline._load_cfg` +
    `_apply_split_index`, and `eval_pipeline._load_cfgs` — and both must feed the SAME
    `split_from_cfg`. If they disagree, eval scores each checkpoint against datasets it may have
    TRAINED on. This check compares split_from_cfg on both, but that is only valid if the training
    LOOP actually calls split_from_cfg — it did not until 26-08-2026 (it called `split_corpus`
    directly and dropped n_test_datasets/split_seed), so the static guard below fails if
    `train_one_config` reintroduces a direct `split_corpus(` call and bypasses the shared path.
    """
    loop_src = (REPO / "src/train/loop.py").read_text(encoding="utf-8")
    if re.search(r"\bsplit_corpus\s*\(", loop_src):
        rep.fail("train_one_config calls split_corpus() directly",
                 "it must use split_from_cfg (the shared path eval uses), or training and eval "
                 "can resolve different held-out datasets — see the 26-08-2026 split-mismatch bug")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import importlib.util

        from src.train.corpus import split_from_cfg
        import scripts.train_pipeline as tp
        spec = importlib.util.spec_from_file_location(
            "_ep_preflight", REPO / "scripts" / "eval_pipeline.py")
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)
    except Exception as exc:
        rep.warn(f"{name}: cannot compare train/eval config paths",
                 f"{type(exc).__name__}: {exc}")
        return

    cfg_path = f"config/{name}.yaml"
    try:
        n_splits = int(tp._load_cfg(None, cfg_path).corpus.get("n_splits") or 1)
    except Exception as exc:
        rep.fail(f"{name}: train config will not load", f"{type(exc).__name__}: {exc}")
        return

    bad = []
    for k in range(n_splits):
        try:
            t = tp._apply_split_index(tp._load_cfg(None, cfg_path), k)
            _, e = ep._load_cfgs([], [], config_path=cfg_path, split_index=k)
            t_ids = sorted(c.dataset_id for c in split_from_cfg(t).test)
            e_ids = sorted(c.dataset_id for c in split_from_cfg(e).test)
            if (t.run_name, int(t.corpus.split_seed), t_ids) != \
               (e.run_name, int(e.corpus.split_seed), e_ids):
                bad.append(
                    f"split {k}: train {t.run_name}/seed={t.corpus.split_seed}/{t_ids} != "
                    f"eval {e.run_name}/seed={e.corpus.split_seed}/{e_ids}")
        except Exception as exc:
            bad.append(f"split {k}: {type(exc).__name__}: {exc}")
    if bad:
        rep.fail(f"{name}: train and eval disagree on the dataset draw",
                 "\n".join(bad[:4]) + "\neval would score checkpoints on data they trained on")
    else:
        rep.ok(f"{name}: train and eval agree on all {n_splits} split(s)",
               "same run_name, split_seed and held-out datasets")


def check_launchers_pass_config(rep: Report) -> None:
    """Any launcher that submits eval_*.slurm must also export CREDITPFN_CONFIG.

    eval_pipeline rebuilds the held-out dataset draw from the training corpus block, so an eval
    job that never receives --config/--split-index silently falls back to config/train.yaml and
    scores each checkpoint against a DIFFERENT four datasets than it was held out from.
    Measured 25-08-2026: for split 7 of experiment 1, four of the five datasets the old path
    would have used were in that model's own training set.

    Static check, because the failure is in the shell plumbing rather than in Python, and it
    yields better-looking numbers instead of an error.
    """
    slurm = REPO / "scripts/slurm"
    problems = []
    for sh in sorted(slurm.glob("*.sh")):
        text = sh.read_text(encoding="utf-8", errors="ignore")
        submits_eval = any(t in text for t in (
            "eval_${TRACK}.slurm", "eval_${TR}.slurm", "eval_pd.slurm", "eval_lgd.slurm"))
        if submits_eval and "CREDITPFN_CONFIG" not in text:
            problems.append(f"{sh.name} submits an eval job without exporting CREDITPFN_CONFIG")
    for job in ("eval_pd.slurm", "eval_lgd.slurm"):
        text = (slurm / job).read_text(encoding="utf-8", errors="ignore")
        if "--config" not in text or "--split-index" not in text:
            problems.append(f"{job} does not forward --config / --split-index to eval_pipeline")
    if problems:
        rep.fail("eval launchers can lose the experiment config", "\n".join(problems))
    else:
        rep.ok("every eval launcher forwards the config and split index")


def check_storage_layout(rep: Report) -> None:
    """Report every resolved path and which VSC tier it landed on.

    docs/TEMPLATE.md splits storage in two, and a path resolved against the wrong tier looks
    exactly like a missing file. `data/`, `checkpoints/` and `output/results/` belong on PROJECT
    storage (/lustre1/project/stg_00211/<Project>/); the repository and the rest of `output/` —
    including `output/manifests/`, which holds the dataset REGISTRY the corpus is built from —
    belong on $VSC_DATA. Two debugging rounds were lost to `ls`-ing the wrong one, so print the
    map instead of inferring it.
    """
    sys.path.insert(0, str(REPO))
    try:
        from omegaconf import OmegaConf

        from src.utils.paths import (
            apply_data_source_from_cfg, data_root, manifests_dir, processed_dir, raw_dir,
            results_dir, staging_root,
        )
        from src.utils.stage_checkpoints import checkpoints_root
        apply_data_source_from_cfg(OmegaConf.load(REPO / "config" / "data.yaml"))
    except Exception as exc:                                       # pragma: no cover
        rep.fail("cannot resolve the storage layout", f"{type(exc).__name__}: {exc}")
        return

    stag, dat = pathlib.Path(staging_root()), pathlib.Path(data_root())
    same = stag == dat                                             # true off-cluster

    def tier(path: pathlib.Path) -> str:
        if same:
            return "local"
        try:
            path.relative_to(stag)
            return "project"
        except ValueError:
            return "vsc_data"

    entries = [
        ("data/raw/pd", pathlib.Path(raw_dir("pd")), "project"),
        ("data/raw/lgd", pathlib.Path(raw_dir("lgd")), "project"),
        ("data/processed/pd", pathlib.Path(processed_dir("pd")), "project"),
        ("data/processed/lgd", pathlib.Path(processed_dir("lgd")), "project"),
        ("checkpoints", pathlib.Path(checkpoints_root()), "project"),
        ("output/results", pathlib.Path(results_dir()), "project"),
        ("output/manifests", pathlib.Path(manifests_dir()), "vsc_data"),
    ]
    lines = [f"project storage : {stag}", f"personal data   : {dat}", ""]
    misplaced = []
    for label, path, want in entries:
        n = len(list(path.glob("*"))) if path.is_dir() else -1
        got = tier(path)
        flag = "" if (same or got == want) else f"  << expected {want}, got {got}"
        lines.append(f"{label:20s} {'MISSING' if n < 0 else f'{n:4d} entries':>12s}  "
                     f"[{got:8s}] {path}{flag}")
        if flag:
            misplaced.append(label)
    if misplaced:
        rep.warn(f"{len(misplaced)} path(s) on an unexpected storage tier", "\n".join(lines))
    else:
        rep.ok("storage layout resolved", "\n".join(lines))


def check_data(rep: Report, proc: "dict[str, pathlib.Path]") -> None:
    """Ask the TRAINING code what the corpus is, not the filesystem.

    Globbing `data/processed/<track>/*.csv` once reported "17 processed datasets" while training
    then died with an empty corpus. `build_dataset_pool` is the real function, so calling it is
    the only check that cannot drift from what training sees. Since 26-08-2026 the pool is built
    from DATASET_METADATA (code) + the processed CSVs, with NO manifest file — so a missing
    manifest can no longer produce this mismatch. The check remains as a guard against a genuinely
    empty/unstaged `data/processed`.
    """
    sys.path.insert(0, str(REPO))
    try:
        from src.train.corpus import build_dataset_pool
    except Exception as exc:                                       # pragma: no cover
        rep.fail("cannot import build_dataset_pool", f"{type(exc).__name__}: {exc}")
        return

    for track, need in (("pd", 5), ("lgd", 3)):
        d = proc.get(track, REPO / "data/processed" / track)
        on_disk = len(list(d.glob("*.csv"))) if d.is_dir() else 0
        try:
            pool = build_dataset_pool(track)
        except Exception as exc:
            rep.fail(f"{track}: build_dataset_pool raised", f"{type(exc).__name__}: {exc}")
            continue
        n = len(pool)
        if n < need:
            hint = (f"{on_disk} CSV(s) are in {d} but the pool has {n} — some processed CSV is "
                    f"missing its DATASET_METADATA entry, or a target column is absent."
                    if on_disk >= need else
                    f"only {on_disk} sanitized CSV(s) in {d}; run scripts/data_pipeline.py "
                    f"(processed data must be staged to project storage)")
            rep.fail(f"{track}: corpus has {n} usable dataset(s), need {need}", hint)
        else:
            extra = f" ({on_disk} CSVs on disk)" if on_disk != n else ""
            rep.ok(f"{track}: {n} datasets in the corpus{extra}", f"processed dir: {d}")


def check_predictions_writer(rep: Report) -> None:
    from omegaconf import OmegaConf
    ev = OmegaConf.load(REPO / "config/eval.yaml")
    flat = OmegaConf.to_container(ev, resolve=False)

    def _find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = _find(v, key)
                if r is not None:
                    return r
        return None

    wants = bool(_find(flat, "save_predictions") or False)
    try:
        import pyarrow  # noqa: F401
        have = True
    except ImportError:
        have = False
    if wants and not have:
        rep.warn("save_predictions is on but pyarrow is absent",
                 "falls back to gzipped CSV (~5x larger); pip install -e '.[dev,notebooks]'")
    else:
        rep.ok(f"prediction writer ready (save_predictions={wants}, pyarrow={have})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", default=None,
                    help="experiment config(s); default = all five")
    ap.add_argument("--trials-per-task", type=int, default=2)
    args = ap.parse_args(argv)

    names = args.config or list(EXPERIMENTS)
    rep = Report()
    ckpt_dir, proc_dirs = _resolved_roots()
    check_storage_layout(rep)      # first: every later path failure reads against this
    check_launchers_pass_config(rep)
    loaded = []
    for n in names:
        try:
            cfg, path = _load(n)
        except Exception as exc:
            rep.fail(f"{n}: will not load", f"{type(exc).__name__}: {exc}")
            continue
        label = path.stem
        loaded.append((cfg, label))
        check_required_axes(cfg, label, rep)
        grid = check_grid(cfg, label, rep)
        check_name_collisions(cfg, label, grid, rep)
        check_checkpoints(cfg, label, grid, rep, ckpt_dir)
        check_step_budget(cfg, label, rep)
        check_train_eval_agree(label, rep)

    check_row_caps(rep)
    check_eval_caps(rep)
    check_l2sp_applies(rep)
    check_stale_knobs(rep)
    check_slurm(rep)
    check_data(rep, proc_dirs)
    check_predictions_writer(rep)
    if loaded:
        exp1 = [(c, n) for c, n in loaded if "experiment1" in n] or loaded
        check_job_count(exp1, rep, args.trials_per_task)
        check_packing_divides(exp1, rep, args.trials_per_task)

    rep.render("CreditPFN PREFLIGHT")
    print(f"\n{'=' * 78}")
    if rep.n_fail:
        print(f"  {rep.n_fail} FAILURE(S), {rep.n_warn} warning(s) — DO NOT SUBMIT")
    else:
        print(f"  0 failures, {rep.n_warn} warning(s) — safe to submit")
    print("=" * 78)
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
