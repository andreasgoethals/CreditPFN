"""Independent, artifact-bound VSC acceptance gate for the v2/v2.6 additions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from omegaconf import OmegaConf

from src.utils.atomic import write_json
from src.utils.experiment import code_identity, environment_versions, digest_json, file_digest
from src.utils.legacy_design import root, scope, validate_id, limits, write_configs, base_name
from src.utils.paths import REPO_ROOT, results_dir, training_dir, resolve_base_checkpoint
from src.utils.lora_checks import _binding, verify_bindings


def fingerprint():
    files = [REPO_ROOT / "config/experiment0/legacy.yaml"]
    files += [p for e in (5, 6, 8, 9, 10) for p in (REPO_ROOT / f"config/experiment{e}").glob("*.yaml")]
    # Reused job bodies, activation and logging are part of runtime acceptance too.
    files += sorted((REPO_ROOT / "scripts/slurm").glob("*.sh"))
    files += sorted((REPO_ROOT / "scripts/slurm").glob("*.slurm"))
    return digest_json(dict(train=code_identity(), evaluation=code_identity(stage="eval"),
        files={str(p.relative_to(REPO_ROOT)): file_digest(p) for p in files}))


def read_workflow(identifier, *, hashes=False):
    data = json.loads((root(identifier) / "workflow.json").read_text(encoding="utf-8"))
    if (data.get("kind") != "creditpfn_legacy" or data.get("id") != identifier
            or data.get("prepared") is not True or data.get("fingerprint") != fingerprint()
            or data.get("versions") != environment_versions(stage="eval")):
        raise RuntimeError("Legacy extension preparation is missing or stale; use a fresh ID")
    verify_bindings(data["files"])
    verify_bindings(data["inputs"], full_hash=hashes)
    return data


def receipts(identifier, workflow, *, hashes=False):
    result = {}
    for stage in ("smoke", "capacity"):
        for task in range(len(workflow[stage])):
            path = root(identifier) / "receipts" / f"{stage}_{task}.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            if any(item.get(key) != value for key, value in dict(
                id=identifier, stage=stage, task=task, passed=True,
                workflow_sha256=digest_json(workflow)).items()):
                raise RuntimeError("Failed, missing or mismatched legacy validation receipt")
            with scope(identifier):
                expected = results_dir(experiment="experiment0") / f"{stage}_{task}.json"
            if item["evidence"]["path"] != str(expected.resolve()):
                raise RuntimeError("Legacy evidence escaped its owned output scope")
            verify_bindings([item["evidence"]])
            evidence = json.loads(expected.read_text(encoding="utf-8"))
            verify_bindings(evidence["artifacts"], full_hash=hashes)
            result[path.name] = file_digest(path)
    return result


def inspect(identifier):
    workflow = read_workflow(identifier, hashes=True)
    checked = receipts(identifier, workflow, hashes=True)
    inspection = dict(id=identifier, passed=True, workflow_sha256=digest_json(workflow), receipts=checked)
    write_json(root(identifier) / "inspection.json", inspection)
    print(f"PASS: {len(checked)} independent checks; all source, environment and artifact bindings match.", flush=True)
    return inspection


def require_gate(identifier):
    workflow = read_workflow(validate_id(identifier))
    checked = receipts(identifier, workflow)
    expected = dict(id=identifier, passed=True, workflow_sha256=digest_json(workflow), receipts=checked)
    if json.loads((root(identifier) / "inspection.json").read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Legacy extension needs its successful independent inspection")
    return workflow


def _save_debug(identifier, name, cfg, files):
    from src.utils.prepare_experiment import prepare
    path = root(identifier) / "configs" / "debug" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        OmegaConf.save(cfg, stream)
    report = prepare(path, write=True)
    files.extend([_binding(path), _binding(report["plan"])])
    return str(path)


def debug_config(source, identifier, name, *, rate, minimum, tiny=False):
    from src.utils.auxiliary_check import select_tables
    cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    budget = limits()
    mode = str(cfg.train.get("adaptation_mode", "full"))
    prefix = {"lora": "cpt_lora_debug", "diagscale": "cpt_diagscale_debug"}.get(mode, "cpt_large_debug")
    cfg.run_name = f"{prefix}_{identifier}_{name}"
    cfg.experiment.update(output_group="experiment0", legacy_gate=False, size_study=None)
    cfg.experiment.pop("reference_config", None)
    cfg.checkpoint.trained_dir = f"checkpoints/trained/experiment0/extensions/{identifier}"
    cfg.tunable.learning_rates = [float(rate)]
    cfg.corpus.min_train_rows = int(minimum)
    cfg.train.retention_panel = "smoke"
    cfg.train.epoch_eval_subsample_samples = budget.monitor_rows
    cfg.train.epoch_eval_n_estimators = budget.monitor_members
    cfg.train.recovery_every_updates = 1
    cfg.train.target_total_steps = int(budget.target_updates if tiny else budget.capacity_updates)
    cfg.train.trajectory_steps = ([0, budget.interrupt_after, budget.target_updates] if tiny else [0, budget.capacity_updates])
    if tiny:
        train, held = select_tables(cfg, budget)
        cfg.corpus.train_dataset_ids = [d.dataset_id for d in train]
        cfg.corpus.test_dataset_ids = [d.dataset_id for d in held]
        cfg.corpus.n_splits = 1
        cfg.train.deterministic = True
        cfg.train.max_rows_per_step = budget.row_cap
    return cfg


def prepare(identifier):
    with scope(identifier):
        return _prepare(identifier)


def _prepare(identifier):
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    from src.utils.prepare_experiment import prepare as training_plan
    from src.eval.context_curve import prepare as context_plan, policy, reference_checkpoint
    from src.data.retention import load_refs
    marker = root(identifier) / "workflow.json"
    if marker.exists():
        raise RuntimeError("This extension already has a workflow; preserve it or choose a fresh ID")
    write_json(marker, dict(kind="creditpfn_legacy", id=identifier, prepared=False), exclusive=True)
    paths = write_configs(identifier)
    files, inputs, smoke, capacity = [], {}, [], []
    # Context checks first; prepare E8 before E9's shared controls.
    for key in sorted(paths, key=lambda k: (0 if k.startswith('e10_') else 1, int(k.split('_')[0][1:]), k)):
        path = paths[key]
        files.append(_binding(path))
        if key.startswith("e10_"):
            cfg = policy(path)
            with scope(identifier, "experiment4"):
                planned, count = context_plan(path)
                files.append(_binding(planned))
            for track in cfg["tracks"]:
                for partition in range(4):
                    pair, _ = reference_checkpoint(track, "tabpfn", partition, cfg)
                    for p in (pair["base"], pair["cpt"], pair["cpt"] + ".provenance.json"):
                        inputs[p] = _binding(p)
                case = dict(kind="context", config=str(path), track=track, version=cfg["base_version"])
                smoke.append(case)
                capacity.append(case)
            continue
        source = load_train_config(config_path=str(path))
        report = training_plan(path, write=True)
        files.append(_binding(report["plan"]))
        base = resolve_base_checkpoint(source.tunable.classifier_base_paths[0] if source.track == "pd"
                                       else source.tunable.regressor_base_paths[0])
        inputs[str(base)] = _binding(base)
        for partition in range(4):
            split = split_from_cfg(apply_split_index(OmegaConf.create(OmegaConf.to_container(source)), partition))
            for ref in [*split.train, *split.test, *load_refs(str(source.train.retention_panel), str(source.track))]:
                inputs[str(ref.processed_csv)] = _binding(ref.processed_csv)
        mode = str(source.train.get("adaptation_mode", "full"))
        if mode in ("lora", "diagscale"):
            cases = []
            for i, rate in enumerate(source.tunable.learning_rates):
                pair = {}
                for arm in ("reference", "resumed"):
                    name = f"{key}_r{i}_{arm}"
                    debug = debug_config(source, identifier, name, rate=rate, minimum=0, tiny=True)
                    pair[arm] = _save_debug(identifier, name, debug, files)
                cases.append(dict(rate=float(rate), **pair))
            smoke.append(dict(kind="adapter", mode=mode, config=str(path), rates=cases))
        minima = source.corpus.min_train_rows
        minima = list(minima) if not isinstance(minima, int) else [minima]
        for minimum in minima:
            name = f"{key}_min{minimum}_capacity"
            debug = debug_config(source, identifier, name, rate=max(source.tunable.learning_rates), minimum=minimum)
            cfgpath = _save_debug(identifier, name, debug, files)
            for partition in range(4):
                capacity.append(dict(kind="training", config=cfgpath, split=partition,
                    rows=int(source.train.max_rows_per_step), full_episodes=minimum == 10001))
    workflow = dict(kind="creditpfn_legacy", id=identifier, prepared=True, fingerprint=fingerprint(),
                    versions=environment_versions(stage="eval"), files=files, inputs=list(inputs.values()),
                    production={k: str(v) for k, v in paths.items()}, smoke=smoke, capacity=capacity)
    if (len(smoke), len(capacity)) != (12, 68):
        raise RuntimeError("Prepared case counts differ from the submitted safety arrays")
    write_json(marker, workflow)
    print(f"Prepared {len(smoke)} correctness and {len(capacity)} capacity cases, no production started.", flush=True)
    return workflow


def run_training(identifier, config, *, split=0, stop=None):
    evidence = results_dir(experiment="experiment0") / "gradients" / f"{Path(config).stem}_{split}_{uuid.uuid4().hex}.json"
    command = [sys.executable, "-u", "-m", "src.utils.legacy_probe", "--report", str(evidence),
               "--", "--config", str(config), "--split-index", str(split), "--trial-index", "0"]
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        command.extend(["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]])
    env = dict(os.environ, CREDITPFN_LEGACY_ID=identifier, CREDITPFN_SEGMENT_SECONDS="0", PYTHONHASHSEED="42")
    for key in ("CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED", "CREDITPFN_FLOW_ID", "CREDITPFN_FLOW_PHASE"):
        env.pop(key, None)
    if stop is not None:
        env["CREDITPFN_STOP_AFTER_UPDATES"] = str(stop)
    result = subprocess.run(command, env=env, check=False)
    if result.returncode != (75 if stop is not None else 0):
        raise RuntimeError(f"Legacy debug training returned unexpected exit {result.returncode}")
    return evidence


def audited(config, split):
    from src.utils.audit_experiment import audit
    report = audit(Path(config))
    row = report["trials"][split]
    problems = [p for p in report["problems"] if p.startswith(row["trial"] + ":")]
    if row["status"] != "OK" or problems:
        raise RuntimeError(f"Legacy training audit failed: {row['trial']}: {problems}")
    return row


def smoke_adapter(identifier, case):
    import torch
    import pandas as pd
    from src.utils.legacy_probe import adapter_probe
    from src.utils.recovery_check import compare, compare_trajectories
    from src.train.config import load_train_config
    cfg = load_train_config(config_path=case["config"])
    probe = adapter_probe(identifier, cfg)
    artifacts = list(probe.pop("artifacts"))
    rates = []
    for pair in case["rates"]:
        rows, curves = {}, {}
        for arm in ("reference", "resumed"):
            path = pair[arm]
            if arm == "resumed":
                run_training(identifier, path, stop=limits().interrupt_after)
            gradient = run_training(identifier, path)
            row = rows[arm] = audited(path, 0)
            curve = training_dir(str(cfg.track), row["trial"] + ".trajectory.csv", experiment="experiment0")
            curves[arm] = pd.read_csv(curve)
            artifacts += [_binding(gradient), _binding(row["path"]), _binding(curve)]
        if case["mode"] == "diagscale":
            a, b = [torch.load(rows[arm]["path"], map_location="cpu", weights_only=False)["adapter_state"]
                    for arm in ("reference", "resumed")]
            weight_equal = a.keys() == b.keys() and all(torch.allclose(v, b[k], atol=1e-7, rtol=1e-6) for k, v in a.items())
        else:
            weight_equal = compare(Path(rows["reference"]["path"]), Path(rows["resumed"]["path"]))["passed"]
        monitor = compare_trajectories(curves["reference"], curves["resumed"],
                                      milestones=[0, limits().interrupt_after, limits().target_updates])
        if not weight_equal or not monitor["passed"]:
            raise AssertionError("Resumed adapter weights or monitored predictions differ")
        rates.append(dict(rate=pair["rate"], recovery_equal=True, monitor=monitor))
    return dict(probe=probe, rates=rates, artifacts=artifacts)


def capacity_training(identifier, case):
    import pandas as pd
    from src.train.config import load_train_config
    cfg = load_train_config(config_path=case["config"])
    gradient = run_training(identifier, case["config"], split=case["split"])
    row = audited(case["config"], case["split"])
    curve = training_dir(str(cfg.track), row["trial"] + ".trajectory.csv", experiment="experiment0")
    frame = pd.read_csv(curve)
    if row["successful_updates"] != limits().capacity_updates or frame.iloc[-1].processed_rows <= 0:
        raise AssertionError("Production-shape probe did not finish its update budget")
    if case["full_episodes"] and frame.iloc[-1].processed_rows != case["rows"] * limits().capacity_updates:
        raise AssertionError("Training silently shortened the declared episode size")
    return dict(audit=row, case=case, artifacts=[_binding(gradient), _binding(curve), _binding(row["path"])])


def context_probe(case, *, capacity=False):
    import io
    import numpy as np
    import pandas as pd
    import torch
    from src.eval.context_curve import policy, reference_checkpoint, fold_indices, score_cell
    from src.eval.dataset_loader import ProcessedDataset
    cfg = policy(case["config"])
    pair, _ = reference_checkpoint(case["track"], "tabpfn", 0, cfg)
    rng = np.random.default_rng(518)
    maximum = cfg["max_context_rows"]
    n = maximum + 2 * cfg["query_batch_rows"] if capacity else 160
    features = 8
    if capacity:
        from src.train.corpus import build_dataset_pool
        # Conservative combination: full context cap and the widest real table.
        # Header-only reads; no private rows enter diagnostic output.
        features = max([64] + [len(pd.read_csv(ref.processed_csv, nrows=0).columns) - 1
                              for ref in build_dataset_pool(case["track"])])
    x = rng.normal(size=(n, features)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int) if case["track"] == "pd" else x[:, 0]
    kind = "classification" if case["track"] == "pd" else "regression"
    data = ProcessedDataset(pd.DataFrame(x), y, [], kind, "synthetic_context", case["track"])
    cells, seen = [], []
    if not capacity:
        cfg["inference_views"]["tabpfn"] = 2
    for fold in (range(1) if capacity else range(5)):
        if capacity:
            context, validation, test = np.arange(maximum), np.arange(maximum, maximum + cfg["query_batch_rows"]), np.arange(maximum + cfg["query_batch_rows"], n)
            sizes = [maximum]
        else:
            context, validation, test = fold_indices(y, kind, fold, cfg)
            seen.extend(test.tolist())
            sizes = [8, 32]
        for size in sizes:
            for state in ("base", "cpt"):
                torch.cuda.reset_peak_memory_stats()
                metrics, blob = score_cell(data, context[:size], validation, test, pair[state], "tabpfn", cfg)
                with np.load(io.BytesIO(blob)) as saved:
                    if not np.array_equal(saved["test"], test) or len(saved["prediction"]) != len(test):
                        raise AssertionError("Context evaluation truncated or changed the test set")
                cells.append(dict(fold=fold, rows=size, features=features, state=state, test_rows=len(test),
                                  peak_bytes=torch.cuda.max_memory_allocated(), **metrics))
    if not capacity and sorted(seen) != list(range(n)):
        raise AssertionError("Five test folds do not cover every row exactly once")
    return dict(cells=cells, all_test_rows=True, artifacts=[_binding(pair["base"]), _binding(pair["cpt"])])


def run_case(identifier, stage, task):
    import torch
    if (os.environ.get("SLURM_CLUSTER_NAME") != "mindwell" or not os.environ.get("SLURM_JOB_ID")
            or not torch.cuda.is_available() or "B200" not in torch.cuda.get_device_name()):
        raise RuntimeError("Legacy acceptance checks require a submitted Mindwell B200 job")
    workflow = read_workflow(identifier)
    case = workflow[stage][task]
    with scope(identifier):
        if case["kind"] == "context":
            evidence = context_probe(case, capacity=stage == "capacity")
        elif stage == "smoke":
            evidence = smoke_adapter(identifier, case)
        else:
            evidence = capacity_training(identifier, case)
        destination = results_dir(experiment="experiment0") / f"{stage}_{task}.json"
        write_json(destination, evidence, exclusive=True)
        receipt = dict(id=identifier, stage=stage, task=task, passed=True,
                       workflow_sha256=digest_json(workflow), evidence=_binding(destination))
        write_json(root(identifier) / "receipts" / f"{stage}_{task}.json", receipt, exclusive=True)
    print(f"PASS {stage} task {task}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "capacity", "inspect"))
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", type=int)
    args = parser.parse_args(argv)
    validate_id(args.id)
    if args.action == "prepare":
        prepare(args.id)
    elif args.action == "inspect":
        inspect(args.id)
    else:
        run_case(args.id, args.action, args.task)


if __name__ == "__main__":
    main()
