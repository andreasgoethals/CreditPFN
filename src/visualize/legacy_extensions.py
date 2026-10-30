"""Read one explicitly identified v2/v2.6 addition, including downloaded ZIPs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

from src.utils.legacy_design import scope, validate_id
from src.utils.paths import outputs_dir
from src.visualize.inputs import analysis_location
from src.visualize import campaign as cp


def selected():
    base = analysis_location("experiment0", "extensions", outputs_dir() / "experiment0/extensions")
    candidates = list(base.glob("*/workflow.json"))
    wanted = os.environ.get("CREDITPFN_ANALYSIS_EXTENSION_ID")
    if wanted:
        validate_id(wanted)
        candidates = [p for p in candidates if p.parent.name == wanted]
        if not candidates:
            raise ValueError("Selected extension was not included in the analysis input")
    ready = []
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") == "creditpfn_legacy" and data.get("prepared") is True:
            ready.append((path.parent, data))
    if len(ready) > 1:
        raise ValueError("Multiple v2/v2.6 additions found: set CREDITPFN_ANALYSIS_EXTENSION_ID; never select by score")
    return ready[0] if ready else None


def configs(selection):
    folder, workflow = selection
    result = {}
    for key, recorded in workflow["production"].items():
        name = Path(recorded).name
        if name != key + ".yaml":
            raise ValueError("Unexpected extension config filename")
        path = folder / "configs" / "production" / name
        result[key] = OmegaConf.create(path.read_text(encoding="utf-8"))
        cfg = result[key]
        identifier = cfg.get("extension_id") if key.startswith("e4_") else cfg.experiment.extension_id
        if identifier != workflow["id"]:
            raise ValueError("Extension config ownership differs from its workflow")
    return result


def size_arms(experiment, configuration, version):
    from src.visualize import size_studies as ss
    tag = version.replace(".", "p")
    mapping = {6: configuration[f"e6_pd_{tag}"]}
    if experiment == 7:
        for key, cfg in configuration.items():
            if key.startswith(f"e7_pd_{tag}_"):
                mapping[int(cfg.train.max_rows_per_step)] = cfg
    return ss.load_arms(experiment, configs=mapping)


def studies(track, *, campaigns=None):
    """Add only the missing bases to the cross-experiment synthesis."""
    selection = selected()
    if selection is None:
        return []
    configuration = configs(selection)
    result = []
    for experiment in (5, 6, 7, 8):
        if campaigns is not None and experiment not in campaigns:
            continue
        if track != "pd" and experiment in (6, 7):
            continue
        with scope(selection[1]["id"], f"experiment{experiment}"):
            for version in ("v2", "v2.6"):
                if experiment in (6, 7):
                    result += [(experiment, label, run) for label, run in size_arms(experiment, configuration, version).items()
                               if experiment != 7 or label != "8,192 rows"]
                else:
                    cfg = configuration[f"e{experiment}_{track}_{version.replace('.', 'p')}"]
                    result.append((experiment, None, cp.load_campaign(experiment, track, cfg_override=cfg)))
    return result


def render(report, sink, experiment, phase):
    selection = selected()
    title = "Additional TabPFN v2 and v2.6 coverage"
    if selection is None:
        report.add(title, "No v2/v2.6 extension evidence is present in the selected input. Original results are unchanged.")
        return
    configuration = configs(selection)
    identifier = selection[1]["id"]
    from src.visualize import context_curves as cc, size_studies as ss, adapter_study as ds, lora_study as ls
    with scope(identifier, f"experiment{experiment}"):
        for version in ("v2", "v2.6"):
            tag = version.replace(".", "p")
            pages = []
            if experiment == 4:
                study = cc.load_context_study(cfg_override=configuration[f"e4_{tag}"])
                report.table(f"{version}: context protocol", cc.protocol_table(study))
                report.table(f"{version}: verified context exports", study.inventory)
                pages = cc.plot_context_curves(study) + cc.plot_table_curves(study)
                for metric, table in cc.secondary_tables(study).items():
                    if not table.empty:
                        report.table(f"{version}: {metric} context pairs", table)
            elif experiment in (6, 7):
                arms = size_arms(experiment, configuration, version)
                report.table(f"{version}: coverage", ss.coverage(arms))
                report.table(f"{version}: corpus", ss.corpus_inventory(arms))
                if phase == "training":
                    pages = ss.plot_optimization(arms) + ss.plot_monitors(arms)
                    report.table(f"{version}: training costs", ss.cost_table(arms))
                else:
                    pages = ss.plot_benchmark(arms)
                    for metric in ("brier_score", "log_loss"):
                        report.table(f"{version}: {metric} paired values", ss.matched_observations(arms, benchmark=True, metric=metric))
            else:
                module = ds if experiment == 5 else ls
                for track in ("pd", "lgd"):
                    run = cp.load_campaign(experiment, track, cfg_override=configuration[f"e{experiment}_{track}_{tag}"])
                    report.table(f"{version} {track.upper()}: trial coverage", run.trials)
                    if phase == "training":
                        pages += module.plot_optimization(run) + module.plot_monitors(run)
                        report.table(f"{version} {track.upper()}: costs", module.costs(run))
                    else:
                        pages += ds.plot_benchmark(run) if experiment == 5 else ls.plot_primary(run)
                        for metric in (("brier_score", "log_loss") if track == "pd" else ("rmse", "mae")):
                            report.table(f"{version} {track.upper()}: {metric}", cp.benchmark_effects(run, metric))
            for page in pages:
                page.name = f"extension_{identifier}_{tag}_{page.name}"
                page.provenance["extension_id"] = identifier
                page.caption = "TabPFN " + version + ". " + page.caption
            if pages:
                cp.show(sink, pages)
    report.add(title, f"Extension {identifier}. This model-coverage extension was added after inspecting the original study; "
        "it is exploratory. The original tables and partitions are preserved. Experiment 10 caps inference context at "
        "10,000 rows for v2 and 50,000 for v2.6. Experiment 9 uses 2,048, the shared 8,192-row control, and "
        "the model-specific training ceiling (10,000 or 11,000 total context-plus-query rows). Missing evidence is not a zero effect.")


def render_debug(report):
    selection = selected()
    if selection is None:
        report.add("TabPFN v2/v2.6 safety checks", "No extension safety evidence in the selected download.")
        return
    folder, workflow = selection
    rows = []
    for stage in ("smoke", "capacity"):
        for task, case in enumerate(workflow[stage]):
            path = folder / "receipts" / f"{stage}_{task}.json"
            receipt = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            rows.append(dict(stage=stage, task=task, kind=case["kind"],
                             passed=receipt.get("passed", False), evidence="receipt present" if receipt else "unavailable"))
    report.table("Extension safety receipts", pd.DataFrame(rows))
    report.add("TabPFN v2/v2.6 safety checks", "Downloaded receipt coverage; the VSC inspection additionally verifies source, environment and artifact hashes.")
