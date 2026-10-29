"""Preview/remove one extension's owned files on both tiers; never original runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.utils.legacy_design import root, scope, validate_id
from src.utils.paths import outputs_dir, results_dir, checkpoints_dir, resolve_output_path
from src.utils.clean_run import validate_tree, measure, wipe, check_cluster_writers
from src.utils.atomic import write_json


def publication_files(identifier, experiment):
    """Only explicitly tagged extension pages; preserve shared historical figures."""
    from src.visualize.paths import figures_dir
    groups = [0, 4, 5, 6, 7, 8] if experiment == "all" else [int(experiment)]
    edits, files = {}, []
    for number in groups:
        from src.utils.study_layout import NUMBERS
        folder = figures_dir() / f"experiment{NUMBERS.get(number, number)}"
        for manifest in (folder / "_metadata").glob("*.json"):
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            owned = [e for e in entries if e.get("provenance", {}).get("extension_id") == identifier]
            if not owned:
                continue
            validate_tree(manifest)
            for entry in owned:
                stem = entry["stem"]
                if Path(stem).name != stem or "/" in stem or "\\" in stem:
                    raise ValueError("Invalid publication filename; refusing cleanup")
                path = folder / (stem + ".pdf")
                validate_tree(path)
                files.append(path)
            edits[manifest] = [e for e in entries if e not in owned]
    return files, edits


def targets(identifier, experiment):
    validate_id(identifier)
    groups = [0, 4, 5, 6, 7, 8] if experiment == "all" else [int(experiment)]
    if set(groups) - {0, 4, 5, 6, 7, 8}:
        raise ValueError("Only extension experiments 0 and 4–8 can be cleaned")
    selected = []
    for number in groups:
        group = f"experiment{number}"
        with scope(identifier, group):
            from src.utils.paths import manifests_dir
            selected += [manifests_dir(group).parent,
                         results_dir(experiment=group).parent,
                         checkpoints_dir("trained", group, "extensions", identifier),
                         resolve_output_path(f"checkpoints/trained/{group}/extensions/{identifier}")]
    return list(dict.fromkeys(selected))


def clean(identifier, experiment="all", *, apply=False):
    marker = root(identifier) / ".scope.json"
    if (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) !=
            dict(kind="creditpfn_legacy_scope", id=identifier)):
        raise ValueError("Unknown extension ownership; refusing cleanup")
    chosen = targets(identifier, experiment)
    for path in chosen:
        validate_tree(path)
    figures, metadata = publication_files(identifier, experiment)
    # Gate receipts are inputs to all production jobs; E6 is E7's shared control.
    protected = [4, 5, 6, 7, 8] if experiment == "0" else [7] if experiment == "6" else []
    for dependent in protected:
        if any(path.is_dir() and any(path.rglob("*")) for path in targets(identifier, str(dependent))):
            raise RuntimeError(f"Extension experiment {dependent} still depends on these files; clean it first or select all")
    if list(root(identifier).glob("*.submit.lock")):
        raise RuntimeError("An extension submitter owns a lock; stop/reconcile it before cleanup")
    for path in chosen:
        count, size = measure(path)
        print(f"{count:>6} files {size/1e6:>9.1f} MB  {path}")
    for path in figures:
        print(f"Extension figure: {path}")
    if not apply:
        print("Preview only. Use --apply to delete only these extension trees.")
        return chosen
    # Conservative fail-closed guard includes unknown ownership and consumers.
    check_cluster_writers()
    active = os.environ.get("CREDITPFN_ACTIVE_LOG")
    if active and any(Path(active).resolve().is_relative_to(p.resolve()) for p in chosen):
        raise RuntimeError("Run scoped cleanup outside the extension's active log tree")
    for path in chosen:
        wipe(path)
    for path in figures:
        path.unlink(missing_ok=True)
    for path, entries in metadata.items():
        write_json(path, entries)
    if metadata:
        from src.utils.run_notebooks import write_captions, discover
        write_captions(discover(), migrate=False)
    print("Extension cleanup complete; original campaigns, datasets and base checkpoints preserved.")
    print("Rerun notebooks to refresh shared overview figures and All_Results.md from the remaining evidence.")
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--experiment", choices=("all", "0", "4", "5", "6", "7", "8"), default="all")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    clean(args.id, args.experiment, apply=args.apply)


if __name__ == "__main__":
    main()
