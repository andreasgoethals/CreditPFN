"""v2/v2.6 isolation, packed projections, study design and scheduler invariants."""
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.model import lora, diagscale
from src.utils import legacy_design as design
from src.utils import legacy_submit as launch


class AlongColumnAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for p in ("q", "k", "v", "out"):
            setattr(self, p + "_projection", nn.Linear(8, 8, bias=False))

    def forward(self, x):
        return self.out_projection(F.scaled_dot_product_attention(
            self.q_projection(x), self.k_projection(x), self.v_projection(x)))


class AlongRowAttention(AlongColumnAttention):
    pass


class TabPFNV2p6(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(AlongRowAttention(), AlongColumnAttention()) for _ in range(2)])
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class PerFeatureTransformer(nn.Module):
    def __init__(self, layout="qkv"):
        super().__init__()
        from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
        from tabpfn.architectures.base.config import ModelConfig
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList([nn.Sequential(*[MultiHeadAttention(
            d_k=4, d_v=3 if layout == "q_k_v" else 4, device="cpu", dtype=torch.float32,
            config=ModelConfig(emsize=8, nhead=2, max_num_classes=2, num_buckets=10),
            initialize_output_to_zero=False, share_kv_across_n_heads=1 if layout == "qkv" else 2) for _ in range(2)])])
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.transformer_encoder.layers[0](x))


@pytest.mark.parametrize("layout", ["qkv", "q_kv", "q_k_v", "separate"])
@pytest.mark.parametrize("mode", ["lora", "diagscale"])
def test_real_packed_and_separate_adapters(layout, mode):
    torch.manual_seed(41)
    model = TabPFNV2p6() if layout == "separate" else PerFeatureTransformer(layout)
    pristine = copy.deepcopy(model)
    x, y = torch.randn(2, 12, 8), torch.randn(2, 12, 2)
    before = model(x).detach()
    originals = [(p, p.detach().clone()) for p in model.parameters()]
    module = lora if mode == "lora" else diagscale
    parameters = module.inject(model, family="tabpfn")
    assert torch.equal(before, model(x))
    assert set(map(id, parameters)) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    for _ in range(3):
        F.mse_loss(model(x), y).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(p, value) and p.grad is None for p, value in originals)
    assert not torch.equal(before, model(x))
    if mode == "lora":
        merged = lora.merged_state_dict(model)
        pristine.load_state_dict(merged, strict=True)
        assert torch.allclose(model(x), pristine(x), atol=1e-7, rtol=1e-6)
        for name, value in merged.items():
            prior = dict(pristine.state_dict())[name]
            assert value.shape == prior.shape
        assert lora.weight_movement(model) > 0
        # Packed K rows have exactly zero additive change.
        for _, owner, seq in lora._installed(model):
            adapter = seq[0]
            if adapter.projections == ("q", "v"):
                matrix = adapter.delta().reshape(-1, adapter.shape[-1])
                assert torch.count_nonzero(matrix[adapter.slices[0][1]:adapter.slices[1][0]]) == 0
    else:
        module.inject(pristine, family="tabpfn")
        module.restore(pristine, module.adapter_state(model))
        assert torch.equal(model(x), pristine(x))
    values = [p.detach().clone() for p in parameters]
    model.to(torch.bfloat16)
    assert all(p.dtype == torch.float32 and torch.equal(p, value) for p, value in zip(parameters, values))


def test_legacy_query_scale_is_query_multiplier_only():
    weight = torch.randn(3, 2, 4, 8)
    adapter = diagscale.QueryWeightScale(weight, 0, 8)
    with torch.no_grad():
        adapter.delta.copy_(torch.linspace(-.2, .3, 8))
    x = torch.randn(7, 8)
    scaled = F.linear(x, adapter(weight).flatten(0, 2))
    original = F.linear(x, weight.flatten(0, 2))
    assert torch.allclose(scaled[:, :8], original[:, :8] * (1 + adapter.delta), atol=1e-6)
    assert torch.equal(scaled[:, 8:], original[:, 8:])


@pytest.mark.parametrize("version", ["v2", "v2p6"])
@pytest.mark.parametrize("track", ["pd", "lgd"])
def test_training_entry_reaches_validated_legacy_diagscale(version, track, monkeypatch, tmp_path):
    """Exercise the production entry point, not just the standalone adapter mapper."""
    from src.train import loop
    from src.utils import paths

    cfg = design.configurations("entryprobe")[f"e5_{track}_{version}"]
    cfg.device = "cpu"
    cfg.train.dataloader_workers = 0
    base = tmp_path / "base.ckpt"
    base.write_bytes(b"mock checkpoint; the native loader is replaced below")
    model = PerFeatureTransformer() if version == "v2" else TabPFNV2p6()
    monkeypatch.setattr(paths, "resolve_base_checkpoint", lambda _: base)
    monkeypatch.setattr(loop, "split_from_cfg", lambda *a, **kw: SimpleNamespace(
        train=[SimpleNamespace(dataset_id="synthetic", n_rows=32)], test=[], summary={}))
    monkeypatch.setattr(loop, "load_tabpfn_for_training", lambda *a, **kw: (model, None, None, None))
    original = diagscale.inject

    class InjectionVerified(Exception):
        pass

    def verify_injection(*args, **kwargs):
        parameters = original(*args, **kwargs)
        assert parameters and all(p.requires_grad for p in parameters)
        assert model._creditpfn_adapter["base_sha256"]
        raise InjectionVerified

    monkeypatch.setattr(diagscale, "inject", verify_injection)
    with pytest.raises(InjectionVerified):
        loop.train_one_config(cfg)


@pytest.mark.parametrize("mode", [lora, diagscale])
def test_real_legacy_cache_preserves_adapted_predictions(mode):
    model = PerFeatureTransformer()
    module = model.transformer_encoder.layers[0][0]
    parameters = mode.inject(model, family="tabpfn")
    with torch.no_grad():
        for p in parameters:
            p.add_(.013)
        x = torch.randn(1, 12, 8)
        first = module(x[:, 8:], x[:, :8], cache_kv=True)
        repeat = module(x[:, 8:], use_cached_kv=True)
    assert torch.allclose(first, repeat, atol=1e-6, rtol=1e-5)


def test_unknown_layout_rejected_before_mutation():
    model = TabPFNV2p6()
    model.blocks[1][0].q_projection = nn.Identity()
    before = copy.deepcopy(model.state_dict())
    for mode in (lora, diagscale):
        with pytest.raises(ValueError, match="Incomplete"):
            mode.inject(model, family="tabpfn")
        assert all(p.requires_grad for p in model.parameters())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_extension_design_preserves_old_files_and_declared_factors(monkeypatch, tmp_path):
    from src.train.config import resolve_grid, validate_lora_config
    from src.train.size_studies import validate_config
    monkeypatch.setattr(design, "root", lambda identifier: tmp_path / identifier)
    before = {p: p.read_bytes() for n in range(4, 9) for p in Path(f"config/experiment{n}").glob("*.yaml")}
    configs = design.configurations("unit1")
    assert len(configs) == 16
    counts = {5: 0, 6: 0, 7: 0, 8: 0}
    for name, cfg in configs.items():
        if name.startswith("e4"):
            assert cfg.context_rows[-1] == (10000 if cfg.base_version == "v2" else 50000)
            assert cfg.outer_folds == 5 and cfg.families == ["tabpfn"]
            continue
        counts[int(name[1])] += len(resolve_grid(cfg, single=False)) * 4
        assert cfg.experiment.extension_id == "unit1" and cfg.experiment.legacy_gate
        assert "extensions/unit1" in cfg.checkpoint.trained_dir
        assert cfg.corpus.n_folds == 4 and cfg.corpus.split_seed == 1729
        validate_config(cfg)
        if name.startswith("e8"):
            validate_lora_config(cfg)
        assert cfg.train.max_rows_per_step <= (11000 if "v2p6" in name else 10000)
    assert counts == {5: 48, 6: 16, 7: 16, 8: 48}
    assert before == {p: p.read_bytes() for p in before}


def test_scope_routing_and_restoration(monkeypatch, tmp_path):
    from src.utils.paths import results_dir, activate_experiment, training_dir
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("CREDITPFN_LEGACY_ID", raising=False)
    with design.scope("one", "experiment5"):
        assert "experiment5/extensions/one" in results_dir().as_posix()
        assert "experiment1/extensions" not in results_dir(experiment="experiment1").as_posix()
        cfg = design.configurations("one")["e5_pd_v2"]
        activate_experiment(cfg)
        assert "extensions/one" in training_dir().as_posix()
    assert "extensions" not in results_dir(experiment="experiment5").as_posix()
    with design.scope("one"):
        from src.train.config import load_train_config
        activate_experiment(load_train_config(config_path="config/experiment1/pd.yaml"))
        assert "extensions" not in results_dir().as_posix()


def test_single_gate_contains_all_cases_and_only_true_dependencies():
    plan = launch.test_plan("scope1")
    assert [p["key"] for p in plan] == ["prepare", "smoke", "capacity", "inspect"]
    assert len(plan[1]["tasks"]) == 12 and len(plan[2]["tasks"]) == 68
    assert not any("%" in arg for p in plan for arg in p["args"])


@pytest.mark.parametrize("refusal", ["QOSMaxSubmitJobPerUserLimit", "maximum number of submitted jobs"])
def test_submission_quota_split_resume_and_dependencies(tmp_path, refusal):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr=refusal)
        return SimpleNamespace(returncode=0, stdout=f"{100 + len(calls)};mindwell\n", stderr="")
    plan = [dict(key="a", needs=[], tasks=[3, 4, 5, 6], args=["job.slurm"]),
            dict(key="b", needs=["a"], tasks=None, args=["inspect.slurm"])]
    path = tmp_path / "receipt.json"
    state = launch.submit_plan(plan, path, runner=runner, sleep=lambda _: None, max_array_size=1001)
    assert [r["tasks"] for r in state["accepted"]] == [[3, 4], [5, 6], None]
    assert "--dependency=afterok:102:103" in calls[-1]
    assert not any("%" in a for c in calls for a in c)
    launch.submit_plan(plan, path, runner=lambda *a, **k: pytest.fail("duplicate submission"))


def test_generic_qos_error_never_retries_even_previously_accepted_resources(tmp_path):
    calls, sleeps = [], []
    message = "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)"
    plan = [dict(key="a", needs=[], tasks=[0], args=["--mem=64G", "--export=ALL,A=1", "job.slurm"]),
            dict(key="b", needs=[], tasks=[0, 1], args=["--mem=64G", "--export=ALL,A=2", "job.slurm"])]
    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) in (2, 3):
            return SimpleNamespace(returncode=1, stdout="", stderr=message)
        return SimpleNamespace(returncode=0, stdout=f"{200 + len(calls)};mindwell", stderr="")
    with pytest.raises(RuntimeError, match="inspect Slurm"):
        launch.submit_plan(plan, tmp_path / "known.json", runner=runner, sleep=sleeps.append, max_array_size=1001)
    assert sleeps == [] and len(calls) == 2
    assert [r["tasks"] for r in json.loads((tmp_path/"known.json").read_text())["accepted"]] == [[0]]
    path = tmp_path / "unknown.json"
    with pytest.raises(RuntimeError, match="inspect Slurm"):
        launch.submit_plan(plan[:1], path,
            runner=lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr=message),
            sleep=lambda _: pytest.fail("unbounded retry of invalid allocation"), max_array_size=1001)
    assert json.loads(path.read_text())["in_flight"] is None


def test_uncertain_submission_is_never_retried(tmp_path):
    path = tmp_path / "receipt.json"
    plan = [dict(key="a", needs=[], tasks=[0], args=["job.slurm"])]
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("sbatch", 60)
    with pytest.raises(subprocess.TimeoutExpired):
        launch.submit_plan(plan, path, runner=timeout, max_array_size=1001)
    with pytest.raises(RuntimeError, match="uncertain"):
        launch.submit_plan(plan, path, runner=lambda *a, **k: pytest.fail("unsafe retry"))


def test_cleanup_only_selected_extension(monkeypatch, tmp_path):
    from src.utils import legacy_clean as cleaner
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_STAGING_ROOT", str(tmp_path / "project"))
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.delenv("CREDITPFN_ACTIVE_LOG", raising=False)
    marker = design.root("unit") / ".scope.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(kind="creditpfn_legacy_scope", id="unit")))
    ours = cleaner.targets("unit", "5")
    for p in ours:
        p.mkdir(parents=True, exist_ok=True)
        (p / "owned.txt").write_text("ours")
    other = tmp_path / "output CreditPFN/experiment5/results/keep.txt"
    other.parent.mkdir(parents=True)
    other.write_text("historical")
    cleaner.clean("unit", "5")
    assert all((p / "owned.txt").exists() for p in ours)
    cleaner.clean("unit", "5", apply=True)
    assert other.read_text() == "historical" and marker.exists()
    assert all(not (p / "owned.txt").exists() for p in ours)
    with pytest.raises(ValueError):
        cleaner.clean("../escape", "all", apply=True)


def test_preparation_generates_all_production_debug_plans_and_shared_controls(monkeypatch, tmp_path):
    from src.train import corpus
    from src.data import preprocessing, retention
    from src.utils import experiment, legacy_checks as checks, prepare_experiment as prep
    from src.eval import context_curve
    from src.train.corpus import DatasetRef
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("CREDITPFN_LEGACY_ID", raising=False)
    refs = {}
    for track, count in (("pd", 17), ("lgd", 8)):
        refs[track] = []
        for i in range(count):
            path = tmp_path / f"{track}_{i}.csv"
            path.write_text("x,y\n1,0\n2,1\n")
            refs[track].append(DatasetRef(f"{track}{i}", track, "classification" if track == "pd" else "regression",
                "y", (), path, 40000 if i % 3 else 999))
    monkeypatch.setattr(corpus, "build_dataset_pool", lambda track: refs[track])
    monkeypatch.setattr(preprocessing, "DATASET_METADATA", {d.dataset_id: {"track": track} for track in refs for d in refs[track]})
    monkeypatch.setattr(retention, "load_refs", lambda *args: [])
    base = tmp_path / "base.ckpt"
    base.write_bytes(b"test-only binding")
    (tmp_path / "base.ckpt.provenance.json").write_text("{}")
    monkeypatch.setattr("src.utils.paths.resolve_base_checkpoint", lambda path: base)
    monkeypatch.setattr(checks, "resolve_base_checkpoint", lambda path: base)
    monkeypatch.setattr(context_curve, "reference_checkpoint", lambda track, family, partition, cfg:
        (dict(base=str(base), cpt=str(base), base_sha256="test", cpt_sha256="test", provenance_sha256="test"),
         refs[track][partition::4]))
    monkeypatch.setattr(experiment, "code_identity", lambda **kwargs: "test-source")
    monkeypatch.setattr(prep, "code_identity", lambda **kwargs: "test-source")
    workflow = checks.prepare("safety1")
    import os
    assert not os.environ.get("CREDITPFN_LEGACY_ID")
    assert (len(workflow["smoke"]), len(workflow["capacity"])) == (12, 68)
    assert sum(c["kind"] == "adapter" for c in workflow["smoke"]) == 8
    for key, path in workflow["production"].items():
        if key.startswith("e7_"):
            cfg = OmegaConf.load(path)
            with design.scope("safety1", "experiment7"):
                plan = prep.read_plan(prep.plan_path(cfg.run_name, "pd"))
            assert len(plan["shared_reference"]["trials"]) == 4
    # The production launcher queues all training, context evaluation and
    # dependent benchmarks, retaining only genuine per-partition prerequisites.
    monkeypatch.setattr(checks, "require_gate", lambda _: workflow)
    monkeypatch.setattr(prep, "check_prepared", lambda _: None)
    plan = launch.research_plan("safety1")
    assert not os.environ.get("CREDITPFN_LEGACY_ID")
    trained = [p for p in plan if p["key"].startswith(("e5_", "e6_", "e7_", "e8_")) and not p["needs"]]
    benchmarks = [p for p in plan if p["needs"]]
    contexts = [p for p in plan if p["key"].startswith("e4_")]
    assert sum(len(p["tasks"]) for p in trained) == 128
    assert sum(len(p["tasks"]) for p in benchmarks) == 896
    assert sum(len(p["tasks"]) for p in contexts) == 250
    assert {p["needs"][0] for p in benchmarks} == {p["key"] for p in trained}
    assert not any("%" in arg for p in plan for arg in p["args"])
    with pytest.raises(RuntimeError, match="already"):
        checks.prepare("safety1")


def test_gate_rejects_missing_receipts_and_changed_source(monkeypatch, tmp_path):
    from src.utils import legacy_checks as checks
    from src.utils.atomic import write_json
    monkeypatch.setattr(checks, "root", lambda _: tmp_path)
    monkeypatch.setattr(checks, "fingerprint", lambda: "current")
    monkeypatch.setattr(checks, "environment_versions", lambda **kw: {"runtime": "test"})
    payload = dict(kind="creditpfn_legacy", id="one", prepared=True, fingerprint="current",
        versions={"runtime": "test"}, files=[], inputs=[], smoke=[{}], capacity=[])
    evidence = tmp_path / "bound.txt"
    evidence.write_text("binding fixture")
    payload.update(files=[checks._binding(evidence)], inputs=[checks._binding(evidence)])
    write_json(tmp_path / "workflow.json", payload)
    with pytest.raises(FileNotFoundError):
        checks.require_gate("one")
    payload["fingerprint"] = "old"
    write_json(tmp_path / "workflow.json", payload)
    with pytest.raises(RuntimeError, match="stale"):
        checks.require_gate("one")


def test_publication_cleanup_preserves_original_pages(monkeypatch, tmp_path):
    from src.utils import legacy_clean as cleaner
    from src.utils.atomic import write_json
    from src.visualize.paths import figures_dir
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("VSC_DATA", raising=False)
    monkeypatch.delenv("CREDITPFN_ACTIVE_LOG", raising=False)
    write_json(design.root("unit") / ".scope.json", dict(kind="creditpfn_legacy_scope", id="unit"))
    folder = figures_dir() / "experiment5"
    entries = [dict(index=1, stem="old", name="old", caption="Original"),
               dict(index=2, stem="new", name="new", caption="Extension", provenance=dict(extension_id="unit"))]
    manifest = folder / "_metadata/01_training.json"
    write_json(manifest, entries)
    for entry in entries:
        (folder / (entry["stem"] + ".pdf")).write_bytes(b"fixture")
    cleaner.clean("unit", "5", apply=True)
    assert (folder / "old.pdf").is_file() and not (folder / "new.pdf").exists()
    assert json.loads(manifest.read_text()) == entries[:1]


def test_report_selection_is_explicit_and_scoped(monkeypatch, tmp_path):
    from src.visualize import legacy_extensions as report
    from src.utils.atomic import write_json
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("CREDITPFN_ANALYSIS_ROOT", str(tmp_path / "output CreditPFN"))
    (tmp_path / "output CreditPFN").mkdir()
    monkeypatch.delenv("CREDITPFN_ANALYSIS_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CREDITPFN_ANALYSIS_EXTENSION_ID", raising=False)
    monkeypatch.delenv("CREDITPFN_LEGACY_ID", raising=False)
    assert report.selected() is None
    for identifier in ("one", "two"):
        paths = design.write_configs(identifier)
        write_json(design.root(identifier) / "workflow.json", dict(kind="creditpfn_legacy", id=identifier,
            prepared=True, production={k: str(p) for k, p in paths.items()}))
    with pytest.raises(ValueError, match="Multiple"):
        report.selected()
    monkeypatch.setenv("CREDITPFN_ANALYSIS_EXTENSION_ID", "one")
    selection = report.selected()
    assert selection[1]["id"] == "one"
    configs = report.configs(selection)
    with design.scope("one", "experiment7"):
        arms = report.size_arms(7, configs, "v2.6")
    assert list(arms) == ["2,048 rows", "8,192 rows", "11,000 rows"]
    assert all(r.trials.status.eq("PENDING").all() for r in arms.values())
    assert sum(len(r.trials) for r in arms.values()) == 12


@pytest.mark.parametrize("experiment", [5, 8])
@pytest.mark.parametrize("phase", ["training", "results"])
def test_render_legacy_adapters_keeps_values_captions_and_single_model_panels(monkeypatch, tmp_path, experiment, phase):
    from src.visualize import legacy_extensions as report, campaign as cp, style
    from tests.test_new_experiment_reports import adapter_fixture
    from tests.test_lora_reports import lora_fixture
    from src.visualize.reporting import page_text
    style.apply()
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    configuration = design.configurations("render1")
    monkeypatch.setattr(report, "selected", lambda: (tmp_path, {"id": "render1"}))
    monkeypatch.setattr(report, "configs", lambda _: configuration)
    fixture = adapter_fixture if experiment == 5 else lora_fixture
    monkeypatch.setattr(cp, "load_campaign", lambda exp, track, cfg_override: fixture(track, cfg_override))
    pages, tables = [], []
    monkeypatch.setattr(cp, "show", lambda sink, value: pages.extend(value))
    recorder = SimpleNamespace(table=lambda title, data: tables.append((title, data)), add=lambda *args: None)
    report.render(recorder, None, experiment, phase)
    assert pages and tables
    for page in pages:
        assert page.tables and page_text(page) and page.caption.startswith("TabPFN v2")
        assert page.provenance["extension_id"] == "render1"
        assert page.name.startswith("extension_render1_")
        assert page.figure.get_figwidth() == style.WIDTH_FULL
        assert not any("v3" in ax.get_title().lower() for ax in page.figure.axes)
        page.figure.canvas.draw()
