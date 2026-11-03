"""CreditPFN's CPT loss must equal upstream TabPFN's own predictive loss at update zero.

A training pipeline that feeds the model differently from upstream inference (targets in the
wrong space, a missing preprocessing step) shows up as a gap between the two numbers before
any weight has moved. Diagnosed 03-11-2026: v2/v3 agree within ~0.02 nats; the v2.6 regressor
is far from both because its own bucket density is a comb, which upstream reproduces too.
Requires the base checkpoints and processed data; runs on CPU with one estimator and no
softmax temperature so both paths compute the same quantity.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow
N_CTX, N_QRY = 500, 300


def _episode(track, table, seed=0):
    path = ROOT / f"data/processed/{track}/{table}.sanitized.csv"
    if not path.is_file():
        pytest.skip(f"processed table unavailable: {path.name}")
    df = pd.read_csv(path)
    y, X = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1:]
    idx = np.random.default_rng(seed).permutation(len(df))[: N_CTX + N_QRY]
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _checkpoint(name):
    path = ROOT / "checkpoints" / name
    if not path.is_file():
        pytest.skip(f"base checkpoint unavailable: {name}")
    return str(path)


def _creditpfn(ckpt, track, X, y, seed):
    from src.train.model import load_tabpfn_for_training
    from src.train import tabpfn_preprocessing as tp
    from src.train.loop import _ensemble_step_loss
    model, criterion, _, cfg = load_tabpfn_for_training(ckpt, track=track, device="cpu")
    model.eval()
    task = "regression" if track == "lgd" else "classification"
    n_classes = None if track == "lgd" else int(len(np.unique(y)))
    tp._CLEAN_CACHE.clear()
    cleaned = tp.clean_loaded_dataset(X_full_df=X, y_full=y if track == "lgd" else y.astype(int),
        cat_columns=[], task_type=task, dataset_id=f"{ckpt}-{seed}", n_estimators=1,
        n_classes=n_classes, inference_config=cfg, base_seed=seed)
    Xc = np.asarray(cleaned.X_clean, dtype=np.float64)
    yy = y if track == "lgd" else y.astype(int)
    batch = tp.build_ensemble_members(X_ctx=Xc[:N_CTX], y_ctx_raw=yy[:N_CTX], X_qry=Xc[N_CTX:],
        y_qry_raw=yy[N_CTX:], feature_schema=cleaned.feature_schema,
        ensemble_configs=cleaned.ensemble_configs, outlier_removal_std=cleaned.outlier_removal_std,
        task_type=task, n_classes=n_classes, inference_config=cfg, n_estimators=1, rng_seed=seed,
        dataset_id="parity")
    with torch.no_grad():
        return float(_ensemble_step_loss(model, batch, criterion=criterion)), model, batch


def _upstream(ckpt, track, X, y, seed):
    if track == "lgd":
        from tabpfn import TabPFNRegressor
        reg = TabPFNRegressor(model_path=ckpt, n_estimators=1, device="cpu", random_state=seed,
                              ignore_pretraining_limits=True, softmax_temperature=1.0)
        reg.fit(X.iloc[:N_CTX], y[:N_CTX])
        out = reg.predict(X.iloc[N_CTX:], output_type="full")
        z = torch.as_tensor((y[N_CTX:] - reg.y_train_mean_) / reg.y_train_std_, dtype=torch.float32)
        nll = reg.znorm_space_bardist_(out["logits"].float().unsqueeze(1), z.reshape(-1, 1)).squeeze()
        return float(nll[torch.isfinite(nll)].mean())
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(model_path=ckpt, n_estimators=1, device="cpu", random_state=seed,
                           ignore_pretraining_limits=True, softmax_temperature=1.0)
    clf.fit(X.iloc[:N_CTX], y[:N_CTX].astype(int))
    proba = clf.predict_proba(X.iloc[N_CTX:])
    labels = np.searchsorted(clf.classes_, y[N_CTX:].astype(int))
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(labels)), labels], 1e-12, 1))))


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("table", ["0001.heloc", "0007.lgd_lendingclub"])
def test_regression_training_loss_matches_upstream_density(version, table):
    ckpt = _checkpoint(f"tabpfn-{version}-regressor-{version}_default.ckpt")
    X, y = _episode("lgd", table)
    ours = np.mean([_creditpfn(ckpt, "lgd", X, y, s)[0] for s in range(2)])
    theirs = np.mean([_upstream(ckpt, "lgd", X, y, s) for s in range(2)])
    assert abs(ours - theirs) < 0.15, (ours, theirs)


@pytest.mark.parametrize("version", ["v2", "v2.6", "v3"])
def test_classification_training_loss_matches_upstream_log_loss(version):
    ckpt = _checkpoint(f"tabpfn-{version}-classifier-{version}_default.ckpt")
    X, y = _episode("pd", "0010.thomas")
    ours = np.mean([_creditpfn(ckpt, "pd", X, y, s)[0] for s in range(2)])
    theirs = np.mean([_upstream(ckpt, "pd", X, y, s) for s in range(2)])
    assert abs(ours - theirs) < 0.05, (ours, theirs)


def test_v26_regressor_density_is_flagged_by_the_bar_nll_guard():
    """Documents the released checkpoint's property; fails if Prior Labs changes it."""
    from src.train.density_checks import ROUGHNESS_LIMIT, bucket_roughness
    from src.train.loop import _forward_one_member
    ckpt = _checkpoint("tabpfn-v2.6-regressor-v2.6_default.ckpt")
    X, y = _episode("lgd", "0007.lgd_lendingclub")
    _, model, batch = _creditpfn(ckpt, "lgd", X, y, 0)
    m = batch.members[0]
    with torch.no_grad():
        logits = _forward_one_member(model, X_ctx=m.X_context, y_ctx=m.y_context, X_qry=m.X_query,
                                     cat_idx=m.categorical_idx, outlier_removal_std=m.outlier_removal_std)
    assert float(bucket_roughness(logits.squeeze(1).softmax(-1)).median()) > ROUGHNESS_LIMIT
