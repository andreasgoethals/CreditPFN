import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, mean_squared_error

from src.utils.evaluation_evidence import predictions, check_scores


def saved(tmp_path, y, p):
    root = tmp_path / 'LGD/tabpfn-trained__v2'
    root.mkdir(parents=True)
    metric = root / 'campaign.csv'
    metric.write_text('metric\n')
    blob = root / 'campaign.csv.gz'
    pd.DataFrame({'y_true': y.astype('float64'), 'y_pred': p.astype('float64')}).to_csv(blob, index=False)
    metric.with_name(metric.name+'.evaluation.json').write_text(json.dumps({
        '__predictions__': {'file': blob.name, 'bytes': blob.stat().st_size}}))
    return metric


def test_original_fp32_r2_near_zero_is_reproduced_without_relaxing_tolerance(tmp_path):
    y = np.random.default_rng(42).uniform(size=1000).astype('float32')
    p = np.full_like(y, y.mean()+7e-5)
    record = dict(r2=float(r2_score(y,p)), rmse=float(np.sqrt(mean_squared_error(y,p))))
    assert not np.isclose(r2_score(y.astype('float64'),p.astype('float64')),record['r2'],rtol=2e-6,atol=1e-8)
    frame = predictions(saved(tmp_path,y,p))
    assert frame.y_true.dtype == frame.y_pred.dtype == np.float32
    np.testing.assert_array_equal(frame.y_true,y)
    np.testing.assert_array_equal(frame.y_pred,p)
    assert check_scores(record,frame.y_true.to_numpy(),frame.y_pred.to_numpy(),'lgd') == record


def test_native_float64_predictions_are_never_rounded_to_pass_an_audit(tmp_path):
    y = np.array([0.1,0.2,0.3,0.4],dtype='float32')
    p = np.array([0.123456789123,0.234567891234,0.345678912345,0.456789123456],dtype='float64')
    frame = predictions(saved(tmp_path,y,p))
    assert frame.y_pred.dtype == np.float64
    np.testing.assert_array_equal(frame.y_pred,p)
    original = dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))))
    check_scores(original,frame.y_true.to_numpy(),frame.y_pred.to_numpy(),'lgd')
    tampered = frame.y_pred.to_numpy().copy();tampered[0] += .01
    with pytest.raises(ValueError,match='disagree'):
        check_scores(original,frame.y_true.to_numpy(),tampered,'lgd')
