"""RQ compaction preserves paired scientific units and publication ownership."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualize import figures, questions as rq
from src.visualize import paths as publication_paths
from src.utils import paths as scientific_paths


def test_rq_publication_does_not_change_scientific_routing(isolated_output):
    assert publication_paths.notebook_parts('rq2/01_training')[0] == 'rq2'
    assert scientific_paths.notebook_parts('rq2/01_training')[0] == 'general'
    assert publication_paths.notebook_figures_dir('rq2/01_training').name == 'rq2'
    for invalid in ('../rq2/a', 'rq2/../a', 'rq2'):
        with pytest.raises(ValueError):
            publication_paths.notebook_parts(invalid)


def test_retirement_cleans_owned_figures_not_results_or_other_notebooks(isolated_output):
    fig, ax = plt.subplots(); ax.plot([0, 1])
    old = figures.FigureSaver('experiment3/02_results')
    old(fig, 'factorial', caption='Old factorial.')
    current = figures.FigureSaver('rq2/01_training')
    current(fig, 'training', caption='Training.')
    peer = figures.FigureSaver('rq2/02_results')
    peer(fig, 'results', caption='Results.')
    numerical = scientific_paths.outputs_dir() / 'shared/main_grid/results/evidence.csv'
    numerical.parent.mkdir(parents=True); numerical.write_text('preserve')
    figures.clear('rq2/01_training')
    assert peer.last_path.is_file() and old.last_path.is_file()
    figures.clear_collection(('rq2/02_results',))
    assert not old.last_path.exists()
    assert numerical.read_text() == 'preserve'
    # The runner clears current owners before executing them too.
    assert not peer.last_path.exists()
    plt.close(fig)


def test_fixed_table_cohort_excludes_partial_arms_and_rejects_duplicates():
    data = pd.DataFrame([dict(base='v2', config=c, dataset=t, effect=e)
        for c,t,e in [('A','one',.1),('B','one',.2),('A','two',99.)]])
    matched = rq.fixed_cohort(data)
    assert set(matched.dataset) == {'one'}
    assert matched.groupby('config').effect.mean().to_dict() == {'A':.1,'B':.2}
    with pytest.raises(ValueError, match='Duplicate'):
        rq.fixed_cohort(pd.concat([data,data.iloc[:1]]))


def test_seed_contrast_is_matched_alternative_minus_reference():
    data = pd.DataFrame([dict(base='v2', dataset=t, metric='roc_auc', domain='credit',
        experiment=e, reference=e == 1, effect=value)
        for e,t,value in [(1,'one',.01),(2,'one',.015),(1,'missing',.8)]])
    result = rq.choice_differences(data,1)
    assert result.dataset.tolist() == ['one']
    assert result.effect.iloc[0] == pytest.approx(.005)
    assert result.effect_reference.iloc[0] == .01
    with pytest.raises(ValueError, match='No recipe-mechanism'):
        rq.choice_differences(data,2)


def test_episode_contrast_uses_exact_shared_large_table_control():
    data = pd.DataFrame([dict(base='v3',dataset='one',metric='roc_auc',domain='credit',
        experiment=e,arm=a,sampling=s,effect=v)
        for e,a,s,v in [(3,None,'one_sample',.01),(3,None,'full_pass',.03),
                       (3,None,'accumulate',.005),(6,'All tables','full_pass',.01),
                       (6,'Large tables','full_pass',.02),(7,'2,048 rows','full_pass',.015),
                       (7,'26,000 rows','full_pass',.04)]])
    result = rq.choice_differences(data,3)
    assert result.effect.tolist() == pytest.approx([.02,-.005,.01,-.005,.02])
    assert len(result) == 5


def test_compact_factorial_keeps_interaction_and_fixed_cohort():
    from tests.test_study_reorganization import factorial
    frame = factorial()
    frame = pd.concat([frame, frame.assign(learning_rate=1e-5)], ignore_index=True)
    # Lose one cell for B at one rate: B must leave the whole curve.
    frame = frame.drop(frame.index[-1])
    page = rq.plot_factorial(frame,'pd')[0]
    cells = page.tables['All four paired cell values']
    assert set(cells.dataset) == {'A'} and len(cells) == 8
    interaction = page.tables['Anchoring simple effects and interaction']
    assert interaction[interaction.condition.eq('Interaction')].contrast.eq(3).all()
    page.figure.canvas.draw()
    plt.close(page.figure)


def test_loss_curve_requires_every_planned_trial(monkeypatch):
    meta=dict(base='v2',learning_rate=3e-7,l2sp_lambda=.003,frozen=False,sampling='one_sample')
    trials=pd.DataFrame([dict(**meta,trial_name=n) for n in ('one','two')])
    run=SimpleNamespace(trials=trials,track='pd')
    raw=pd.DataFrame([dict(**meta,trial_name=n,window_end=u,train_loss=l)
                     for n,u,l in [('one',10,1.),('two',10,3.),('one',20,.1)]])
    monkeypatch.setattr(rq.cp,'optimization_curves',lambda *args:raw)
    monkeypatch.setattr(rq.cp,'optimization_grid',lambda *args:[10,20])
    data=rq.training_curves([(1,None,run)],'loss')
    assert data['mean'].iloc[0] == 2.
    assert np.isnan(data['mean'].iloc[1])
    with pytest.raises(ValueError,match='counted twice'):
        rq.training_curves([(1,None,run),(1,None,run)],'loss')


def test_reused_trials_count_once_across_views():
    trials=pd.DataFrame([dict(trial_name='shared',status='OK')])
    run=SimpleNamespace(trials=trials,track='pd',cfg=SimpleNamespace(run_name='same'))
    assert len(rq.coverage([(6,'Large tables',run),(7,'8,192 rows',run)])) == 1
