"""Separate policy and episode-size analyses with an explicitly reused control."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.train.config import load_train_config
from src.train.size_studies import match_reference
from src.utils.experiment import digest_json, scientific_config
from src.utils.paths import REPO_ROOT, group_for_run, manifests_dir
from src.visualize import campaign as cp, comparisons as cmp, dashboards as db, eval_viz, style
from src.visualize.inputs import analysis_location
from src.visualize.paper_figures import effect_label


def _select(run, minimum):
    trials = run.trials[run.trials.min_train_rows.eq(minimum)].copy()
    names = set(trials.trial_name)
    def select(frame):
        return frame[frame.trial_name.isin(names)].copy() if 'trial_name' in frame else frame.copy()
    evaluation = run.evaluation.copy()
    if not evaluation.empty:
        keep = ~evaluation.source.str.endswith('-trained', na=False) | evaluation.method_dirname.map(
            lambda name: eval_viz._decode_method_dirname(name).get('min_train_rows', 0) == minimum)
        evaluation = evaluation[keep]
    return cp.Campaign(run.cfg, trials, select(run.trajectories), select(run.histories), evaluation)


def plan(run):
    group = group_for_run(str(run.cfg.run_name))
    path = analysis_location(group, 'manifests', manifests_dir(group)) / 'plans' / f'{run.cfg.run_name}_pd.json'
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('sha256') != digest_json({k:v for k,v in payload.items() if k != 'sha256'}):
        raise ValueError('Size-study plan checksum failed')
    recorded = OmegaConf.create(payload['config'])
    if (scientific_config(recorded) != scientific_config(run.cfg)
            or recorded.tunable != run.cfg.tunable or recorded.evaluation != run.cfg.evaluation):
        raise ValueError('Downloaded size-study plan differs from this notebook configuration')
    return payload


def load_arms(experiment, *, configs=None):
    if experiment not in (6, 7):
        raise ValueError('Expected experiment 6 or 7')
    cfg = configs[6] if configs else load_train_config(config_path=str(REPO_ROOT/'config/experiment6/pd.yaml'))
    corpus = cp.load_campaign(6, 'pd', cfg_override=cfg)
    if experiment == 6:
        return {'All tables':_select(corpus, 0), 'Large tables':_select(corpus, 10001)}
    arms = {'8,192 rows':_select(corpus, 10001)}
    reference = plan(corpus)
    caps = sorted(k for k in configs if k != 6) if configs else (2048, 26000)
    for cap in caps:
        cfg = configs[cap] if configs else load_train_config(config_path=str(REPO_ROOT/f'config/experiment7/pd_rows{cap}.yaml'))
        current = cp.load_campaign(7, 'pd', cfg_override=cfg)
        payload = plan(current)
        if payload is not None:
            if reference is None or payload.get('shared_reference', {}).get('sha256') != reference['sha256']:
                raise ValueError('Experiment 7 needs its original experiment-6 reference plan in the download')
            if match_reference(payload, reference) != payload['shared_reference']['trials']:
                raise ValueError('Shared-control trial mapping does not match the downloaded plans')
        elif current.trials.status.ne('PENDING').any() or not current.evaluation.empty or not current.trajectories.empty:
            raise ValueError('Experiment-7 evidence is missing its immutable plan')
        arms[f'{cap:,} rows'] = current
    return {f'{cap:,} rows':arms[f'{cap:,} rows'] for cap in sorted([8192, *caps])}


def coverage(arms):
    return pd.DataFrame([dict(arm=name, source_experiment=group_for_run(str(run.cfg.run_name)),
        source_run=str(run.cfg.run_name), planned=len(run.trials), completed=int(run.trials.status.eq('OK').sum()),
        other_outcomes=int((~run.trials.status.isin(['OK','PENDING'])).sum()),
        pending=int(run.trials.status.eq('PENDING').sum()), trajectory_records=len(run.trajectories),
        benchmark_fold_records=len(run.evaluation)) for name,run in arms.items()])


def corpus_inventory(arms):
    rows=[]
    for name,run in arms.items():
        payload=plan(run)
        if payload is None:
            continue
        minimum=int(run.trials.min_train_rows.iloc[0])
        for partition in payload['partitions']:
            for arm in partition.get('corpus_arms',[]):
                if arm['min_train_rows'] != minimum:
                    continue
                for table in arm['tables']:
                    rows.append(dict(arm=name, partition=partition['index'], **table,
                                     held_out_tables=len(partition['test'])))
    return pd.DataFrame(rows)


def matched_observations(arms, *, benchmark=False, metric='roc_auc'):
    """Same table/partition/update across all arms; five complete folds for endpoints."""
    pieces=[]
    for label,run in arms.items():
        for domain in ('credit','ood'):
            data=cmp.observations(run, domain=domain, benchmark=benchmark, metric=metric)
            data=cmp.dataset_means(data,run,['effect'])
            if not data.empty:
                pieces.append(data.assign(arm=label))
    if not pieces:
        return pd.DataFrame()
    data=pd.concat(pieces,ignore_index=True)
    keys=['base','dataset','domain','metric'] + ([] if benchmark else ['updates'])
    if data.duplicated(keys+['arm']).any():
        raise ValueError('Duplicate size-study comparisons')
    support=data.groupby(keys).arm.agg(set)
    complete=support[support.map(lambda values:values==set(arms))].index
    return data.set_index(keys).loc[complete].reset_index() if len(complete) else data.iloc[:0]


def contrasts(data, reference):
    if data.empty:
        return data
    keys=['base','dataset','domain','metric'] + (['updates'] if 'updates' in data else [])
    anchor=data[data.arm.eq(reference)][keys+['effect']]
    paired=data[~data.arm.eq(reference)].merge(anchor,on=keys,suffixes=('', '_reference'),validate='many_to_one')
    return paired.assign(difference=paired.effect-paired.effect_reference)


def _line(ax, x, y, label):
    ax.plot(x,y,label=label,color=style.SIZE_STUDY_COLORS[label],linestyle=style.SIZE_STUDY_STYLES[label],
            linewidth=style.THIN_LINE,marker='o',markersize=style.CURVE_MARKER_SIZE)


def plot_optimization(arms):
    fields={'train_loss':'Query cross-entropy', 'grad_norm_mean':'Gradient norm',
            'l2sp_penalty':'Applied L2-SP penalty', 'weight_drift':'Relative weight displacement'}
    frames=[db.complete_windows(run,field).assign(arm=label) for label,run in arms.items() for field in fields]
    frames=[f for f in frames if not f.empty]
    if not frames:
        return []
    data=pd.concat(frames,ignore_index=True)
    pages=[]
    first=next(iter(arms.values()))
    interpretation = ('The corpus-policy comparison changes the loss population, so a lower '
                      'training loss is not by itself evidence of better transfer. '
                      if 'All tables' in arms else
                      'All arms use the same large-table pool. Larger episodes change both context '
                      'and query counts; training loss is not a held-out transfer score. ')
    for base,group in data.groupby('base'):
        if not group.value.notna().any():
            continue
        fig,axes=db._grid(f'PD / {base}: training dynamics')
        plotted=[]
        for ax,(metric,label) in zip(axes,fields.items()):
            part=group[group.metric.eq(metric)]
            if not part.value.notna().any():
                ax.remove();continue
            for arm in arms:
                curve=part[part.arm.eq(arm)].set_index('window_end').reindex(cp.optimization_grid(first,metric))
                _line(ax,curve.index,curve.value,arm)
                plotted.append(curve.reset_index().assign(arm=arm,metric=metric))
            ax.set_ylabel(label);ax.set_xlabel('Successful updates')
        db._legend(fig,next(ax for ax in axes if ax in fig.axes))
        pages.append(cp.Page(f'size_optimization_{base}',fig,
            'Native query cross-entropy excludes anchoring. Each line averages the four planned dataset partitions; '
            'missing partitions leave gaps. ' + interpretation + 'Other panels show optimization and movement, '
            'not final predictive quality.', {'Window values and partition support':pd.concat(plotted,ignore_index=True)}))
    return pages


def plot_monitors(arms):
    data=matched_observations(arms)
    if data.empty or not data.updates.gt(0).any():
        return []
    domains=[d for d in ('credit','ood') if data.domain.eq(d).any()]
    title=('PD: credit transfer and non-credit retention' if len(domains)==2
           else 'PD: '+cmp.DOMAINS[domains[0]].lower()+' monitoring')
    panels=[(b,d) for d in domains for b in sorted(data.base.unique())
            if (data.base.eq(b)&data.domain.eq(d)).any()]
    fig,axes=(cp._subplots(title,len(panels)) if len(panels)<=2 else db._grid(title))
    tables=[]
    for ax in axes[len(panels):]:
        ax.remove()
    for ax,(base,domain) in zip(axes,panels):
        part=data[data.base.eq(base)&data.domain.eq(domain)]
        expected=set(part.loc[part.updates.eq(0),'dataset'])
        if not expected:
            ax.remove();continue
        for arm in arms:
            arm_data=part[part.arm.eq(arm)]
            curve=arm_data.groupby('updates').effect.agg(mean='mean',datasets='size')
            support=arm_data.groupby('updates').dataset.agg(set)
            curve.loc[~support.map(lambda s:s==expected),'mean']=np.nan
            curve=curve.reindex(list(arms[arm].cfg.train.trajectory_steps))
            _line(ax,curve.index,curve['mean'],arm)
            tables.append(curve.reset_index().assign(arm=arm,base=base,domain=domain))
        ax.set_title(base+'\n'+cmp.DOMAINS[domain]);ax.set_xlabel('Successful updates');ax.set_ylabel('AUC change')
        ax.axhline(0,linestyle=':',color=style.color('reference'))
    if not tables:
        return []
    db._legend(fig,next(ax for ax in axes if ax in fig.axes))
    return [cp.Page('size_monitors',fig,
        'Fixed-context monitoring effects relative to update zero, with equal weight per table. All arms must be '
        'observed for the same table; non-credit tables additionally require all four partitions. The baseline table '
        'cohort is held fixed across milestones. These monitors are distinct from the final five-fold benchmark.',
        {'Matched table effects':data,'Plotted means and counts':pd.concat(tables,ignore_index=True)})]


def plot_benchmark(arms, metric='roc_auc'):
    data=matched_observations(arms,benchmark=True,metric=metric)
    if data.empty:
        return []
    reference='All tables' if 'All tables' in arms else '8,192 rows'
    differences=contrasts(data,reference)
    scale=style.EFFECT_SCALE.get(metric,1.)
    study=6 if 'All tables' in arms else 7
    fig,axes=db._grid(f'PD: experiment {study}, CPT − original {"AUC" if metric=="roc_auc" else "Brier score"}')
    stats=[]
    for ax,(domain,diff) in zip(axes,[(d,c) for d in ('credit','ood') for c in (False,True)]):
        part=(differences if diff else data)
        part=part[part.domain.eq(domain)]
        if part.empty:
            ax.remove();continue
        value='difference' if diff else 'effect'
        labels=[label for label in arms if not diff or label!=reference]
        for i,base in enumerate(sorted(part.base.unique())):
            for j,arm in enumerate(labels):
                group=part[part.base.eq(base)&part.arm.eq(arm)]
                if group.empty:continue
                x=i+(j-(len(labels)-1)/2)*style.JITTER_WIDTH
                ax.scatter(np.full(len(group),x),group[value]*scale,s=style.POINT_SIZE/2,
                           color=style.SIZE_STUDY_COLORS[arm],alpha=style.POINT_ALPHA)
                ax.scatter([x],[group[value].mean()*scale],marker='D',s=style.POINT_SIZE,
                           color=style.SIZE_STUDY_COLORS[arm],label=arm if i==0 else None)
                stats.append(dict(domain=domain,base=base,arm=arm,contrast=diff,mean=group[value].mean(),
                                  improved=int(group[value].gt(0).sum()),datasets=len(group)))
        ax.set_xticks(range(part.base.nunique()),sorted(part.base.unique()))
        ax.set_title(cmp.DOMAINS[domain]+'\n'+('Contrast with '+reference if diff else 'Effect versus original weights'))
        ax.set_ylabel(style.EFFECT_LABELS.get(metric,effect_label(metric)) if not diff else 'Difference in effect (+ favours arm)');ax.axhline(0,linestyle=':',color=style.color('reference'))
    db._legend(fig,next(ax for ax in axes if ax in fig.axes))
    return [cp.Page(f'size_benchmark_{metric}',fig,
        (f'Experiment {study}, PD final five-fold benchmark ({", ".join(sorted(data.base.unique()))}; full updates, LR 3e-7, L2-SP 0.003, '
         'proportional class sampling). '
         + ('Arms: CPT on all training tables versus only tables with more than 10,000 rows, both with 8,192-row episodes. '
            if study == 6 else
            'Arms: ' + ', '.join(arms) + ' on the same large-table corpus; the 8,192-row arm is the '
            'experiment-6 large-table checkpoint, reused rather than retrained. ')
         + f'Left panels: paired change in {style.METRIC_LABELS.get(metric, metric)} ({style.EFFECT_LABELS.get(metric, effect_label(metric))}) '
         'relative to the original checkpoint of the same base; right panels: difference between each arm and the '
         f'{reference} arm within table (same scale), positive favouring the named arm. Upper panels use held-out credit '
         'tables; lower panels the four non-credit retention tables, each first averaged over the four matched CPT partitions. '
         'Points are tables and diamonds equal-table means. Every arm and its original-model control has five finite outer '
         'folds on the same table. Tables from related sources are not independent.'), {'Paired effects':data,'Within-table contrasts':differences,
                                            'Plotted summaries':pd.DataFrame(stats)})]


def cost_table(arms):
    pieces=[]
    for label,run in arms.items():
        data=cp.diagnostics(run)
        fields=[c for c in ('trial_name','base','partition','status','total_optimizer_steps','rows_seen',
                            'gpu_hours','sec_per_step','peak_gpu_gb','training_seconds','compute_seconds') if c in data]
        pieces.append(data[fields].assign(arm=label))
    return pd.concat(pieces,ignore_index=True)
