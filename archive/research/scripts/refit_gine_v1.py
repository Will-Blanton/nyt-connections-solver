"""Refit previous/CV-selected regular GINE on train+val, then select on test once.

Training budgets come only from already-completed validation learning curves.
All six final checkpoints are sealed before any test features or metrics are used.
"""
import argparse
import csv
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from graph_group_loss import ProbabilityFloorLoss
from run_gine_curriculum import make_schedule
from run_gine_relation_witnesses import COLORS, write
from tune_gine_coherence_cv import build_fresh_model
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from nyt_connections_solver.harness import SearchHarness
from nyt_connections_solver.feedback_search import FeedbackSearchHarness, ModelGroupScorer
from nyt_connections_solver.evaluation.search import Puzzle
from nyt_connections_solver.evaluation.feedback import evaluate_feedback_games


ARMS = ('previous', 'cv')
SEEDS = (10, 11, 12)


def refit_budget(selected_updates, source_train_counts, pool_size):
    """Median prior checkpoint exposure, adjusted to the final training pool."""
    if not selected_updates or len(selected_updates) != len(source_train_counts):
        raise ValueError('Require paired validation-selected updates and training counts')
    if pool_size <= 0 or any(u <= 0 or n <= 0 for u, n in zip(selected_updates, source_train_counts)):
        raise ValueError('Updates and puzzle counts must be positive')
    scaled = [u * pool_size / n for u, n in zip(selected_updates, source_train_counts)]
    return max(6, 6 * round(median(scaled) / 6))


def train_fixed(model, loaders, artifacts, device, *, seed, updates, lr, weight_decay,
                callback=None, log_every=500):
    """Train to the fixed final update; no validation, scheduler, or checkpoint search."""
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    loss_fn = ProbabilityFloorLoss(weight=.1, beta=.1)
    labels = {s: {k: v.to(device) for k, v in a.items()} for s, a in artifacts.items()}
    iterators = {s: iter(loader) for s, loader in loaders.items()}
    schedule = make_schedule('mixed', seed, updates=updates)
    history = []
    running = count = 0
    started = perf_counter()
    for step, size in enumerate(schedule, 1):
        batch = experiment.next_batch(iterators, loaders, size)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = experiment.score_batch(model, batch, labels[size]['idx'], device)
        loss = loss_fn(scores, labels[size]['y'])
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss at {step}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        count += 1
        if step % log_every == 0 or step == updates:
            row = dict(update=step, train_loss=running/count, lr=lr, seconds=perf_counter()-started)
            history.append(row)
            print(f"Update {step}/{updates} | train loss {row['train_loss']:.4f}", flush=True)
            if callback is not None:
                callback(row)
            running = count = 0
            started = perf_counter()
    return model.cpu().eval(), dict(updates_ran=updates, history=history,
        checkpoint_rule='final fixed update; no validation or test selection',
        state_updates=dict(Counter(schedule)), loss='probability_mass_with_floor',
        floor_weight=.1, floor_beta=.1)


def plans(reference):
    pool = sorted(set(reference['split_dates']['train']) | set(reference['split_dates']['val']))
    previous_root = Path('experiments/gine-relation-witnesses-regular')
    previous_manifest = json.loads((previous_root/'manifest.json').read_text())
    previous = [json.loads((previous_root/f'control-seed-{seed}.json').read_text()) for seed in SEEDS]
    cv_root = Path('experiments/gine-coherence-cv')
    best = json.loads((cv_root/'best-baseline.json').read_text())
    folds = json.loads((cv_root/'manifest.json').read_text())['folds']
    cv = [json.loads(Path(p).read_text()) for p in best['fold_results']]
    assert all(r['status'] == 'complete' for r in previous + cv)
    assert previous_manifest['edge_coherence_scale'] == 0
    result = {}
    for name, rows, counts, params in (
        ('previous', previous, [len(reference['split_dates']['train'])]*len(previous), previous_manifest['params']),
        ('cv', cv, [len(folds[r['fold']]['train']) for r in cv], best['params']),
    ):
        selected = [r['training']['best_update'] for r in rows]
        result[name] = dict(params=params, selected_updates=selected, source_train_counts=counts,
            updates=refit_budget(selected, counts, len(pool)))
    return result, pool


def fresh_model(reference, arm, params, idx, seed):
    if arm == 'cv':
        return build_fresh_model(reference['model_config'], {**params, 'edge_coherence_scale':0.}, idx, seed)
    experiment.seed_everything(seed)
    config = deepcopy(reference['model_config'])
    config.update(edge_hidden=params['hidden_width'], gine_layers=params['gine_layers'],
        readout_width=params['readout_width'], gine_dropout=params['dropout'], edge_bias_scale=0.,
        train_text_projection=True, train_input_norm=True, edge_coherence_scale=0.)
    return StandaloneCandidateGINE(group_idx=idx, **config), config


def evaluate_saved(path, nodes, edges, frame, test_dates, loaders, artifacts, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = StandaloneCandidateGINE(group_idx=artifacts[4]['idx'], **payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    ranking = experiment.evaluate(model.to(device), loaders, artifacts, device, by_color=True)
    model.cpu()
    harness = SearchHarness.from_model(model, None, input_dim=nodes.dim, device=device,
        beam_width=20, node_feature_builder=nodes, edge_feature_builder=edges.encode)
    scorer = ModelGroupScorer(harness)
    puzzles, gold_ranks = [], []
    for date in test_dates:
        groups = tuple(tuple(g) for g in frame.loc[frame.date == date, 'connections'])
        words = tuple(w for g in groups for w in g)
        scorer.prepare(words)
        candidates = sorted(scorer.score_states([words])[0], key=lambda c:c.log_probability, reverse=True)
        lookup = {frozenset(c.words):rank for rank,c in enumerate(candidates,1)}
        gold_ranks.append(dict(date=date, ranks=[lookup[frozenset(g)] for g in groups]))
        puzzles.append(Puzzle(date, words, frozenset(map(frozenset, groups)), groups))
    hit1 = mean(1 in r['ranks'] for r in gold_ranks)
    assert abs(hit1-ranking[4]['hit1']) < 1e-7
    policy = FeedbackSearchHarness.from_harness(harness, beam_width=20, capture_trace=False, score_mode='full_board')
    feedback = evaluate_feedback_games(policy, puzzles, max_mistakes=4, seed=0)
    direct, forced = Counter(), Counter()
    for game, puzzle in zip(feedback['games'], puzzles):
        colors = {frozenset(g):c for g,c in zip(puzzle.ordered_groups, COLORS)}
        remaining, history = 16, []
        for row in game['history']:
            item = asdict(row)
            item['outcome'] = row.outcome.value
            item['color'] = colors.get(frozenset(row.guess))
            item['forced'] = remaining == 4
            if row.outcome.value == 'correct':
                (forced if remaining == 4 else direct)[item['color']] += 1
                remaining -= 4
            history.append(item)
        game['history'] = history
    return dict(ranking={f's{s}':r for s,r in ranking.items()}, gold_ranks=gold_ranks,
                harness=feedback, direct_colors=dict(direct), forced_colors=dict(forced))


def summarize(root, manifest):
    rows = [json.loads((root/f'{arm}-seed-{seed}.json').read_text()) for arm in ARMS for seed in SEEDS]
    assert all(r['status'] == 'complete' for r in rows)
    aggregate = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        ev = [r['evaluation'] for r in selected]
        aggregate[arm] = dict(
            appearances=sum(e['harness']['puzzles'] for e in ev),
            solves=sum(e['harness']['solved'] for e in ev),
            first_correct=sum(g['first_guess_correct'] for e in ev for g in e['harness']['games']),
            mean_groups=mean(e['harness']['mean_correct_groups'] for e in ev),
            ranking={s:{m:mean(e['ranking'][s][m] for e in ev) for m in ('hit1','mrr','hit4')}
                     for s in ('s4','s3','s2')},
            direct_colors={c:sum(e['direct_colors'].get(c,0) for e in ev) for c in COLORS},
            forced_colors={c:sum(e['forced_colors'].get(c,0) for e in ev) for c in COLORS})
    paired = []
    for seed in SEEDS:
        a,b = ({g['date']:g['won'] for g in next(r for r in rows if r['arm']==arm and r['seed']==seed)['evaluation']['harness']['games']}
               for arm in ARMS)
        assert a.keys() == b.keys()
        paired.append(dict(seed=seed, cv_gains=sum(b[d] and not a[d] for d in a), cv_losses=sum(a[d] and not b[d] for d in a)))
    winner = max(ARMS, key=lambda arm:(aggregate[arm]['solves'],aggregate[arm]['first_correct'],
                                      aggregate[arm]['ranking']['s4']['hit1'],arm=='cv'))
    primary = next(r['checkpoint'] for r in rows if r['arm']==winner and r['seed']==10)
    selection = dict(winner=winner, primary_seed=10, checkpoint=primary,
                     criterion=manifest['selection'], test_used_for_setup_selection=True)
    write(root/'selection.json', selection)
    write(root/'summary.json', dict(aggregate=aggregate, paired=paired, selection=selection))
    curves=[dict(arm=r['arm'],seed=str(r['seed']),**h) for r in rows for h in r['training']['history']]
    with (root/'training-curves.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    import altair as alt
    import pandas as pd
    alt.Chart(pd.DataFrame(curves)).mark_line().encode(x='update:Q',y='train_loss:Q',color='arm:N',
        tooltip=['arm','seed','update','train_loss']).properties(width=330,height=200).facet(
        column='seed:N').interactive().save(root/'training-curves.html')
    lines = ['# Train+validation refit: previous versus CV-selected regular GINE', '',
        'Both setups initialize all active learned layers from scratch on the original train+validation pool '
        '(1,089 puzzles). Only original cached Qwen vectors stay frozen. Core 15 edge features only; '
        'coherence and direct edge-summary branch disabled. The previous setup omits its earlier '
        '11 unused, zero-padded/masked edge columns; all 15 active channels and learned architecture '
        'are preserved. Three fixed seeds: 10, 11, 12.', '',
        'No validation set or scheduler is used during refitting. The fixed final update is selected '
        'before test evaluation: median prior validation-selected checkpoint exposure, scaled by '
        'final/source training puzzle counts and rounded to a multiple of six. Every state size '
        'gets equal updates, batch 16. Constant AdamW learning rate and original floor loss (.1/.1).', '',
        '| Setup | Source selected updates | Fixed refit updates |', '|---|---|---:|']
    for arm in ARMS:
        p=manifest['plans'][arm]
        lines.append(f"| {arm} | {p['selected_updates']} | {p['updates']} |")
    lines += ['', '| Parameter | Previous | CV |', '|---|---:|---:|']
    for key in ('hidden_width','gine_layers','readout_width','projection_width','dropout','lr','weight_decay'):
        lines.append(f"| {key} | {manifest['plans']['previous']['params'].get(key,'legacy text MLP')} | "
                     f"{manifest['plans']['cv']['params'].get(key,'legacy text MLP')} |")
    lines += ['', 'All six checkpoints were saved and SHA-256 sealed before test feature preparation '
        'or evaluation. Test chooses only between the two frozen setups; it does not tune their '
        'individual hyperparameters, update budgets, or seeds. The original test split has been inspected in '
        'earlier experiments, so this comparison is not a pristine final holdout. Future live puzzles '
        'remain the prospective evaluation.', '',
        'Harness: beam 20, original full-board scores, four mistakes, shuffle seed 0. Reduced-state '
        'ranking uses oracle remaining groups. Three seeds repeat the same 121 test puzzles; '
        '363 appearances are correlated.', '',
        '| Metric | Previous | CV |', '|---|---:|---:|']
    for s in ('s4','s3','s2'):
        for m in ('hit1','mrr','hit4'):
            lines.append(f"| {s} {m} | {aggregate['previous']['ranking'][s][m]:.4f} | {aggregate['cv']['ranking'][s][m]:.4f} |")
    for m in ('solves','first_correct','mean_groups'):
        lines.append(f"| {m} | {aggregate['previous'][m]} | {aggregate['cv'][m]} |")
    lines += ['', '| Seed | Previous Hit@1 | CV Hit@1 | Previous solves | CV solves | CV gains / losses |',
              '|---|---:|---:|---:|---:|---:|']
    for seed,p in zip(SEEDS,paired):
        a,b=[next(r['evaluation'] for r in rows if r['arm']==arm and r['seed']==seed) for arm in ARMS]
        lines.append(f"| {seed} | {a['ranking']['s4']['hit1']:.2%} | {b['ranking']['s4']['hit1']:.2%} | "
                     f"{a['harness']['solved']} | {b['harness']['solved']} | {p['cv_gains']} / {p['cv_losses']} |")
    lines += ['', '| Color | Previous direct / forced | CV direct / forced |', '|---|---:|---:|']
    for c in COLORS:
        a,b=[aggregate[arm] for arm in ARMS]
        lines.append(f"| {c} | {a['direct_colors'][c]} / {a['forced_colors'][c]} | {b['direct_colors'][c]} / {b['forced_colors'][c]} |")
    lines += ['', f"Selected setup: **{winner}** by the predeclared solve-count, first-guess, Hit@1 "
              'tie-break rule. Deployment checkpoint uses predeclared seed 10, not the best test seed.', '',
              f"Checkpoint: `{primary}`", '', '## Primary seed-10 checkpoint', '']
    primary_eval=next(r['evaluation'] for r in rows if r['arm']==winner and r['seed']==10)
    lines += [f"Hit@1 {primary_eval['ranking']['s4']['hit1']:.2%}; "
              f"{primary_eval['harness']['solved']}/121 solves; "
              f"first-guess accuracy {primary_eval['harness']['first_guess_accuracy']:.2%}.", '',
              'Complete histories, independent group ranks, loss curves, configuration and budget '
              'provenance are retained. Checkpoints truthfully list all 1,089 fitted dates as training '
              'and no validation dates; the original split is separately preserved. Test was not used '
              'for training or scheduling. Notebooks are unchanged.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(aggregate=aggregate, selection=selection, paired=paired),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/gine-v1-refit'))
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    root=args.output
    if args.summarize_only:
        summarize(root,json.loads((root/'manifest.json').read_text()))
        return
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for refitting')
    device=torch.device('cuda')
    reference_path=Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    reference.pop('model_state_dict')
    plan,pool=plans(reference)
    test_dates=reference['split_dates']['test']
    assert not set(pool)&set(test_dates)
    manifest=dict(version=1,reference_metadata_only=str(reference_path),plans=plan,seeds=list(SEEDS),
        train_dates=pool,test_dates=test_dates,primary_seed=10,batch_size=16,
        budget_rule='median(selected_updates * final_train_puzzles/source_train_puzzles), nearest multiple of six',
        selection='aggregate test solves; then first guesses correct; then s4 Hit@1; final tie CV',
        harness=dict(beam_width=20,score_mode='full_board',max_mistakes=4,shuffle_seed=0))
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:
        raise ValueError('Manifest changed; use a new output directory')
    write(root/'manifest.json',manifest)
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):return reference['model_config']['in_dim']
        def encode(self,*a,**kw):raise RuntimeError('Missing cached original Qwen embedding')
    encoder=CacheOnly()
    nodes,edges,metadata=build_graph_features(encoder,reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph',**{**reference['feature_metadata']['options'],
        'use_spelling':False,'include_phonetics':False,'include_letter_edits':False,
        'phonetic_nodes':False,'transformation_nodes':False})
    assert list(edges.feature_names)==reference['edge_feature_names'] and edges.dim==15
    frame=experiment.load_connections(Path(reference['dataset_path']))
    subset=frame[frame.date.isin(pool)]
    assert subset.date.nunique()==len(pool)
    datasets,_=make_state_loaders(subset,encoder,None,16,shuffle=False,feature_builder=nodes,edge_providers=edges)
    artifacts=experiment.make_artifacts()
    print('PLANS',json.dumps(plan),flush=True)
    print('PREPARED TRAIN',len(pool),{s:len(d) for s,d in datasets.items()},flush=True)
    final_dates=dict(train=pool,val=[],test=test_dates)
    for seed in SEEDS:
        for arm in ARMS:
            path=root/f'{arm}-seed-{seed}.json'
            if path.exists() and json.loads(path.read_text())['status'] in ('trained','complete'):continue
            write(root/'status.json',dict(status='training',arm=arm,seed=seed))
            p=plan[arm]
            model,config=fresh_model(reference,arm,p['params'],artifacts[4]['idx'],seed)
            loaders={s:DataLoader(d,batch_size=16,shuffle=True,collate_fn=collate_keep_wordlists,
                      generator=torch.Generator().manual_seed(seed+s)) for s,d in datasets.items()}
            def progress(row):write(root/'progress.json',dict(arm=arm,seed=seed,**row))
            print('START',arm,seed,p['updates'],flush=True)
            model,training=train_fixed(model,loaders,artifacts,device,seed=seed,updates=p['updates'],
                lr=p['params']['lr'],weight_decay=p['params']['weight_decay'],callback=progress)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights=Path('weights/experiments')/f'{stamp}-v1-refit-{arm}-seed-{seed}.pt'
            torch.save(dict(model_class='StandaloneCandidateGINE',model_config=config,model_state_dict=model.state_dict(),
                feature_metadata=metadata,edge_feature_names=list(edges.feature_names),split_dates=final_dates,
                original_split_dates=reference['split_dates'],dataset_path=reference['dataset_path'],
                embedding_model_name=reference['embedding_model_name'],training_config={**manifest,'arm':arm,'seed':seed},
                training_summary=training),weights)
            write(path,dict(status='trained',arm=arm,seed=seed,checkpoint=str(weights),config=config,training=training))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    sealed=[]
    for seed in SEEDS:
        for arm in ARMS:
            r=json.loads((root/f'{arm}-seed-{seed}.json').read_text())
            path=Path(r['checkpoint'])
            sealed.append(dict(arm=arm,seed=seed,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if (root/'sealed-checkpoints.json').exists():
        assert json.loads((root/'sealed-checkpoints.json').read_text())==sealed
    write(root/'sealed-checkpoints.json',sealed)
    # Test is first prepared here, after every final model is immutable.
    subset=frame[frame.date.isin(test_dates)]
    assert subset.date.nunique()==len(test_dates)
    test,_=make_state_loaders(subset,encoder,None,16,shuffle=False,feature_builder=nodes,edge_providers=edges)
    loaders={s:DataLoader(d,batch_size=16,shuffle=False,collate_fn=collate_keep_wordlists) for s,d in test.items()}
    for seed in SEEDS:
        for arm in ARMS:
            path=root/f'{arm}-seed-{seed}.json'
            r=json.loads(path.read_text())
            if r['status']=='complete':continue
            write(root/'status.json',dict(status='test_evaluation',arm=arm,seed=seed))
            print('TEST',arm,seed,flush=True)
            r['evaluation']=evaluate_saved(r['checkpoint'],nodes,edges,frame,test_dates,loaders,artifacts,device)
            r['status']='complete'
            write(path,r)
            print('COMPLETE',arm,seed,r['evaluation']['harness']['solved'],flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    summarize(root,manifest)
    write(root/'status.json',dict(status='complete'))


if __name__=='__main__':main()
