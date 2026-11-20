"""Matched sequential validation-only tests of concept and exact edit witnesses."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import gc
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch.utils.data import DataLoader

import run_graph_feature_experiment as experiment
from graph_group_loss import ProbabilityFloorLoss
from run_gine_curriculum import make_schedule
from nyt_connections_solver.data_loader import make_state_loaders,collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from nyt_connections_solver.harness import SearchHarness
from nyt_connections_solver.feedback_search import FeedbackSearchHarness,ModelGroupScorer
from nyt_connections_solver.evaluation.search import Puzzle
from nyt_connections_solver.evaluation.feedback import evaluate_feedback_games


VARIANTS=('control','concept','edit_targets')
COLORS=('purple','blue','green','yellow')


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def fresh_expanded_model(base_config,params,idx,seed,edge_dim,active_channels,candidate_channels,*,coherence_scale=1.,edge_summary_width=0):
    """Fresh trainable projections; zero new input columns, matched base weights."""
    experiment.seed_everything(seed)
    original_config=deepcopy(base_config)
    original_config.update(edge_hidden=params['hidden_width'],gine_layers=params['gine_layers'],
        readout_width=params['readout_width'],gine_dropout=params['dropout'],edge_bias_scale=0.,
        train_text_projection=True,train_input_norm=True,edge_coherence_scale=coherence_scale)
    original=StandaloneCandidateGINE(group_idx=idx,**original_config)
    old_dim=original_config['edge_dim']
    config={**original_config,'edge_dim':edge_dim,'residual_edge_channels':list(active_channels),
            'candidate_edge_channels':list(candidate_channels)}
    if edge_summary_width:config['edge_summary_width']=edge_summary_width
    with torch.random.fork_rng():model=StandaloneCandidateGINE(group_idx=idx,**config)
    target=model.state_dict()
    for name,value in original.state_dict().items():
        if target[name].shape==value.shape:target[name].copy_(value)
        elif name=='group_readout.0.weight' and edge_summary_width:
            target[name].zero_();target[name][...,:value.shape[-1]].copy_(value)
        elif 'edge_projection' in name and value.ndim==2 and value.shape[-1]==old_dim and target[name].shape[-1]==edge_dim:
            target[name].zero_();target[name][...,:old_dim].copy_(value)
        else:raise ValueError(f'Unexpected shape change: {name}')
    model.load_state_dict(target,strict=True)
    return model,config


def rank_and_play(model,config,nodes,edges,metadata,reference,frame,artifacts,device):
    """Independently reload saved parameters, rank gold groups, then play games."""
    reloaded=StandaloneCandidateGINE(group_idx=artifacts[4]['idx'],**config)
    reloaded.load_state_dict(model.state_dict(),strict=True)
    harness=SearchHarness.from_model(reloaded,None,input_dim=nodes.dim,device=device,beam_width=20,
        node_feature_builder=nodes,edge_feature_builder=edges.encode)
    scorer=ModelGroupScorer(harness)
    puzzles=[];ranks=[]
    for date in reference['split_dates']['val']:
        groups=tuple(tuple(group) for group in frame.loc[frame.date==date,'connections'])
        words=tuple(w for group in groups for w in group)
        scorer.prepare(words)
        candidates=scorer.score_states([words])[0]
        ordered=sorted(candidates,key=lambda c:c.log_probability,reverse=True)
        lookup={frozenset(c.words):i for i,c in enumerate(ordered,1)}
        ranks.append(dict(date=date,gold_ranks=[lookup[frozenset(g)] for g in groups]))
        puzzles.append(Puzzle(date,words,frozenset(map(frozenset,groups)),groups))
    policy=FeedbackSearchHarness.from_harness(harness,beam_width=20,capture_trace=False,score_mode='full_board')
    feedback=evaluate_feedback_games(policy,puzzles,max_mistakes=4,seed=0)
    direct=Counter();forced=Counter()
    for game,puzzle in zip(feedback['games'],puzzles):
        colors={frozenset(group):color for group,color in zip(puzzle.ordered_groups,COLORS)}
        remaining=16;history=[]
        for row in game['history']:
            item=asdict(row);item['outcome']=row.outcome.value
            color=colors.get(frozenset(row.guess))
            item['color']=color;item['forced']=remaining==4
            if row.outcome.value=='correct':
                (forced if remaining==4 else direct)[color]+=1
                remaining-=4
            history.append(item)
        game['history']=history
    return dict(split='val',gold_ranks=ranks,harness=feedback,
                direct_colors=dict(direct),forced_colors=dict(forced))


def summarize(root,manifest):
    records=[]
    for seed in manifest['seeds']:
        for variant in VARIANTS:
            path=root/f'{variant}-seed-{seed}.json'
            if path.exists():
                payload=json.loads(path.read_text())
                if payload.get('status')=='complete':records.append(payload)
    aggregate={}
    for variant in VARIANTS:
        runs=[r for r in records if r['variant']==variant]
        if not runs:continue
        audit=[x for r in runs for x in r['evaluation']['gold_ranks']]
        aggregate[variant]=dict(runs=len(runs),appearances=len(audit),
            ranking={s:{metric:mean(r['training']['validation'][s][metric] for r in runs)
                        for metric in ('hit1','mrr','hit4')} for s in ('s4','s3','s2')},
            solves=sum(r['evaluation']['harness']['solved'] for r in runs),
            first_correct=sum(g['first_guess_correct'] for r in runs for g in r['evaluation']['harness']['games']),
            direct_colors={c:sum(r['evaluation']['direct_colors'].get(c,0) for r in runs) for c in COLORS},
            forced_colors={c:sum(r['evaluation']['forced_colors'].get(c,0) for r in runs) for c in COLORS},
            color_top4={c:mean(a['gold_ranks'][i]<=4 for a in audit) for i,c in enumerate(COLORS)},
            best_updates=[r['training']['best_update'] for r in runs])
    paired={}
    for variant in VARIANTS[1:]:
        gains=losses=0
        for seed in manifest['seeds']:
            control=next((r for r in records if r['variant']=='control' and r['seed']==seed),None)
            other=next((r for r in records if r['variant']==variant and r['seed']==seed),None)
            if control and other:
                a={g['date']:g for g in control['evaluation']['harness']['games']}
                b={g['date']:g for g in other['evaluation']['harness']['games']}
                gains+=sum(b[d]['won'] and not a[d]['won'] for d in a)
                losses+=sum(a[d]['won'] and not b[d]['won'] for d in a)
        paired[variant]=dict(solve_gains=gains,solve_losses=losses)
    summary=dict(manifest=manifest,completed_runs=len(records),aggregate=aggregate,paired=paired)
    write(root/'summary.json',summary)
    lines=['# Shared concept and letter-edit witnesses','',
           f"Completed {len(records)}/{len(manifest['seeds'])*3} runs. Validation only; no test evaluation.",'',
           'Three matched arms: original core edges, added shared concept, or added exact edit targets. '
           'No Wiktionary, phonetic, or semantic-edit additions. Every arm has the same 26-channel '
           'parameter shape, zero-initialized added columns, matched fresh base parameters, schedule '
           'and batches. Unused channels are masked. Frozen Qwen vectors; all active projections, '
           'LayerNorm, GINE and readout trainable. No learned reference parameters are loaded. '
           'Candidate-specific witness values override only their corresponding channels.', '',
           f"Budget {manifest['updates']:,} updates per run, balanced s2/s3/s4, batch 16, evaluation "
           f"every {manifest['eval_every']}, no pruning/early stopping/scheduler. Checkpoint selection "
           'by validation s4 Hit@1, MRR for ties. Existing probability-floor loss (.1/.1) and '
           f"coherence scale {manifest['edge_coherence_scale']:g} held constant. Fixed tuned architecture/AdamW settings; no retuning. "
           'The ongoing CV sweep shares GPU resources, so elapsed times are not benchmark comparisons.','',
           '| Arm | Runs | s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 | Solves | Direct Purple | Forced Purple |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant,a in aggregate.items():
        lines.append(f"| {variant} | {a['runs']} | {a['ranking']['s4']['hit1']:.4f} | {a['ranking']['s4']['mrr']:.4f} | {a['ranking']['s4']['hit4']:.4f} | {a['ranking']['s3']['hit1']:.4f} | {a['ranking']['s2']['hit1']:.4f} | {a['solves']}/{a['appearances']} | {a['direct_colors']['purple']} | {a['forced_colors']['purple']} |")
    lines+=['','Repeated seeds reuse the same 121 validation puzzles; appearances are correlated. '
            'Mechanism annotations are model-reviewed diagnostic labels, not training inputs. '
            'Full histories, group ranks, checkpoint paths, and training curves are retained in '
            'the per-run JSON files. Partial-run aggregates are not final comparisons.','']
    if manifest.get('edge_summary_width'):
        lines += [f"Direct edge-summary branch: width {manifest['edge_summary_width']}; per-channel mean, "
                  'minimum, and weakest-word average on raw candidate edges after exact-witness overrides. '
                  'Small two-linear-layer MLP, concatenated into the GINE readout; added readout columns '
                  'initialize zero. No edge amplification or gate. See edge-summary-comparison.md for '
                  'the matched regular-GINE comparison.','']
    (root/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/gine-relation-witnesses'))
    parser.add_argument('--updates',type=int,default=30000)
    parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--seeds',type=int,nargs='+',default=[10,11,12])
    parser.add_argument('--coherence-scale',type=float,default=1.)
    parser.add_argument('--edge-summary-width',type=int,default=0)
    args=parser.parse_args()
    if not math.isfinite(args.coherence_scale) or args.coherence_scale<0:raise ValueError('Coherence scale must be finite and nonnegative')
    if args.edge_summary_width<0:raise ValueError('Edge summary width must be nonnegative')
    if args.updates<15000 or args.updates%6 or args.updates%args.eval_every:raise ValueError('Use >=15k balanced updates divisible by evaluation interval')
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for full matched runs')
    device=torch.device('cuda')
    reference_path=Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    params=json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    manifest=dict(reference_metadata_only=str(reference_path),params=params,seeds=args.seeds,updates=args.updates,
                  eval_every=args.eval_every,variants=list(VARIANTS),split='val',batch_size=16,
                  initialization='fresh; zero padded added columns; active text projection and norm trainable',
                  edge_coherence_scale=args.coherence_scale,floor_weight=.1,floor_beta=.1,
                  harness=dict(beam_width=20,score_mode='full_board',max_mistakes=4,shuffle_seed=0))
    if args.edge_summary_width:manifest['edge_summary_width']=args.edge_summary_width
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:
        raise ValueError('Manifest differs; use a separate output directory')
    write(root/'manifest.json',manifest);write(root/'status.json',dict(status='preparing'))
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):return reference['model_config']['in_dim']
        def encode(self,*args,**kwargs):raise RuntimeError('Frozen embedding missing from existing cache')
    encoder=CacheOnly()
    options={**reference['feature_metadata']['options'],'use_spelling':False,
             'include_concept_witnesses':True,'include_edit_targets':True}
    nodes,edges,metadata=build_graph_features(encoder,reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph',**options)
    write(root/'feature-metadata.json',dict(metadata=metadata,names=list(edges.feature_names)))
    frame=experiment.load_connections(Path(reference['dataset_path']))
    datasets={}
    for split in ('train','val'):
        subset=frame[frame.date.isin(reference['split_dates'][split])]
        assert subset.date.nunique()==len(reference['split_dates'][split])
        datasets[split],_=make_state_loaders(subset,encoder,None,16,shuffle=False,feature_builder=nodes,edge_providers=edges)
        print('PREPARED',split,{s:len(d) for s,d in datasets[split].items()},flush=True)
    artifacts=experiment.make_artifacts()
    val_loaders={s:DataLoader(d,batch_size=16,shuffle=False,collate_fn=collate_keep_wordlists) for s,d in datasets['val'].items()}
    assert edges.dim==26 and list(edges.feature_names[:15])==reference['edge_feature_names']
    # Signal diagnostics: pair maxima versus exact common witnesses on validation.
    idx=artifacts[4]['idx'];gold=artifacts[4]['y'].bool();diag={}
    for offset,name in enumerate(edges.feature_names[15:]):
        positive=negative=0;values=[]
        for board in datasets['val'][4]:
            value=board['candidate_edge_features'][:,offset]
            positive+=int((value[gold]>0).sum());negative+=int((value[~gold]>0).sum())
            values.append(float(value[gold].mean()))
        diag[name]=dict(gold_groups_nonzero=positive,incorrect_quartets_nonzero=negative,mean_gold_value=mean(values))
    write(root/'feature-diagnostics.json',diag)
    for seed in args.seeds:
        schedule=make_schedule('mixed',seed,updates=args.updates)
        for variant in VARIANTS:
            path=root/f'{variant}-seed-{seed}.json'
            if path.exists() and json.loads(path.read_text()).get('status')=='complete':continue
            active=list(range(15)) + ([15] if variant=='concept' else list(range(16,26)) if variant=='edit_targets' else [])
            model,config=fresh_expanded_model(reference['model_config'],params,idx,seed,edges.dim,active,edges.candidate_feature_channels,
                coherence_scale=args.coherence_scale,edge_summary_width=args.edge_summary_width)
            loaders={s:DataLoader(d,batch_size=16,shuffle=True,collate_fn=collate_keep_wordlists,
                     generator=torch.Generator().manual_seed(seed+s)) for s,d in datasets['train'].items()}
            history=[]
            def progress(step,metrics,best,row):
                history.append(row)
                write(root/'progress.json',dict(variant=variant,seed=seed,update=step,best_hit1=best[0],metrics=metrics))
                write(path,dict(status='training',variant=variant,seed=seed,history=history))
            write(root/'status.json',dict(status='training',variant=variant,seed=seed))
            started=perf_counter()
            model,summary=experiment.train(model,loaders,val_loaders,artifacts,device,lr=params['lr'],projection_lr=None,
                weight_decay=params['weight_decay'],updates=args.updates,eval_every=args.eval_every,
                min_updates_before_stop=args.updates,patience=1000,min_delta=1e-3,evaluation_callback=progress,
                state_schedule=schedule,ranking_loss=ProbabilityFloorLoss(weight=.1,beta=.1))
            elapsed=perf_counter()-started
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            weights=Path('weights/experiments')/f'{stamp}-relation-witness-summary-{args.edge_summary_width}-coherence-{args.coherence_scale:g}-{variant}-seed-{seed}.pt'
            weights.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model_class='StandaloneCandidateGINE',model_config=config,model_state_dict=model.state_dict(),
                feature_metadata=metadata,edge_feature_names=list(edges.feature_names),split_dates=reference['split_dates'],
                dataset_path=reference['dataset_path'],embedding_model_name=reference['embedding_model_name'],
                training_config={**manifest,'variant':variant,'seed':seed},training_summary=summary),weights)
            write(root/'status.json',dict(status='evaluating',variant=variant,seed=seed))
            evaluation=rank_and_play(model,config,nodes,edges,metadata,reference,frame,artifacts,device)
            measured=mean(1 in row['gold_ranks'] for row in evaluation['gold_ranks'])
            assert abs(measured-summary['validation']['s4']['hit1'])<1e-7,(measured,summary['validation']['s4']['hit1'])
            write(path,dict(status='complete',variant=variant,seed=seed,checkpoint=str(weights),config=config,
                training=summary,evaluation=evaluation,train_seconds=elapsed))
            summarize(root,manifest)
            print('COMPLETE',variant,seed,summary['validation']['s4']['hit1'],evaluation['harness']['solve_rate'],flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    summarize(root,manifest);write(root/'status.json',dict(status='complete'))


if __name__=='__main__':main()
