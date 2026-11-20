"""Sequential validation-only transformation edge/node ablations, 30k updates."""
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import run_graph_feature_experiment as experiment
from tune_gine import build_model
from run_gine_curriculum import make_schedule
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.transformation_models import TransformationGINE

ROOT=Path('experiments/gine-transformations')
VARIANTS={'control':('text',False),'edges-text':('text',True),
          'edges-fused':('fused',True),'edges-constant':('constant',True)}


def build(reference,params,idx,seed,dim,variant):
    base, config=build_model(reference,params,idx,seed)
    mode, edges=VARIANTS[variant]
    config.update(in_dim=dim,text_dim=reference['model_config']['in_dim'],
        base_edge_dim=reference['model_config']['edge_dim'],max_transformations=8,
        transformation_width=64,learned_edge_dim=16,node_mode=mode,learned_edges=edges)
    experiment.seed_everything(seed)
    model=TransformationGINE(group_idx=idx,**config)
    weights={n:v for n,v in base.state_dict().items() if not n.startswith('edge_encoder.edge_projections.')}
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if unexpected: raise ValueError(unexpected)
    for target,source in zip(model.edge_encoder.edge_projections,base.edge_encoder.edge_projections):
        with torch.no_grad(): target.weight[:,:source.in_features].copy_(source.weight)
    del base
    return model,config


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    reference_path=Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    params=json.loads(Path('experiments/gine-tuning/best.json').read_text())['params']
    dates=reference['split_dates']
    manifest=dict(reference=str(reference_path),params=params,seeds=[10,11,12],variants=list(VARIANTS),
        updates=30000,eval_every=500,batch_size=16,cpu_threads=4,split='val',loss='probability_mass',
        early_stopping=False,scheduler=False,floor_penalty=False,
        max_transformations=8,min_length=3,transformation_width=64,learned_edge_dim=16,
        position_layout='fixed32 left/right offsets plus normalized span and retained fraction',
        feature_set='original 15 scalar edges plus learned transformation pair vectors',
        pair_pool='masked elementwise max', node_pool='sigmoid-gated mean plus max, parallel text fusion',
        selection='s4 Hit@1 with MRR tie-break, full 30000 updates',
        training='matched initialization, data permutations and balanced 10000 updates per state size')
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest: raise ValueError('Manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Full experiment requires GPU')
    frame=experiment.load_connections(Path(reference['dataset_path']))
    options={**reference['feature_metadata']['options'],'use_spelling':False,
             'transformation_nodes':True,'max_transformations':8,'transformation_min_length':3}
    embedder=SentenceTransformer(reference['embedding_model_name'],revision=options['revision'],device=str(device))
    nodes,edges,metadata=build_graph_features(embedder,reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph',**options)
    datasets={}
    old_threads=torch.get_num_threads(); torch.set_num_threads(4)
    for split in ('train','val'):
        subset=frame[frame.date.isin(dates[split])]
        assert subset.date.nunique()==len(dates[split])
        started=perf_counter()
        datasets[split],_=make_state_loaders(subset,embedder,None,16,shuffle=False,
            feature_builder=nodes,edge_providers=edges,cache=edges.cache)
        words=sorted({w for d in datasets[split][4] for w in d['word_list']})
        counts=[len(nodes.alternatives(w,truncate=False)) for w in words]
        coverage=dict(words=len(words),mean_alternatives=sum(counts)/len(counts),
            with_proper_transformations=sum(c>1 for c in counts),truncated_words=sum(c>8 for c in counts),
            max_alternatives=max(counts),histogram=dict(Counter(counts)),seconds=perf_counter()-started)
        (ROOT/f'coverage-{split}.json').write_text(json.dumps(coverage,indent=2)+'\n')
        print('PREPARED',split,{s:len(d) for s,d in datasets[split].items()},coverage,flush=True)
    torch.set_num_threads(4)
    embedder.to('cpu'); torch.cuda.empty_cache()
    val_loaders={s:DataLoader(d,batch_size=16,shuffle=False,collate_fn=collate_keep_wordlists) for s,d in datasets['val'].items()}
    artifacts=experiment.make_artifacts()
    assert list(edges.feature_names)==reference['edge_feature_names']
    runs=[]
    for seed in manifest['seeds']:
        for variant in VARIANTS:
            path=ROOT/f'{variant}-seed-{seed}.json'
            payload=json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status')!='complete':
                model,config=build(reference,params,artifacts[4]['idx'],seed,nodes.dim,variant)
                schedule=make_schedule('mixed',seed,updates=30000)
                loaders={s:DataLoader(d,batch_size=16,shuffle=True,collate_fn=collate_keep_wordlists,
                    generator=torch.Generator().manual_seed(seed+s)) for s,d in datasets['train'].items()}
                counts=dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
                print('START',variant,seed,'PARAMETERS',counts,flush=True)
                def progress(step,metrics,best,row):
                    (ROOT/'progress.json').write_text(json.dumps(dict(seed=seed,variant=variant,update=step,
                        best_hit1=best[0],metrics=metrics),indent=2)+'\n')
                started=perf_counter()
                model,summary=experiment.train(model,loaders,val_loaders,artifacts,device,
                    lr=params['lr'],projection_lr=None,weight_decay=params['weight_decay'],
                    updates=30000,eval_every=500,min_updates_before_stop=30000,patience=1000,
                    min_delta=1e-3,evaluation_callback=progress,state_schedule=schedule)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights=Path('weights/experiments')/f'{stamp}-transformations-{variant}-seed-{seed}.pt'
                checkpoint=dict(model_class='TransformationGINE',model_config=config,model_state_dict=model.state_dict(),
                    feature_metadata=metadata,edge_feature_names=list(edges.feature_names),split_dates=dates,
                    dataset_path=reference['dataset_path'],embedding_model_name=reference['embedding_model_name'],
                    training_config=dict(variant=variant,seed=seed,params=params,updates=30000,
                        reference_checkpoint=str(reference_path),state_schedule=schedule,ranking_loss='probability_mass'),
                    training_summary=summary)
                torch.save(checkpoint,weights)
                payload=dict(status='complete',variant=variant,seed=seed,checkpoint=str(weights),
                    training=summary,model_config=config,parameters=counts,training_seconds=perf_counter()-started)
                path.write_text(json.dumps(payload,indent=2)+'\n')
                del model,loaders; gc.collect(); torch.cuda.empty_cache()
            harness=ROOT/f'harness-{variant}-seed-{seed}-val.json'
            if not harness.exists():
                with harness.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable,'scripts/evaluate_graph_feedback.py',payload['checkpoint'],
                        '--split','val','--beam-width','20','--score-mode','full_board','--output',str(harness)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            runs.append(dict(variant=variant,seed=seed,result=str(path),harness=str(harness)))
            (ROOT/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            print('COMPLETE',variant,seed,payload['training']['validation']['s4']['hit1'],flush=True)
    print('ALL RUNS COMPLETE',flush=True)


if __name__=='__main__': main()
