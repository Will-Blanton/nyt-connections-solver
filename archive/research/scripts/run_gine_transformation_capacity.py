"""Sequential, validation-only width/depth ablations of learned transformation edges."""
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import run_graph_feature_experiment as experiment
from run_gine_curriculum import make_schedule
from tune_gine import build_model
from tune_gine_transformations import diagnostic_metrics, write
from nyt_connections_solver.data_loader import make_state_loaders, collate_keep_wordlists
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.transformation_models import TransformationGINE

ROOT=Path('experiments/gine-transformation-capacity')
VARIANTS={
    'full':dict(transformation_width=64,pair_hidden=64,pair_layers=2,hidden_width=128,gine_layers=2,readout_width=128),
    'edge-small':dict(transformation_width=16,pair_hidden=16,pair_layers=2,hidden_width=128,gine_layers=2,readout_width=128),
    'all-small':dict(transformation_width=16,pair_hidden=16,pair_layers=2,hidden_width=64,gine_layers=2,readout_width=64),
    'one-gine':dict(transformation_width=16,pair_hidden=16,pair_layers=2,hidden_width=64,gine_layers=1,readout_width=64),
    'linear-pair':dict(transformation_width=16,pair_hidden=16,pair_layers=1,hidden_width=64,gine_layers=1,readout_width=64),
}


def build(reference,params,idx,seed,dim,variant):
    shape=VARIANTS[variant]
    params={**params,**{k:shape[k] for k in ('hidden_width','gine_layers','readout_width')}}
    base,config=build_model(reference,params,idx,seed)
    config.update(in_dim=dim,text_dim=reference['model_config']['in_dim'],
        base_edge_dim=reference['model_config']['edge_dim'],max_transformations=8,
        transformation_width=shape['transformation_width'],pair_hidden=shape['pair_hidden'],
        pair_layers=shape['pair_layers'],learned_edge_dim=16,node_mode='text',learned_edges=True)
    experiment.seed_everything(seed)
    model=TransformationGINE(group_idx=idx,**config)
    common={n:v for n,v in base.state_dict().items() if not n.startswith('edge_encoder.edge_projections.')}
    missing,unexpected=model.load_state_dict(common,strict=False)
    assert not unexpected
    for target,source in zip(model.edge_encoder.edge_projections,base.edge_encoder.edge_projections):
        with torch.no_grad():target.weight[:,:source.in_features].copy_(source.weight)
    # The reused text projection and LayerNorm are active in every arm.
    assert all(p.requires_grad for m in (model.input_norm,model.proj) for p in m.parameters())
    counts=dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        by_module={n:sum(p.numel() for p in m.parameters() if p.requires_grad)
            for n,m in model.named_children()},
        active_linear_layers={n:[list(m.weight.shape)[::-1] for m in module.modules()
            if isinstance(m,nn.Linear) and m.weight.requires_grad] for n,module in model.named_children()})
    del base
    return model,config,counts


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    reference_path=Path('weights/experiments/20261023T150230Z-text-only.pt')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    params=json.loads(Path('experiments/gine-transformation-tuning/best.json').read_text())['params']
    manifest=dict(reference=str(reference_path),params=params,variants=VARIANTS,seeds=[10,11,12],
        updates=30000,batch_size=16,eval_every=500,cpu_threads=4,split='val',
        loss='probability_mass',scheduler=False,early_stopping=False,pruning=False,
        frozen='Qwen vectors only; all active projection/LN/GINE/edge modules train',
        text_projection='keep shared pretrained initialization and 1024->256 width unchanged',
        feature_payload='same eight alternatives, position metadata, base15 edges and learned16 edge dimensions',
        comparison='fresh full-size control per seed in this process; previous ordinary GINE for reference',
        selection='best s4 Hit@1; MRR tie-break',
        diagnostics='same fixed 96 training states/size as tuning, full validation every2k plus first/last',
        interpretation='width/depth ladder at fixed tuned optimizer; not independently retuned per size')
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Manifest mismatch')
    write(path,manifest)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    device=torch.device('cuda')
    dates=reference['split_dates']
    frame=experiment.load_connections(Path(reference['dataset_path']))
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):return reference['model_config']['in_dim']
        def encode(self,*a,**kw):raise RuntimeError('Expected frozen feature cache')
    encoder=CacheOnly()
    nodes,edges,metadata=build_graph_features(encoder,reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph',**{**reference['feature_metadata']['options'],
            'use_spelling':False,'transformation_nodes':True,'max_transformations':8,
            'transformation_min_length':3})
    datasets={}
    for split in ('train','val'):
        subset=frame[frame.date.isin(dates[split])]
        assert subset.date.nunique()==len(dates[split])
        datasets[split],_=make_state_loaders(subset,encoder,None,16,shuffle=False,
            feature_builder=nodes,edge_providers=edges,cache=edges.cache)
        print('PREPARED',split,{s:len(d) for s,d in datasets[split].items()},flush=True)
    assert list(edges.feature_names)==reference['edge_feature_names']
    indices=json.loads(Path('experiments/gine-transformation-tuning/diagnostic-indices.json').read_text())
    write(ROOT/'diagnostic-indices.json',indices)
    train_eval={s:DataLoader(Subset(d,indices[str(s)]),batch_size=16,shuffle=False,
        collate_fn=collate_keep_wordlists) for s,d in datasets['train'].items()}
    val_loaders={s:DataLoader(d,batch_size=16,shuffle=False,collate_fn=collate_keep_wordlists)
        for s,d in datasets['val'].items()}
    artifacts=experiment.make_artifacts()
    runs=[]
    for seed in manifest['seeds']:
        runs.append(dict(variant='ordinary',seed=seed,
            result=f'experiments/gine-transformations/control-seed-{seed}.json',
            harness=f'experiments/gine-transformations/harness-control-seed-{seed}-val.json'))
        for variant in VARIANTS:
            path=ROOT/f'{variant}-seed-{seed}.json'
            payload=json.loads(path.read_text()) if path.exists() else None
            if not payload or payload.get('status')!='complete':
                model,config,counts=build(reference,params,artifacts[4]['idx'],seed,nodes.dim,variant)
                print('START',variant,seed,'PARAMETERS',counts,flush=True)
                loaders={s:DataLoader(d,batch_size=16,shuffle=True,collate_fn=collate_keep_wordlists,
                    generator=torch.Generator().manual_seed(seed+s)) for s,d in datasets['train'].items()}
                history=[];diagnostics=[]
                def callback(step,metrics,best,row):
                    history.append(row)
                    if step==500 or step%2000==0 or step==30000:
                        diagnostics.append(dict(update=step,
                            train=diagnostic_metrics(model,train_eval,artifacts,device),
                            validation=diagnostic_metrics(model,val_loaders,artifacts,device)))
                    write(path,dict(status='running',variant=variant,seed=seed,model_config=config,
                        parameters=counts,history=history,diagnostics=diagnostics))
                    write(ROOT/'progress.json',dict(variant=variant,seed=seed,update=step,best_hit1=best[0]))
                started=perf_counter()
                model,summary=experiment.train(model,loaders,val_loaders,artifacts,device,
                    lr=params['lr'],projection_lr=None,weight_decay=params['weight_decay'],updates=30000,
                    eval_every=500,min_updates_before_stop=30000,patience=10000,min_delta=1e-3,
                    evaluation_callback=callback,state_schedule=make_schedule('mixed',seed,updates=30000),
                    diagnose_gradients=True)
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                weights=Path('weights/experiments')/f'{stamp}-transformation-capacity-{variant}-seed-{seed}.pt'
                torch.save(dict(model_class='TransformationGINE',model_config=config,
                    model_state_dict=model.state_dict(),feature_metadata=metadata,
                    edge_feature_names=list(edges.feature_names),split_dates=dates,
                    dataset_path=reference['dataset_path'],embedding_model_name=reference['embedding_model_name'],
                    training_config=dict(variant=variant,seed=seed,params=params,updates=30000,
                        architecture=VARIANTS[variant],reference_checkpoint=str(reference_path),
                        ranking_loss='probability_mass',batch_size=16,scheduler=False),training_summary=summary),weights)
                payload=dict(status='complete',variant=variant,seed=seed,checkpoint=str(weights),
                    model_config=config,parameters=counts,training=summary,diagnostics=diagnostics,
                    elapsed_seconds=perf_counter()-started)
                write(path,payload)
                del model,loaders;gc.collect();torch.cuda.empty_cache()
            harness=ROOT/f'harness-{variant}-seed-{seed}-val.json'
            if not harness.exists():
                with harness.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable,'scripts/evaluate_graph_feedback.py',payload['checkpoint'],
                        '--split','val','--beam-width','20','--score-mode','full_board','--output',str(harness)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
            runs.append(dict(variant=variant,seed=seed,result=str(path),harness=str(harness)))
            write(ROOT/'runs.json',runs)
            print('COMPLETE',variant,seed,payload['training']['validation']['s4']['hit1'],flush=True)
    print('ALL RUNS COMPLETE',flush=True)


if __name__=='__main__':main()
