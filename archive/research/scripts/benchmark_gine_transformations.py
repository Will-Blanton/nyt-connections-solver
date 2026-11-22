"""Warm model-only full-board latency on precomputed validation inputs."""
import json
from pathlib import Path
from statistics import median
import torch
import run_graph_feature_experiment as experiment
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.checkpoints import GRAPH_MODEL_TYPES

ROOT=Path('experiments/gine-transformations')


def main():
    torch.set_num_threads(4)
    runs=json.loads((ROOT/'runs.json').read_text())
    runs=[r for r in runs if r['seed']==10]
    first=torch.load(json.loads(Path(runs[0]['result']).read_text())['checkpoint'],map_location='cpu',weights_only=True)
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self): return first['model_config']['text_dim']
        def encode(self,*a,**k): raise RuntimeError('Expected cached frozen vectors')
    encoder=CacheOnly()
    nodes,edges,metadata=build_graph_features(encoder,first['embedding_model_name'],cache_dir='data/feature_cache/graph',**first['feature_metadata']['options'])
    frame=experiment.load_connections(Path(first['dataset_path']))
    frame=frame[frame.date.isin(first['split_dates']['val'])]
    boards=ConnectionsDataset.prepare(frame,encoder,None,nodes,edges)
    device=torch.device('cuda')
    x=torch.stack([boards[i]['x'] for i in range(10)]).to(device)
    e=torch.stack([boards[i]['edge_features'] for i in range(10)]).to(device)
    idx=experiment.make_artifacts()[4]['idx'].to(device)
    results=[]
    for run in runs:
        checkpoint=torch.load(json.loads(Path(run['result']).read_text())['checkpoint'],map_location='cpu',weights_only=True)
        model=GRAPH_MODEL_TYPES[checkpoint['model_class']](group_idx=idx,**checkpoint['model_config']).to(device).eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        timings=[]
        with torch.inference_mode():
            for step in range(120):
                i=step%10
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                model(x[i:i+1],group_idx=idx,edge_features=e[i:i+1])
                end.record(); end.synchronize()
                if step>=20: timings.append(start.elapsed_time(end))
        results.append(dict(variant=run['variant'],median_ms=median(timings),mean_ms=sum(timings)/len(timings),samples=len(timings)))
        del model; torch.cuda.empty_cache()
    (ROOT/'latency.json').write_text(json.dumps(dict(method='100 warm CUDA-event forwards, batch1, all1820quartets, ten validation boards; excludes transfer, features and harness',results=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
