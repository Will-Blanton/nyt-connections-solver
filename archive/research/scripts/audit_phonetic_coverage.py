import json
from pathlib import Path
import sys
from statistics import mean
import torch
sys.path.insert(0,str(Path('scripts').resolve()))
from run_graph_feature_experiment import load_connections,make_artifacts
from audit_graph_edge_coverage import candidate_evidence
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.data_loader import ConnectionsDataset

torch.set_num_threads(2)
r=torch.load('weights/experiments/20261023T150230Z-text-only.pt',map_location='cpu',weights_only=True)
class CacheOnly(torch.nn.Module):
 def get_embedding_dimension(self): return r['model_config']['in_dim']
 def encode(self,*args,**kwargs): raise RuntimeError('Audit requires existing embedding cache; do not run a second embedding job')
embedder=CacheOnly()
nodes,edges,metadata=build_graph_features(embedder,r['embedding_model_name'],cache_dir='data/feature_cache/graph',**{**r['feature_metadata']['options'],'include_phonetics':True})
frame=load_connections(Path(r['dataset_path']))
frame=frame[frame.date.isin(r['split_dates']['val'])]
boards=ConnectionsDataset.prepare(frame,embedder,None,nodes,edges)
a=make_artifacts()[4];targets=a['y'].nonzero().flatten()
inventory=json.loads(Path('experiments/gine-phonetics/mechanism-inventory.json').read_text())
lookup={(x['date'],x['color']):x for x in inventory['groups_inventory']}
rows=[]
for board in boards:
 values,pooled,percentiles=candidate_evidence(board['edge_features'],a['idx'],a['y'])
 for ci,color in enumerate(('purple','blue','green','yellow')):
  target=targets[ci]; source=lookup[(str(board['date']),color)]
  rows.append(dict(date=str(board['date']),color=color,category=source['category'],mechanism=source['mechanism'],words=source['words'],
    channels={name:dict(any_pair=bool((values[target,:,j]>0).any()),all_pairs=bool((values[target,:,j]>0).all()),
        mean=float(pooled[target,j]),incorrect_percentile=float(percentiles[ci,j])) for j,name in enumerate(edges.feature_names)}))
summary={}
for subset in ('all_purple','sound','embedded_fragment'):
 selected=[row for row in rows if (row['color']=='purple' if subset=='all_purple' else row['mechanism']==subset)]
 summary[subset]=dict(groups=len(selected),channels={name:dict(any_pair=mean(x['channels'][name]['any_pair'] for x in selected),
    all_pairs=mean(x['channels'][name]['all_pairs'] for x in selected),incorrect_percentile=mean(x['channels'][name]['incorrect_percentile'] for x in selected))
    for name in edges.feature_names[-6:]})
Path('experiments/gine-phonetics/feature-coverage.json').write_text(json.dumps(dict(summary=summary,groups=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
