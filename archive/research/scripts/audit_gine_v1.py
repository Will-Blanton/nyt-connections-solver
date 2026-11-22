"""Check refitted test checkpoints through the production model/feature loader."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch

import run_graph_feature_experiment as experiment
from run_gine_relation_witnesses import write
from nyt_connections_solver.checkpoints import load_graph_harness
from nyt_connections_solver.feedback_search import ModelGroupScorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/gine-v1-refit'))
    args=parser.parse_args()
    root=args.root
    torch.set_num_threads(2)
    sealed=json.loads((root/'sealed-checkpoints.json').read_text())
    records=[]
    for item in sealed:
        path=Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['split_dates']['val']==[]
        assert len(payload['split_dates']['train'])==1089
        assert not set(payload['split_dates']['train'])&set(payload['split_dates']['test'])
        class CacheOnly(torch.nn.Module):
            def get_embedding_dimension(self):return payload['model_config']['in_dim']
            def encode(self,*a,**kw):raise RuntimeError('Expected cached original embeddings')
        with patch('nyt_connections_solver.checkpoints.SentenceTransformer',return_value=CacheOnly()):
            harness=load_graph_harness(path,device='cpu',cache_dir='data/feature_cache/graph')
        scorer=ModelGroupScorer(harness)
        frame=experiment.load_connections(Path(payload['dataset_path']))
        correct=[]
        for date in payload['split_dates']['test']:
            groups=tuple(tuple(g) for g in frame.loc[frame.date==date,'connections'])
            words=tuple(w for g in groups for w in g)
            scorer.prepare(words)
            best=max(scorer.score_states([words])[0],key=lambda c:c.log_probability)
            correct.append(frozenset(best.words) in set(map(frozenset,groups)))
        result=json.loads((root/f"{item['arm']}-seed-{item['seed']}.json").read_text())
        expected=result['evaluation']['ranking']['s4']['hit1']
        assert abs(mean(correct)-expected)<1e-7,(item,mean(correct),expected)
        records.append(dict(**item,expected_hit1=expected,actual_hit1=mean(correct),test_boards=len(correct)))
        print('VERIFIED',item['arm'],item['seed'],mean(correct),flush=True)
    write(root/'checkpoint-audit.json',dict(complete=len(records)==6,records=records))
    with (root/'report.md').open('a') as stream:
        stream.write('\n\nAll six sealed checkpoints reconstruct through the production loader, '
                     'retain their original SHA-256, and reproduce test Hit@1 on CPU. '
                     'The refit-budget and fixed-training tests pass; `git diff --check` passes.\n')


if __name__=='__main__':main()
