"""Reconstruct experiment checkpoints through the production loader, cache-only."""
import json
import argparse
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch

import run_graph_feature_experiment as experiment
from run_gine_relation_witnesses import VARIANTS
from nyt_connections_solver.checkpoints import load_graph_harness
from nyt_connections_solver.feedback_search import ModelGroupScorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available',action='store_true',help='Audit completed runs while training continues')
    parser.add_argument('--root',type=Path,default=Path('experiments/gine-relation-witnesses'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    records=[]
    for seed in manifest['seeds']:
        for variant in VARIANTS:
            path=root/f'{variant}-seed-{seed}.json'
            if not path.exists() and args.available:continue
            result=json.loads(path.read_text())
            if result['status']!='complete':
                if args.available:continue
                raise ValueError('Wait for all matched runs')
            checkpoint=torch.load(result['checkpoint'],map_location='cpu',weights_only=True)
            class CacheOnly(torch.nn.Module):
                def get_embedding_dimension(self):return checkpoint['model_config']['in_dim']
                def encode(self,*args,**kwargs):raise RuntimeError('Audit requires precomputed frozen embeddings')
            with patch('nyt_connections_solver.checkpoints.SentenceTransformer',return_value=CacheOnly()):
                harness=load_graph_harness(result['checkpoint'],device='cpu',cache_dir='data/feature_cache/graph')
            scorer=ModelGroupScorer(harness)
            branch_activity=[]
            width=getattr(harness.model,'edge_summary_width',0)
            hook=(harness.model.edge_summary_encoder.register_forward_hook(
                lambda module,inputs,output:branch_activity.append(
                    float(output.float().std(dim=0,unbiased=False).mean()))) if width else None)
            frame=experiment.load_connections(Path(checkpoint['dataset_path']))
            rows=[]
            for date in checkpoint['split_dates']['val']:
                groups=[tuple(g) for g in frame.loc[frame.date==date,'connections']]
                words=tuple(w for g in groups for w in g)
                scorer.prepare(words)
                scored=sorted(scorer.score_states([words])[0],key=lambda c:c.log_probability,reverse=True)
                lookup={frozenset(c.words):rank for rank,c in enumerate(scored,1)}
                rows.append(dict(date=date,gold_ranks=[lookup[frozenset(g)] for g in groups]))
            measured=mean(1 in row['gold_ranks'] for row in rows)
            expected=result['training']['validation']['s4']['hit1']
            assert abs(measured-expected)<1e-7,(variant,seed,measured,expected)
            if hook is not None:hook.remove()
            record=dict(variant=variant,seed=seed,checkpoint=result['checkpoint'],
                        measured_hit1=measured,expected_hit1=expected,groups=rows)
            if width:
                record['summary_branch']=dict(
                    readout_weight_norm=float(harness.model.group_readout[0].weight[:,-width:].detach().norm()),
                    mean_within_board_activation_std=mean(branch_activity))
            records.append(record)
            print('VERIFIED',variant,seed,measured,flush=True)
    (root/'checkpoint-audit.json').write_text(json.dumps(dict(records=records,
        complete=len(records)==len(manifest['seeds'])*len(VARIANTS)),indent=2)+'\n')


if __name__=='__main__':main()
