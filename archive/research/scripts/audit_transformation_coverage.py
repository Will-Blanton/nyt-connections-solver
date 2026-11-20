"""Analysis-only intended readings; never imported by feature construction."""
import json
from pathlib import Path
import torch
import run_graph_feature_experiment as experiment
from nyt_connections_solver.graph_features import build_graph_features

ROOT=Path('experiments/gine-transformations')
EXAMPLES={
    '2024-03-20':['cream','journey','kiss','rush'],
    '2025-11-18':['place','point','site','spot'],
    '2026-03-08':['jack','nada','squat','zip'],
    '2026-04-05':['goo','mush','paste','pulp'],
}


def main():
    result=json.loads((ROOT/'control-seed-10.json').read_text())
    ref=torch.load(result['checkpoint'],map_location='cpu',weights_only=True)
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self): return ref['model_config']['text_dim']
        def encode(self,*a,**k): raise RuntimeError('Only lexical extraction requested')
    nodes,_,_=build_graph_features(CacheOnly(),ref['embedding_model_name'],cache_dir='data/feature_cache/graph',**ref['feature_metadata']['options'])
    inventory=json.loads(Path('experiments/gine-phonetics/mechanism-inventory.json').read_text())
    records=[]
    for row in inventory['groups_inventory']:
        if row['date'] not in EXAMPLES or row['color']!='purple': continue
        targets=EXAMPLES[row['date']]
        readings=[]
        for word,target in zip(row['words'],targets):
            all_rows=nodes.alternatives(word,truncate=False)
            kept=nodes.alternatives(word)
            readings.append(dict(word=word,target=target,attested=target in nodes.lexicon.words,
                present=any(r[0]==target for r in kept),
                truncated=any(r[0]==target for r in all_rows) and not any(r[0]==target for r in kept),
                matches=[list(r) for r in kept if r[0]==target]))
        records.append(dict(date=row['date'],category=row['category'],readings=readings,
            all_targets_present=all(r['present'] for r in readings)))
    out=dict(method='Manually specified intended readings, analysis only; not used for features or checkpoint selection',records=records)
    (ROOT/'target-coverage.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
