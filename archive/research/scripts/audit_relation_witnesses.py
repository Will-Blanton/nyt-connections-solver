"""Train/validation-only signal audit using the same cached witness providers."""
import csv
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import run_graph_feature_experiment as experiment
from nyt_connections_solver.edge_features import (
    EnglishLexicon,SharedWordNetConcept,SharedLetterEditTargets,WordNetSimilarity,EdgeFeatureComposer)
from nyt_connections_solver.feature_cache import TensorCache


def main():
    torch.set_num_threads(2)
    root=Path('experiments/gine-relation-witnesses')
    reference=torch.load('weights/experiments/20261023T150230Z-text-only.pt',map_location='cpu',weights_only=True)
    frame=experiment.load_connections(Path(reference['dataset_path']))
    lex=EnglishLexicon();concepts=SharedWordNetConcept(lex);targets=SharedLetterEditTargets(lex)
    cache=TensorCache('data/feature_cache/graph')
    composer=EdgeFeatureComposer([concepts,targets],cache=cache)
    old=EdgeFeatureComposer([WordNetSimilarity(lex)],cache=cache)
    idx=np.array(list(itertools.combinations(range(16),4)))
    gold=np.array([len(set(q//4))==1 for q in idx])
    names=[*concepts.feature_names,*targets.feature_names,'edit_target_family_max','existing_rare_concept_mean','existing_rare_concept_minimum']
    aggregate={};covered=[]
    for split in ('train','val'):
        rows=[];positives=np.zeros(len(names),int);negatives=positives.copy()
        for date in reference['split_dates'][split]:
            groups=frame.loc[frame.date==date,'connections'].tolist()
            words=sum(groups,[])
            values=composer.encode_candidates(words).numpy()
            wordnet=old.encode(words).numpy()[...,1]
            pairs=np.stack([wordnet[idx[:,i],idx[:,j]] for i,j in itertools.combinations(range(4),2)],axis=1)
            values=np.column_stack([values,values[:,1:].max(1),pairs.mean(1),pairs.min(1)])
            maxima=values.max(0);tied=values==maxima
            rows.append((tied[gold].sum(0)/tied.sum(0)).tolist())
            positives+=(values[gold]>0).sum(0);negatives+=(values[~gold]>0).sum(0)
            for group_idx,quartet in enumerate(idx[gold]):
                if values[gold][group_idx,11]>0:
                    readings={}
                    for channel,name in enumerate(targets.feature_names):
                        if values[gold][group_idx,channel+1]<=0:continue
                        from nyt_connections_solver.edge_features.letter_edits import OPERATIONS,ALIGNMENTS
                        operation=OPERATIONS[channel//len(ALIGNMENTS)];alignment=ALIGNMENTS[channel%len(ALIGNMENTS)]
                        shared=frozenset.intersection(*(targets.witnesses(words[i],operation,alignment) for i in quartet))
                        readings[name]=[list(value) for value in sorted(shared)]
                    covered.append(dict(split=split,date=date,source_group_idx=group_idx,
                                        words=[words[i] for i in quartet],witnesses=readings))
        aggregate[split]={name:dict(nonzero_gold_groups=int(positives[i]),nonzero_incorrect_quartets=int(negatives[i]),
                                   standalone_expected_hit1=mean(row[i] for row in rows)) for i,name in enumerate(names)}
    (root/'signal-audit.json').write_text(json.dumps(dict(aggregate=aggregate,covered_edit_groups=covered),indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))
    print('Covered edit groups',json.dumps(covered,indent=2))


if __name__=='__main__':main()
