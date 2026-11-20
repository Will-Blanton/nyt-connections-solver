"""CPU-only group-mechanism labeling and deterministic edge discrimination audit.

Creates new analysis artifacts; original data, features, checkpoints and notebooks
are unchanged. Labels and answers are diagnostics, never model feature inputs.
"""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path
from statistics import mean

import torch

import run_graph_feature_experiment as experiment
from group_mechanisms import classify, validate_review, MECHANISMS
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.edge_features.resources import phrase_form


COLORS = ('purple','blue','green','yellow')
AGGREGATIONS = ('mean','minimum','weakest_word')
REVIEW_FILES = ('purple-review.csv','blue-review.csv','green-yellow-review.csv')


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError('Need a schema for an empty CSV')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_edges(edges, idx):
    pairs = torch.triu_indices(4,4,offset=1)
    values = edges[idx[:,pairs[0]],idx[:,pairs[1]]]
    matrix = edges[idx[:,:,None],idx[:,None,:]]
    diagonal = torch.eye(4,dtype=torch.bool)
    matrix = matrix.masked_fill(diagonal[...,None],0)
    return values, dict(mean=values.mean(1), minimum=values.amin(1),
                       weakest_word=(matrix.sum(-2)/3).amin(-2))


def tie_percentile(positive, negative):
    """Tie-aware within-board discrimination, .5 for constant channels."""
    if not len(negative):
        raise ValueError('Need competing quartets')
    return ((negative[None] < positive[:,None]).float() +
            .5*(negative[None] == positive[:,None])).mean(1)


def standalone_top_choice(scores, gold):
    """Expected first-choice success under uniform tie breaking, per channel."""
    maximum=scores.amax(0)
    tied=scores.eq(maximum[None])
    count=tied.sum(0)
    contributions=tied[gold].float()/count
    return contributions.sum(0), contributions, count, maximum


def phrase_consistency(words, index, idx, *, closed=False):
    """All-pair witnesses versus one completion shared by all four words."""
    forms = [phrase_form(w).replace(' ','') if closed else phrase_form(w) for w in words]
    sets = [set(index.get(w,())) for w in forms]
    pairs = torch.triu_indices(4,4,offset=1)
    pair = torch.tensor([[bool(a & b) for b in sets] for a in sets])
    all_pairs = pair[idx[:,pairs[0]],idx[:,pairs[1]]].all(-1)
    by_witness = defaultdict(list)
    for i, choices in enumerate(sets):
        for witness in choices:
            by_witness[witness].append(i)
    lookup = {tuple(group):i for i,group in enumerate(idx.tolist())}
    shared = torch.zeros(len(idx),dtype=torch.bool)
    for positions in by_witness.values():
        for group in itertools.combinations(positions,4):
            shared[lookup[group]] = True
    return all_pairs, shared, sets


def label_groups(frame, reference, folds, root):
    train, val = (set(reference['split_dates'][s]) for s in ('train','val'))
    fold_for_date = {d:f['fold'] for f in folds for d in f['val']}
    reviews = {}
    for name in REVIEW_FILES:
        path = root/name
        if not path.exists():
            raise ValueError('Complete the model review before final audit: '+str(path))
        with path.open(newline='') as handle:
            for review in csv.DictReader(handle):
                validate_review(review)
                key = review['date'],review['color']
                if key in reviews or review['date'] not in val:
                    raise ValueError('Duplicate/out-of-scope review: '+str(key))
                reviews[key] = review
    assert len(reviews) == 4*len(val)
    assert set(reviews) == {(date,color) for date in val for color in COLORS}
    rows = []
    for date, board in frame.groupby('date',sort=True):
        assert len(board) == 4
        for color,(_,source) in zip(COLORS,board.iterrows()):
            annotation = classify(source['category'])
            row = dict(date=str(date), original_split='train' if date in train else 'val',
                cv_fold=fold_for_date[date], color=color, category=source['category'],
                words=json.dumps(source['connections']), **annotation,
                regex_primary_mechanism=annotation['primary_mechanism'],
                regex_subtype=annotation['subtype'], reviewed=False)
            for i,word in enumerate(source['connections'],1):
                row[f'word_{i}'] = word
            if (date,color) in reviews:
                reviewed = reviews[date,color]
                row.update(primary_mechanism=reviewed['primary_mechanism'], subtype=reviewed['subtype'],
                    secondary_mechanisms=reviewed['secondary_mechanisms'],
                    label_confidence=reviewed['confidence'], label_source='model_review',
                    review_notes=reviewed['review_notes'], reviewed=True)
            row['needs_review'] = not row['reviewed'] or row['label_confidence']=='low'
            rows.append(row)
    assert len(rows) == 4*(len(train)+len(val))
    assert len({(r['date'],r['color']) for r in rows}) == len(rows)
    write_csv(root/'group_mechanisms.csv',rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/group-mechanism-audit'))
    parser.add_argument('--scope',choices=('val','pool'),default='pool')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    reference = torch.load('weights/experiments/20261023T150230Z-text-only.pt',map_location='cpu',weights_only=True)
    reference.pop('model_state_dict')
    folds = json.loads(Path('experiments/gine-coherence-cv/manifest.json').read_text())['folds']
    dates = set(reference['split_dates']['train'])|set(reference['split_dates']['val'])
    assert not dates & set(reference['split_dates']['test'])
    frame = experiment.load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(dates)]
    labels = label_groups(frame,reference,folds,root)
    label_lookup = {(r['date'],r['color']):r for r in labels}
    if args.scope == 'val':
        frame = frame[frame.date.isin(reference['split_dates']['val'])]
    class CacheOnly(torch.nn.Module):
        def get_embedding_dimension(self):
            return reference['model_config']['in_dim']

        def encode(self,*args,**kwargs):
            raise RuntimeError('CPU audit requires cached embeddings; no second embedding job allowed')
    encoder = CacheOnly()
    nodes,edges,metadata = build_graph_features(encoder,reference['embedding_model_name'],
        cache_dir='data/feature_cache/graph',**{**reference['feature_metadata']['options'],
        'use_spelling':False,'include_phonetics':True,'include_letter_edits':True,
        'phonetic_nodes':False,'transformation_nodes':False})
    names = list(edges.feature_names)
    assert len(names) == 41 and names[:15] == reference['edge_feature_names']
    lexicon = edges.providers[1].lexicon
    artifacts = experiment.make_artifacts()[4]
    idx,gold,near = (artifacts[k] for k in ('idx','y','near_miss'))
    targets = gold.nonzero().flatten()
    combinations = idx.tolist()
    summary_sums, witness_counts = {}, defaultdict(Counter)
    board_phrase_rows, phrase_group_rows, group_maxima, univariate_rows = [], [], [], []
    out_fields = ['date','original_split','cv_fold','color','category','primary_mechanism',
        'label_source','channel','active_in_current_cv','pair_support',
        *[f'{a}_{m}' for a in AGGREGATIONS for m in ('value','percentile','near_miss_percentile')]]
    with (root/'group_edge_evidence.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=out_fields)
        writer.writeheader()
        for board_index,(date,board_frame) in enumerate(frame.groupby('date',sort=True)):
            words = list(itertools.chain.from_iterable(board_frame.connections))
            board_edges = edges.encode(words)
            values,pooled = aggregate_edges(board_edges,idx)
            support = (values[gold]>0).float().mean(1)
            percentiles = {a:tie_percentile(v[gold],v[~gold]) for a,v in pooled.items()}
            near_percentiles = {a:tie_percentile(v[gold],v[near]) for a,v in pooled.items()}
            split=label_lookup[date,'purple']['original_split']
            for aggregation,scores in pooled.items():
                expected,contributions,total,maximum=standalone_top_choice(scores,gold)
                for k,name in enumerate(names):
                    univariate_rows.append(dict(date=date,original_split=split,channel=name,
                        active_in_current_cv=k<15,aggregation=aggregation,
                        expected_hit1=float(expected[k]),
                        maximum_value=float(maximum[k]),tied_at_maximum=int(total[k]),
                        **{f'{color}_expected_hit1':float(contributions[ci,k])
                           for ci,color in enumerate(COLORS)}))
            for ci,color in enumerate(COLORS):
                label = label_lookup[date,color]
                for k,name in enumerate(names):
                    row = {field:label[field] for field in ('date','original_split','cv_fold','color',
                                                          'category','primary_mechanism','label_source')}
                    row.update(channel=name,active_in_current_cv=k<15,pair_support=float(support[ci,k]))
                    for aggregation in AGGREGATIONS:
                        row[f'{aggregation}_value'] = float(pooled[aggregation][targets[ci],k])
                        row[f'{aggregation}_percentile'] = float(percentiles[aggregation][ci,k])
                        row[f'{aggregation}_near_miss_percentile'] = float(near_percentiles[aggregation][ci,k])
                    writer.writerow(row)
                    for aggregation in AGGREGATIONS:
                        key=(label['original_split'],color,label['primary_mechanism'],name,aggregation)
                        stats=summary_sums.setdefault(key,Counter())
                        stats['groups']+=1
                        p=row[f'{aggregation}_percentile']
                        n=row[f'{aggregation}_near_miss_percentile']
                        stats.update(value_sum=row[f'{aggregation}_value'],percentile_sum=p,
                            near_percentile_sum=n,strong_count=int(p>=.95),
                            nonzero_count=int(row[f'{aggregation}_value']>0),
                            all_pairs_count=int(support[ci,k]==1))
                group_maxima.append(dict(date=date,color=color,original_split=label['original_split'],
                    primary_mechanism=label['primary_mechanism'],
                    best_core_channel_percentile=float(percentiles['weakest_word'][ci,:15].max()),
                    best_optional_channel_percentile=float(percentiles['weakest_word'][ci,15:].max())))
            for direction,index in (('preceding',lexicon.preceding),('following',lexicon.following),
                                    ('closed_preceding',lexicon.closed_preceding),
                                    ('closed_following',lexicon.closed_following)):
                all_pairs,shared,sets=phrase_consistency(words,index,idx,closed=direction.startswith('closed'))
                split=label_lookup[date,'purple']['original_split']
                row=dict(date=date,original_split=split,direction=direction,
                    gold_all_pairs=int((all_pairs&gold).sum()),gold_shared=int((shared&gold).sum()),
                    gold_inconsistent=int((all_pairs&~shared&gold).sum()),
                    negative_all_pairs=int((all_pairs&~gold).sum()),
                    negative_shared=int((shared&~gold).sum()),
                    negative_inconsistent=int((all_pairs&~shared&~gold).sum()))
                board_phrase_rows.append(row)
                witness_counts[split,direction].update({k:v for k,v in row.items() if isinstance(v,int)})
                for ci,color in enumerate(COLORS):
                    group=combinations[targets[ci]]
                    common=sorted(set.intersection(*(sets[i] for i in group)))
                    label=label_lookup[date,color]
                    phrase_group_rows.append(dict(date=date,color=color,original_split=split,
                        category=label['category'],primary_mechanism=label['primary_mechanism'],
                        direction=direction,all_pairs_supported=bool(all_pairs[targets[ci]]),
                        common_witness_count=len(common),common_witnesses=json.dumps(common)))
            if (board_index+1)%25==0:
                print('AUDITED',board_index+1,'/',frame.date.nunique(),flush=True)
    summary_rows=[]
    for (split,color,mechanism,channel,aggregation),sums in summary_sums.items():
        n=sums['groups']
        summary_rows.append(dict(original_split=split,color=color,primary_mechanism=mechanism,
            channel=channel,active_in_current_cv=names.index(channel)<15,aggregation=aggregation,groups=n,
            mean_value=sums['value_sum']/n,mean_percentile=sums['percentile_sum']/n,
            mean_near_miss_percentile=sums['near_percentile_sum']/n,
            strong_fraction=sums['strong_count']/n,nonzero_fraction=sums['nonzero_count']/n,
            all_pairs_fraction=sums['all_pairs_count']/n))
    write_csv(root/'mechanism_channel_summary.csv',summary_rows)
    write_csv(root/'phrase_witness_groups.csv',phrase_group_rows)
    write_csv(root/'phrase_witness_boards.csv',board_phrase_rows)
    write_csv(root/'oracle_best_channel.csv',group_maxima)
    write_csv(root/'standalone_feature_rankings.csv',univariate_rows)
    ranks_path=Path('experiments/gine-edge-coherence/probability-audit.json')
    model_rows=[]
    for row in json.loads(ranks_path.read_text())['records']:
        for ci,color in enumerate(COLORS):
            label=label_lookup[row['date'],color]
            model_rows.append(dict(variant=row['variant'],seed=row['seed'],date=row['date'],color=color,
                primary_mechanism=label['primary_mechanism'],rank=row['gold_ranks'][ci],
                category=label['category'],words=label['words']))
    write_csv(root/'model_group_ranks.csv',model_rows)
    metadata.update(audit=dict(scope=args.scope,boards=frame.date.nunique(),groups=4*frame.date.nunique(),
        all_label_rows=len(labels),model_reviewed=sum(r['reviewed'] for r in labels),
        cpu_threads=2,model_inference=False,
        current_cv_channels=names[:15],optional_channels=names[15:],
        percentile='tie-aware vs 1816 incorrect quartets in the same board; .5 is uninformative',
        near_misses='192 quartets sharing exactly three words with a true group',
        labels='diagnostic only; never feature inputs or training targets'))
    (root/'audit_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('AUDIT COMPLETE',metadata['audit'],flush=True)


if __name__=='__main__':
    main()
