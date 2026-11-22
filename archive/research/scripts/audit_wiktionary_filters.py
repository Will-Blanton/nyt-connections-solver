"""Validation-only audit of common-witness and dictionary-specificity filtering."""
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon


# Evaluation annotations only: intended witnesses checked against the previous
# model review. These NEVER enter the index, frequencies, or filtering rules.
TARGETS = {
    "___ fish that aren't fish": ('following','fish'),
    '___ triangle': ('following','triangle'), 'happy ___': ('preceding','happy'),
    '___ chair': ('following','chair'), 'sea ___': ('preceding','sea'),
    '___ pad': ('following','pad'), '___ ball': ('following','ball'),
    'words with “fruit”': ('following','fruit'),
    'weights in boxing': ('following','weight'),
    'art movements, with -ism': ('following','ism'), 'tele___': ('preceding','tele'),
    'words after “eye”': ('preceding','eye'), '___horn': ('following','horn'),
    'kinds of knives': ('following','knife'), '___ control': ('following','control'),
    '___ of time': ('following','oftime'), 'double ___': ('preceding','double'),
    'long___': ('preceding','long'), 'pot___': ('preceding','pot'),
    '___fish': ('following','fish'), 'words after “golden”': ('preceding','golden'),
    '___ chain': ('following','chain'), '___box': ('following','box'),
    'kinds of trucks': ('following','truck'), 'words before “bed”': ('following','bed'),
    'apple products': ('preceding','apple'), 'star___': ('preceding','star'),
    'dirty ___': ('preceding','dirty'), 'kinds of bears': ('following','bear'),
    '___ zone': ('following','zone'), 'words before "light"': ('following','light'),
    'draw ___': ('preceding','draw'), '___life': ('following','life'),
    '"long" things': ('preceding','long'), 'ski ___': ('preceding','ski'),
    'words before "ball" in sport names': ('following','ball'),
    '___boards you ride on': ('following','board'),
}


def dictionary_frequency(lexicon, witness, direction):
    """Count distinct compact dictionary titles containing this end completion.

    Prefix/suffix direction matters. Exclude the title equal to the witness
    itself (empty source). No puzzle words or labels define the frequency.
    """
    return lexicon.completion_frequency(witness, direction)


def quartet_support(completions, quartets):
    """Return six-pair clique support and actual common-witness support."""
    n = len(completions)
    pairs = np.zeros((n,n), dtype=bool)
    witness_words = defaultdict(list)
    for i, choices in enumerate(completions):
        for witness in choices:
            witness_words[witness].append(i)
        for j in range(i):
            pairs[i,j] = pairs[j,i] = bool(choices & completions[j])
    clique = np.ones(len(quartets), dtype=bool)
    for i,j in itertools.combinations(range(4),2):
        clique &= pairs[quartets[:,i],quartets[:,j]]
    matched = set()
    for positions in witness_words.values():
        matched.update(itertools.combinations(positions,4))
    common = np.array([tuple(q) in matched for q in quartets], dtype=bool)
    return clique,common


def main():
    started = time.perf_counter()
    root = Path('experiments/wiktionary-filters')
    root.mkdir(parents=True, exist_ok=True)
    lex = CompletionLexicon.from_headword_cache('data/wiktionary/headwords.jsonl.gz')
    spaced = CompletionLexicon.from_headword_cache('data/wiktionary/headwords.jsonl.gz', spaced_only=True)
    with open('experiments/group-mechanism-audit/group_mechanisms.csv') as stream:
        rows = [r for r in csv.DictReader(stream) if r['original_split']=='val']
    boards = defaultdict(list)
    for row in rows:
        boards[row['date']].append(row)
    cutoffs = [None,10000,5000,2500,1000,500,250,100,50,25]
    counts = {(cap,stage):defaultdict(int) for cap in cutoffs for stage in ('pair_clique','common_witness')}
    retained = {cap:set() for cap in cutoffs}
    witness_rows, targets = [], []
    frequencies = {}
    quartets = np.array(list(itertools.combinations(range(16),4)))
    gold = np.array([len({int(i)//4 for i in q})==1 for q in quartets])
    for date,groups in boards.items():
        words = list(itertools.chain.from_iterable(json.loads(r['words']) for r in groups))
        choices = {d:[lex.completions(w,d) for w in words] for d in ('preceding','following')}
        for direction,word_choices in choices.items():
            for witness in set().union(*word_choices):
                key = (direction,witness)
                if key not in frequencies:
                    frequencies[key] = dictionary_frequency(lex,witness,direction)
        for gi,row in enumerate(groups):
            if row['primary_mechanism']!='phrase_completion' or row['category'] not in TARGETS:
                continue
            direction,witness = TARGETS[row['category']]
            if not all(witness in c for c in choices[direction][gi*4:gi*4+4]):
                continue
            frequency = frequencies[direction,witness]
            needs_closed = not all(witness in spaced.completions(w,direction) for w in words[gi*4:gi*4+4])
            target = dict(date=date,color=row['color'],category=row['category'],witness=witness,
                          direction=direction,dictionary_frequency=frequency,needs_closed_or_mixed=needs_closed)
            targets.append(target)
            for cap in cutoffs:
                if cap is None or frequency <= cap:
                    retained[cap].add((date,row['category']))
        for cap in cutoffs:
            unique = {stage:set() for stage in ('pair_clique','common_witness')}
            for direction,word_choices in choices.items():
                filtered = [set(c) if cap is None else {w for w in c if frequencies[direction,w]<=cap} for c in word_choices]
                clique,common = quartet_support(filtered,quartets)
                for stage,mask in [('pair_clique',clique),('common_witness',common)]:
                    counts[cap,stage]['incorrect_direction_occurrences'] += int((mask & ~gold).sum())
                    counts[cap,stage]['gold_direction_occurrences'] += int((mask & gold).sum())
                    unique[stage].update(tuple(q) for q in quartets[mask & ~gold])
            for stage in unique:
                counts[cap,stage]['incorrect_unique_quartets'] += len(unique[stage])
        print(f'{date}: {len(targets)} intended groups audited',flush=True)
    # Ensure the manually reviewed targets reproduce the prior intended coverage.
    assert len(targets)==38, f'Expected 38 reviewed intended templates; got {len(targets)}'
    results = []
    for cap in cutoffs:
        intended = [t for t in targets if (t['date'],t['category']) in retained[cap]]
        for stage in ('pair_clique','common_witness'):
            results.append(dict(max_dictionary_frequency=cap,stage=stage,**counts[cap,stage],
                intended_groups_retained=len(intended),
                intended_purple_retained=sum(t['color'].casefold()=='purple' for t in intended),
                closed_or_mixed_retained=sum(t['needs_closed_or_mixed'] for t in intended)))
    for (direction,witness),frequency in frequencies.items():
        witness_rows.append(dict(direction=direction,witness=witness,dictionary_frequency=frequency))
    for name,data in [('results',results),('intended_targets',targets),('witness_frequencies',witness_rows)]:
        with (root/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    payload = dict(boards=len(boards), dictionary_entries=len(lex.entries),
                   lexicon_identity=lex.identity,results=results,seconds=time.perf_counter()-started)
    (root/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    baseline = counts[None,'pair_clique']['incorrect_direction_occurrences']
    common_base = counts[None,'common_witness']['incorrect_direction_occurrences']
    total_closed = sum(t['needs_closed_or_mixed'] for t in targets)
    lines=['# Wiktionary witness filtering', '',
           f'Validation only: {len(boards)} boards. All entries remain available as spaced/closed '
           'compounds and substrings; no boundary-only restriction. No training or test evaluation.', '',
           'A pair clique means all six pairs share some completion in the same direction. '
           'A common witness means all four share the exact same completion. Counts below are '
           'quartet-direction occurrences, not unique quartets, guesses, or solves.', '',
           f'Unfiltered pair cliques: **{baseline:,}** incorrect occurrences. Common-witness '
           f'filter: **{common_base:,}**, removing **{baseline-common_base:,} '
           f'({100*(baseline-common_base)/baseline:.1f}%)**. The previous 47,629 figure already '
           'required a common witness; it is not a pair-clique baseline.', '',
           'Specificity is measured by the number of distinct compact lexical dictionary titles '
           'with a completion at that end, excluding an empty source. Frequencies are global '
           'dictionary statistics, computed without puzzle labels. Smaller counts are more specific. '
           'IDF weighting alone removes nothing; the following grid measures hard maximum-frequency '
           'cutoffs, equivalent to thresholding a monotonic IDF score. Cutoffs are exploratory, '
           'not independently validated operating points.', '',
           f'Intended coverage uses explicit, model-reviewed witness annotations: 38 groups, '
           f'32 Purple, {total_closed} requiring closed or mixed spellings. Unrelated witnesses '
           'do not count as retained intended coverage. Annotation is not human-verified.', '',
           '| Max dictionary frequency | Incorrect after common witness | Removed vs common witness | Intended retained | Purple retained | Closed/mixed retained |',
           '|---|---:|---:|---:|---:|---:|']
    for result in results:
        if result['stage']!='common_witness':continue
        cap=result['max_dictionary_frequency']; n=result['incorrect_direction_occurrences']
        lines.append(f"| {cap if cap is not None else 'Unlimited'} | {n:,} | {100*(1-n/common_base):.1f}% | {result['intended_groups_retained']}/38 | {result['intended_purple_retained']}/32 | {result['closed_or_mixed_retained']}/{total_closed} |")
    lines += ['', '## Intended completions and their dictionary frequencies', '']
    for target in sorted(targets,key=lambda t:t['dictionary_frequency'],reverse=True):
        lines.append(f"- {target['date']} {target['category']}: `{target['witness']}` ({target['direction']}), {target['dictionary_frequency']:,} titles; closed/mixed={target['needs_closed_or_mixed']}")
    lines += ['', 'These filters suppress phrase evidence only. They must not hard-prune entire '
              'candidate groups or harness paths, since valid groups need not have phrase evidence. '
              'Exact witness identity is also essential: pairwise vectors alone cannot establish '
              'quartet-wide consistency.', '',
              'Reproduce: `nix develop -c uv run python scripts/audit_wiktionary_filters.py`', '']
    (root/'report.md').write_text('\n'.join(lines))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=='__main__':main()
