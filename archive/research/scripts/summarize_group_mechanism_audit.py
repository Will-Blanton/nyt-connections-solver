"""Summarize label provenance, edge discrimination, and common phrase witnesses."""
import json
from pathlib import Path

import pandas as pd


ROOT=Path('experiments/group-mechanism-audit')


def intended_channels(mechanism,names):
    if mechanism in ('semantic_category','synonym','polysemy','named_entity_or_cultural'):
        return [n for n in names if n in ('semantic_cosine','sense_jaccard','rare_concept_jaccard')]
    if mechanism=='phrase_completion':
        return [n for n in names if n in ('shared_preceding_word','shared_following_word')]
    if mechanism=='word_containment':
        return [n for n in names if n in ('prefix_semantic_similarity','suffix_semantic_similarity')]
    if mechanism=='letter_edit':
        return [n for n in names if n.startswith(('deletion_','substitution_')) and n.endswith('_semantic')]
    if mechanism=='sound':
        return [n for n in names if n in ('homophones','phoneme_edit_similarity','shared_rhyme',
            'homophone_alternative_semantic_similarity','prefix_phonetic_similarity','suffix_phonetic_similarity')]
    if mechanism=='anagram':
        return [n for n in names if n in ('exact_anagram','character_multiset_overlap')]
    if mechanism=='orthographic_pattern':
        return [n for n in names if n in ('both_palindromes','both_initial_double',
            'both_consecutive_double_pairs','common_prefix','common_suffix','character_multiset_overlap')]
    return []


def main():
    labels=pd.read_csv(ROOT/'group_mechanisms.csv')
    evidence=pd.read_csv(ROOT/'group_edge_evidence.csv')
    phrases=pd.read_csv(ROOT/'phrase_witness_groups.csv')
    phrase_boards=pd.read_csv(ROOT/'phrase_witness_boards.csv')
    ranks=pd.read_csv(ROOT/'model_group_ranks.csv')
    standalone=pd.read_csv(ROOT/'standalone_feature_rankings.csv')
    metadata=json.loads((ROOT/'audit_metadata.json').read_text())
    validation=labels[labels.original_split=='val']
    assert len(labels)==4356 and len(validation)==484 and validation.reviewed.all()
    assert len(evidence)==4356*41
    assert not labels.duplicated(['date','color']).any()
    assert not evidence.duplicated(['date','color','channel']).any()
    metric_rows=[]
    for (split,mechanism,channel),rows in evidence.groupby(['original_split','primary_mechanism','channel']):
        for aggregation in ('mean','minimum','weakest_word'):
            metric_rows.append(dict(original_split=split,primary_mechanism=mechanism,channel=channel,
                aggregation=aggregation,groups=len(rows),
                mean_percentile=rows[f'{aggregation}_percentile'].mean(),
                mean_near_miss_percentile=rows[f'{aggregation}_near_miss_percentile'].mean(),
                strong_fraction=rows[f'{aggregation}_percentile'].ge(.95).mean(),
                all_pairs_fraction=rows.pair_support.eq(1).mean(),
                active_in_current_cv=bool(rows.active_in_current_cv.iloc[0])))
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(ROOT/'mechanism_feature_discrimination.csv',index=False)
    val_metrics=metrics[metrics.original_split=='val']
    names=evidence.channel.unique().tolist()
    family_stats=[]
    for mechanism,groups in validation.groupby('primary_mechanism'):
        channels=intended_channels(mechanism,names)
        rows=evidence[(evidence.original_split=='val') & evidence.primary_mechanism.eq(mechanism) & evidence.channel.isin(channels)]
        stats=dict(mechanism=mechanism,groups=len(groups),channels=channels)
        for aggregation in ('mean','minimum','weakest_word'):
            best=rows.groupby(['date','color'])[f'{aggregation}_percentile'].max()
            stats[aggregation]=dict(mean_best_percentile=float(best.mean()),
                any_channel_strong=int(best.ge(.95).sum()),fraction=float(best.ge(.95).mean()))
        family_stats.append(stats)
    pd.DataFrame([dict(mechanism=s['mechanism'],groups=s['groups'],channels=json.dumps(s['channels']),
        **{f'{a}_{k}':v for a in ('mean','minimum','weakest_word') for k,v in s[a].items()})
        for s in family_stats]).to_csv(ROOT/'intended_family_evidence.csv',index=False)
    # Current model ranks are from previous fixed-split controls, not the ongoing CV search.
    median_ranks=ranks.groupby(['date','color','variant'])['rank'].median().unstack('variant')
    gaps=[]
    for _,label in validation.iterrows():
        channels=intended_channels(label.primary_mechanism,names)
        rows=evidence[evidence.date.eq(label.date) & evidence.color.eq(label.color) & evidence.channel.isin(channels)]
        row={k:label[k] for k in ('date','color','category','words','primary_mechanism','subtype','review_notes')}
        for kind,mask in (('core',rows.active_in_current_cv),('optional',~rows.active_in_current_cv)):
            selected=rows[mask]
            if len(selected):
                best=selected.loc[selected.weakest_word_percentile.idxmax()]
                row[f'{kind}_channel']=best.channel
                row[f'{kind}_percentile']=float(best.weakest_word_percentile)
                row[f'{kind}_near_miss_percentile']=float(best.weakest_word_near_miss_percentile)
            else:
                row[f'{kind}_channel']=''
                row[f'{kind}_percentile']=None
                row[f'{kind}_near_miss_percentile']=None
            row[f'{kind}_midrank_among_incorrect']=(None if row[f'{kind}_percentile'] is None else
                1+1816*(1-row[f'{kind}_percentile']))
        row['baseline_median_rank']=float(median_ranks.loc[(label.date,label.color),'control'])
        row['amplified_median_rank']=float(median_ranks.loc[(label.date,label.color),'amplified'])
        row['strong_core_signal_but_both_models_miss']=(row['core_percentile'] is not None and
            row['core_percentile']>=.95 and min(row['baseline_median_rank'],row['amplified_median_rank'])>4)
        row['core_midrank_top4_but_models_outside_top20']=(row['core_midrank_among_incorrect'] is not None and
            row['core_midrank_among_incorrect']<=4.001 and
            min(row['baseline_median_rank'],row['amplified_median_rank'])>20)
        gaps.append(row)
    gap_frame=pd.DataFrame(gaps)
    gap_frame.to_csv(ROOT/'feature_model_gaps.csv',index=False)
    phrase_results={}
    for split in ('train','val'):
        totals={}
        for direction,rows in phrase_boards[phrase_boards.original_split.eq(split)].groupby('direction'):
            totals[direction]={k:int(rows[k].sum()) for k in ('gold_all_pairs','gold_shared','gold_inconsistent',
                'negative_all_pairs','negative_shared','negative_inconsistent')}
            totals[direction]['boards_with_negative_inconsistency']=int(rows.negative_inconsistent.gt(0).sum())
        selected_labels=labels[labels.original_split.eq(split) & labels.primary_mechanism.eq('phrase_completion')]
        keys=set(zip(selected_labels.date,selected_labels.color))
        scope=phrases[phrases.original_split.eq(split)]
        current=set(zip(scope[scope.direction.isin(['preceding','following']) & scope.common_witness_count.gt(0)].date,
                        scope[scope.direction.isin(['preceding','following']) & scope.common_witness_count.gt(0)].color))
        closed=set(zip(scope[scope.direction.str.startswith('closed') & scope.common_witness_count.gt(0)].date,
                       scope[scope.direction.str.startswith('closed') & scope.common_witness_count.gt(0)].color))
        phrase_results[split]=dict(totals=totals,phrase_groups=len(keys),
            phrase_groups_with_current_witness=len(keys & current),
            phrase_groups_with_closed_witness=len(keys & closed),
            phrase_groups_added_by_closed=len((keys & closed)-current))
    summary=dict(metadata=metadata['audit'],labels=dict(total=len(labels),reviewed=int(labels.reviewed.sum()),
        confidence=validation.label_confidence.value_counts().to_dict(),
        changed_primary_from_regex=int(validation.primary_mechanism.ne(validation.regex_primary_mechanism).sum())),
        validation_mechanisms=validation.primary_mechanism.value_counts().to_dict(),
        intended_family_evidence=family_stats,phrase_consistency=phrase_results,
        strong_core_model_misses=int(gap_frame.strong_core_signal_but_both_models_miss.sum()),
        core_midrank_top4_model_outside20=int(gap_frame.core_midrank_top4_but_models_outside_top20.sum()),
        strict_gap_colors=gap_frame[gap_frame.core_midrank_top4_but_models_outside_top20].color.value_counts().to_dict())
    standalone_summary=standalone.groupby(['original_split','channel','aggregation']).agg(
        expected_hit1=('expected_hit1','mean'),purple_expected_hit1=('purple_expected_hit1','mean'),
        blue_expected_hit1=('blue_expected_hit1','mean'),green_expected_hit1=('green_expected_hit1','mean'),
        yellow_expected_hit1=('yellow_expected_hit1','mean')).reset_index()
    standalone_summary.to_csv(ROOT/'standalone_feature_summary.csv',index=False)
    import altair as alt
    plot=gap_frame[gap_frame.core_percentile.notna()].copy()
    plot=plot.melt(id_vars=['date','color','category','words','primary_mechanism','core_channel','core_percentile'],
        value_vars=['baseline_median_rank','amplified_median_rank'],var_name='model',value_name='median_rank')
    alt.Chart(plot).mark_circle(size=55,opacity=.65).encode(
        x=alt.X('core_percentile:Q',title='Best relevant core channel percentile',scale=alt.Scale(domain=[0,1])),
        y=alt.Y('median_rank:Q',title='Median model rank (lower is better)',scale=alt.Scale(type='log')),
        color='primary_mechanism:N',shape='model:N',
        tooltip=['date','color','category','words','core_channel','core_percentile','model','median_rank']
    ).properties(width=800,height=440,title='Existing feature evidence versus previous fixed-split model ranking').interactive().save(
        ROOT/'evidence-vs-model.html')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Group mechanisms and similarity evidence', '',
        'New annotations cover 4,356 groups from 1,089 original training/validation puzzles. '
        'The 484 original validation groups were individually model-reviewed by three reviewers; '
        '3,872 training groups retain regex/fallback labels and are explicitly marked unreviewed. '
        'The old test set was excluded. Original source data, notebooks and training features were not changed.', '',
        '## Annotation quality and scope', '',
        f"Model review changed {summary['labels']['changed_primary_from_regex']}/484 primary regex/fallback assignments. "
        'This is disagreement with the draft labels, not measured annotation accuracy. '
        'Model-reviewed labels are not human ground truth. Multiple mechanisms can apply; '
        'secondary labels, subtype, confidence and notes preserve those distinctions. '
        'Training fallbacks are broad hypotheses, not a complete verified taxonomy. '
        'All unreviewed rows are flagged `needs_review`.', '',
        'Shared-phrase constructions can overlap ordinary categories: clear compounds such as '
        'butter/pocket/etc. + knife are phrase-first, while ordinary category membership is '
        'semantic-first with phrase secondary when relevant. Named entities versus familiar '
        'general categories also require judgment. These labels are analysis only and never '
        'supplied to training, feature construction or the harness.', '',
        '| Primary mechanism | Purple | Blue | Green | Yellow |', '|---|---:|---:|---:|---:|']
    for mechanism in sorted(validation.primary_mechanism.unique()):
        lines.append('| '+mechanism+' | '+' | '.join(str(len(validation[validation.color.eq(c) & validation.primary_mechanism.eq(mechanism)]))
            for c in ('purple','blue','green','yellow'))+' |')
    lines+=['', '## What the feature audit measures', '',
        'CPU-only, two threads, cached frozen embeddings; no model inference or second GPU job. '
        'All 41 existing scalar channels were inspected: 15 core channels used by the current CV '
        'models plus six optional sound and twenty optional letter-edit channels. '
        'Optional channels are audited evidence, not inputs those current models already possess.', '',
        'For every true quartet, compare each channel aggregated by mean, minimum, or weakest-word '
        'mean against all 1,816 incorrect quartets on the same board. Separately compare against '
        'the 192 three-of-four near misses. Percentiles give ties half credit: .5 is uninformative, '
        '.95 beats/ties approximately 95% of competing candidates and does not imply Top4. '
        'Nonzero values alone do not establish useful discrimination. These are univariate '
        'diagnostics, not estimates of multichannel-model accuracy or causal ablations.', '',
        '## Evidence in the intended feature families', '',
        'The table counts true validation groups with at least one relevant channel at or above '
        'the 95th percentile. Channel choice is post-hoc per group, so this is an optimistic '
        'diagnostic, not an executable prediction rule. Different mechanisms have different '
        'numbers of candidate channels. Sound/letter-edit rows include optional features.', '',
        '| Mechanism | Groups | Mean strong | Minimum strong | Weakest-word strong |',
        '|---|---:|---:|---:|---:|']
    for stats in family_stats:
        lines.append('| '+stats['mechanism']+' | '+str(stats['groups'])+' | '+' | '.join(
            f"{stats[a]['any_channel_strong']}/{stats['groups']}" for a in ('mean','minimum','weakest_word'))+' |')
    lines+=['','## Standalone feature rankings','',
        'Rank all 1,820 quartets by just one channel, with no training. Hit@1 below is the '
        'probability of a correct first choice if ties at the maximum are broken uniformly. '
        'Thus an all-zero channel receives chance credit (4/1,820), rather than an artificial '
        'win or loss from candidate order. These are diagnostic channel rankings, not harness solves. '
        'The table shows the ten strongest channel/aggregation combinations, selected post-hoc '
        'on validation; this selection makes them optimistic as recommendations.', '',
        '| Channel | Aggregation | Expected Hit@1 | Purple contribution |', '|---|---|---:|---:|']
    for _,row in standalone_summary[standalone_summary.original_split.eq('val')].sort_values(
            'expected_hit1',ascending=False).head(10).iterrows():
        lines.append(f'| {row.channel} | {row.aggregation} | {row.expected_hit1:.2%} | {row.purple_expected_hit1:.2%} |')
    lines+=['', '## Strongest individual channels by mechanism', '',
        'Channels below are selected post-hoc within the intended family, using mean validation '
        'percentile under weakest-word aggregation; small group counts and multiple comparisons '
        'limit interpretation.', '',
        '| Mechanism | Channel | Current CV input? | Mean percentile | Versus near misses |',
        '|---|---|---|---:|---:|']
    for stats in family_stats:
        rows=val_metrics[val_metrics.primary_mechanism.eq(stats['mechanism']) &
            val_metrics.channel.isin(stats['channels']) & val_metrics.aggregation.eq('weakest_word')]
        if len(rows):
            row=rows.loc[rows.mean_percentile.idxmax()]
            lines.append(f"| {stats['mechanism']} | {row.channel} | {'yes' if row.active_in_current_cv else 'optional'} | "
                         f"{row.mean_percentile:.3f} | {row.mean_near_miss_percentile:.3f} |")
    lines+=['', '## Shared phrase witnesses', '',
        'Check whether all six pairwise matches use one common completion. A clique of '
        'pairwise phrase edges need not logically have a global witness. Actual corpus '
        'counts below test how often that issue arises here. Closed compounds are audited '
        'separately and are not among the 41 feature channels inspected.', '',
        '| Split | Direction | True all-pair quartets | True shared witness | Incorrect all-pair quartets | Incorrect without shared witness |',
        '|---|---|---:|---:|---:|---:|']
    for split,results in phrase_results.items():
        for direction,sums in results['totals'].items():
            lines.append(f"| {split} | {direction} | {sums['gold_all_pairs']} | {sums['gold_shared']} | "
                         f"{sums['negative_all_pairs']} | {sums['negative_inconsistent']} |")
    result=phrase_results['val']
    lines+=['',f"Among {result['phrase_groups']} model-reviewed validation phrase groups, the current spaced-phrase "
        f"index has a common completion for {result['phrase_groups_with_current_witness']}; closed compounds have one "
        f"for {result['phrase_groups_with_closed_witness']} and add {result['phrase_groups_added_by_closed']} distinct groups. "
        'An attested completion can be incidental rather than the intended explanation.', '',
        '## Strong core evidence missed by the previous model', '',
        f"{summary['strong_core_model_misses']} validation groups have a relevant core channel at the "
        '95th percentile under weakest-word aggregation but median true-group rank below Top4 '
        'for both previous control and amplified models. These ranks reuse the completed '
        'three-seed fixed-split experiment; they are not results from the ongoing CV search. '
        'High percentile does not prove that all witnesses are consistent or that Top4 is attainable.', '',
        f"A stricter check finds {summary['core_midrank_top4_model_outside20']} groups whose best relevant "
        'core channel has a tie-aware average rank of at most four against incorrect quartets, '
        'while both models have median rank worse than twenty. This diagnostic excludes the '
        'other three true groups from the single-channel competitor pool. It is stronger '
        'evidence of an integration opportunity, but still does not prove a causal model failure.', '',
        '| Date | Color | Category | Core channel | Percentile | Baseline median rank | Amplified median rank |',
        '|---|---|---|---|---:|---:|---:|']
    candidates=gap_frame[gap_frame.strong_core_signal_but_both_models_miss].sort_values(
        ['core_percentile','amplified_median_rank'],ascending=[False,False]).head(15)
    for _,row in candidates.iterrows():
        lines.append(f"| {row.date} | {row.color} | {row.category.replace('|','/')} | {row.core_channel} | "
            f"{row.core_percentile:.3f} | {row.baseline_median_rank:g} | {row.amplified_median_rank:g} |")
    lines+=['','## Conclusions supported by this audit','',
        '1. **There is useful deterministic signal.** Mean semantic cosine alone reaches '
        '16.53% expected first-choice accuracy on these validation boards; minimum rare-concept '
        'overlap reaches 15.71%, against 0.22% chance. Their successes may overlap and '
        'must not be added. Neither statistic measures complete-puzzle solves.',
        '2. **No aggregation dominates.** Semantic mean beats semantic weakest-word pooling '
        '(16.53% versus 10.74% standalone Hit@1). Phrase-family evidence clears the 95th '
        'percentile for 27/68 groups under mean, 13/68 under weakest-word and 6/68 under '
        'minimum. Missing pair evidence is common; this argues for supplying multiple '
        'summaries rather than enforcing one hard coherence rule.',
        '3. **Phrase-resource coverage is a concrete gap.** Only 6/68 phrase groups have a '
        'shared current-index completion. Closed compounds add seven distinct groups, '
        'but previous training experiments with those features did not establish a gain. '
        'Coverage improvements still need a matched model experiment.',
        '4. **Witness consistency is useful but narrower.** Of 108 incorrect validation '
        'quartet–direction occurrences with all-six-pair spaced-phrase support, 21 have '
        'no global completion witness. None of the eight supported true quartet–direction '
        'occurrences lose support in validation; three training occurrences do. This is '
        'a candidate evidence feature, not a justified hard pruning rule.',
        '5. **There are integration opportunities as well as missing information.** '
        f"The stricter core-evidence/model-rank check identifies {summary['core_midrank_top4_model_outside20']} "
        'groups, including ordinary semantic cases. This supports trying explicit group '
        'summary features at scoring time, not only adding more edge types. The relevant '
        'channel is selected post-hoc here, so the model must still learn when to trust it.',
        '6. **Optional sound/edit signals are not absent.** At least one intended-family '
        'channel has mean evidence at or above the 95th percentile for 11/16 sound groups '
        'and 9/11 letter-edit groups. Those are optimistic per-group choices among multiple '
        'channels and are not proof of a solve benefit. These optional channels are not '
        'enabled in the ongoing core-15 CV sweep.', '',
        'The strongest bounded follow-up is the proposed separate group-summary scoring '
        'channel (retain mean and coherence information), followed by a common-phrase-witness '
        'feature and targeted phrase-resource coverage work. Avoid treating a label taxonomy '
        'or a high feature percentile as an oracle for group correctness.', '']
    lines+=['', '## Files and reproduction', '',
        '- `group_mechanisms.csv`: new annotation dataset, including words, provenance, confidence, '
        'secondary mechanisms, review status and CV fold.',
        '- `group_edge_evidence.csv`: all per-group/per-channel values and percentiles.',
        '- `mechanism_feature_discrimination.csv`: per-mechanism channel summaries.',
        '- `intended_family_evidence.csv`: descriptive family coverage by aggregation.',
        '- `phrase_witness_groups.csv` and `phrase_witness_boards.csv`: actual completions and inconsistencies.',
        '- `feature_model_gaps.csv`: relevant core/optional evidence beside previous model ranks.',
        '- `standalone_feature_rankings.csv` and `standalone_feature_summary.csv`: '
        'tie-aware single-channel first-choice performance.', '',
        '- `evidence-vs-model.html`: interactive core-evidence versus model-rank diagnostic.', '',
        'Reproduce with `nix develop -c uv run python scripts/audit_group_mechanisms.py`, then '
        '`nix develop -c uv run python scripts/summarize_group_mechanism_audit.py`. '
        'The three `*-review.csv` files preserve reviewer decisions; rerunning regex does not overwrite them. '
        'The ongoing five-fold sweep is unchanged.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
