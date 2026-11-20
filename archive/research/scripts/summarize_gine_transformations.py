"""Matched transformation ablations, color recovery and learning-budget checks."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import altair as alt
from summarize_gine_group_contrastive import purple_status

ROOT=Path('experiments/gine-transformations')
VARIANTS=('control','edges-text','edges-fused','edges-constant')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=json.loads((ROOT/'runs.json').read_text())
    assert len(entries)==12
    inventory=json.loads(Path('experiments/gine-phonetics/mechanism-inventory.json').read_text())
    targets={r['date'] for r in inventory['groups_inventory'] if r['color']=='purple' and r['mechanism']=='embedded_fragment'}
    coverage=json.loads((ROOT/'target-coverage.json').read_text())
    concrete={r['date'] for r in coverage['records'] if r['all_targets_present']}
    runs=[]; games_by_key={}; curves=[]
    for entry in entries:
        result=json.loads(Path(entry['result']).read_text())
        harness=json.loads(Path(entry['harness']).read_text())
        assert result['status']=='complete' and result['training']['updates_ran']==30000
        assert harness['split']=='val' and len(harness['games'])==121
        games=harness['games']; history=result['training']['history']
        assert len(history)==60
        curves.extend(dict(seed=entry['seed'],variant=entry['variant'],update=h['update'],
            validation_hit1=h['val_s4_hit1'],train_loss=h['train_loss']) for h in history)
        games_by_key[entry['variant'],entry['seed']]={g['date']:g for g in games}
        runs.append(dict(**entry,parameters=result['parameters'],ranking=result['training']['validation'],
            checkpoint=result['checkpoint'],best_update=result['training']['best_update'],
            best_first_15k=max(h['val_s4_hit1'] for h in history if h['update']<=15000),
            solves=sum(g['won'] for g in games),first_correct=sum(g['first_guess_correct'] for g in games),
            colors=dict(Counter(c for g in games for c in g['solved_colors'])),
            purple=dict(Counter(purple_status(g) for g in games)),
            embedded_purple=dict(Counter(purple_status(g) for g in games if g['date'] in targets)),
            concrete_purple=dict(Counter(purple_status(g) for g in games if g['date'] in concrete)),
            train_seconds=sum(h['train_seconds'] for h in history),
            validation_seconds=sum(h['validation_seconds'] for h in history),
            harness_seconds=mean(g['seconds'] for g in games)))
    aggregate={}; paired={}
    audit=json.loads((ROOT/'probability-audit.json').read_text()) if (ROOT/'probability-audit.json').exists() else None
    for v in VARIANTS:
        rows=[r for r in runs if r['variant']==v]
        assert len(rows)==3
        aggregate[v]=dict(
            ranking={s:{k:mean(r['ranking'][s][k] for r in rows) for k in ('hit1','mrr','hit4')} for s in ('s4','s3','s2')},
            solves=sum(r['solves'] for r in rows),first_correct=sum(r['first_correct'] for r in rows),
            colors={c:sum(r['colors'].get(c,0) for r in rows) for c in ('yellow','green','blue','purple')},
            purple={k:sum(r['purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            embedded_purple={k:sum(r['embedded_purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            concrete_purple={k:sum(r['concrete_purple'].get(k,0) for r in rows) for k in ('selected','forced','missed')},
            trainable_parameters=rows[0]['parameters']['trainable'],
            training_seconds=mean(r['train_seconds'] for r in rows),
            validation_seconds=mean(r['validation_seconds'] for r in rows),
            harness_seconds=mean(r['harness_seconds'] for r in rows),
            best_first_15k=mean(r['best_first_15k'] for r in rows))
        if audit:
            boards=[r for r in audit['records'] if r['variant']==v]
            assert len(boards)==363
            assert abs(mean(1 in r['gold_ranks'] for r in boards)-aggregate[v]['ranking']['s4']['hit1'])<1e-8
            aggregate[v]['color_top4']={c:mean(r['gold_ranks'][i]<=4 for r in boards) for i,c in enumerate(audit['colors'])}
        if v!='control':
            count=Counter()
            for seed in manifest['seeds']:
                a,b=games_by_key['control',seed],games_by_key[v,seed]
                assert a.keys()==b.keys()
                for date in a:
                    count['solve_gain']+=b[date]['won'] and not a[date]['won']
                    count['solve_loss']+=a[date]['won'] and not b[date]['won']
                    p,q=purple_status(a[date]),purple_status(b[date])
                    count['purple_gain']+=p=='missed' and q!='missed'
                    count['purple_loss']+=q=='missed' and p!='missed'
            paired[v]=dict(count)
    summary=dict(manifest=manifest,aggregate=aggregate,runs=runs,paired=paired,
        coverage={s:json.loads((ROOT/f'coverage-{s}.json').read_text()) for s in ('train','val')},
        embedded_purple_dates=sorted(targets),concrete_purple_dates=sorted(concrete),
        concrete_recovery=[dict(variant=v,seed=seed,date=date,status=purple_status(games_by_key[v,seed][date]))
            for v in VARIANTS for seed in manifest['seeds'] for date in sorted(concrete)])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    data=alt.Data(values=curves)
    hit=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('validation_hit1:Q',axis=alt.Axis(format='.0%')),
        color='variant:N',tooltip=['seed:O','variant:N','update:Q','validation_hit1:Q']).properties(width=330,height=220).facet(column='seed:O')
    loss=alt.Chart(data).mark_line().encode(x='update:Q',y=alt.Y('train_loss:Q',scale=alt.Scale(type='log')),
        color='variant:N',tooltip=['seed:O','variant:N','update:Q','train_loss:Q']).properties(width=330,height=220).facet(column='seed:O')
    alt.vconcat(hit,loss).save(ROOT/'learning-curves.html')
    lines=['# Learned transformation edges and node representations','',
        'Keep the existing baseline. Learned transformation matching fits training '
        'data readily but reduces aggregate validation ranking and harness solves '
        'under these fixed settings. Shared-vector nodes are the strongest new '
        'variant, but still below the control.','',
        'Validation only: 121 puzzles, three matched seeds (10/11/12), four arms, '
        '30,000 updates per run. Retrain the control at the same budget. '
        'Repeated seeds reuse the same validation puzzles and prior selection history; '
        '363 appearances are not 363 independent puzzles. No notebooks or test-set evaluation.','',
        '## Features and architecture','',
        'Use the original word plus proper contiguous substrings of at least three '
        'letters that are English WordNet words. Only ASCII single words up to 32 '
        'characters receive substring extraction. The original remains available '
        'for all inputs. Cache frozen Qwen embeddings for every alternative. '
        'Keep eight slots, original first, then round-robin prefix/suffix/internal '
        'alternatives ordered longest first within each type. Padding is explicitly masked.','',
        'Each alternative carries its extraction type, left/right removed-character '
        'counts divided by a fixed 32, normalized start/end span and retained fraction. '
        'Fixed-scale offsets preserve exact matches such as removing one leading '
        'character across different word lengths. No puzzle labels enter features.','',
        'Learn a 64-dimensional alternative representation. For each unordered word '
        'pair, compare all valid alternative pairs through a shared MLP using '
        'absolute embedding differences, products, position differences, and exact '
        'type/left-offset/right-offset compatibility. Elementwise max yields a '
        '16-dimensional edge vector; concatenate with the original 15 scalar edges. '
        'Edges are symmetric with zero diagonal and computed once per state before '
        'candidate extraction. New GINE edge-projection columns start at zero; '
        'the learned pair network begins receiving gradients once those columns move.','',
        'The text arm keeps the original active text projection. The fused arm '
        'combines text projection with sigmoid-gated mean and max transformation '
        'summaries through a learned parallel fusion layer. Edges retain the '
        'unpooled alternatives. The constant arm starts all nodes from one shared '
        'learned vector, disables the original text projection, and relies on '
        'edge evidence and subsequent GINE messages. The control ignores the '
        'extra payload and supplies zero learned edge vectors.','',
        'This remains the existing candidate-local GINE, with two width-128 '
        'layers and readout 128. It does not add global message passing or '
        'relation nodes. Learned edges depend on the two words, not on a '
        'joint explanation shared by four words. Original alternatives are '
        'included, so a gain could reflect a learned original-word pair scorer '
        'as well as substring information; an original-only learned-edge '
        'control would be needed to separate those explanations.','',
        '## Training and evaluation','',
        'Ordinary probability-mass loss; no floor penalty or auxiliary loss. '
        'Batch 16, fixed LR 0.0003016685, weight decay 0.0000950089, dropout zero. '
        'Frozen Qwen vectors; active projection/GINE/transformation networks '
        'trainable as appropriate to each arm. Matched initialization of shared '
        'weights, data permutations and balanced schedules (10,000 updates '
        'per 8/12/16-word state size, roughly 165 full-board training passes). Full 30,000 updates, no pruning or early '
        'stopping; evaluate every 500 updates and restore best s4 Hit@1 with '
        'MRR tie-break. All arms use four CPU threads. New modules increase '
        'capacity; parameter counts below are not a capacity-matched ablation.','',
        'Normal harness: beam 20, full-board score sums, four mistakes, shuffle '
        'seed zero and public-feedback pruning. No harness-based checkpoint '
        'selection. Streamlit and harness use registered self-contained '
        'TransformationGINE checkpoints and reconstructable feature metadata.','',
        'Two incomplete startup controls were restarted before any learned '
        'arm completed: CPU batching oversubscription was fixed, then absolute '
        'offset matching was added. Logs and recovery records are retained. '
        'All reported runs use the final feature version and full matched budget.','',
        '## Coverage','', '| Split | Unique words | Proper alternatives available | Truncated at eight |', '|---|---:|---:|---:|']
    for s,c in summary['coverage'].items():
        lines.append(f"| {s} | {c['words']} | {c['with_proper_transformations']} ({c['with_proper_transformations']/c['words']:.2%}) | {c['truncated_words']} |")
    lines+=['','## Aggregate results','', '| Metric | Control | Learned edges + text | Learned edges + fused nodes | Learned edges + constant nodes |','|---|---:|---:|---:|---:|']
    for s in ('s4','s3','s2'):
        for k in ('hit1','mrr','hit4'):
            lines.append(f'| {s} {k} | '+' | '.join(f"{aggregate[v]['ranking'][s][k]:.4f}" if k=='mrr' else f"{aggregate[v]['ranking'][s][k]:.2%}" for v in VARIANTS)+' |')
    for k in ('solves','first_correct'):
        lines.append(f'| Harness {k} | '+' | '.join(f"{aggregate[v][k]}/363 ({aggregate[v][k]/363:.2%})" for v in VARIANTS)+' |')
    for c in ('yellow','green','blue','purple'):
        lines.append(f'| {c} recovered | '+' | '.join(str(aggregate[v]['colors'][c])+'/363' for v in VARIANTS)+' |')
    for k in ('selected','forced'):
        lines.append(f'| Purple {k} | '+' | '.join(str(aggregate[v]['purple'][k])+'/363' for v in VARIANTS)+' |')
    denominator=len(targets)*3
    lines.append('| Embedded-fragment Purple recovered | '+' | '.join(str(denominator-aggregate[v]['embedded_purple']['missed'])+f'/{denominator}' for v in VARIANTS)+' |')
    lines.append('| Embedded-fragment Purple directly selected | '+' | '.join(str(aggregate[v]['embedded_purple']['selected'])+f'/{denominator}' for v in VARIANTS)+' |')
    denominator=len(concrete)*3
    lines.append('| Concrete covered Purple recovered | '+' | '.join(str(denominator-aggregate[v]['concrete_purple']['missed'])+f'/{denominator}' for v in VARIANTS)+' |')
    lines.append('| Concrete covered Purple directly selected | '+' | '.join(str(aggregate[v]['concrete_purple']['selected'])+f'/{denominator}' for v in VARIANTS)+' |')
    lines+=['','Embedded-fragment mechanism labels are the earlier regex inventory, '
        'used only for audit. They are approximate and do not establish coverage '
        'of every intended transformation. Concrete coverage separately checks '
        'four manually specified groups: rock-band prefixes (cream/journey/kiss/rush), '
        'location suffixes (place/point/site/spot), zero prefixes (jack/nada/squat/zip), '
        'and slush prefixes (goo/mush/paste/pulp). All sixteen intended readings '
        'are present within the eight-slot cap. These labels never enter training '
        'or feature construction. Per-date recovery is recorded in summary.json.','', '| Variant | Trainable parameters | Mean train minutes | Mean validation minutes | Mean harness seconds/puzzle |', '|---|---:|---:|---:|---:|']
    for v,a in aggregate.items():
        lines.append(f"| {v} | {a['trainable_parameters']:,} | {a['training_seconds']/60:.2f} | {a['validation_seconds']/60:.2f} | {a['harness_seconds']:.3f} |")
    lines+=['','Timings exclude initial feature preparation and cold embedding inference. '
        'Harness timings include variable search work, so they are not model-only '
        'microbenchmarks. Cached lexical vectors are reused across subsets and runs.','',
        '## Per seed and update budget','', '| Seed | Variant | s4 Hit@1 | Best through 15k | Best update | Solves | Purple direct / forced |', '|---|---|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ranking']['s4']['hit1']:.2%} | {r['best_first_15k']:.2%} | {r['best_update']} | {r['solves']}/121 | {r['purple'].get('selected',0)} / {r['purple'].get('forced',0)} |")
    lines+=['','## Paired differences from control','']
    for v,p in paired.items():
        lines.append(f"- {v}: {p['solve_gain']} new solves, {p['solve_loss']} lost; {p['purple_gain']} new Purple recoveries, {p['purple_loss']} lost.")
    lines+=['','## Concrete intended-mechanism recovery','',
        'Each cell is direct selections / forced final-group recoveries across '
        'three seeds. These four cases were selected for lexical coverage, '
        'not because the learned model solved them.','',
        '| Date and mechanism | Control | Edges + text | Edges + fused | Edges + constant |',
        '|---|---:|---:|---:|---:|']
    for row in coverage['records']:
        values=[]
        for v in VARIANTS:
            counts=Counter(purple_status(games_by_key[v,seed][row['date']]) for seed in manifest['seeds'])
            values.append(f"{counts['selected']} / {counts['forced']}")
        lines.append(f"| {row['date']} {row['category']} | "+' | '.join(values)+' |')
    lines+=['','## Interpretation','',
        'The ordinary control wins aggregate 16-word Hit@1, Hit@4, MRR and harness '
        'solve rate. Learned edges with text nodes lose solves in all three seeds '
        '(35/33/28 to 17/20/20). Fusion improves the new-edge aggregate but remains '
        'below control (23/18/29). Shared-vector nodes are stronger still '
        '(23/25/29), but do not recover the control (96 to 77 total solves). '
        'They match aggregate 12-word Hit@1 and slightly exceed 8-word Hit@1, '
        'while trailing on the most important 16-word state and full solves.','',
        'This does not establish that original text node embeddings are inherently '
        'misleading. Constant nodes remove about 264,000 trainable text-projection '
        'parameters, so regularization/capacity changes are entangled with input '
        'choice. Their learned edges still include original-word text embeddings. '
        'None of these runs removes ordinary semantic information from the '
        'system entirely. A shared-node model with only the original scalar '
        'edges would be a clean next isolation before further widening networks.','',
        'All runs completed 30,000 updates and sixty validation observations. '
        'Four selected checkpoints occur after 15,000 updates; three runs have '
        'strictly better Hit@1 in the second half. Thus the extra budget mattered, '
        'but did not reverse the aggregate result. Final training losses are '
        'small and validation often peaks much earlier, consistent with '
        'overfitting rather than a model unable to fit the objective. These '
        'are fixed existing optimizer settings, not tuned upper bounds for '
        'the new modules. More updates alone are not the clearest next lever.','',
        'There is one concrete direct Purple gain: fused seed 10 selects '
        'CREAMSICLE/JOURNEYMAN/KISSCAM/RUSHMORE in the rock-band-prefix puzzle '
        'that all three controls miss. It does not repeat in the other seeds. '
        'Overall Purple recovery falls from 106 to 70/78/88 and direct selections '
        'from 59 to 46/45/44. All sixteen intended readings are available in the '
        'four coverage-audited cases, yet location and slush groups remain '
        'unsolved in every arm. Availability is not reliable use of the intended '
        'mechanism, and this experiment does not justify promoting the features.','',
        'Twenty targeted regression tests pass: lexical extraction/cache layout, '
        'absolute-offset matching across lengths, padding masks, symmetric edges, '
        'transformation-slot and word permutation invariance, end-to-end edge '
        'and fusion/shared-node gradients, checkpoint reload, control equivalence '
        'and existing diagnostics. All twelve saved checkpoints ran through '
        'the normal validation harness; an independent audit of 1,452 '
        'board/model evaluations reproduces every full-board Hit@1. No notebooks '
        'or test-set evaluation changed.','']
    if (ROOT/'latency.json').exists():
        latency=json.loads((ROOT/'latency.json').read_text())
        lines+=['','## Warm model-only latency','',latency['method'],'','| Variant | Median milliseconds |','|---|---:|']
        for r in latency['results']:
            lines.append(f"| {r['variant']} | {r['median_ms']:.3f} |")
    lines+=['','## Artifacts','',
        'Full metrics, parameter counts, checkpoint paths and timing are in '
        '`experiments/gine-transformations/summary.json`. Per-run JSON retains '
        'every validation observation and loss interval. Normal harness JSON '
        'retains all guesses and feedback. `probability-audit.json` independently '
        'recomputes full-board ranks for every selected checkpoint. Interactive '
        'loss/validation curves are in `learning-curves.html`.','',
        'Runner: `nix develop -c uv run python scripts/run_gine_transformations.py`. '
        'Audit: `nix develop -c uv run python scripts/audit_gine_probability_floor.py '
        '--results experiments/gine-transformations`. Summary: '
        '`nix develop -c uv run python scripts/summarize_gine_transformations.py`.','']
    Path('experiments/gine-transformations.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()
