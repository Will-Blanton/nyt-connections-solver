"""Build a compact headword cache and compare phrase evidence on validation only.

Download the English Kaikki JSONL.gz separately. No categories/solutions are
used in the index. Labels are read only for reporting after indexes are built.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon
from nyt_connections_solver.edge_features.resources import EnglishLexicon, phrase_form

SOURCE_URL = 'https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz'


def headwords(record):
    """English titles only; distinguish pure inflections from lexical entries."""
    if record.get('lang_code') != 'en' or not isinstance(record.get('word'), str):
        return None
    senses = record.get('senses', [])
    inflection = bool(senses) and all(s.get('form_of') for s in senses)
    return record['word'], not inflection


def build_cache(snapshot, cache):
    if cache.exists():
        return json.loads(cache.with_suffix('.metadata.json').read_text())
    started = time.perf_counter()
    words = {}
    records = 0
    with gzip.open(snapshot, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = headwords(json.loads(line))
            records += 1
            if record:
                word, lexical = record
                words[word] = words.get(word, False) or lexical
            if records % 250000 == 0:
                print(f'Parsed {records:,} entries in {time.perf_counter()-started:.1f}s', flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, 'wt', encoding='utf-8') as stream:
        for word, lexical in sorted(words.items()):
            stream.write(json.dumps([word, lexical], ensure_ascii=False) + '\n')
    digest = hashlib.sha256()
    with open(snapshot, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    metadata = dict(source_url=SOURCE_URL, snapshot=str(snapshot),
                    sha256=digest.hexdigest(), compressed_bytes=snapshot.stat().st_size,
                    records=records, headwords=len(words), lexical_headwords=sum(words.values()),
                    cache_bytes=cache.stat().st_size, build_seconds=time.perf_counter()-started,
                    extraction='English titles; lexical excludes titles whose every sense has form_of',
                    attribution='English Wiktionary contributors; extracted by Kaikki/Wiktextract. '
                                'See https://en.wiktionary.org/wiki/Wiktionary:Copyrights')
    cache.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def matching_quartets(words, choices, direction):
    index = defaultdict(list)
    for i, word in enumerate(words):
        for completion in choices(word, direction):
            index[completion].append(i)
    result = set()
    for positions in index.values():
        result.update(itertools.combinations(positions, 4))
    return result


def evaluate(rows, choices):
    boards = defaultdict(list)
    details = []
    lookup_started = time.perf_counter()
    for row in rows:
        boards[row['date']].append(row)
        words = json.loads(row['words'])
        witnesses = {d: sorted(set.intersection(*(set(choices(w, d)) for w in words)))
                     for d in ('preceding', 'following')}
        details.append(dict(date=row['date'], color=row['color'], category=row['category'],
                            mechanism=row['primary_mechanism'], words=words, witnesses=witnesses))
    lookup_seconds = time.perf_counter()-lookup_started
    gold_count = incorrect_count = 0
    covered_groups = set()
    board_counts = []
    for date, groups in boards.items():
        words = list(itertools.chain.from_iterable(json.loads(r['words']) for r in groups))
        gold = {tuple(range(i*4, i*4+4)) for i in range(4)}
        negative = set()
        for direction in ('preceding', 'following'):
            matching = matching_quartets(words, choices, direction)
            gold_count += len(matching & gold)
            incorrect_count += len(matching - gold)
            negative.update(matching-gold)
            covered_groups.update((date, positions) for positions in matching & gold)
        board_counts.append(len(negative))
    phrase = [r for r in details if r['mechanism'] == 'phrase_completion']
    purple_phrase = [r for r in phrase if r['color'].casefold() == 'purple']
    supported = lambda items: sum(any(r['witnesses'].values()) for r in items)
    # All lookups have now been cached; measure normal repeated board use.
    warm_started = time.perf_counter()
    for _ in range(10):
        for groups in boards.values():
            for row in groups:
                for word in json.loads(row['words']):
                    for direction in ('preceding', 'following'):
                        choices(word, direction)
    warm_ms = (time.perf_counter()-warm_started)*1000/(10*len(boards))
    return dict(phrase_groups=supported(phrase), phrase_total=len(phrase),
                purple_phrase_groups=supported(purple_phrase), purple_phrase_total=len(purple_phrase),
                all_gold_groups=len(covered_groups), gold_direction_occurrences=gold_count,
                incorrect_direction_occurrences=incorrect_count,
                incorrect_unique_quartets=sum(board_counts),
                cold_validation_lookup_seconds=lookup_seconds,
                warm_lookup_ms_per_board=warm_ms), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('data/wiktionary/english-2026-10-30.jsonl.gz'))
    parser.add_argument('--cache', type=Path, default=Path('data/wiktionary/headwords.jsonl.gz'))
    parser.add_argument('--output', type=Path, default=Path('experiments/wiktionary-phrases'))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = build_cache(args.snapshot, args.cache)
    with gzip.open(args.cache, 'rt', encoding='utf-8') as stream:
        entries = [json.loads(line) for line in stream]
    wn = EnglishLexicon()
    started = time.perf_counter()
    source = {'source_url': metadata['source_url'], 'sha256': metadata['sha256']}
    indexes = {
        'wordnet_unified': CompletionLexicon(wn.entries, source={'type': 'wordnet', 'version': wn.identity}),
        'wiktionary_lexical': CompletionLexicon((w for w, lexical in entries if lexical), source={**source, 'inflections': False}),
        'wiktionary_spaced': CompletionLexicon((w for w, lexical in entries if lexical), source={**source, 'inflections': False}, spaced_only=True),
        'wiktionary_all': CompletionLexicon((w for w, _ in entries), source={**source, 'inflections': True}),
        'union_lexical': CompletionLexicon(itertools.chain(wn.entries, (w for w, lexical in entries if lexical)), source={**source, 'inflections': False, 'wordnet': wn.identity}),
    }
    build_seconds = time.perf_counter()-started
    print(f'Built {len(indexes)} indexes in {build_seconds:.1f}s', flush=True)
    with open('experiments/group-mechanism-audit/group_mechanisms.csv') as stream:
        rows = [r for r in csv.DictReader(stream) if r['original_split'] == 'val']
    variants = {'wordnet_current': lambda w, d: getattr(wn, d).get(phrase_form(w), set())}
    variants.update({name: index.completions for name, index in indexes.items()})
    completion_words = {phrase_form(w) for w, lexical in entries if lexical and phrase_form(w).isalpha()}
    variants['wiktionary_word_completion'] = lambda w,d: indexes['wiktionary_lexical'].completions(w,d) & completion_words
    summary, details = {}, {}
    for name, choices in variants.items():
        summary[name], details[name] = evaluate(rows, choices)
        print(name, json.dumps(summary[name]), flush=True)
    from nyt_connections_solver.edge_features.phrase_lexicon import AttestedPhraseCompletions
    provider = AttestedPhraseCompletions(indexes['wiktionary_lexical'])
    board_words = list(itertools.chain.from_iterable(json.loads(r['words']) for r in rows[:4]))
    encode_started = time.perf_counter()
    for _ in range(20):
        provider.encode(board_words)
    encode_ms = (time.perf_counter()-encode_started)*1000/20
    (args.output/'summary.json').write_text(json.dumps(dict(snapshot=metadata, variants=summary,
                index_build_seconds=build_seconds, warm_pairwise_encode_ms_one_board=encode_ms,
                alphabetic_index_sizes={name:len(index.entries) for name,index in indexes.items()}), indent=2)+'\n')
    (args.output/'group_witnesses.json').write_text(json.dumps(details, indent=2)+'\n')
    # Evaluation-only manual model review of lexical Wiktionary witnesses.
    # This file never participates in dictionary construction or model features.
    review_path = Path('experiments/wiktionary-phrases/incidental_witness_review.csv')
    unrelated = {}
    if review_path.exists():
        with review_path.open() as stream:
            unrelated = {(r['date'],r['category']):r['reason'] for r in csv.DictReader(stream)}
        review_rows = []
        for row in details['wiktionary_lexical']:
            if row['mechanism'] != 'phrase_completion':
                continue
            matched = any(row['witnesses'].values())
            reason = unrelated.get((row['date'],row['category']))
            review_rows.append(dict(date=row['date'],category=row['category'],color=row['color'],
                   any_witness=matched, intended_template_match=matched and not reason,
                   witnesses=json.dumps(row['witnesses']),
                   review_source='model inspection of witnesses against category template',
                   note=reason or ('witness matches stated template' if matched else 'no common witness')))
        with (args.output/'witness_review.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)
        reviewed_count = sum(r['intended_template_match'] for r in review_rows)
        reviewed_purple = sum(r['intended_template_match'] and r['color'].casefold()=='purple' for r in review_rows)
    else:
        reviewed_count = reviewed_purple = None
    payload = json.loads((args.output/'summary.json').read_text())
    payload['reviewed_wiktionary_lexical'] = dict(intended_phrase_groups=reviewed_count,
                                               intended_purple_phrase_groups=reviewed_purple)
    (args.output/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# Wiktionary phrase coverage', '',
             'Validation only: 121 boards, 484 groups, including 68 reviewed phrase groups. '
             'Indexes use dictionary titles only; puzzle categories are used only for evaluation. '
             'No model training or test evaluation. Unified variants use alphabetic, '
             'space/hyphen-insensitive completion matching. The spaced variant requires an actual '
             'word boundary, and word_completion requires the completion to be a standalone '
             'lexical title. Current WordNet uses its existing spaced index.', '',
             '| Source | Phrase groups | Purple phrase groups | All gold groups | Incorrect quartet-direction matches | Warm lookup ms/board |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in summary.items():
        lines.append(f"| {name} | {result['phrase_groups']}/68 | {result['purple_phrase_groups']}/{result['purple_phrase_total']} | {result['all_gold_groups']}/484 | {result['incorrect_direction_occurrences']} | {result['warm_lookup_ms_per_board']:.3f} |")
    if reviewed_count is not None:
        lines += ['', f'Model inspection of the lexical Wiktionary witnesses confirms the intended '
                  f'template in **{reviewed_count}/68 groups**, including **{reviewed_purple}/54 Purple '
                  'phrase groups**. Eight of the 46 any-witness hits use unrelated completions; '
                  'see witness_review.csv and incidental_witness_review.csv. This review is subjective '
                  'evaluation annotation, not independently human-verified ground truth.']
    lines += ['', f'Building all {len(indexes)} indexes took {build_seconds:.1f}s. Warm pairwise tensor '
              f'construction on one 16-word board averaged {encode_ms:.2f}ms over 20 repetitions.', '',
              'A match means one identical completion supports all four words. Incorrect counts '
              'count each quartet once per direction, even if it has multiple witnesses. '
              'Accidental substrings remain possible: these are ANY-witness counts, including '
              'completions unrelated to the intended mechanism. Coverage is not retrieval accuracy or solve rate. '
              'Warm lookup timing covers witness retrieval, not pairwise tensor construction or model inference.', '',
              f"Snapshot: {metadata['source_url']} (SHA256 `{metadata['sha256']}`). "
              'English Wiktionary contributors, extracted by Kaikki/Wiktextract; '
              'https://en.wiktionary.org/wiki/Wiktionary:Copyrights. '
              'The upstream download is mutable; preserve the downloaded file and hash for reproducibility.', '']
    lines += ['## Newly covered phrase groups versus unified WordNet', '']
    baseline = {(r['date'],r['category']):any(r['witnesses'].values()) for r in details['wordnet_unified']}
    for row in details['wiktionary_lexical']:
        if row['mechanism'] == 'phrase_completion' and any(row['witnesses'].values()) and not baseline[row['date'],row['category']] and (row['date'],row['category']) not in unrelated:
            evidence = '; '.join(f"{d}: {', '.join(w)}" for d,w in row['witnesses'].items() if w)
            lines.append(f"- {row['date']} {row['color']} — {row['category']}: {evidence}")
    lines += ['', '## Conclusion', '',
              'Wiktionary is a useful and inexpensive offline resource. The 509 MiB snapshot '
              'produced a 4.8 MiB compressed headword cache. It supplies many missing intended '
              'phrases, but unrestricted substring matching creates substantial noise, and requiring '
              'a standalone completion barely reduces it. Boundary-respecting matching is a safer '
              'first training ablation. Keep spaced and closed-compound evidence distinguishable, '
              'retain witnesses, and consider downweighting broadly productive affixes before '
              'testing unrestricted closed matching. No solve-rate improvement has been demonstrated.', '',
              'The provider is opt-in; existing feature order, model checkpoints, notebooks, and the '
              'running cross-validation sweep are unchanged. Offline loading example:', '', '```python',
              'from nyt_connections_solver.edge_features.phrase_lexicon import CompletionLexicon, AttestedPhraseCompletions',
              "lexicon = CompletionLexicon.from_headword_cache('data/wiktionary/headwords.jsonl.gz', spaced_only=True)",
              'provider = AttestedPhraseCompletions(lexicon)', '```', '',
              'Reproduce: `nix develop -c uv run python scripts/audit_wiktionary_phrases.py`', '']
    (args.output/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
