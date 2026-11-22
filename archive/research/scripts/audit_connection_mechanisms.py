"""Validation-only mechanism inventory and dictionary coverage; no new labels enter features."""
from collections import Counter
import json
from pathlib import Path
import re

from run_graph_feature_experiment import load_connections
from nyt_connections_solver.edge_features import PronunciationLexicon, EnglishLexicon


def mechanism(category, date):
    """Coarse, reviewable label-based inventory, not an exhaustive taxonomy."""
    label = category.casefold()
    if label == 'bands minus the number five':
        return 'named_or_cultural_association'
    if re.search(r'homophon|sounds? like|rhymes?|heteronym|silent [“"\']', label):
        return 'sound'
    if re.search(r'plus|minus|\+\s*letter|letters? changed|letters? (?:are )?changed|with -ism', label):
        return 'letter_edit'
    if 'anagram' in label:
        return 'anagram'
    if re.search(r'starting|ending|starts of|beginnings|ends of|last words of', label):
        return 'embedded_fragment'
    if re.search(r'_{2,}|words (?:after|before|with)', label):
        return 'phrase_template'
    if re.search(r'what .*may mean|what .*might refer', label):
        return 'polysemy'
    if (date == '2023-09-07' or re.search(
            r'movies?|films?|hollywood|oscars?|grammy|bands?|rappers?|hall of fame|superhero|detectives|characters|'
            r'brands?|record labels|teams|web browsers|musicals?|companies named|famous|baseball greats|inventors|'
            r'lees of|roosevelt|hudsons|broadway|dickens|looney|reality shows|n\.b\.a\.|w\.n\.b\.a\.|'
            r'spelling bee ranks|septet|lyrics|tarot|apple products|photoshop|monopoly|comic strips|'
            r'multi-time nba|rooms in clue|cgi|goatees', label)):
        return 'named_or_cultural_association'
    return 'other_semantic_or_associative'


def main():
    import torch
    reference = torch.load('weights/experiments/20261023T150230Z-text-only.pt', map_location='cpu', weights_only=True)
    frame = load_connections(Path(reference['dataset_path']))
    frame = frame[frame.date.isin(reference['split_dates']['val'])]
    assert frame.date.nunique() == 121
    sounds = PronunciationLexicon()
    lexicon = EnglishLexicon()
    rows = []
    for date, board in frame.groupby('date', sort=True):
        for color, (_, row) in zip(('purple', 'blue', 'green', 'yellow'), board.iterrows()):
            words = row['connections']
            rows.append(dict(date=date, color=color, category=row['category'], words=words,
                             mechanism=mechanism(row['category'], date),
                             missing_pronunciations=[w for w in words if not sounds.phones(w)],
                             homophone_alternatives={w: sounds.alternatives(w) for w in words},
                             direct_homophone_pairs=sum(bool(set(sounds.phones(a)) & set(sounds.phones(b)))
                                                        for i, a in enumerate(words) for b in words[i+1:])))
    counts = {color: dict(Counter(r['mechanism'] for r in rows if r['color'] == color))
              for color in ('purple', 'blue', 'green', 'yellow')}
    output = Path('experiments/gine-phonetics')
    output.mkdir(exist_ok=True, parents=True)
    result = dict(split='val', puzzles=121, groups=484, counts=counts,
                  method='First-match label regex; overlapping mechanisms reduced to one bucket; labels used only for audit',
                  pronunciation_identity=sounds.identity, groups_inventory=rows)
    # Audit-only explanation witnesses, never supplied to the experiment.
    transformations = {
        '2025-03-07': ['apple', 'date', 'lemon', 'peach'],
        '2026-01-04': ['rear', 'tail', 'rump', 'can'],
        '2026-01-13': ['toe', 'arch', 'ankle', 'heel'],
        '2026-02-28': ['comet', 'nova', 'star', 'sun'],
        '2026-05-01': ['five', 'six', 'four', 'eight'],
        '2026-05-12': ['franc', 'rand', 'real', 'won'],
        '2026-05-31': ['ash', 'oak', 'pine', 'teak'],
        '2026-09-25': ['order', 'class', 'family', 'genus'],
    }
    result['audited_letter_targets'] = [dict(date=date, targets=words,
        attested=[w for w in words if w in lexicon.words]) for date, words in transformations.items()]
    (output / 'mechanism-inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Validation group-mechanism inventory', '',
             '121 validation puzzles / 484 groups. No test groups inspected. This is a coarse, '
             'first-match inventory inferred from historical labels, not a trained category classifier '
             'or a complete taxonomy. Buckets can overlap in reality; ambiguous labels remain in the '
             'broad semantic/association bucket. The JSON lists every assignment for review. '
             'Labels and bucket assignments never enter feature construction or training.', '',
             '| Mechanism | Purple | Blue | Green | Yellow |', '|---|---:|---:|---:|---:|']
    for family in sorted({r['mechanism'] for r in rows}):
        lines.append('| ' + family + ' | ' + ' | '.join(str(counts[c].get(family, 0)) for c in counts) + ' |')
    lines += ['', '## Sound-labelled examples and dictionary coverage', '',
              'A shared exact pronunciation is different from four homophones of different category members. '
              'Dialect/stress variation and dictionary omissions are limitations. Word coverage below '
              'counts any pronunciation, not whether the intended reading or spelling is present.', '',
              '| Date | Color | Label | Words | Direct homophone pairs /6 | Words with pronunciation /4 |',
              '|---|---|---|---|---:|---:|']
    for row in rows:
        if row['mechanism'] == 'sound':
            lines.append(f"| {row['date']} | {row['color']} | {row['category']} | {', '.join(row['words'])} | "
                         f"{row['direct_homophone_pairs']} | {4-len(row['missing_pronunciations'])} |")
    lines += ['', '## Mechanisms without dedicated evidence in the original 15 channels', '',
              '- **Semantic categories after one-letter edits.** Examples: `ample/dote/lumon/poach` '
              '(fruits with second letters changed), `hive/mix/pour/wight` (numbers with first letters changed), '
              '`cometh/novak/start/sung` (astronomical terms plus a letter). Existing character overlap '
              'compares puzzle words; it does not retrieve their transformed lexical readings.',
              '- **Anagrams of different category members.** `abut/bulge/groan/voila` rearrange to '
              '`tuba/bugle/organ/viola`. The existing exact-anagram edge only links words with the same '
              'letter multiset, so it does not directly cover this task.',
              '- **Truncated names and hidden entity fragments.** `ear/mar/mer/sat` start planet names; '
              '`autoharp/contrabass/pepperoni/rosebud` end with beer brands. Prefix/suffix semantic channels '
              'cover only WordNet-attested proper fragments and do not resolve those fragments to entities.',
              '- **Pronunciation properties rather than pair similarity.** Heteronyms `bow/row/sow/wind` '
              'share a property of having distinct readings; silent-G `gnat/gnaw/gnocchi/gnome` requires '
              'spelling-to-sound alignment. Direct homophone/rhyme/edit edges do not explicitly encode either property.',
              '- **Implicit shared senses.** `assignment/defeat/tempo/tired` are readings of “beat”; '
              '`carapace/gas station/pastry crust/rowing boat` refer to “shell”. Pairwise semantic similarity '
              'does not explicitly identify one shared ambiguous headword.',
              '- **Entity membership, titles, roles, and cultural facts.** `glass/gross/king/stern` '
              '(radio hall of fame), `brave/flow/frozen/up` (animated-feature winners), and '
              '`berry/madison/stone/swank` (actress winners) lack an explicit knowledge relation. '
              'Embeddings may contain some of this information; that is different from dedicated coverage.', '',
              '## Next bounded experiments suggested by the inventory', '',
              '1. **One-letter transformation semantics:** retrieve valid words after a deletion or substitution, '
              'then compare their frozen embeddings; keep operation types separate and cap candidate counts. '
              'Use confidence/rarity and controls: taking maxima over many lexical alternatives creates false positives.',
              '2. **Anagram-neighbor semantics:** index the existing vocabulary by sorted letters and compare '
              'embeddings of alternative anagrams. This reuses WordNet and the existing embedding cache.',
              '3. **Pronunciation-property node features:** a few cheap indicators for distinct pronunciations, '
              'syllable count, and stress patterns; do not assume these alone identify silent letters.',
              '4. **Broader phrase resources:** improves dictionary coverage, but pairwise witnesses can differ '
              'across a group and an earlier closed-compound experiment did not improve overall ranking.', '',
              'These are hypotheses, not established improvements. Existing nonzero edge values are not proof '
              'of explanatory coverage, and lack of a dedicated edge does not prove the model lacks the information.', '',
              '## Lexical feasibility of selected letter-transform cases', '',
              'Eight manually interpreted Purple examples have the following intended transformed words '
              'in the current WordNet vocabulary. This checks resource availability, not whether a '
              'retrieval/scoring algorithm would identify the intended alternatives among distractors. '
              'The examples are selected, not an unbiased coverage estimate.', '',
              '| Date | Intended targets | WordNet-attested /4 |', '|---|---|---:|']
    for example in result['audited_letter_targets']:
        lines.append(f"| {example['date']} | {', '.join(example['targets'])} | {len(example['attested'])} |")
    lines += ['',
              '## A bounded knowledge-graph option', '',
              'Wikipedia does not require a full graph download for a first test: resolve puzzle entries to '
              'candidate pages, cache outgoing links/categories, and compare neighbor overlap or verified '
              'two-hop connections on those local neighborhoods. This is bounded evidence, not exact global '
              'shortest-path distance. Entity ambiguity (e.g. “up”, “flow”, or “stone”) and generic high-degree '
              'pages are likely bigger challenges than raw graph size. It is a separate experiment; no '
              'Wikipedia-derived features or network retrieval were added here. MediaWiki reference: '
              '[API:Links](https://www.mediawiki.org/wiki/API:Links).', '',
              'Reproduce: `nix develop -c uv run python scripts/audit_connection_mechanisms.py`.']
    Path('experiments/connection-mechanism-gaps.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
