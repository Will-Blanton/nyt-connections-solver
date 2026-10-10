"""Compare WordNet concept overlap within Connections groups and across groups."""

import argparse
import ast
import csv
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def load_puzzles(path: Path):
    puzzles = defaultdict(list)
    invalid_dates = set()
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            try:
                words = ast.literal_eval(row["connections"])
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"Invalid connections list on CSV row {row}") from error
            if not isinstance(words, list) or len(words) != 4:
                invalid_dates.add(row["date"])
                continue
            words = [str(word).strip() for word in words]
            if any(not word for word in words):
                invalid_dates.add(row["date"])
            puzzles[row["date"]].append((row["category"], words))

    if not puzzles:
        raise ValueError(f"No puzzles found in {path}")
    for date, groups in puzzles.items():
        flat_words = [word for _, words in groups for word in words]
        if len(groups) != 4 or len(set(flat_words)) != 16:
            invalid_dates.add(date)
    for date in invalid_dates:
        puzzles.pop(date, None)
    if not puzzles:
        raise ValueError(f"No valid 16-word puzzles found in {path}")
    return puzzles, sorted(invalid_dates)


def load_wordnet(languages):
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("Could not download the NLTK WordNet corpus")
        wn.ensure_loaded()

    if any(language != "eng" for language in languages):
        try:
            available = wn.langs()
        except LookupError:
            if not nltk.download("omw-1.4", quiet=True):
                raise RuntimeError("Non-English lookup requires the NLTK omw-1.4 corpus")
            available = wn.langs()
        missing = sorted(set(languages) - set(available))
        if missing:
            raise ValueError(f"WordNet languages unavailable: {', '.join(missing)}")
    return wn


def concept_closure(wn, word, languages):
    normalized = word.strip().lower()
    lemmas = dict.fromkeys((normalized, normalized.replace(" ", "_"), normalized.replace("-", "_")))
    senses = {}
    for language in languages:
        for lemma in lemmas:
            for sense in wn.synsets(lemma, lang=language):
                senses[sense.name()] = sense

    concepts = set()
    for sense in senses.values():
        pending = [sense]
        while pending:
            concept = pending.pop()
            if concept not in concepts:
                concepts.add(concept)
                pending.extend(concept.hypernyms())
                pending.extend(concept.instance_hypernyms())
    return concepts, bool(senses)


def puzzle_pair_scores(groups, wn, languages):
    words = [word for _, group in groups for word in group]
    closures, covered = zip(*(concept_closure(wn, word, languages) for word in words))

    # Puzzle-local IDF: broad concepts supported by all 16 words get zero weight.
    support = Counter(concept for closure in closures for concept in closure)
    idf = {
        concept: math.log(len(words) / count)
        for concept, count in support.items()
        if 2 <= count < len(words)
    }

    pair_scores = {}
    for left, right in combinations(range(len(words)), 2):
        shared = closures[left] & closures[right]
        shared_weight = sum(idf.get(concept, 0.0) for concept in shared)
        total_weight = sum(idf.get(concept, 0.0) for concept in closures[left] | closures[right])
        pair_scores[left, right] = shared_weight / total_weight if total_weight else 0.0

    within, across = [], []
    offset = 0
    for group_index, (_, group) in enumerate(groups):
        indices = range(offset, offset + len(group))
        within.extend(pair_scores[min(i, j), max(i, j)] for i, j in combinations(indices, 2))
        offset += len(group)
    for i, j in combinations(range(len(words)), 2):
        left_group, right_group = i // 4, j // 4
        if left_group != right_group:
            across.append(pair_scores[i, j])

    return float(np.mean(within)), float(np.mean(across)), sum(covered)


def bootstrap_mean_ci(values, *, seed=0, samples=5000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("data/connections.csv"))
    parser.add_argument(
        "--languages", default="eng",
        help="comma-separated NLTK WordNet language codes (default: eng); e.g. eng,spa",
    )
    args = parser.parse_args()
    languages = list(dict.fromkeys(code.strip() for code in args.languages.split(",") if code.strip()))
    if not languages:
        parser.error("--languages must contain at least one language code")

    puzzles, invalid_dates = load_puzzles(args.dataset)
    wn = load_wordnet(languages)
    results = [puzzle_pair_scores(groups, wn, languages) for groups in puzzles.values()]
    differences = [within - across for within, across, _ in results]
    ci_low, ci_high = bootstrap_mean_ci(differences)
    within_scores = [within for within, _, _ in results]
    across_scores = [across for _, across, _ in results]
    words_covered = sum(covered for _, _, covered in results)
    total_words = len(puzzles) * 16

    print(f"Dataset: {args.dataset} ({len(puzzles):,} puzzles)")
    if invalid_dates:
        print(f"Excluded malformed puzzles ({len(invalid_dates)}): {', '.join(invalid_dates)}")
    print(f"WordNet languages: {', '.join(languages)}")
    print(f"Words with at least one WordNet sense: {words_covered:,}/{total_words:,} ({words_covered / total_words:.1%})")
    print(f"Mean within-group weighted Jaccard: {np.mean(within_scores):.4f}")
    print(f"Mean across-group weighted Jaccard: {np.mean(across_scores):.4f}")
    print(f"Mean per-puzzle difference: {np.mean(differences):+.4f} (95% bootstrap CI {ci_low:+.4f}, {ci_high:+.4f})")
    print(f"Puzzles where within-group overlap is higher: {np.mean(np.asarray(differences) > 0):.1%}")
    print("Score: weighted Jaccard of each word pair's WordNet synset + hypernym closures;")
    print("weights are puzzle-local log(16 / distinct-word support), with concepts supported by all 16 ignored.")


if __name__ == "__main__":
    main()
