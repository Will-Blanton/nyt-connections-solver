"""Reviewable explanation labels for analysis only; never model inputs."""
import json
import re


MECHANISMS = (
    'semantic_category', 'synonym', 'phrase_completion', 'word_containment',
    'letter_edit', 'anagram', 'sound', 'polysemy', 'named_entity_or_cultural',
    'orthographic_pattern', 'other_or_uncertain',
)

# Specific mechanical operations precede topic/entity heuristics. The matched
# span is retained, and broad unmatched topics remain explicitly low confidence.
RULES = [
    ('anagram', 'anagram', 'rearrangement', r'\banagrams?\b', 'high'),
    ('homophone', 'sound', 'homophone', r'homophon|sounds? like', 'high'),
    ('rhyme', 'sound', 'rhyme', r'\brhym', 'high'),
    ('pronunciation', 'sound', 'pronunciation_property',
     r'heteronym|silent (?:letter|[“\"\x27‘]?[a-z][”\"\x27’]?\b)|pronounc|spoken|phonetic', 'high'),
    ('letter_edit', 'letter_edit', 'letter_transformation',
     r'(?:plus|minus|without|remove|dropp?ing|chang|replac|swap|add|subtract|losing|missing).{0,30}\bletters?\b|'
     r'letters?.{0,25}(?:chang|replac|swap|remov|add|dropp|missing)|with -ism|\bplus a[n]?\b', 'high'),
    ('orthography', 'orthographic_pattern', 'spelling_property',
     r'palindrom|double letters?|double consonants?|repeated letters?|consecutive letters?|'
     r'alphabetical|roman numerals?|vowels?|consonants?|letter pattern', 'medium'),
    ('blank', 'phrase_completion', 'blank_template', r'_{2,}', 'high'),
    ('phrase', 'phrase_completion', 'shared_word',
     r'words? (?:after|before|following|preceding)|preceded by|followed by|with .{1,30} (?:before|after)|'
     r'can precede|can follow|complet(?:e|ing) the phrase', 'high'),
    ('fragment', 'word_containment', 'embedded_fragment',
     r'\b(?:starting|ending|beginnings?|ends?) (?:of|with)|\bstarts of\b|\blast words of\b|'
     r'\bhidden\b|\bcontain(?:ing|s)?\b|\bprefix(?:es)?\b|\bsuffix(?:es)?\b', 'medium'),
    ('polysemy', 'polysemy', 'shared_headword_senses',
     r'what .{1,50}(?:may mean|might refer|can mean|can refer)|meanings? of|senses of', 'high'),
    ('synonyms', 'synonym', 'same_definition', r'synonyms?|ways to say|another word for', 'high'),
    ('quoted_definition', 'synonym', 'quoted_definition', r'^[“\"‘\x27][^“\"”‘’\x27]{1,35}[”\"’\x27]$', 'medium'),
    ('cultural', 'named_entity_or_cultural', 'named_membership',
     r'\b(?:nba|nfl|nhl|mlb|n\.b\.a\.|w\.n\.b\.a\.)|\b(?:teams?|brands?|companies|rappers?|'
     r'actors?|actresses|superheroes|superhero|characters|inventors|presidents|celebrities|magazines|'
     r'sitcoms?|musicals?|albums?|novels?|cities|countries|states|planets)\b|'
     r'hall of fame|oscar|grammy|looney|monopoly|tarot|broadway|dickens|movies? (?:starring|with|by)|'
     r'films? (?:starring|with|by)|\bbands?\b|\b(?:first|last) names\b', 'medium'),
]


def classify(category):
    label = category.casefold().strip()
    matches = []
    for rule_id, mechanism, subtype, pattern, confidence in RULES:
        match = re.search(pattern, label)
        if match:
            matches.append((rule_id, mechanism, subtype, confidence, match.group()))
    if not matches:
        return dict(primary_mechanism='semantic_category', subtype='unresolved_topic_or_definition',
            secondary_mechanisms='[]', label_confidence='low', label_source='fallback',
            rule_id='unmatched', matched_text='', review_notes='Unreviewed broad fallback; may be synonym, entity, or another mechanism.')
    rule, primary, subtype, confidence, text = matches[0]
    secondary = list(dict.fromkeys(m[1] for m in matches if m[1] != primary))
    return dict(primary_mechanism=primary, subtype=subtype,
        secondary_mechanisms=json.dumps(secondary), label_confidence=confidence,
        label_source='regex', rule_id=rule, matched_text=text,
        review_notes='Category-label heuristic; not manually verified.')


def validate_review(row):
    if row['primary_mechanism'] not in MECHANISMS:
        raise ValueError('Unknown primary mechanism: '+str(row))
    secondary = json.loads(row['secondary_mechanisms'])
    if not isinstance(secondary, list) or any(m not in MECHANISMS for m in secondary):
        raise ValueError('Invalid secondary mechanisms: '+str(row))
    if row['confidence'] not in ('high', 'medium', 'low'):
        raise ValueError('Invalid confidence: '+str(row))
