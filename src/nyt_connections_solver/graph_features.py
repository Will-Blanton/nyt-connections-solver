"""Reconstructable default feature configuration for graph training and search."""
from .feature_cache import TensorCache
from .node_features import TextSpellingFeatures
from .edge_features import EnglishLexicon, EmbeddingStore, EdgeFeatureComposer, default_edge_providers
from .edge_features import PronunciationSimilarity
from .edge_features import LetterEditSimilarity
from .edge_features import CompletionLexicon, SpecificPhraseCompletions
from .edge_features import SharedWordNetConcept, SharedLetterEditTargets
from .phonetic_nodes import PhoneticNodeFeatures
from .transformation_nodes import TransformationNodeFeatures


def build_graph_features(model, model_id, *, cache_dir=None, refresh=False,
                         revision=None, char_dim=128, use_spelling=True,
                         include_wordnet=True, min_fragment_length=3, include_phonetics=False,
                         phonetic_nodes=False, phonetic_node_dim=128,
                         phonetic_max_pronunciations=8, phonetic_max_phones=32,
                         include_letter_edits=False, letter_edit_min_length=3,
                         letter_edit_max_length=32, transformation_nodes=False,
                         max_transformations=12, transformation_min_length=3,
                         include_wiktionary=False,
                         wiktionary_cache_path='data/wiktionary/headwords.jsonl.gz',
                         wiktionary_max_frequency=None, include_concept_witnesses=False,
                         include_edit_targets=False, edit_target_min_length=3,
                         edit_target_max_length=32):
    cache = TensorCache(cache_dir, refresh=refresh)
    embeddings = EmbeddingStore(model, model_id, revision=revision, cache=cache)
    lexicon = EnglishLexicon()
    nodes = TextSpellingFeatures(embeddings, char_dim=char_dim, use_spelling=use_spelling)
    if transformation_nodes:
        if use_spelling or phonetic_nodes:
            raise ValueError('Transformation experiments require text-only base features')
        nodes = TransformationNodeFeatures(nodes, lexicon, max_transformations=max_transformations,
                                          min_length=transformation_min_length)
    if phonetic_nodes:
        if use_spelling:
            raise ValueError('Phonetic node experiment expects text-only base nodes')
        nodes = PhoneticNodeFeatures(nodes, ngram_dim=phonetic_node_dim,
                                    max_pronunciations=phonetic_max_pronunciations,
                                    max_phones=phonetic_max_phones)
    providers = default_edge_providers(embeddings, lexicon, include_wordnet=include_wordnet,
                                       min_fragment_length=min_fragment_length)
    if include_phonetics:
        providers.append(PronunciationSimilarity(lexicon, embeddings, min_length=min_fragment_length))
    if include_letter_edits:
        providers.append(LetterEditSimilarity(lexicon, embeddings, min_length=letter_edit_min_length,
                                             max_length=letter_edit_max_length))
    if include_wiktionary:
        phrase_lexicon = CompletionLexicon.from_headword_cache(wiktionary_cache_path)
        providers.append(SpecificPhraseCompletions(phrase_lexicon, max_frequency=wiktionary_max_frequency))
    if include_concept_witnesses:
        providers.append(SharedWordNetConcept(lexicon))
    if include_edit_targets:
        providers.append(SharedLetterEditTargets(lexicon,min_length=edit_target_min_length,
                                                max_length=edit_target_max_length))
    edges = EdgeFeatureComposer(providers, cache=cache)
    metadata = {
        'version': 1,
        'options': {'revision': embeddings.identity['revision'], 'char_dim': char_dim,
                    'use_spelling': use_spelling, 'include_wordnet': include_wordnet,
                    'min_fragment_length': min_fragment_length},
        'nodes': nodes.cache_identity, 'edges': edges.configuration,
    }
    # Omit disabled additions to preserve existing checkpoint/cache identities.
    if include_phonetics:
        metadata['options']['include_phonetics'] = True
    if include_letter_edits:
        metadata['options'].update(include_letter_edits=True, letter_edit_min_length=letter_edit_min_length,
                                   letter_edit_max_length=letter_edit_max_length)
    if phonetic_nodes:
        metadata['options'].update(phonetic_nodes=True, phonetic_node_dim=phonetic_node_dim,
                                   phonetic_max_pronunciations=phonetic_max_pronunciations,
                                   phonetic_max_phones=phonetic_max_phones)
    if transformation_nodes:
        metadata['options'].update(transformation_nodes=True, max_transformations=max_transformations,
                                   transformation_min_length=transformation_min_length)
    if include_wiktionary:
        metadata['options'].update(include_wiktionary=True,
                                   wiktionary_cache_path=str(wiktionary_cache_path),
                                   wiktionary_max_frequency=wiktionary_max_frequency)
    if include_concept_witnesses:
        metadata['options']['include_concept_witnesses'] = True
    if include_edit_targets:
        metadata['options'].update(include_edit_targets=True,edit_target_min_length=edit_target_min_length,
                                   edit_target_max_length=edit_target_max_length)
    return nodes, edges, metadata
