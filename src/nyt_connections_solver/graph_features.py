"""Reconstructable default feature configuration for graph training and search."""
from .feature_cache import TensorCache
from .node_features import TextSpellingFeatures
from .edge_features import EnglishLexicon, EmbeddingStore, EdgeFeatureComposer, default_edge_providers


def build_graph_features(model, model_id, *, cache_dir=None, refresh=False,
                         revision=None, char_dim=128, use_spelling=True,
                         include_wordnet=True, min_fragment_length=3):
    cache = TensorCache(cache_dir, refresh=refresh)
    embeddings = EmbeddingStore(model, model_id, revision=revision, cache=cache)
    lexicon = EnglishLexicon()
    nodes = TextSpellingFeatures(embeddings, char_dim=char_dim, use_spelling=use_spelling)
    providers = default_edge_providers(embeddings, lexicon, include_wordnet=include_wordnet,
                                       min_fragment_length=min_fragment_length)
    edges = EdgeFeatureComposer(providers, cache=cache)
    metadata = {
        'version': 1,
        'options': {'revision': embeddings.identity['revision'], 'char_dim': char_dim,
                    'use_spelling': use_spelling, 'include_wordnet': include_wordnet,
                    'min_fragment_length': min_fragment_length},
        'nodes': nodes.cache_identity, 'edges': edges.configuration,
    }
    return nodes, edges, metadata
