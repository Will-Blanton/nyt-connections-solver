"""Reusable providers and the standard text/spelling graph feature setup."""
from .base import EdgeFeatureProvider, CandidateEdgeFeatureProvider
from .composer import EdgeFeatureComposer
from .resources import EnglishLexicon, EmbeddingStore
from .semantic import SemanticSimilarity, WordNetSimilarity
from .character_overlap import CharacterOverlap
from .substrings import SubstringSimilarity
from .phrases import PhraseCompletions, ClosedCompoundCompletions
from .phrase_lexicon import CompletionLexicon, SpecificPhraseCompletions
from .anagrams import Anagrams
from .spelling_structure import SpellingStructure
from .pronunciation import PronunciationLexicon, PronunciationSimilarity
from .letter_edits import LetterEditSimilarity
from .concept_witnesses import SharedWordNetConcept
from .edit_targets import SharedLetterEditTargets


def default_edge_providers(embeddings, lexicon, *, include_wordnet=True, min_fragment_length=3):
    """Share the embedding store and English lexicon across feature families."""
    providers = [SemanticSimilarity(embeddings)]
    if include_wordnet:
        providers.append(WordNetSimilarity(lexicon))
    providers.extend([CharacterOverlap(), SubstringSimilarity(lexicon, embeddings, min_length=min_fragment_length),
                      PhraseCompletions(lexicon), Anagrams(), SpellingStructure()])
    return providers
