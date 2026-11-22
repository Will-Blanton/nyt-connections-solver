"""Load graph model checkpoints without executing notebook cells."""

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from .graph_features import build_graph_features
from .graph_models import GraphSetTransformer, GatedEdgeResidual, GatedGINEResidual, StandaloneCandidateGINE
from .phonetic_models import PhoneticFusionGINE
from .additive_models import AdditiveGINE
from .transformation_models import TransformationGINE
from .edge_features import EdgeFeatureComposer, ClosedCompoundCompletions
from .harness import SearchHarness


GRAPH_MODEL_TYPES = {cls.__name__: cls for cls in (
    GraphSetTransformer, GatedEdgeResidual, GatedGINEResidual, StandaloneCandidateGINE, PhoneticFusionGINE,
    AdditiveGINE, TransformationGINE)}


def register_graph_model(model_type, *, name=None):
    """Register an nn.Module using forward(x, group_idx=..., edge_features=...).

    Constructors accept group_idx plus saved model_config. Outputs are one finite
    score per candidate, shaped (batch, candidates) or (batch, candidates, 1).
    Optional diagnostic_representations(nodes, edges) returns named (words, D)
    tensors. Registration uses trusted Python code, never imports from checkpoints.
    """
    key = name or model_type.__name__
    if key in GRAPH_MODEL_TYPES and GRAPH_MODEL_TYPES[key] is not model_type:
        raise ValueError(f"Model type already registered: {key}")
    if not issubclass(model_type, torch.nn.Module):
        raise TypeError("Graph model must be a torch.nn.Module")
    GRAPH_MODEL_TYPES[key] = model_type
    return model_type


def supports_graph_checkpoint(checkpoint):
    """Shared eligibility check for checkpoint selectors and reconstruction."""
    return (checkpoint.get("model_class") in GRAPH_MODEL_TYPES
            and all(key in checkpoint for key in (
                "model_config", "embedding_model_name", "feature_metadata", "edge_feature_names")))


def load_graph_harness(weights_path, device="auto", beam_width=20, state_batch_size=4,
                       cache_dir="data/feature_cache/graph"):
    """Load a text/spelling graph checkpoint with its original feature providers."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not supports_graph_checkpoint(checkpoint):
        raise ValueError("Expected a registered graph model checkpoint with feature metadata")
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    options = checkpoint["feature_metadata"]["options"]
    embedder = SentenceTransformer(checkpoint["embedding_model_name"],
                                  revision=options["revision"], device=str(resolved))
    nodes, edges, metadata = build_graph_features(
        embedder, checkpoint["embedding_model_name"], cache_dir=cache_dir, **options
    )
    if checkpoint["edge_feature_names"][-2:] == ["shared_closed_compound_prefix", "shared_closed_compound_suffix"]:
        providers = (*edges.providers, ClosedCompoundCompletions(edges.providers[1].lexicon))
        edges = EdgeFeatureComposer(providers, cache=edges.cache)
        metadata["edges"] = edges.configuration
    if metadata != checkpoint["feature_metadata"] or list(edges.feature_names) != checkpoint["edge_feature_names"]:
        raise ValueError("Feature definitions or lexical resources differ from the checkpoint")
    config = checkpoint["model_config"]
    if tuple(config.get('candidate_edge_channels', ())) != tuple(getattr(edges, 'candidate_feature_channels', ())):
        raise ValueError('Checkpoint candidate edge channels do not match feature definitions')
    if config["in_dim"] != nodes.dim or config["edge_dim"] != edges.dim:
        raise ValueError("Checkpoint feature dimensions do not match")
    model = GRAPH_MODEL_TYPES[checkpoint["model_class"]](group_idx=torch.tensor([[0, 1, 2, 3]]), **config)
    model.load_state_dict(checkpoint["model_state_dict"])
    harness = SearchHarness.from_model(
        model, embedder, input_dim=nodes.dim, device=resolved,
        node_feature_builder=nodes, edge_feature_builder=edges.encode,
        beam_width=beam_width, state_batch_size=state_batch_size,
    )
    harness.checkpoint = checkpoint
    harness.weights_path = Path(weights_path)
    return harness
