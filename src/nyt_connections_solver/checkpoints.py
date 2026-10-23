"""Load graph model checkpoints without executing notebook cells."""

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from .graph_features import build_graph_features
from .graph_models import GraphSetTransformer
from .harness import SearchHarness


def load_graph_harness(weights_path, device="auto", beam_width=20, state_batch_size=4,
                       cache_dir="data/feature_cache/graph"):
    """Load a text/spelling graph checkpoint with its original feature providers."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if checkpoint["model_class"] != "GraphSetTransformer" or "feature_metadata" not in checkpoint:
        raise ValueError("Expected a text/spelling graph checkpoint with feature metadata")
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    options = checkpoint["feature_metadata"]["options"]
    embedder = SentenceTransformer(checkpoint["embedding_model_name"],
                                  revision=options["revision"], device=str(resolved))
    nodes, edges, metadata = build_graph_features(
        embedder, checkpoint["embedding_model_name"], cache_dir=cache_dir, **options
    )
    if metadata != checkpoint["feature_metadata"] or list(edges.feature_names) != checkpoint["edge_feature_names"]:
        raise ValueError("Feature definitions or lexical resources differ from the checkpoint")
    config = checkpoint["model_config"]
    if config["in_dim"] != nodes.dim or config["edge_dim"] != edges.dim:
        raise ValueError("Checkpoint feature dimensions do not match")
    model = GraphSetTransformer(group_idx=torch.tensor([[0, 1, 2, 3]]), **config)
    model.load_state_dict(checkpoint["model_state_dict"])
    harness = SearchHarness.from_model(
        model, embedder, input_dim=nodes.dim, device=resolved,
        node_feature_builder=nodes, edge_feature_builder=edges.encode,
        beam_width=beam_width, state_batch_size=state_batch_size,
    )
    harness.checkpoint = checkpoint
    harness.weights_path = Path(weights_path)
    return harness
