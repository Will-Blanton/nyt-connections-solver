"""Standalone GINE with learned fusion of text and phonetic word features."""
import torch
from torch import nn
import torch.nn.functional as F

from .graph_models import StandaloneCandidateGINE


class PhoneticFusionGINE(StandaloneCandidateGINE):
    def __init__(self, *, text_dim, phonetic_dim=128, max_pronunciations=8, max_phones=32,
                 phone_vocab_size=40, phonetic_encoding='ngram', phonetic_fusion='linear',
                 phonetic_enabled=True, **config):
        if phonetic_encoding not in {'ngram', 'cnn'} or phonetic_fusion not in {'linear', 'gated'}:
            raise ValueError('Unknown phonetic encoding or fusion')
        if config['in_dim'] != text_dim + phonetic_dim + max_pronunciations * max_phones:
            raise ValueError('Node payload does not match the saved phonetic layout')
        if config.get('raw_text_embeddings') or config.get('adapter_hidden') is not None:
            raise ValueError('Phonetic fusion expects the ordinary trainable text projection')
        # Preserve the original text LayerNorm/projection and checkpoint shapes.
        # Saved in_dim describes the full input tensor, while this branch sees text.
        super().__init__(**{**config, 'in_dim': text_dim})
        self.text_dim, self.phonetic_dim = text_dim, phonetic_dim
        self.max_pronunciations, self.max_phones = max_pronunciations, max_phones
        self.phonetic_encoding, self.phonetic_fusion = phonetic_encoding, phonetic_fusion
        self.phonetic_enabled = phonetic_enabled
        if phonetic_encoding == 'cnn':
            self.phone_embedding = nn.Embedding(phone_vocab_size, 24, padding_idx=0)
            self.phone_convs = nn.ModuleList([nn.Conv1d(24, 32, k, padding=k//2) for k in (3, 5, 7)])
            self.phone_readout = nn.Linear(96, phonetic_dim)
        output_dim = self.edge_encoder.node_projection.in_features
        self.phonetic_projection = nn.Linear(phonetic_dim, output_dim, bias=False)
        nn.init.zeros_(self.phonetic_projection.weight)
        if phonetic_fusion == 'gated':
            self.phonetic_gate = nn.Linear(output_dim * 2, 1)
            nn.init.zeros_(self.phonetic_gate.weight)
            nn.init.zeros_(self.phonetic_gate.bias)  # initially 0.5 for known words

    def encode_phonetics(self, x):
        shape = x.shape[:-1]
        token_start = self.text_dim + self.phonetic_dim
        ids = x[..., token_start:].long().reshape(-1, self.max_pronunciations, self.max_phones)
        known = ids.ne(0).any(-1)
        if self.phonetic_encoding == 'ngram':
            encoded = x[..., self.text_dim:token_start]
        else:
            flattened = ids.reshape(-1, self.max_phones)
            valid = flattened.ne(0)
            embedded = self.phone_embedding(flattened).transpose(1, 2)
            pooled = [F.relu(conv(embedded)).masked_fill(~valid[:, None], -1e4).amax(-1)
                      for conv in self.phone_convs]
            encoded = self.phone_readout(torch.cat(pooled, dim=-1))
            encoded = encoded.reshape(-1, self.max_pronunciations, self.phonetic_dim)
            encoded = (encoded * known[..., None]).sum(1) / known.sum(1).clamp_min(1)[..., None]
            encoded = F.normalize(encoded, dim=-1).reshape(*shape, self.phonetic_dim)
        return encoded, known.any(-1).reshape(*shape, 1)

    def fusion_parts(self, x):
        semantic = super().encode_input_words(x[..., :self.text_dim])
        encoded, known = self.encode_phonetics(x)
        projected = self.phonetic_projection(encoded) * known
        gate = (self.phonetic_gate(torch.cat([semantic, projected], dim=-1)).sigmoid()
                if self.phonetic_fusion == 'gated' else torch.ones_like(known, dtype=semantic.dtype))
        gate = gate * known * self.phonetic_enabled
        return semantic, projected, gate

    def encode_input_words(self, x):
        if not self.phonetic_enabled:
            return super().encode_input_words(x[..., :self.text_dim])
        semantic, projected, gate = self.fusion_parts(x)
        return semantic + gate * projected

    def diagnostic_representations(self, nodes, edges):
        with torch.inference_mode():
            semantic, projected, gate = self.fusion_parts(nodes[None])
        return {'Text projection': semantic[0], 'Phonetic projection': projected[0],
                'GINE input words': (semantic + gate * projected)[0]}
