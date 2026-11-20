"""Rank/pruning overlays, checkpoint loading, and real Streamlit interactions."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nyt_connections_solver.evaluation.diagnostics import (
    DiagnosticSession, candidate_rankings, project_words, summarize_record,
)
from nyt_connections_solver.evaluation import load_puzzles
from nyt_connections_solver.feedback_search import FeedbackSearchHarness, ScoredGroup
from nyt_connections_solver.game import LocalConnectionsEnvironment, Outcome
from nyt_connections_solver.graph_models import GraphSetTransformer, StandaloneCandidateGINE
from nyt_connections_solver.checkpoints import load_graph_harness, register_graph_model, supports_graph_checkpoint
from nyt_connections_solver.harness import SearchHarness
from nyt_connections_solver.models import MultiheadAttentionBlock, SetDecoder, SetTransformer


WORDS = tuple("ABCDEFGHIJKLMNOP")
GROUPS = tuple(WORDS[i:i + 4] for i in range(0, 16, 4))


def tiny_harness(model_type=GraphSetTransformer):
    torch.manual_seed(0)
    config = dict(in_dim=4, edge_dim=1, edge_bias_scale=.5, lin_layers=1, lin_width=8,
                  attn_layers=1, attn_width=8, attn_heads=2, dropout=0.)
    if model_type is StandaloneCandidateGINE:
        config.update(edge_bias_scale=0., edge_hidden=8, gine_layers=1,
                      train_text_projection=True, train_input_norm=True)
    model = model_type(group_idx=torch.tensor([[0, 1, 2, 3]]), **config)
    def encode(words, **kwargs):
        return torch.tensor([[float(ord(w)), float(ord(w) % 7), 0., 1.] for w in words])
    nodes = SimpleNamespace(embeddings=SimpleNamespace(dim=4), encode=encode)
    harness = SearchHarness.from_model(
        model, None, input_dim=4, node_feature_builder=nodes,
        edge_feature_builder=lambda words, _: torch.zeros(16, 16, 1),
        device="cpu", beam_width=1,
    )
    harness.checkpoint = dict(
        model_class=model_type.__name__, model_config=config,
        model_state_dict=model.state_dict(), embedding_model_name="fake",
        feature_metadata={"options": {"revision": "test"}}, edge_feature_names=["test_edge"],
        split_dates={"val": ["2026-01-01"], "test": ["2026-01-01"]}, training_config={},
    )
    return harness


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pruning_overlays_keep_truth_and_normalize_surviving_mass(self):
        env = LocalConnectionsEnvironment(WORDS, GROUPS)
        env.submit(tuple("ABCE"))
        candidates = (ScoredGroup(tuple("ABCE"), np.log(.6)),
                      ScoredGroup(GROUPS[0], np.log(.3)), ScoredGroup(GROUPS[1], np.log(.1)))
        rows = candidate_rankings(env.observe(), candidates, GROUPS, GROUPS[0])
        self.assertFalse(rows[0]["feasible"])
        self.assertEqual(rows[1]["feasible_rank"], 1)
        self.assertTrue(rows[1]["true_group"])
        self.assertTrue(rows[1]["chosen"])
        self.assertAlmostEqual(rows[1]["probability_after_pruning"], .75)
        self.assertAlmostEqual(sum(row["probability_after_pruning"] for row in rows), 1.)
        self.assertEqual(rows[0]["max_gold_overlap"], 3)

    def test_projection_handles_constant_vectors(self):
        points, variance = project_words(np.ones((16, 4)))
        self.assertEqual(points.shape, (16, 2))
        self.assertTrue(np.isfinite(points).all())
        self.assertEqual(variance, 0.)

    def test_gine_uses_real_input_views_and_same_full_board_policy(self):
        harness = tiny_harness(StandaloneCandidateGINE)
        puzzle = SimpleNamespace(words=WORDS, groups=GROUPS, date="2026-01-01")
        with patch.object(harness.model.full_encoder, 'forward', side_effect=AssertionError('Unused attention')):
            session = DiagnosticSession(harness, puzzle, beam_width=1)
            self.assertEqual(set(session.projections), {'Text embeddings', 'GINE input words'})
            policy = FeedbackSearchHarness.from_harness(harness, beam_width=1, score_mode='full_board')
            self.assertEqual(session.analyze().chosen, policy.choose(session.environment.observe()))
            self.assertEqual(session.record().score_mode, 'full_board')

    def test_registered_model_without_internal_encoder_or_diagnostic_hook(self):
        from nyt_connections_solver.checkpoints import GRAPH_MODEL_TYPES

        class OtherScorer(torch.nn.Module):
            def __init__(self, group_idx, **config):
                super().__init__()

            def forward(self, x, group_idx=None, edge_features=None):
                return x[:, group_idx, :].sum(dim=(-1, -2))

        with patch.dict(GRAPH_MODEL_TYPES):
            register_graph_model(OtherScorer)
            harness = tiny_harness()
            harness.model = OtherScorer(None)
            harness.node_feature_builder = SimpleNamespace(encode=harness.node_feature_builder.encode)
            harness.edge_feature_builder = None
            session = DiagnosticSession(harness, SimpleNamespace(words=WORDS, groups=GROUPS, date='custom'), beam_width=1)
            self.assertEqual(set(session.projections), {'Node features'})
            self.assertEqual(session.edge_names, ())
            self.assertIsNotNone(session.advance())
            metadata = {**harness.checkpoint, 'model_class': 'OtherScorer'}
            self.assertTrue(supports_graph_checkpoint(metadata))

    def test_saved_gine_checkpoint_uses_shared_loader(self):
        harness = tiny_harness(StandaloneCandidateGINE)
        nodes = harness.node_feature_builder
        nodes.dim = 4
        edges = SimpleNamespace(dim=1, feature_names=['test_edge'], encode=harness.edge_feature_builder)
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'arbitrary-name.pt'
            torch.save(harness.checkpoint, weights)
            with patch('nyt_connections_solver.checkpoints.SentenceTransformer', return_value=torch.nn.Identity()), \
                 patch('nyt_connections_solver.checkpoints.build_graph_features',
                       return_value=(nodes, edges, harness.checkpoint['feature_metadata'])):
                loaded = load_graph_harness(weights, device='cpu', beam_width=1)
            x, e = harness.encode_puzzle(WORDS)
            with torch.inference_mode():
                torch.testing.assert_close(harness.model(x[None], edge_features=e[None]),
                                           loaded.model(x[None], edge_features=e[None]))

    def test_historical_colors_follow_csv_order_not_alphabetical_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "puzzles.csv"
            with path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['date', 'connections'])
                writer.writeheader()
                for group in reversed(GROUPS):
                    writer.writerow(dict(date='2026-01-01', connections=repr(list(group))))
            puzzle = load_puzzles(path, ['2026-01-01'])[0]
        self.assertEqual(puzzle.ordered_groups, tuple(reversed(GROUPS)))
        record = DiagnosticSession(tiny_harness(), puzzle, beam_width=1).record()
        self.assertEqual(record.gold_group_labels, ('Purple', 'Blue', 'Green', 'Yellow'))
        self.assertEqual(record.gold_groups[0], GROUPS[3])
        self.assertEqual(record.gold_groups[3], GROUPS[0])

    def test_trace_does_not_change_policy_choice(self):
        harness = tiny_harness()
        state = LocalConnectionsEnvironment(WORDS, GROUPS).observe()
        ordinary = FeedbackSearchHarness.from_harness(harness, capture_trace=False)
        traced = FeedbackSearchHarness.from_harness(harness, capture_trace=True)
        self.assertEqual(ordinary.choose(state), traced.choose(state))
        self.assertIsNone(ordinary.last_trace)
        self.assertEqual(len(traced.last_trace.candidates), 1820)
        self.assertEqual(len(traced.last_trace.beams_by_depth), 4)

    def test_preview_is_cached_and_only_submission_changes_environment(self):
        harness = tiny_harness()
        puzzle = SimpleNamespace(words=WORDS, groups=GROUPS, date="2026-01-01")
        session = DiagnosticSession(harness, puzzle, beam_width=1)
        before = session.environment.observe()
        decision = session.analyze()
        self.assertIs(session.analyze(), decision)
        self.assertEqual(session.environment.observe(), before)
        session.advance()
        self.assertIsNotNone(decision.feedback)
        record = session.finish()
        self.assertTrue(record.final_state.finished)
        self.assertLessEqual(len(record.decisions), 7)
        self.assertEqual(record.projections["Text embeddings"][0].shape, (16, 2))
        self.assertEqual(summarize_record(record)["guesses"], len(record.decisions))



class StreamlitTests(unittest.TestCase):
    def test_load_step_finish_batch_and_replay(self):
        self.exercise_app(GraphSetTransformer)

    def test_gine_load_step_finish_batch_and_replay(self):
        self.exercise_app(StandaloneCandidateGINE)

    def exercise_app(self, model_type):
        from streamlit.testing.v1 import AppTest

        torch.set_num_threads(1)
        harness = tiny_harness(model_type)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = path / "experiments" / "tuned-seed.pt"
            weights.parent.mkdir()
            torch.save(harness.checkpoint, weights)
            dataset = path / "puzzles.csv"
            with dataset.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['date', 'connections'])
                writer.writeheader()
                for group in reversed(GROUPS):
                    writer.writerow(dict(date='2026-01-01', connections=repr(list(group))))
            app = AppTest.from_file(str(ROOT / 'scripts/diagnose.py'), default_timeout=30)
            app.session_state['weights_dir'] = directory
            app.session_state['dataset_path'] = str(dataset)
            with patch('nyt_connections_solver.checkpoints.load_graph_harness', return_value=harness):
                app.run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label == 'Load model').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertFalse(app.error, str(app.error))
                record = app.session_state['active'].record()
                self.assertEqual(record.gold_groups[0], GROUPS[3])
                self.assertEqual(record.gold_group_labels[0], 'Purple')
                # Reloading the same checkpoint must recreate the active game.
                next(b for b in app.button if b.label == 'Load model').click().run()
                self.assertFalse(app.error, str(app.error))
                app.number_input[0].set_value(1)
                self.assertEqual(app.session_state['active'].policy.beam_width, 20)
                next(b for b in app.button if b.label == 'Apply run settings').click().run()
                self.assertEqual(app.session_state['active'].policy.beam_width, 1)
                next(b for b in app.button if b.label == 'Submit next harness guess').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(len(app.session_state['active'].decisions), 1)
                next(b for b in app.button if b.label == 'Run puzzle to completion').click().run()
                self.assertFalse(app.error, str(app.error))
                self.assertTrue(app.session_state['active'].environment.observe().finished)
                app.select_slider[0].set_value(0).run()
                metrics = {metric.label: metric.value for metric in app.metric}
                self.assertEqual(metrics['Incorrect guesses left before this guess'], '4')
                self.assertEqual(metrics['Accepted before this guess'], '0/4')
                next(b for b in app.button if b.label == 'Run selected puzzles').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertFalse(app.error, str(app.error))
                self.assertIn('2026-01-01', app.session_state['batch_records'])
                record = app.session_state['batch_records']['2026-01-01']
                self.assertEqual(record.beam_width, 1)
                self.assertEqual(record.max_mistakes, 4)
                # Editing controls does not mutate the completed batch. Applying
                # marks it stale; only rerunning actually replaces its results.
                app.number_input[0].set_value(2)
                app.number_input[1].set_value(1)
                next(b for b in app.button if b.label == 'Apply run settings').click().run()
                self.assertEqual(app.session_state['active'].policy.beam_width, 2)
                self.assertEqual(app.session_state['active'].environment.observe().mistakes_remaining, 1)
                self.assertEqual(app.session_state['batch_records']['2026-01-01'].max_mistakes, 4)
                self.assertTrue(app.warning)
                next(b for b in app.button if b.label == 'Run selected puzzles').click().run()
                self.assertFalse(app.error, str(app.error))
                record = app.session_state['batch_records']['2026-01-01']
                self.assertEqual(record.beam_width, 2)
                self.assertEqual(record.max_mistakes, 1)
                self.assertLessEqual(summarize_record(record)['mistakes'], 1)
                self.assertFalse(app.warning)


if __name__ == '__main__':
    unittest.main()
