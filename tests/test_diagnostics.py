"""Rank/pruning overlays, checkpoint loading, and real Streamlit interactions."""

import ast
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
from nyt_connections_solver.graph_models import GraphSetTransformer
from nyt_connections_solver.harness import SearchHarness
from nyt_connections_solver.models import MultiheadAttentionBlock, SetDecoder, SetTransformer


WORDS = tuple("ABCDEFGHIJKLMNOP")
GROUPS = tuple(WORDS[i:i + 4] for i in range(0, 16, 4))


def tiny_harness():
    torch.manual_seed(0)
    config = dict(in_dim=4, edge_dim=1, edge_bias_scale=.5, lin_layers=1, lin_width=8,
                  attn_layers=1, attn_width=8, attn_heads=2, dropout=0.)
    model = GraphSetTransformer(group_idx=torch.tensor([[0, 1, 2, 3]]), **config)
    def encode(words, **kwargs):
        return torch.tensor([[float(ord(w)), float(ord(w) % 7), 0., 1.] for w in words])
    nodes = SimpleNamespace(embeddings=SimpleNamespace(dim=4), encode=encode)
    harness = SearchHarness.from_model(
        model, None, input_dim=4, node_feature_builder=nodes,
        edge_feature_builder=lambda words, _: torch.zeros(16, 16, 1),
        device="cpu", beam_width=1,
    )
    harness.checkpoint = dict(
        model_class="GraphSetTransformer", model_config=config,
        model_state_dict=model.state_dict(), embedding_model_name="fake",
        feature_metadata={"options": {}}, edge_feature_names=["test_edge"],
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

    def test_shared_graph_model_matches_notebook_weights_and_outputs(self):
        notebook = json.loads((ROOT / "connections-prob-mass.ipynb").read_text())
        source = next(''.join(c.get('source', [])) for c in notebook['cells']
                      if 'class GraphAttentionBlock' in ''.join(c.get('source', [])))
        definitions = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
        namespace = dict(torch=torch, nn=torch.nn, SetTransformer=SetTransformer,
                         MultiheadAttentionBlock=MultiheadAttentionBlock, SetDecoder=SetDecoder)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'notebook model', 'exec'), namespace)
        harness = tiny_harness()
        original = namespace['GraphSetTransformer'](group_idx=torch.tensor([[0, 1, 2, 3]]),
                                                   **harness.checkpoint['model_config']).eval()
        original.load_state_dict(harness.model.state_dict())
        x, edges = harness.encode_puzzle(WORDS)
        with torch.inference_mode():
            torch.testing.assert_close(harness.model(x[None], edge_features=edges[None]),
                                       original(x[None], edge_features=edges[None]), rtol=0, atol=0)


class StreamlitTests(unittest.TestCase):
    def test_load_step_finish_batch_and_replay(self):
        from streamlit.testing.v1 import AppTest

        torch.set_num_threads(1)
        harness = tiny_harness()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = path / "graph_text_spelling_test.pt"
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
