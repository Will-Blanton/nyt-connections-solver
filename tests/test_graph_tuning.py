"""Exercise notebook stopping/selection logic without launching training trials."""
import ast
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import unittest

import torch
import numpy as np
import optuna


def definitions():
    notebook = json.loads(Path('connections-graph.ipynb').read_text())
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n, ast.FunctionDef))
    return nodes


class SimpleLoss(torch.nn.Module):
    def forward(self, scores, relevance, lengths):
        return (scores.squeeze(-1) - relevance).square().mean(dim=-1)


class GraphTuningTests(unittest.TestCase):
    def namespace(self, values):
        states = []
        iterator = iter(values)
        def validate(model, loaders, artifacts, device):
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            value = next(iterator)
            return {size: value for size in loaders}
        namespace = dict(torch=torch, np=np, random=random, optuna=optuna,
                         LambdaNDCGLoss2=SimpleLoss,
                         score_batch=lambda model, x, batch, group_idx: model(x))
        wanted = [n for n in definitions() if n.name in {'_next_batch', 'train_mixed_states'}]
        exec(compile(ast.Module(body=wanted, type_ignores=[]), 'tuning', 'exec'), namespace)
        namespace['evaluate_state_ndcg'] = validate
        return namespace, states

    def inputs(self):
        batch = {'x': torch.ones(1, 2, 1)}
        loaders = {2: [batch]*6, 3: [batch]*4, 4: [batch]}
        artifacts = {s: {'idx': torch.tensor([[0]]), 'y': torch.tensor([1., 0.]),
                         'n': torch.tensor(2), 'k': s} for s in loaders}
        return loaders, artifacts

    def test_plateau_stops_and_restores_best_epoch(self):
        namespace, states = self.namespace([.2, .4, .3, .3])
        loaders, artifacts = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            model = namespace['train_mixed_states'](
                torch.nn.Linear(1, 1), loaders, loaders, artifacts, epochs=20,
                device='cpu', lr=.01, weight_decay=0., early_stopping_patience=2)
        self.assertEqual(model.training_summary['epochs_ran'], 4)
        self.assertEqual(model.training_summary['best_epoch'], 2)
        self.assertEqual(model.training_summary['steps_per_epoch'], 1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, states[1][key])

    def test_small_gains_do_not_reset_patience_but_best_weights_are_kept(self):
        namespace, states = self.namespace([.2, .20001, .20002])
        loaders, artifacts = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            model = namespace['train_mixed_states'](
                torch.nn.Linear(1, 1), loaders, loaders, artifacts, epochs=20,
                device='cpu', lr=.01, weight_decay=0., early_stopping_patience=2)
        self.assertEqual(model.training_summary['epochs_ran'], 3)
        self.assertEqual(model.training_summary['best_epoch'], 3)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, states[-1][key])

    def test_pruning_reports_ndcg(self):
        namespace, _ = self.namespace([.25])
        loaders, artifacts = self.inputs()
        class Trial:
            def report(self, value, step): self.reported = (value, step)
            def set_user_attr(self, key, value): pass
            def should_prune(self): return True
        trial = Trial()
        with self.assertRaises(optuna.TrialPruned):
            namespace['train_mixed_states'](torch.nn.Linear(1, 1), loaders, loaders,
                artifacts, epochs=20, device='cpu', lr=.01, weight_decay=0., trial=trial)
        self.assertEqual(trial.reported, (.25, 0))

    def test_objective_has_no_search_and_returns_best_ndcg(self):
        objective = next(n for n in definitions() if n.name == 'objective')
        source = ast.unparse(objective)
        self.assertNotIn('SearchHarness', source)
        self.assertNotIn('evaluate_puzzles', source)
        self.assertIn("summary['best_validation_ndcg']", source)
        self.assertIn('finally:', source)

    def test_ndcg_validation_runs_without_loss_or_search(self):
        from pytorchltr.evaluation import ndcg
        namespace = dict(torch=torch, ndcg=ndcg,
                         score_batch=lambda model, x, batch, group_idx: model(x))
        node = next(n for n in definitions() if n.name == 'evaluate_state_ndcg')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'validation', 'exec'), namespace)
        batch = {'x': torch.tensor([[[2.], [1.]]])}
        artifacts = {1: {'idx': torch.tensor([[0]]), 'y': torch.tensor([1, 0]), 'n': torch.tensor(2)}}
        metrics = namespace['evaluate_state_ndcg'](torch.nn.Identity(), {1: [batch]}, artifacts, 'cpu')
        self.assertAlmostEqual(metrics[1], 1.)

    def test_validation_subset_is_fixed_bounded_and_ordered(self):
        from torch.utils.data import DataLoader, Subset
        namespace = dict(np=np, DataLoader=DataLoader, Subset=Subset,
                         collate_keep_wordlists=lambda batch: batch)
        node = next(n for n in definitions() if n.name == 'tuning_validation_loaders')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'subset', 'exec'), namespace)
        loaders = namespace['tuning_validation_loaders']({2: list(range(100)), 4: list(range(5))}, 12, 4)
        self.assertEqual(len(loaders[2].dataset), 12)
        self.assertEqual(len(loaders[4].dataset), 5)
        self.assertEqual(loaders[2].dataset.indices, sorted(set(loaders[2].dataset.indices)))
        self.assertEqual(loaders[2].dataset.indices[0], 0)
        self.assertEqual(loaders[2].dataset.indices[-1], 99)

    def test_learning_diagnostic_uses_fixed_budget_and_separate_metrics(self):
        import pandas as pd
        from types import SimpleNamespace
        from torch.utils.data import DataLoader
        class TinyGraph(torch.nn.Linear):
            def __init__(self, **config):
                super().__init__(1, 1)
                self.config = config
        datasets = {size: [{'x': torch.ones(2, 1)}] * 3 for size in (2, 3, 4)}
        _, artifacts = self.inputs()
        namespace = dict(torch=torch, np=np, pd=pd, random=random, DataLoader=DataLoader,
                         BATCH_SIZE=2, train_state_datasets=datasets, val_state_datasets=datasets,
                         base_feature_builder=SimpleNamespace(dim=1), graph_edge_builder=SimpleNamespace(dim=1),
                         state_artifacts=artifacts, GraphSetTransformer=TinyGraph, device='cpu',
                         collate_keep_wordlists=torch.utils.data.default_collate, LambdaNDCGLoss2=SimpleLoss,
                         score_batch=lambda model, x, batch, group_idx: model(x),
                         evaluate_state_ndcg=lambda model, loaders, artifacts, device: {s: .5 for s in loaders})
        wanted = [node for node in definitions() if node.name in {'run_learning_diagnostic', '_next_batch'}]
        exec(compile(ast.Module(body=wanted, type_ignores=[]), 'diagnostic', 'exec'), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            model, history = namespace['run_learning_diagnostic'](total_updates=4, eval_every=2, sample_states=2)
        self.assertEqual(history['updates'].tolist(), [0, 2, 4])
        self.assertEqual(model.config['dropout'], 0.)
        self.assertEqual(history['lr'].tolist(), [3e-4] * 3)
        counts = [history.iloc[-1][f'updates_s{s}'] for s in (2, 3, 4)]
        self.assertEqual(sum(counts), 4)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertTrue(all(f'{split}_ndcg_s{s}' in history for split in ('train', 'val') for s in (2, 3, 4)))


if __name__ == '__main__': unittest.main()
