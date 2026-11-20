import sys
from pathlib import Path
import unittest

import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tune_gine_coherence_cv import make_folds, make_pruner, suggest, build_fresh_model, STARTUP, fold_subsets


class CoherenceCVTests(unittest.TestCase):
    def test_folds_are_puzzle_disjoint_balanced_reproducible_and_exclude_test(self):
        dates = dict(train=[f't{i}' for i in range(21)], val=[f'v{i}' for i in range(10)], test=['heldout'])
        folds = make_folds(dates)
        self.assertEqual(folds, make_folds(dates))
        self.assertEqual(len(folds), 5)
        visited = []
        for fold in folds:
            self.assertFalse(set(fold['train']) & set(fold['val']))
            self.assertNotIn('heldout', fold['train']+fold['val'])
            self.assertEqual(set(fold['train']+fold['val']), set(dates['train']+dates['val']))
            visited.extend(fold['val'])
        self.assertEqual(len(visited), len(set(visited)))
        self.assertLessEqual(max(len(f['val']) for f in folds)-min(len(f['val']) for f in folds), 1)
        with self.assertRaises(ValueError):
            make_folds({**dates, 'test':['t0']})

    def test_fresh_initialization_is_matched_and_trainable_with_reload(self):
        torch.set_num_threads(1)
        base = dict(in_dim=8, edge_dim=3, edge_bias_scale=0., attn_width=8,
            attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1, dropout=0.)
        params = {**STARTUP[0], 'hidden_width':8, 'projection_width':8,
                  'readout_width':8, 'edge_coherence_scale':0.}
        idx = torch.tensor([[0,1,2,3], [0,1,2,4]])
        a, config = build_fresh_model(base, params, idx, 3)
        b, _ = build_fresh_model(base, {**params, 'edge_coherence_scale':1.}, idx, 3)
        for name,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        c,_ = build_fresh_model(base, params, idx, 4)
        self.assertFalse(torch.equal(a.proj[0].weight,c.proj[0].weight))
        x, edges = torch.randn(2,5,8), torch.rand(2,5,5,3)
        a.eval()
        score = a(x,edge_features=edges)
        c.load_state_dict(a.state_dict())
        c.eval()
        torch.testing.assert_close(score,c(x,edge_features=edges))
        score.square().sum().backward()
        for module in (a.input_norm,a.proj,a.edge_encoder,a.group_readout):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))

    def test_startup_params_in_space_and_baseline_has_no_strength_search(self):
        for common in STARTUP:
            baseline = suggest(optuna.trial.FixedTrial(common),'baseline')
            amplified = suggest(optuna.trial.FixedTrial({**common,'edge_coherence_scale':1.}),'coherence')
            self.assertEqual(baseline['edge_coherence_scale'],0.)
            self.assertEqual({k:v for k,v in amplified.items() if k!='edge_coherence_scale'},common)

    def test_pruning_waits_for_six_configurations_and_three_complete_folds(self):
        study = optuna.create_study(direction='maximize',pruner=make_pruner())
        for _ in range(6):
            trial = study.ask()
            for folds in range(1,6):
                trial.report(.4,folds)
                self.assertFalse(trial.should_prune())
            study.tell(trial,.4)
        trial = study.ask()
        for folds in (1,2):
            trial.report(.1,folds)
            self.assertFalse(trial.should_prune())
        trial.report(.1,3)
        self.assertTrue(trial.should_prune())

    def test_state_views_keep_all_partial_states_with_their_puzzle(self):
        datasets = {size:[dict(date=date) for date in ('a','b') for _ in range(n)]
                    for size,n in ((2,6),(3,4),(4,1))}
        views = fold_subsets(datasets,dict(train=['a'],val=['b']))
        for side,date in (('train','a'),('val','b')):
            for size,view in views[side].items():
                self.assertEqual(len(view),{2:6,3:4,4:1}[size])
                self.assertTrue(all(sample['date']==date for sample in view))


if __name__ == '__main__':
    unittest.main()
