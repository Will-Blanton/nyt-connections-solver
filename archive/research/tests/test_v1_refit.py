"""Refit budgets derive from validation history; training never evaluates test."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refit_gine_v1 import refit_budget, train_fixed


class RefitTests(unittest.TestCase):
    def test_budget_preserves_exposure_and_is_robust_to_late_outlier(self):
        self.assertEqual(refit_budget([100,200,10000],[100,100,100],200),402)
        self.assertEqual(refit_budget([100,200,10000],[100,200,10000],200),198)
        for updates,counts,pool in (([],[],100),([100],[0],100),([100],[100],0),([1,2],[1],10)):
            with self.assertRaises(ValueError):refit_budget(updates,counts,pool)

    def test_fixed_training_finishes_without_evaluating_or_restoring_checkpoint(self):
        torch.set_num_threads(1)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.zeros(2,1))
            def forward(self,x,group_idx=None,edge_features=None):return x@self.weight
        model=Toy()
        batch=dict(x=torch.eye(2)[None],edge_features=torch.zeros(1,2,2,1))
        loaders={s:[batch] for s in (2,3,4)}
        artifacts={s:dict(idx=torch.tensor([[0],[1]]),y=torch.tensor([True,False])) for s in loaders}
        progress=[]
        with patch('refit_gine_v1.experiment.evaluate',side_effect=AssertionError('No validation/test during refit')):
            trained,summary=train_fixed(model,loaders,artifacts,torch.device('cpu'),seed=10,
                updates=6,lr=.01,weight_decay=0.,log_every=3,callback=progress.append)
        self.assertEqual(summary['updates_ran'],6)
        self.assertEqual(summary['state_updates'],{2:2,3:2,4:2})
        self.assertEqual([r['update'] for r in progress],[3,6])
        self.assertGreater(trained.weight[0].item(),trained.weight[1].item())
        self.assertFalse(trained.training)


if __name__=='__main__':unittest.main()
