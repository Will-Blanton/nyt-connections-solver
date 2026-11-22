"""Search-space coverage and conservative pruning despite distinct final metric."""
from pathlib import Path
import sys
import unittest

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tune_gine_probability_floor import suggest, make_pruner


class FloorTuningTests(unittest.TestCase):
    def test_known_good_configuration_is_in_search_space(self):
        params = dict(hidden_width=128, gine_layers=2, readout_width=128, dropout=0.,
                      lr=.0003016684667937828, weight_decay=.00009500890359387789, floor_weight=.1)
        self.assertEqual(suggest(optuna.trial.FixedTrial(params)), params)

    def test_pruning_uses_intermediate_hit1_and_waits_8000_updates(self):
        study = optuna.create_study(direction='maximize', pruner=make_pruner())
        for _ in range(6):
            trial = study.ask()
            for step in range(250, 15001, 250):
                trial.report(.3, step)
                if step < 8000:
                    self.assertFalse(trial.should_prune())
            # Final objective is harness solve rate, not the reported Hit@1.
            study.tell(trial, .05)
        trial = study.ask()
        for step in range(250, 8000, 250):
            trial.report(.2, step)
            self.assertFalse(trial.should_prune())
        trial.report(.2, 8000)
        self.assertTrue(trial.should_prune())

    def test_middle_trial_is_retained_below_upper_quartile(self):
        study = optuna.create_study(direction='maximize', pruner=make_pruner())
        for value in (.20, .25, .30, .35, .40, .45):
            trial = study.ask()
            trial.report(value, 8000)
            study.tell(trial, .1)
        trial = study.ask()
        trial.report(.32, 8000)
        self.assertFalse(trial.should_prune())
