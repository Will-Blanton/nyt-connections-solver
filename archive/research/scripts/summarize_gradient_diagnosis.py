"""Summarize sampled gradients on the common first 1500 training updates."""
import json
from pathlib import Path
from statistics import median


def summarize(directory=Path('experiments/gradient-diagnosis')):
    runs = []
    for path in sorted(directory.glob('*.json')):
        run = json.loads(path.read_text())
        records = run['training']['gradient_diagnostics']
        common = [row for row in records if row['update'] <= 1500]
        modules = {}
        for name in common[0]['parameters']:
            rows = [r['parameters'][name] for r in common]
            modules[name] = {
                'trainable': rows[0]['trainable'],
                **{f'median_{key}': median(r[key] for r in rows) for key in (
                    'gradient_norm', 'gradient_rms', 'gradient_nonzero_fraction', 'update_to_weight')},
                'zero_gradient_samples': sum(r['gradient_norm'] == 0 for r in rows),
                'end_drift': records[-1]['parameters'][name]['drift_from_initial'],
            }
        relu = {name: {key: median(r['relu'][name][key] for r in common)
                       for key in ('active_fraction', 'inactive_unit_fraction')}
                for name in common[0]['relu']}
        runs.append({'source': str(path), 'variant': run['config']['variant'],
                     'seed': run['config']['seed'], 'sample_count': len(common),
                     'updates_ran': run['training']['updates_ran'],
                     'best_update': run['training']['best_update'],
                     's4_hit1': run['training']['validation']['s4']['hit1'],
                     'loss_1500': next(r['train_loss'] for r in run['training']['history'] if r['update'] == 1500),
                     'modules': modules, 'relu': relu})
    return runs


if __name__ == '__main__':
    results = summarize()
    Path('experiments/gradient-diagnosis-summary.json').write_text(json.dumps(results, indent=2) + '\n')
    for run in results:
        print(run['variant'], run['seed'], 'Hit@1', round(run['s4_hit1'], 4))
        for name, row in run['modules'].items():
            if row['trainable']:
                print(name, 'gradient RMS', f"{row['median_gradient_rms']:.3g}",
                      'update/weight', f"{row['median_update_to_weight']:.3g}",
                      'zero samples', row['zero_gradient_samples'])
