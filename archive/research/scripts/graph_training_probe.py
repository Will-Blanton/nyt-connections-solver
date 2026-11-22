"""Sparse training measurements without changing the optimization objective."""
import torch
from torch import nn


class TrainingProbe:
    def __init__(self, model):
        names = ['input_norm', 'proj', 'edge_encoder.node_projection']
        for collection in ('edge_projections', 'updates'):
            names.extend(f'edge_encoder.{collection}.{i}'
                         for i in range(len(getattr(model.edge_encoder, collection))))
        names.append('group_readout')
        if hasattr(model, 'embedding_adapter'):
            names.append('embedding_adapter')
        for name in ('transform_projection', 'type_embedding', 'transform_encoder', 'pair_encoder'):
            if hasattr(model, name):
                names.append(name)
        modules = dict(model.named_modules())
        self.modules = {name: modules[name] for name in names}
        self.initial = {name: [p.detach().clone() for p in m.parameters()]
                        for name, m in self.modules.items()}
        self.current = None
        self.records = []
        self.handles = []
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self.activation_hook(name)))
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU) and name.startswith(('proj', 'edge_encoder', 'group_readout', 'embedding_adapter')):
                self.handles.append(module.register_forward_hook(self.relu_hook(name)))

    def begin(self, step, size):
        self.current = {'update': step, 'state_size': size, 'activations': {}, 'relu': {}}

    def activation_hook(self, name):
        def hook(module, inputs, output):
            if self.current is None:
                return
            values = output.detach().float()
            row = {'rms': values.square().mean().sqrt().item()}
            self.current['activations'][name] = row
            if output.requires_grad:
                output.register_hook(lambda grad: row.update(
                    gradient_rms=grad.detach().float().square().mean().sqrt().item()))
        return hook

    def relu_hook(self, name):
        def hook(module, inputs, output):
            if self.current is not None:
                active = output.detach() > 0
                self.current['relu'][name] = {
                    'active_fraction': active.float().mean().item(),
                    'inactive_unit_fraction': (~active.reshape(-1, active.shape[-1]).any(0)).float().mean().item(),
                }
        return hook

    def before_step(self):
        self.before = {name: [p.detach().clone() for p in m.parameters()]
                       for name, m in self.modules.items()}
        self.current['parameters'] = {}
        for name, module in self.modules.items():
            ps = list(module.parameters())
            count = sum(p.numel() for p in ps)
            sq = sum(p.detach().float().square().sum().item() for p in ps)
            grads = [p.grad.detach().float() for p in ps if p.grad is not None]
            gsq = sum(g.square().sum().item() for g in grads)
            self.current['parameters'][name] = {
                'trainable': any(p.requires_grad for p in ps),
                'weight_norm': sq ** .5, 'gradient_norm': gsq ** .5,
                'gradient_rms': (gsq / max(count, 1)) ** .5,
                'gradient_nonzero_fraction': sum(g.count_nonzero().item() for g in grads) / max(count, 1),
            }

    def after_step(self, loss):
        for name, module in self.modules.items():
            ps = list(module.parameters())
            delta = sum((p.detach().float() - b.float()).square().sum().item()
                        for p, b in zip(ps, self.before[name])) ** .5
            drift = sum((p.detach().float() - b.float()).square().sum().item()
                        for p, b in zip(ps, self.initial[name])) ** .5
            initial_norm = sum(b.float().square().sum().item() for b in self.initial[name]) ** .5
            row = self.current['parameters'][name]
            row.update(update_to_weight=delta / max(row['weight_norm'], 1e-12),
                       drift_from_initial=drift / max(initial_norm, 1e-12))
        self.current['loss'] = loss.item()
        self.records.append(self.current)
        self.current = None

    def close(self):
        for handle in self.handles:
            handle.remove()
