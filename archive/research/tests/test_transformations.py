import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.transformation_nodes import TransformationNodeFeatures
from nyt_connections_solver.transformation_models import TransformationGINE
from nyt_connections_solver.graph_models import StandaloneCandidateGINE
from nyt_connections_solver.feature_cache import TensorCache
from nyt_connections_solver.node_features import TextSpellingFeatures


class TransformationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.config = dict(in_dim=8+4*15, text_dim=8, edge_dim=3, base_edge_dim=3,
            max_transformations=4, transformation_width=8, learned_edge_dim=4,
            attn_width=8, attn_heads=2, attn_layers=1, lin_width=16, lin_layers=1,
            edge_bias_scale=0., dropout=0., edge_hidden=8, train_text_projection=True, train_input_norm=True)
        self.idx=torch.tensor([[0,1,2,3],[0,1,2,4]])
        self.x=torch.randn(2,5,self.config['in_dim'])
        slots=self.x[...,8:].reshape(2,5,4,15)
        slots[...,8:14]=torch.rand(2,5,4,6)
        slots[...,8]=torch.randint(4,(2,5,4)).float()
        slots[...,-1]=1
        self.edges=torch.rand(2,5,5,3)
        self.edges=(self.edges+self.edges.transpose(1,2))/2

    def test_extraction_position_and_cap(self):
        class Base:
            dim=8
            embeddings=type('E',(),dict(dim=8))()
            cache_identity={}
        lexicon=type('L',(),dict(words={'art','cart','car','tar','star'},identity={}))()
        features=TransformationNodeFeatures(Base(),lexicon,max_transformations=4)
        rows=features.alternatives('CARTS',truncate=False)
        self.assertIn(('cart',1,0,4),rows)
        self.assertIn(('art',3,1,4),rows)
        self.assertEqual(len(features.alternatives('CARTS')),4)
        self.assertEqual(features.alternatives('two words'), [('two words',0,0,9)])
        self.assertEqual(features.alternatives('x'), [('x',0,0,1)])

    def test_payload_cache_and_exact_offsets_across_lengths(self):
        class Embeddings:
            dim=8
            identity={'type':'test_vectors'}
            cache=TensorCache()
            def encode(self,words):
                return torch.stack([torch.arange(8).float()+len(w) for w in words])
        base=TextSpellingFeatures(Embeddings(),use_spelling=False)
        lexicon=type('L',(),dict(words={'mart','harts'},identity={}))()
        features=TransformationNodeFeatures(base,lexicon,max_transformations=4)
        x=features.encode(['smart','charts'])
        self.assertEqual(x.shape,(2,68))
        torch.testing.assert_close(x,features.encode(['smart','charts']))
        slots=x[:,8:].reshape(2,4,15)
        self.assertEqual(slots[0,1,9],slots[1,1,9])  # one character from left
        self.assertEqual(slots[0,1,9],1/32)
        self.assertNotEqual(slots[0,1,11],slots[1,1,11])  # relative offset differs
        self.assertEqual(slots[:,2:,-1].sum(),0)

    def test_original_only_matches_single_pair_and_masks_other_readings(self):
        torch.manual_seed(9)
        full=TransformationGINE(group_idx=self.idx,**self.config)
        torch.manual_seed(9)
        original=TransformationGINE(group_idx=self.idx,original_only=True,**self.config)
        for name,value in full.state_dict().items():
            torch.testing.assert_close(value,original.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in full.parameters() if p.requires_grad),
                         sum(p.numel() for p in original.parameters() if p.requires_grad))
        x=self.x.clone()
        slots=x[...,8:].reshape(2,5,4,15)
        slots[...,8]=1
        slots[:,:,0,8]=0
        z,valid,pos=original.encode_transformations(x)
        self.assertTrue(valid[:,:,0].all())
        self.assertFalse(valid[:,:,1:].any())
        torch.testing.assert_close(original.pair_edges(z,valid,pos),
            original.pair_edges(z[:,:,:1],valid[:,:,:1],pos[:,:,:1]))
        with torch.no_grad():
            for layer in original.edge_encoder.edge_projections:
                layer.weight[:,-4:].fill_(.1)
        actual=original(x,edge_features=self.edges)
        changed=x.clone()
        changed[...,8:].reshape(2,5,4,15)[:,:,1:,:8]=1000
        torch.testing.assert_close(actual,original(changed,edge_features=self.edges))
        x.requires_grad_()
        original(x,edge_features=self.edges).sum().backward()
        grads=x.grad[...,8:].reshape(2,5,4,15)
        self.assertEqual(grads[:,:,1:,:8].abs().sum(),0)
        self.assertGreater(grads[:,:,0,:8].abs().sum(),0)
        reloaded=TransformationGINE(group_idx=self.idx,original_only=True,**self.config)
        reloaded.load_state_dict(original.state_dict())
        torch.testing.assert_close(actual,reloaded(x.detach(),edge_features=self.edges))

    def test_symmetry_padding_permutation_and_gradients(self):
        model=TransformationGINE(group_idx=self.idx,**self.config)
        z,valid,pos=model.encode_transformations(self.x)
        extra=model.pair_edges(z,valid,pos)
        torch.testing.assert_close(extra,extra.transpose(1,2))
        slots=self.x[...,8:].reshape(2,5,4,15)
        slots[:,:,3,-1]=0
        a=model(self.x,edge_features=self.edges)
        changed=self.x.clone(); changed[...,8:].reshape(2,5,4,15)[:,:,3,:8]=1e3
        torch.testing.assert_close(a,model(changed,edge_features=self.edges))
        # Permuting transformation slots leaves edge pooling invariant.
        changed=self.x.clone(); changed[...,8:]=slots.flip(-2).flatten(-2)
        torch.testing.assert_close(a,model(changed,edge_features=self.edges))
        perm=torch.tensor([2,0,4,1,3])
        idx=perm.argsort()[self.idx]
        torch.testing.assert_close(a,model(self.x[:,perm],group_idx=idx,
            edge_features=self.edges[:,perm][:,:,perm]))
        # New edge columns start at zero; activate them to test end-to-end gradients.
        with torch.no_grad():
            for p in model.edge_encoder.edge_projections: p.weight[:,-4:].fill_(.1)
        loss=model(self.x,edge_features=self.edges).square().sum(); loss.backward()
        self.assertGreater(model.transform_projection.weight.grad.abs().sum(),0)
        self.assertGreater(model.pair_encoder[0].weight.grad.abs().sum(),0)
        empty=torch.zeros_like(valid)
        self.assertEqual(model.pair_edges(z,empty,pos).abs().sum(),0)

    def test_reload_modes_and_legacy_control(self):
        for mode in ('text','fused','constant'):
            config={**self.config,'node_mode':mode}
            model=TransformationGINE(group_idx=self.idx,**config).eval()
            reload=TransformationGINE(group_idx=self.idx,**config).eval()
            reload.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(self.x,edge_features=self.edges),reload(self.x,edge_features=self.edges))
            model.zero_grad()
            model(self.x,edge_features=self.edges).sum().backward()
            if mode=='fused':
                self.assertGreater(model.node_fusion[0].weight.grad.abs().sum(),0)
                self.assertGreater(model.node_gate.weight.grad.abs().sum(),0)
            if mode=='constant':
                self.assertFalse(model.proj[0].weight.requires_grad)
                self.assertGreater(model.shared_node.grad.abs().sum(),0)
            representations=model.diagnostic_representations(self.x[0],self.edges[0])
            self.assertEqual(representations['GINE input words'].shape[0],5)
        config={k:v for k,v in self.config.items() if k not in {'text_dim','base_edge_dim','max_transformations','transformation_width','learned_edge_dim'}}
        config['in_dim']=8
        torch.manual_seed(5); legacy=StandaloneCandidateGINE(group_idx=self.idx,**config).eval()
        torch.manual_seed(5); control=TransformationGINE(group_idx=self.idx,learned_edges=False,**self.config).eval()
        torch.testing.assert_close(legacy(self.x[...,:8],edge_features=self.edges),control(self.x,edge_features=self.edges),rtol=1e-5,atol=1e-6)

    def test_probe_tracks_learned_edges_without_changing_updates(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from graph_training_probe import TrainingProbe
        model=TransformationGINE(group_idx=self.idx,**self.config)
        reference=TransformationGINE(group_idx=self.idx,**self.config)
        reference.load_state_dict(model.state_dict())
        probe=TrainingProbe(model)
        # Two steps: zero-initialized edge columns initially block gradients
        # into the new encoder, then the first update opens that path.
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (model,reference)]
        for step in (1,2):
            for m,optimizer in zip((model,reference),optimizers):
                optimizer.zero_grad()
                if m is model: probe.begin(step,4)
                loss=m(self.x,edge_features=self.edges).square().sum()
                loss.backward()
                if m is model: probe.before_step()
                optimizer.step()
                if m is model: probe.after_step(loss)
        probe.close()
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
        first,second=[r['parameters'] for r in probe.records]
        self.assertEqual(first['transform_projection']['gradient_norm'],0)
        self.assertGreater(second['transform_projection']['gradient_norm'],0)
        self.assertGreater(second['pair_encoder']['update_to_weight'],0)

    def test_smaller_pair_matchers_and_single_gine_layer_reload_and_receive_gradients(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from graph_training_probe import TrainingProbe
        for layers in (1,2):
            config={**self.config,'transformation_width':4,'pair_hidden':4,'pair_layers':layers,
                    'gine_layers':1,'edge_hidden':4,'readout_width':4}
            model=TransformationGINE(group_idx=self.idx,**config)
            with torch.no_grad():
                model.edge_encoder.edge_projections[0].weight[:,-4:].fill_(.1)
            reload=TransformationGINE(group_idx=self.idx,**config)
            reload.load_state_dict(model.state_dict())
            actual=model(self.x,edge_features=self.edges)
            torch.testing.assert_close(actual,reload(self.x,edge_features=self.edges))
            self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.pair_encoder),layers)
            probe=TrainingProbe(model)
            self.assertNotIn('edge_encoder.updates.1',probe.modules)
            probe.begin(1,4)
            loss=model(self.x,edge_features=self.edges).square().sum()
            loss.backward();probe.before_step()
            torch.optim.AdamW(model.parameters(),lr=.001).step();probe.after_step(loss);probe.close()
            self.assertGreater(probe.records[0]['parameters']['pair_encoder']['gradient_norm'],0)
        with self.assertRaises(ValueError):
            TransformationGINE(group_idx=self.idx,pair_layers=0,**self.config)
