# Small residual embedding adapters

Validation only, four matched seeds (0–3). Frozen Qwen 1024-dimensional embeddings feed a trainable LayerNorm, then `z + up(ReLU(down(z)))`, then a learned 1024→32 GINE node projection, two GINE message-passing layers with the original 15 edge channels, mean/max pooling, and the existing readout. Adapter bottlenecks are 16 and 64. The up projection starts at zero, so the adapter initially preserves z. No text-only model weights are loaded; the reference checkpoint supplies split/feature metadata. Only Qwen is frozen in the active computation path. Inherited unused attention modules remain in the experimental checkpoint scaffold.

The no-adapter control uses the same trainable LayerNorm and GINE. Adapter weights are initialized last, preserving identical initial GINE/readout tensors under matched seeds. Control and adapters have the same initial predictions. Adapter bottlenecks change only the residual correction capacity, unlike the previous projection-width experiments.

Same 968 training/121 validation puzzles, state sampling, batch size 16, probability-mass loss, AdamW LR 1e-4, weight decay 1e-6, up to 3000 updates, minimum stopping budget 1500, patience 8 evaluations, evaluation every 100 updates, selection by s4 Hit@1 then MRR. Existing test puzzles and notebooks were untouched.

| Variant | Trainable parameters | Mean s4 Hit@1 | s4 MRR | s4 Hit@4 | s3 Hit@1 | s2 Hit@1 |
|---|---:|---:|---:|---:|---:|---:|
| No adapter | 48,483 | 19.21% | 0.2698 | 30.79% | 20.71% | 28.27% |
| Bottleneck 16 | 82,291 | 18.18% | 0.2619 | 30.58% | 20.56% | 28.31% |
| Bottleneck 64 | 180,643 | 15.50% | 0.2246 | 25.21% | 17.67% | 24.66% |
| Single Linear/ReLU residual | 1,098,083 | 10.74% | 0.1737 | 19.83% | 13.43% | 22.14% |

Per-seed s4 Hit@1:

| Seed | Control | Adapter 16 | Adapter 64 |
|---:|---:|---:|---:|
| 0 | 21.49% | 19.01% | 14.05% |
| 1 | 21.49% | 19.83% | 11.57% |
| 2 | 8.26% | 12.40% | 16.53% |
| 3 | 25.62% | 21.49% | 19.83% |

Neither adapter improves mean ranking under this training setup. Both improve the weak seed 2 but worsen the other three. The larger adapter is worse on average. This is a negative result for these adapters and hyperparameters; it does not establish that adapters generally cannot help. GINE hidden width remains 32, so this experiment does not test widening the message-passing bottleneck. No feedback harness replays were run for these variants.

Reproduce with `nix develop -c uv run python scripts/run_graph_feature_experiment.py --variant standalone-gine-adapter-WIDTH --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt --seed SEED --updates 3000 --min-updates-before-stop 1500 --patience 8 --lr 1e-4 --weight-decay 1e-6`, where WIDTH is 0, 16, or 64. Logs: `experiments/graph-features/adapter-{width}-{seed}.log`; JSON summaries/checkpoints: `*standalone-gine-adapter-{width}.{json,pt}` in the usual experiment directories.

Checks cover identical initial adapter/control outputs, gradients reaching the adapter down projection after its zero-initialized up projection updates, trainable LayerNorm gradients, and checkpoint reload.

## Single-layer residual follow-up

User-requested architecture: `z = LayerNorm(frozen_Qwen_embedding); words = z + ReLU(Linear(1024,1024)(z))`; then the existing learned GINE node projection 1024→32, message-passing MLPs, pooling, and readout. There is exactly one affine layer in the adapter, with a dimension-preserving residual skip. All active non-Qwen layers train from scratch. This has 1,098,083 trainable parameters, so it is shallower but larger than the bottleneck adapters. It uses normal random initialization: zero initialization before ReLU would block adapter weight gradients at startup. Thus unlike the bottleneck adapters, it does not start as the identity.

Same seeds and training/selection settings as above. s4 Hit@1 for seeds 0–3: 12.40%, 5.79%, 6.61%, 18.18%; selected updates 1600, 1500, 1800, 1500. All four seeds underperformed the matched no-adapter control. No harness replays were run for this negative ranking result. These results concern this architecture and training configuration; they do not prove that every single-layer residual adapter is ineffective.

Runner variant: `standalone-gine-adapter-single`. Logs: `experiments/graph-features/adapter-single-{seed}.log`; summaries/checkpoints: `*standalone-gine-adapter-single.{json,pt}`. Tests check that the adapter contains exactly one Linear, that GINE receives the specified residual sum, that gradients reach both the affine layer and LayerNorm, and that reload reproduces scores.
