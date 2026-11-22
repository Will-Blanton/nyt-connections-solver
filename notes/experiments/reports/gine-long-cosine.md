# One global cosine schedule with longer training

Tested one schedule without additional tuning: a global cosine decay from LR 1e-4 on update 1 to zero on update 10000. All trainable layers share the same rate. No warmup, separate encoder LR, early encoder freeze, auxiliary loss, or schedule sweep.

Architecture is the original pretrained-input-projection standalone GINE, fully unfrozen in its active computation: input LayerNorm → Linear(1024,256)/ReLU → GINE node Linear(256,32) → two message-passing layers → pooled readout. Input norm/projection initialize from the text-only checkpoint; both train, along with GINE/readout. Qwen embeddings remain frozen. The original 15 edge channels and same 968 training/121 validation puzzles are used. Four seeds (0–3), batch size 16, AdamW weight decay 1e-6, 10000 updates for every run, validation every 100, selected checkpoint by s4 Hit@1 then MRR. No test evaluations or notebook edits.

Controls are the corresponding constant-rate 10000-update runs from `experiments/long-unfrozen-projection/`.

| Validation metric | Constant LR | Cosine decay |
|---|---:|---:|
| s4 Hit@1 | 24.59% | 21.28% |
| s4 MRR | 0.3183 | 0.2861 |
| s4 Hit@4 | 33.88% | 32.02% |
| s3 Hit@1 | 24.95% | 22.11% |
| s2 Hit@1 | 31.44% | 29.06% |

| Seed | Constant s4 Hit@1 | Cosine s4 Hit@1 | Constant selected update | Cosine selected update |
|---:|---:|---:|---:|---:|
| 0 | 26.45% | 19.83% | 6000 | 2200 |
| 1 | 25.62% | 19.01% | 7400 | 2400 |
| 2 | 24.79% | 23.97% | 8800 | 2000 |
| 3 | 21.49% | 22.31% | 5500 | 4300 |

Cosine improves one seed and worsens three. Its selected checkpoints all precede the second half of training, unlike the constant-rate controls. This does not support replacing the constant LR for this setup. It does not establish that every schedule is ineffective, but no further schedule settings were tested. No harness replays were run because ranking did not improve. GPU numerical determinism is not enforced, and this remains repeated validation analysis.

Implementation: `--cosine-lr` (default off), using the shared `cosine_to_zero` function in `scripts/graph_training_schedule.py`. It cannot combine with the separate projection LR or encoder annealing. The actual LR is logged at every evaluation as `training.history[*].learning_rate`. Tests verify schedule endpoints, monotonicity and the halfway value.

Artifacts: `experiments/long-cosine-unfrozen/*standalone-gine-trainable-projection.json`, matching checkpoints under `weights/experiments/`, `summary.json`, and [learning-curves.html](long-cosine-unfrozen/learning-curves.html). Regenerate summaries with `scripts/summarize_long_gine_training.py --directory experiments/long-cosine-unfrozen --variant standalone-gine-trainable-projection` under `nix develop -c uv run python`.

Reproduce training with the usual text-only reference and `--variant standalone-gine-trainable-projection --train-input-norm --seed SEED --batch-size 16 --updates 10000 --min-updates-before-stop 10000 --patience 8 --lr 1e-4 --weight-decay 1e-6 --cosine-lr --output-dir experiments/long-cosine-unfrozen`.
