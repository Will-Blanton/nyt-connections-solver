# Smaller learned-transformation GINE models (running)

Validation-only comparison of a fresh full-size control, smaller learned-edge branch, narrower GINE/readout, one GINE block and a single-linear-layer pair matcher. Three seeds, 30,000 updates each, batch 16, LR 3e-4, WD .001 and dropout .2. No early stopping or pruning. Frozen Qwen vectors, trainable original text projection and LayerNorm. No notebook or test-set changes.

The text projection/LN retains 264,448 parameters in every arm. The overall active capacity decreases from 484,787 to 422,195 / 327,667 / 317,362 / 317,090 parameters.

Full results and interpretation will replace this note after the runs and checkpoint audit complete.
