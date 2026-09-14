# GuardPO training

This directory contains the complete VERL-based Guard Policy Optimization
(GuardPO) recipe used by HazardAuditor. It keeps the original deterministic
outcome reward, batch-centered advantage, clipped token-level importance
weighting, response-level normalization, and best/latest checkpoint retention.

The implementation targets the following audited stack:

```text
Python       3.10.13
PyTorch      2.8.0
Transformers 4.57.1
vLLM         0.11.0
Ray          2.50.1
VERL         0.7.0.dev0 (tag v0.7.0, commit f9c855f7cf04d603c9546bc01776c74806a879c1)
```

## Prepare data

First create the SFT artifacts described in `../sft/README.md`, then convert
the unique prompts into the balanced GuardPO schedule:

```bash
python -m training.guardpo.cispo.prepare_rl_data \
  --input-dir artifacts/sft_data \
  --output-dir artifacts/guardpo_data \
  --overwrite
```

No judge API, learned reward model, or network service is used during GuardPO.
Gold rationales remain hidden from the actor and do not contribute to reward.

## Launch

Install VERL and the versions above in the GPU training environment, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u -m training.guardpo.train \
  --model artifacts/checkpoints/sft/best_checkpoint \
  --tokenizer artifacts/checkpoints/sft/best_checkpoint \
  --resume-mode disable
```

The default run uses 32 prompts per global batch, eight rollouts per prompt,
16,000 prompt tokens, 384 response tokens, a `2e-7` learning rate, and a
label-loss weight of 2.0. Outputs are written beneath
`artifacts/checkpoints/guardpo/`, which is ignored by Git.

## Compatibility names

Some internal classes, environment variables, and VERL registry keys retain
the historical `RLGuard` / `rlguard_*` names. They are compatibility
interfaces used by the released training recipe, not the public model name.
The public model and method names are HazardAuditor and GuardPO.

See [ALGORITHM.md](ALGORITHM.md) for the mathematical objective and design
constraints.
