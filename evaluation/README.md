# Evaluation

This directory contains the single public evaluation entry point for the
generative HazardAuditor model. Benchmark inputs and predictions are not
distributed with this repository.

Prepare the validation split with `training/sft/prepare_data.py`, then run:

```bash
python -m evaluation.evaluate \
  --model-path artifacts/checkpoints/guardpo/best_checkpoint/actor/huggingface \
  --validation-file artifacts/sft_data/validation.jsonl
```

The evaluator uses the same output parser as the public inference API. An
unparseable generation is recorded with a null label; it is never silently
counted as safe.

See the root README for the paper-reported results. Do not compare new runs to
those values unless the same benchmark splits and evaluation protocol are
used.
