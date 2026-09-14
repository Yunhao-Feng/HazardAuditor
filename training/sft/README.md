# Supervised fine-tuning

HazardAuditor begins with full-parameter supervised fine-tuning of
`Qwen/Qwen3Guard-Gen-8B`. Targets contain an evidence-grounded analysis followed
by one strict binary verdict:

```text
<analysis>
trajectory-grounded safety rationale
</analysis>
<label>safe|unsafe</label>
```

## Data contract

The repository does not distribute training or evaluation trajectories. Supply
explicit JSON or JSONL files whose records contain:

```json
{
  "content": [{"role": "user", "content": "..."}],
  "label": 0,
  "reason": "Evidence-grounded rationale.",
  "source": "optional-source-name"
}
```

`label` uses `0 = safe` and `1 = unsafe`. Training and validation files must be
separate. The preparation command hashes complete trajectories, removes exact
train/validation overlap, deduplicates training rows, and records provenance in
manifests without copying source paths into the model prompt.

```bash
python -m training.sft.prepare_data \
  --train-file SourceA=/path/to/train.json \
  --validation-file SourceA=/path/to/validation.json \
  --output-dir artifacts/sft_data \
  --overwrite
```

Repeat either file option to combine multiple sources.

## Environment

The custom launcher was developed with LLaMA-Factory `0.9.5.dev0` at commit
`6b9df75ab9823d69c8c66a309389350585fbe728`:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 6b9df75ab9823d69c8c66a309389350585fbe728
python -m pip install -e .
python -m pip install -r requirements/metrics.txt -r requirements/deepspeed.txt
```

From the HazardAuditor repository, launch the eight-GPU recipe with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  training/sft/train.py training/sft/configs/qwen3_guard_full_sft.yaml \
  --resume auto
```

Prepared data and checkpoints are written under `artifacts/` and are excluded
from version control.
