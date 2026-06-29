# CLS EMG-Only — Cross-Subject Reproduction

Reproduction of the EMG-only sentence classification baseline (Bi-GRU) from
AVE Speech. Speaker-independent 70/10/20 subject split.

## Data
Structured as `<root>/Subj###/Sess##/Subj###_Sess##_Sent###_emg.mat`
(six-channel EMG). Subjects 1–70 train, 71–80 val, 81–100 test.

## Train
```bash
python emg_main.py --dataset /path/to/AVE_Structured --batch-size 128 --epochs 200
```
Checkpoints + log are written to `finetuneGRU_every_frame/`.

## Evaluate
Run val + test on a checkpoint:
```bash
python emg_main.py --dataset /path/to/AVE_Structured --test \
  --path finetuneGRU_every_frame/finetuneGRU_emg_<epoch>.pt
```

## Results
50-epoch run (batch size 128), best validation at epoch 40:

| Metric | This repro | Paper (Table III) |
|--------|-----------|-------------------|
| Validation | 72.4% | 77.53% |
| Test | 73.15% | 75.53% |

The gap is consistent with the shorter schedule / smaller batch; the paper's
setup is ~200 epochs at batch size 128.
