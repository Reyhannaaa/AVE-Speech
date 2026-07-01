# CLS EMG - audio→EMG crossCon Distillation

The idea here is: EMG-only speech recognition is hard, but during training
we also have the *audio* of the same sentences sitting right there. So why not let
a good audio model quietly teach the EMG model what a sentence "should" look like
in latent space? That's exactly what this does, a frozen audio teacher shapes the
EMG student's features while it trains, using the **crossCon** loss from the MONA
paper (Benster et al., 2024). The audio is only ever seen during training; at test
time the model is pure EMG, and it's evaluated exactly like the plain baseline.

## How it works
- **Student** : the EMG model (`inputDim=256`), trained from scratch.
- **Teacher** : the audio Bi-GRU model (`inputDim=512`) from
  `pretrained-models/sentence-level-recognition/audio_BGRU.pt`, kept frozen. We
  read its 1024-d Bi-GRU features with a forward hook, so the audio code stays
  completely untouched.
- **Where they meet** : both models produce a 1024-d Bi-GRU latent. The EMG branch
  collapses time down to a single token, so rather than aligning frame-by-frame we
  align at the utterance level (the audio's 16 frames are mean-pooled to one).
- **The loss** : `L = CE(emg) + λ_cross · crossCon(z_emg, z_audio)`. crossCon pulls
  each utterance's EMG and audio embeddings together and pushes everything else
  apart (cosine similarity, `τ=0.1`, `λ_cross=1`, following MONA). The paired
  audio+EMG come from the same utterance on the same 70/10/20 cross-subject split,
  and the teacher always sees clean audio.

## Training
```bash
DISTILL_GPU=2 PYTHONPATH=CLS_emg_only:CLS_audio_only:CLS_emg_audio_distill \
  python CLS_emg_audio_distill/distill_main.py \
  --dataset /path/to/AVE_Structured --batch-size 128 --epochs 50 --workers 0
```
Checkpoints and the log land in `finetuneGRU_crosscon_every_frame/`. Keep
`--workers 0`: the GRU relies on a CUDA default tensor type, and forked DataLoader
workers can't initialize CUDA, so multi-worker loading crashes.

## Evaluating (EMG-only)
```bash
DISTILL_GPU=2 PYTHONPATH=CLS_emg_only:CLS_audio_only:CLS_emg_audio_distill \
  python CLS_emg_audio_distill/distill_main.py \
  --dataset /path/to/AVE_Structured --workers 0 \
  --test --path finetuneGRU_crosscon_every_frame/finetuneGRU_emg_<best>.pt
```

## Results
All numbers below are 50 epochs, batch 128, with the best-validation checkpoint
picked for testing, and remember, inference is EMG-only; the teacher is thrown
away once training ends.

Our first run (seed 5) was already encouraging:

| Metric | EMG baseline (CE only) | + crossCon distillation | Δ |
|--------|------------------------|-------------------------|------|
| Validation | 72.36% | 73.44% | +1.08 |
| Test | 73.15% | 74.61% | +1.46 |

### Was it real, or just luck?

One run improving doesn't prove much, it could easily be a lucky initialization.
So we wanted to check this properly and re-ran the whole comparison on two more
seeds. Here's every run side by side:

| Seed | Baseline (CE only) | + crossCon | Δ |
|------|--------------------|------------|------|
| 5 | 73.15% | 74.61% | +1.46 |
| 1 | 72.86% | 73.99% | +1.13 |
| 2 | 72.81% | 73.88% | +1.07 |
| **Mean ± std** | **72.94 ± 0.18** | **74.16 ± 0.39** | **+1.22** |

The gain holds up. Every single seed improves (by +1.07 to +1.46), and the part
that really settles it, the *worst* crossCon run (73.88%) still comes in ahead of
the *best* baseline run (73.15%). No overlap at all. This isn't seed noise; the
audio teacher genuinely helps the EMG model. (Reproduce any seed with `--seed <n>`;
each writes to its own `finetuneGRU_crosscon_every_frame_seed<n>/`.)

These are 50-epoch runs, and the baseline is a bit undertrained
compared to the paper (~75.5% at 200 epochs). A longer schedule should lift the
absolute numbers for both confirming the gain survives to convergence is the
natural next step.
