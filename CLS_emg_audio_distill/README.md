# CLS EMG — audio→EMG crossCon Distillation

Improve the **EMG-only** sentence classifier by aligning its latent space to a
frozen **audio teacher** during training, via the **crossCon** loss from MONA
(Benster et al., 2024). Audio is used *only at training time*; inference is
EMG-only (the trained student is a plain EMG model, evaluated like the baseline).

## Method
- **Student:** EMG `EMGNet` (`inputDim=256`), trained from scratch.
- **Teacher:** audio `ASR` Bi-GRU (`inputDim=512`), loaded from
  `pretrained-models/sentence-level-recognition/audio_BGRU.pt`, frozen. Its
  1024-d Bi-GRU latent is read via a forward hook (audio baseline untouched).
- **Latent:** Bi-GRU output (1024-d). EMG collapses to one temporal token, so
  crossCon is applied at **utterance level** (audio's 16 frames mean-pooled to 1).
- **Loss:** `L = CE(emg) + λ_cross · crossCon(z_emg, z_audio)`, with cosine
  similarity, `τ=0.1`, `λ_cross=1` (per MONA). Paired audio+EMG come from the
  same utterance on the same 70/10/20 cross-subject split; teacher is fed clean audio.

## Train
```bash
DISTILL_GPU=2 PYTHONPATH=CLS_emg_only:CLS_audio_only:CLS_emg_audio_distill \
  python CLS_emg_audio_distill/distill_main.py \
  --dataset /path/to/AVE_Structured --batch-size 128 --epochs 50 --workers 0
```
Checkpoints + log → `finetuneGRU_crosscon_every_frame/`. (`--workers 0` is
required: the GRU uses a CUDA default tensor type, which forked loader workers
cannot initialize.)

## Evaluate (EMG-only)
```bash
DISTILL_GPU=2 PYTHONPATH=CLS_emg_only:CLS_audio_only:CLS_emg_audio_distill \
  python CLS_emg_audio_distill/distill_main.py \
  --dataset /path/to/AVE_Structured --workers 0 \
  --test --path finetuneGRU_crosscon_every_frame/finetuneGRU_emg_<best>.pt
```

## Results
50 epochs, batch 128, best-validation checkpoint selected, single seed:

| Metric | EMG baseline (CE only) | + crossCon distillation | Δ |
|--------|------------------------|-------------------------|------|
| Validation | 72.36% | 73.44% | +1.08 |
| Test | 73.15% | 74.61% | +1.46 |

EMG-only at inference; the audio teacher is discarded after training.
Single-run numbers — multi-seed / longer-schedule confirmation is the next step.
