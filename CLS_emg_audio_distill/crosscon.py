# encoding: utf-8
"""crossCon loss from MONA 

sim(z_i, z_j; tau) = exp(cos(z_i, z_j) / tau)
L^cross_i = -log( sim(z_i, z_{j(i)}) / sum_{j != i} sim(z_i, z_j) )
L^cross   = mean_i L^cross_i

The positive of an EMG embedding is the audio embedding of the SAME utterance
(and vice versa); distractors are every other embedding in the batch, drawn
from BOTH modalities. Cosine similarity, temperature tau = 0.1 per the paper.

Because the EMG encoder collapses to a single temporal token, this is applied
at utterance granularity: one pooled embedding per modality per utterance.
"""
import torch
import torch.nn.functional as F


def crosscon_loss(z_emg, z_audio, tau=0.1):
    """z_emg, z_audio: (B, D) L2-normalizable embeddings. Returns scalar loss."""
    B = z_emg.size(0)
    z = torch.cat([z_emg, z_audio], dim=0)        # (2B, D): rows 0..B-1 EMG, B..2B-1 audio
    z = F.normalize(z, dim=1)                      # cosine similarity via dot product
    logits = (z @ z.t()) / tau                     # (2B, 2B), entry = cos/tau

    # exclude self-comparison (j != i) so it never appears in the denominator
    logits.fill_diagonal_(float('-inf'))

    # positive index: EMG row i <-> audio row i+B, and audio row i <-> EMG row i-B
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)

    # cross_entropy reproduces -log( sim(i,pos) / sum_{j!=i} sim(i,j) ) averaged over rows
    return F.cross_entropy(logits, targets)
