# encoding: utf-8
"""Paired EMG + audio loader for crossCon distillation.

Returns (emg, audio, label) for the same utterance, matched by file path
(..._emg.mat <-> ..._audio.wav), on the same strict 70/10/20 cross-subject
split used by the EMG-only baseline. Reuses the baseline EMG and audio
feature extraction so the student/teacher see identical inputs to their
single-modal counterparts. Audio is loaded clean (no noise) for the teacher.
"""
import os
import random
import numpy as np
import scipy.io as sio
import librosa
import torch

# baseline preprocessing (CLS_emg_only and CLS_audio_only must be on PYTHONPATH)
from emg_dataset import filter as emg_filter, EMG_MFSC
from audio_dataset import get_MFSC


class PairedDataset:
    def build_file_list(self, set, directory):
        subjects = sorted(
            os.path.join(directory, d) for d in os.listdir(directory)
            if d.startswith('Subj')
        )
        # strict cross-subject split, identical to the EMG-only baseline
        split = {'train': subjects[:70], 'val': subjects[70:80], 'test': subjects[80:]}[set]

        items = []
        for subj in split:
            for session in sorted(os.listdir(subj)):
                session_path = os.path.join(subj, session)
                if not os.path.isdir(session_path):
                    continue
                for sample in os.listdir(session_path):
                    if not sample.endswith('_emg.mat'):
                        continue
                    label = sample.split('_emg.mat')[0].split('Sent')[-1]
                    emg_path = os.path.join(session_path, sample)
                    audio_path = emg_path.replace('_emg.mat', '_audio.wav')
                    if os.path.exists(audio_path):
                        items.append((label, emg_path, audio_path))
        random.shuffle(items)
        return items

    def __init__(self, set, directory):
        self.set = set
        self.file_list = self.build_file_list(set, directory)
        print('Total num of paired samples: ', len(self.file_list))

    def __getitem__(self, idx):
        label, emg_path, audio_path = self.file_list[idx]

        # EMG -> (6, 36, 36)
        emg = sio.loadmat(emg_path)
        emg = np.expand_dims(emg['data'], axis=0)
        emg = emg_filter(emg)
        emg = EMG_MFSC(emg)                       # (1, 6, 36, 36)
        emg = torch.FloatTensor(emg)[0]           # (6, 36, 36)

        # Audio (clean, no added noise) -> (1, 60, 64)
        audio, fs = librosa.load(audio_path)
        audio = get_MFSC(fs, audio)               # (60, 64)
        audio = torch.FloatTensor(audio[np.newaxis, :])  # (1, 60, 64)

        return emg, audio, int(label)

    def __len__(self):
        return len(self.file_list)
