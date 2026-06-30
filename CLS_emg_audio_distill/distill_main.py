# coding: utf-8
"""audio -> EMG crossCon distillation.

Trains an EMG student from scratch with  L = CE(emg) + lambda_cross * crossCon,
using a FROZEN audio model (audio_BGRU.pt) as the cross-modal teacher. The
teacher shapes the EMG latent space during training only; inference is
EMG-only and uses the standard EMG classification path (so the resulting
checkpoint is a plain EMGNet, evaluatable like the baseline).
"""
import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lr_scheduler import AdjustLR           # CLS_emg_only on PYTHONPATH
from emg_model import emg_model             # student
from model import audio_model              # teacher (CLS_audio_only on PYTHONPATH)
from paired_dataset import PairedDataset
from crosscon import crosscon_loss

GPU = os.environ.get('DISTILL_GPU', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 5
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    # original GRU code allocates h0 on the default device; keep CUDA default
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def data_loader(args):
    dsets = {x: PairedDataset(x, args.dataset) for x in ['train', 'val', 'test']}
    loaders = {x: torch.utils.data.DataLoader(
        dsets[x], batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, generator=torch.Generator(device='cuda'))
        for x in ['train', 'val', 'test']}
    sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(sizes['train'], sizes['val'], sizes['test']))
    return loaders, sizes


def build_teacher(args, logger):
    """Frozen audio teacher. Returns (model, get_latent) where get_latent()
    yields the pooled (B, 1024) GRU latent captured from the last forward."""
    teacher = audio_model(mode='finetuneGRU', inputDim=512, hiddenDim=512,
                          nClasses=args.nClasses, every_frame=True)
    ck = torch.load(args.teacher, map_location='cpu')
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    md = teacher.state_dict()
    matched = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
    teacher.load_state_dict(matched, strict=False)
    logger.info('*** teacher loaded: {}/{} keys ***'.format(len(matched), len(md)))
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    cache = {}
    # inner nn.GRU returns (output, h_n); output is the (B, T, 1024) pre-fc latent
    teacher.gru.gru.register_forward_hook(lambda m, inp, out: cache.__setitem__('z', out[0]))

    def get_latent():
        return cache['z'].mean(dim=1)   # pool over time -> (B, 1024)
    return teacher, get_latent


def showLR(optimizer):
    return [pg['lr'] for pg in optimizer.param_groups]


def run_epoch(student, teacher, get_latent, loaders, phase, epoch, optimizer,
              criterion, args, logger, save_path):
    train = (phase == 'train')
    student.train() if train else student.eval()
    if train:
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    run_loss = run_ce = run_cc = run_corrects = run_all = 0.
    torch.set_grad_enabled(train)
    for batch_idx, (emg, audio, targets) in enumerate(loaders[phase]):
        emg = emg.float().cuda()
        targets = targets.cuda()

        logits_seq, feat = student(emg, return_features=True)   # (B,1,nCls), (B,1,1024)
        logits = logits_seq.mean(1)                              # (B, nCls)
        ce = criterion(logits, targets)

        if train:
            audio = audio.float().cuda()
            with torch.no_grad():
                teacher(audio)                                   # populates latent cache
            z_audio = get_latent()                               # (B, 1024)
            z_emg = feat.mean(1)                                 # (B, 1024)
            cc = crosscon_loss(z_emg, z_audio, tau=args.tau)
            loss = ce + args.lambda_cross * cc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_cc += cc.data * emg.size(0)
        else:
            loss = ce

        _, preds = torch.max(F.softmax(logits, dim=1).data, 1)
        run_loss += loss.data * emg.size(0)
        run_ce += ce.data * emg.size(0)
        run_corrects += (preds == targets.data).sum().item()
        run_all += emg.size(0)

    n = len(loaders[phase].dataset)
    acc = run_corrects / n
    if train:
        logger.info('train Epoch:\t{:2}\tLoss: {:.4f}\tCE: {:.4f}\tcrossCon: {:.4f}\tAcc:{:.4f}'.format(
            epoch, run_loss / n, run_ce / n, run_cc / n, acc) + '\n')
        torch.save(student.state_dict(), save_path + '/' + args.mode + '_emg_' + str(epoch + 1) + '.pt')
    else:
        logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(phase, epoch, run_loss / n, acc) + '\n')
    return acc


def main():
    parser = argparse.ArgumentParser(description='audio->EMG crossCon distillation')
    parser.add_argument('--nClasses', default=101, type=int)
    parser.add_argument('--dataset', required=True, help='structured dataset root (Subj*/Sess*/*.mat,*.wav)')
    parser.add_argument('--teacher', default='pretrained-models/sentence-level-recognition/audio_BGRU.pt')
    parser.add_argument('--mode', default='finetuneGRU')
    parser.add_argument('--every-frame', default=True, action='store_true')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lambda-cross', default=1.0, type=float)
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--test', default=False, action='store_true', help='eval val+test from --path, no training')
    parser.add_argument('--path', default='', help='student checkpoint for --test')
    args = parser.parse_args()

    save_path = './' + args.mode + '_crosscon_every_frame'
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger('mylog')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(save_path + '/' + args.mode + '_emg_' + str(args.lr) + '.txt', mode='a'))
    logger.addHandler(logging.StreamHandler())

    student = emg_model(mode=args.mode, inputDim=256, hiddenDim=512,
                        nClasses=args.nClasses, frameLen=40, every_frame=args.every_frame).to(device)
    criterion = nn.CrossEntropyLoss()
    loaders, _ = data_loader(args)

    if args.test:
        student.load_state_dict(torch.load(args.path))
        logger.info('*** student loaded from {} ***'.format(args.path))
        run_epoch(student, None, None, loaders, 'val', 0, None, criterion, args, logger, save_path)
        run_epoch(student, None, None, loaders, 'test', 0, None, criterion, args, logger, save_path)
        return

    teacher, get_latent = build_teacher(args, logger)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=0.)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=20, half=5, verbose=1)
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        run_epoch(student, teacher, get_latent, loaders, 'train', epoch, optimizer, criterion, args, logger, save_path)
        run_epoch(student, None, None, loaders, 'val', epoch, None, criterion, args, logger, save_path)


if __name__ == '__main__':
    main()
