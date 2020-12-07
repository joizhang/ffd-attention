import os
import pickle
import re
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch.backends import cudnn

from training import models
from training.datasets.dffd_dataset import get_dffd_dataloader
from training.datasets.face_forensics_dataset import get_face_forensics_test_dataloader
from training.models import encoder, Decoder
from training.tools.metrics import eval_metrics
from training.tools.model_utils import AverageMeter, ProgressMeter, accuracy
from training.tools.train_utils import parse_args

torch.backends.cudnn.benchmark = True

PICKLE_FILE = "plot/{}.pickle"


def test(test_loader, model, decoder, args):
    y_true = []
    y_pred = []
    y_score = []

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, pw_acc], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(test_loader):
            images, labels, masks = sample['images'].cuda(), sample['labels'].cuda(), sample['masks']
            y_true.extend(labels.tolist())

            # masks[masks >= 0.5] = 1.0
            # masks[masks < 0.5] = 0.0
            # masks = masks.long()

            # compute output
            latent = model(images).reshape(-1, 2, 64, 16, 16)
            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)

            one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)

            y = torch.eye(2)
            if args.gpu >= 0:
                y = y.cuda(args.gpu)

            y = y.index_select(dim=0, index=labels.data.long())

            latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

            seg, rect = decoder(latent)

            labels_pred = torch.stack((zero, one), dim=1)
            pred = torch.argmax(labels_pred, dim=1)
            y_pred.extend(pred.tolist())
            score = torch.nn.functional.softmax(labels_pred, dim=1)
            score = score[:, 1]
            y_score.extend(score.tolist())

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            # pixel-wise acc
            seg = torch.argmax(seg, dim=1)
            # masks_pred = torch.argmax(masks_pred, dim=1)
            overall_acc = eval_metrics(masks, seg.cpu(), 256)
            pw_acc.update(overall_acc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(test_loader):
                progress.display(batch_idx + 1)

    pickle.dump([y_true, y_pred, y_score], open(PICKLE_FILE.format(args.arch), "wb"))
    return pw_acc.avg


def show_metrics(args, pw_acc):
    with open(PICKLE_FILE.format(args.arch), "rb") as f:
        y_true, y_pred, y_score = pickle.load(f)
    print(len(y_true), len(y_pred), len(y_score))
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    # print(fpr, tpr)
    fnr = 1 - tpr
    # Equal error rate
    eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    tpr_0_01 = -1
    tpr_0_02 = -1
    tpr_0_05 = -1
    tpr_0_10 = -1
    tpr_0_20 = -1
    tpr_0_50 = -1
    tpr_1_00 = -1
    tpr_2_00 = -1
    tpr_5_00 = -1
    for i in range(len(fpr)):
        if fpr[i] > 0.0001 and tpr_0_01 == -1:
            tpr_0_01 = tpr[i - 1]
        if fpr[i] > 0.0002 and tpr_0_02 == -1:
            tpr_0_02 = tpr[i - 1]
        if fpr[i] > 0.0005 and tpr_0_05 == -1:
            tpr_0_05 = tpr[i - 1]
        if fpr[i] > 0.001 and tpr_0_10 == -1:
            tpr_0_10 = tpr[i - 1]
        if fpr[i] > 0.002 and tpr_0_20 == -1:
            tpr_0_20 = tpr[i - 1]
        if fpr[i] > 0.005 and tpr_0_50 == -1:
            tpr_0_50 = tpr[i - 1]
        if fpr[i] > 0.01 and tpr_1_00 == -1:
            tpr_1_00 = tpr[i - 1]
        if fpr[i] > 0.02 and tpr_2_00 == -1:
            tpr_2_00 = tpr[i - 1]
        if fpr[i] > 0.05 and tpr_5_00 == -1:
            tpr_5_00 = tpr[i - 1]
    roc_auc = auc(fpr, tpr)
    metrics_template = "ACC: {:f} AUC: {:f} EER: {:f} PWA: {:f}"
    print(metrics_template.format(acc, roc_auc, eer, pw_acc))


def main():
    args = parse_args()
    print(args)

    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print("Loading checkpoint '{}'".format(args.resume))
        # model = models.__dict__[args.arch](pretrained=False)
        model = encoder()
        decoder = Decoder()
        model.cuda()
        decoder.cuda()
        checkpoint = torch.load(args.resume, map_location="cpu")
        encoder_state_dict = checkpoint.get("encoder_state_dict", checkpoint)
        model.load_state_dict(encoder_state_dict, strict=True)
        decoder_state_dict = checkpoint.get("decoder_state_dict", checkpoint)
        decoder.load_state_dict(decoder_state_dict, strict=True)

        print("Initializing Data Loader")
        if args.prefix == 'ff++':
            test_loader = get_face_forensics_test_dataloader(model, args)
        else:
            test_loader = get_dffd_dataloader(model, args, 'test', shuffle=False)

        print("Start Testing")
        pw_acc = test(test_loader, model, decoder, args)

        show_metrics(args, pw_acc)


if __name__ == '__main__':
    main()
