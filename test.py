import os
import pickle
import re
import time

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from training import models
from training.datasets import get_test_dataloader
from training.tools.metrics import eval_metrics, show_metrics
from training.tools.model_utils import AverageMeter, ProgressMeter, accuracy
from training.tools.train_utils import parse_args

torch.backends.cudnn.benchmark = True

PICKLE_FILE = "plot/{}.pickle"


def test(test_loader, model, args):
    y_true = []
    y_pred = []
    y_score = []

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    mae = AverageMeter('MAE', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, pw_acc, mae], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(test_loader):
            images, labels, masks = sample['images'].cuda(), sample['labels'].cuda(), sample['masks']
            masks[masks >= 0.25] = 1.0
            y_true.extend(labels.tolist())

            # compute output
            labels_pred, masks_pred = model(images)

            pred = torch.argmax(labels_pred, dim=1)
            y_pred.extend(pred.tolist())
            score = torch.nn.functional.softmax(labels_pred, dim=1)
            score = score[:, 1]
            y_score.extend(score.tolist())

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            # pixel-wise acc
            masks_pred = F.interpolate(masks_pred, scale_factor=16)
            masks_pred[masks_pred >= 0.25] = 1.0
            overall_acc = eval_metrics(masks, masks_pred.cpu(), 256)
            pw_acc.update(overall_acc, images.size(0))
            mean_avg_err = F.l1_loss(masks.cpu(), masks_pred.cpu())
            mae.update(mean_avg_err, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(test_loader):
                progress.display(batch_idx + 1)

    pickle.dump([y_true, y_pred, y_score], open(PICKLE_FILE.format(args.arch), "wb"))

    return pw_acc.avg, mae.avg


def main():
    args = parse_args()
    print(args)

    if args.resume:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        print("Loading checkpoint '{}'".format(args.resume))
        model = models.__dict__[args.arch](pretrained=False)
        model.cuda()
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)

        print("Initializing Data Loader")
        test_loader = get_test_dataloader(model.default_cfg, args)

        print("Start Testing")
        pw_acc, mae = test(test_loader, model, args)
        # pw_acc, mae = 0., 0.

        with open(PICKLE_FILE.format(args.arch, args.prefix), "rb") as f:
            y_true, y_pred, y_score = pickle.load(f)
        print(len(y_true), len(y_pred), len(y_score))
        show_metrics(y_true, y_pred, y_score, args, pw_acc, mae)


if __name__ == '__main__':
    main()
