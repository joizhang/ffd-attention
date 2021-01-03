import os
import pickle
import time

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from training.datasets import get_test_dataloader
from training.models import encoder, Decoder
from training.tools.metrics import eval_metrics, show_metrics
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
            latent = model(images).reshape(-1, 2, 64, 16, 16)
            zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
            zero = zero_abs.mean(dim=1)
            one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
            one = one_abs.mean(dim=1)
            y = torch.eye(2)
            if args.gpu >= 0:
                y = y.cuda()
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
            mean_avg_err = F.l1_loss(masks.squeeze().cpu(), seg.cpu())
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
        test_loader = get_test_dataloader(model.default_cfg, args)

        print("Start Testing")
        pw_acc, mae = test(test_loader, model, decoder, args)
        # pw_acc, mae = 0., 0.

        with open(PICKLE_FILE.format(args.arch, args.prefix), "rb") as f:
            y_true, y_pred, y_score = pickle.load(f)
        print(len(y_true), len(y_pred), len(y_score))
        show_metrics(y_true, y_pred, y_score, args, pw_acc, mae)


if __name__ == '__main__':
    main()
