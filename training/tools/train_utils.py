import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from training import models
from training.tools.metrics import ProgressMeter, AverageMeter, accuracy, eval_metrics

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='vgg16', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg16)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--prefix', type=str, default='dffd')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # parser.add_argument('--signature', default=str(datetime.datetime.now()))
    parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--save_dir', default='./runs', help='directory for result')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--folds-csv', type=str, default='folds.csv')
    # parser.add_argument('--localization', dest='localization', action='store_true',
    #                     help='Manipulation Localization')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=111, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    return args


def train(train_loader, model, optimizer, loss_functions, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_idx, sample in enumerate(train_loader):
        if args.gpu is not None:
            images = sample['images'].cuda(args.gpu, non_blocking=True)
            labels = sample['labels'].cuda(args.gpu, non_blocking=True)
            masks = sample['masks'].cuda(args.gpu, non_blocking=True)
            # masks_input = F.interpolate(masks, size=(16, 16))
            # masks_input = masks_input.cuda(args.gpu, non_blocking=True)
        else:
            images, labels, masks = sample['images'], sample['labels'], sample['masks']
            # masks_input = F.interpolate(masks, (16, 16))

            # compute output
        outputs = model(images)
        if isinstance(outputs, tuple):
            labels_pred, masks_output = outputs
            loss_classifier = loss_functions['classifier_loss'](labels_pred, labels)
            loss_map = loss_functions['map_loss'](masks_output, masks)
            loss = loss_classifier + 10. * loss_map
        else:
            labels_pred = outputs
            loss = loss_functions['classifier_loss'](labels_pred, labels)

        # measure accuracy and record loss
        acc1, = accuracy(labels_pred, labels)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            progress.display(batch_idx + 1)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    pw_acc = AverageMeter('Pixel-wise Acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1, pw_acc], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, sample in enumerate(val_loader):
            if args.gpu is not None:
                images = sample['images'].cuda(args.gpu, non_blocking=True)
                labels = sample['labels'].cuda(args.gpu, non_blocking=True)
                masks = sample['masks'].cuda(args.gpu, non_blocking=True)
                # masks_input = F.interpolate(masks, size=(16, 16))
                # masks_input = masks_input.cuda(args.gpu, non_blocking=True)
            else:
                images, labels, masks = sample['images'], sample['labels'], sample['masks']
                # masks_input = F.interpolate(masks, (16, 16))

            # compute output
            outputs = model(images)
            if isinstance(outputs, tuple):
                labels_pred, masks_output = outputs
            else:
                labels_pred = outputs

            # measure accuracy and record loss
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            if isinstance(outputs, tuple):
                # masks_pred = F.interpolate(masks_pred, size=)
                overall_acc = eval_metrics(masks.cpu(), masks_output.cpu(), 256)
                pw_acc.update(overall_acc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    return top1.avg


def write_tfboard(writer, vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


def plot_image(data):
    plt.figure(figsize=(10, 5))
    for i in range(len(data)):
        image, label = data[i]
        print(i, image.size)
        plt.subplot(1, 5, i + 1)
        # plt.tight_layout()
        plt.title('Label {}'.format(label))
        plt.axis('off')
        plt.imshow(image)
        if i == 4:
            plt.show()
            break


if __name__ == '__main__':
    print(models.__dict__['vgg16'])
