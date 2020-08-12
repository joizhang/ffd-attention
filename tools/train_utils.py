import argparse
import time

import matplotlib.pyplot as plt
import torch

import models
from tools.model_utils import ProgressMeter, AverageMeter, accuracy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', metavar='ARCH', default='vgg16', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=111, help='manual seed')
    # parser.add_argument('--signature', default=str(datetime.datetime.now()))
    parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    # parser.add_argument('--save_dir', default='./runs', help='directory for result')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    opt = parser.parse_args()
    return opt


def train(train_loader, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.cuda(), target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target)
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

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
