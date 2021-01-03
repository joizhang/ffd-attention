import os
import random
import re
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import hub, optim
from torch.backends import cudnn

from config import Config
from training.datasets import get_dataloader
from training.models.ae import Decoder, encoder, ActivationLoss, ReconstructionLoss, SegmentationLoss
from training.tools.metrics import AverageMeter, ProgressMeter, accuracy, eval_metrics
from training.tools.train_utils import parse_args

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])

torch.backends.cudnn.benchmark = True


def train(train_loader, model, decoder, optimizer, loss_functions, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    loss_act_meter = AverageMeter('Act Loss', ':.4e')
    loss_rect_meter = AverageMeter('Rect Loss', ':.4e')
    loss_seg_meter = AverageMeter('Seg Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, loss_act_meter, loss_rect_meter, loss_seg_meter, top1],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    decoder.train()

    end = time.time()
    for batch_idx, sample in enumerate(train_loader):
        optimizer['optimizer_encoder'].zero_grad()
        optimizer['optimizer_decoder'].zero_grad()

        if args.gpu is not None:
            images = sample['images'].cuda(args.gpu, non_blocking=True)
            labels = sample['labels'].cuda(args.gpu, non_blocking=True)
            masks = sample['masks'].cuda(args.gpu, non_blocking=True)
        else:
            images, labels, masks = sample['images'], sample['labels'], sample['masks']

        masks[masks >= 0.25] = 1.0
        masks[masks < 0.25] = 0.0
        masks = masks.long()

        # compute output
        latent = model(images).reshape(-1, 2, 64, 16, 16)
        zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)

        loss_act = loss_functions['act_loss_fn'](zero, one, labels)
        # loss_act_data = loss_act.item()
        loss_act_meter.update(loss_act.item(), images.size(0))

        y = torch.eye(2)
        if args.gpu >= 0:
            y = y.cuda(args.gpu)

        y = y.index_select(dim=0, index=labels.data.long())

        latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)

        seg, rect = decoder(latent)

        loss_seg = loss_functions['seg_loss_fn'](seg, masks)
        loss_seg = loss_seg * 1
        # loss_seg_data = loss_seg.item()
        loss_seg_meter.update(loss_seg.item(), images.size(0))

        loss_rect = loss_functions['rect_loss_fn'](rect, images)
        loss_rect = loss_rect * 1
        # loss_rect_data = loss_rect.item()
        loss_rect_meter.update(loss_rect.item(), images.size(0))

        loss_total = loss_act + loss_seg + loss_rect

        # measure accuracy and record loss
        labels_pred = torch.stack((zero, one), dim=1)
        acc1, = accuracy(labels_pred, labels)
        # losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do Adam step
        loss_total.backward()
        optimizer['optimizer_decoder'].step()
        optimizer['optimizer_encoder'].step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            progress.display(batch_idx + 1)
        if (batch_idx + 1) % 3000 == 0:
            break


def validate(val_loader, model, decoder, args):
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
                masks = sample['masks']
            else:
                images, labels, masks = sample['images'], sample['labels'], sample['masks']

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

            # measure accuracy and record loss
            labels_pred = torch.stack((zero, one), dim=1)
            acc1, = accuracy(labels_pred, labels)
            top1.update(acc1[0], images.size(0))
            # pixel-wise acc
            seg = torch.argmax(seg, dim=1)
            overall_acc = eval_metrics(masks, seg.cpu(), 256)
            pw_acc.update(overall_acc, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % args.print_freq == 0:
                progress.display(batch_idx + 1)
            if (batch_idx + 1) % 1000 == 0:
                break

    return top1.avg


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("Initializing Networks")
    model = encoder()
    model_cfg = model.default_cfg
    decoder = Decoder()

    print("Initializing Distribution")
    if not torch.cuda.is_available():
        print('Using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Only in single gpu
        decoder = decoder.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    print("Initializing Data Loader")
    train_sampler, train_loader, val_loader = get_dataloader(model_cfg, args)

    loss_functions = {
        "act_loss_fn": ActivationLoss(),
        "rect_loss_fn": ReconstructionLoss(),
        "seg_loss_fn": SegmentationLoss()
    }
    optimizer = {
        'optimizer_encoder': optim.Adam(model.parameters(), args.lr),
        'optimizer_decoder': optim.Adam(decoder.parameters(), args.lr),
    }

    start_epoch = 1
    best_acc1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if args.distributed:
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            print("Loaded checkpoint '{}' (epoch {}, best_acc {})".format(args.resume, start_epoch, best_acc1))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, decoder, args)
        return

    print("Start Training")
    better_acc = False
    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, decoder, optimizer, loss_functions, epoch, args)

        if epoch % 2 == 0 or epoch == args.epochs:
            acc1 = validate(val_loader, model, decoder, args)
            better_acc = best_acc1 < acc1
            best_acc1 = max(acc1, best_acc1)

        is_main_node = not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
        save_model = (better_acc or epoch == args.epochs or epoch % 5 == 0) and is_main_node
        if save_model:
            print('Save model')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'encoder_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'best_acc1': best_acc1,
                'optimizer_encoder': optimizer['optimizer_encoder'].state_dict(),
                'optimizer_decoder': optimizer['optimizer_decoder'].state_dict(),
            }, os.path.join('weights', '{}_{}_{}.pt'.format(args.arch, args.prefix, epoch)))
        better_acc = False


def main():
    args = parse_args()
    print(args)
    os.makedirs('weights', exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. ' +
                      'This will turn on the CUDNN deterministic setting, ' +
                      'which can slow down your training considerably! ' +
                      'You may see unexpected behavior when restarting from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(fn=main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    # Notice: Multi-task detection and segmentation can only trained on single gpu!!!
    main()
