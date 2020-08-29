import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import hub
from torch.backends import cudnn

from config import Config
from training import models
from training.datasets.classifier_dataset import get_celeba_df_dataloader, get_dffd_dataloader
from training.tools.train_utils import parse_args, train, validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


def main():
    args = parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # sig = str(datetime.datetime.now()) + args.signature
    # os.makedirs('%s/modules/%s' % (args.save_dir, sig), exist_ok=True)

    print("Initializing Networks")
    model = models.__dict__[args.arch](pretrained=True)
    model.cuda()
    optimizer = optim.Adam(model.parameters())
    loss_functions = {
        "classifier_loss": nn.CrossEntropyLoss().cuda(),
        "map_loss": nn.L1Loss().cuda()
    }
    # writer = SummaryWriter('%s/logs/%s' % (args.save_dir, sig))

    print("Initializing Data Loader")
    if args.prefix == 'dffd':
        train_loader = get_dffd_dataloader(model, args, 'train', num_workers=1)
        val_loader = get_dffd_dataloader(model, args, 'validation', shuffle=False)
    else:
        train_loader, val_loader = get_celeba_df_dataloader(model, args)
    # print(next(iter(val_loader)))

    if args.evaluate:
        validate(val_loader, model, loss_functions, args)
        return

    print("Start Training")
    best_acc1 = 0.
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, loss_functions, epoch, args)
        acc1 = validate(val_loader, model, loss_functions, args)

        best_acc1 = max(acc1, best_acc1)

        if epoch == args.epochs:
            print('Save model')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join('weights', '{}_{}.pt'.format(args.arch, args.prefix)))


if __name__ == '__main__':
    """
    python train.py --data-dir /data/xinlin/dffd --arch xception --prefix dffd --epoch 5 --batch-size 20 --lr 0.0002 --gpu 1 --print-freq 50
    python train.py --data-dir /data/xinlin/dffd --arch xception_reg --prefix dffd --epoch 5 --batch-size 20 --lr 0.0002 --gpu 1 --print-freq 50
    python train.py --data-dir /data/xinlin/dffd --arch xception_butd --prefix dffd --epoch 5 --batch-size 20 --lr 0.0002 --gpu 1 --print-freq 50
    """
    main()
