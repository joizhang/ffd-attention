import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import hub
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import Config
from training import models
from training.datasets.classifier_dataset import DffdDataset, CelebDFV2Dataset
from training.tools.train_utils import parse_args, train, validate

torch.backends.cudnn.benchmark = True

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])


def get_dffd_dataloader(args):
    classes = {'Real': 0, 'Fake': 1}
    # img_paths = {'Real': [], 'Fake': []}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = DffdDataset(data_root=args.data_dir, mode='train', transform=transform, classes=classes)
    # plot_image(train_data)
    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    val_data = DffdDataset(data_root=args.data_dir, mode='validation', transform=transform, classes=classes)
    val_loader = DataLoader(val_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            pin_memory=True)
    return train_loader, val_loader


def get_celeba_df_dataloader(args):
    df = pd.read_csv(args.folds_csv)
    x_train, x_val = train_test_split(df, test_size=0.1)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = CelebDFV2Dataset(data_root=args.data_dir, df=x_train, mode='train', transform=transform)
    train_loader = DataLoader(train_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              pin_memory=True)
    val_data = CelebDFV2Dataset(data_root=args.data_dir, df=x_val, mode='validation', transform=transform)
    val_loader = DataLoader(val_data, num_workers=1, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            pin_memory=True)
    return train_loader, val_loader


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
    criterion = nn.CrossEntropyLoss().cuda()
    # writer = SummaryWriter('%s/logs/%s' % (args.save_dir, sig))

    print("Initializing Data Loader")
    if args.prefix == 'dffd':
        train_loader, val_loader = get_dffd_dataloader(args)
    else:
        train_loader, val_loader = get_celeba_df_dataloader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print("Start Training")
    best_acc1 = 0.
    for epoch in range(1, args.epochs + 1):
        train(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)

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
    python train.py --data-dir /data/xinlin/dffd --arch vgg16 --epoch 10 --batch-size 100 --prefix dffd
    python train.py --data-dir /data/xinlin/dffd --arch xception --epoch 10 --batch-size 50 --prefix celeb_df
    """
    main()
