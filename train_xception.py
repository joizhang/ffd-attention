import argparse
import datetime
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.classifier_dataset import DffdDataset
from models.xception import xception
import models
import matplotlib.pyplot as plt

# def main():
#     opt = parse_args()
#     print(opt)
#
#     sig = str(datetime.datetime.now()) + opt.signature
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
#     random.seed(opt.seed)
#     torch.manual_seed(opt.seed)
#     torch.cuda.manual_seed_all(opt.seed)
#     os.makedirs('%s/modules/%s' % (opt.save_dir, sig), exist_ok=True)
#
#     print("Initializing Data Loader")
#     classes = {'Real': 0, 'Fake_Entire': 1, 'Fake_Partial': 2}
#     img_paths = {'Real': [], 'Fake_Entire': [], 'Fake_Partial': []}
#     train_data = DATA(data_root=(opt.data_dir + 'train/'), normalize=([0.5] * 3, [0.5] * 3), classes=classes,
#                       img_paths=img_paths, seed=opt.seed)
#     train_loader = DataLoader(train_data, num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True,
#                               pin_memory=True)
#     training_batch_generator = get_training_batch(train_loader)
#     test_data = DATA(data_root=(opt.data_dir + 'validation/'))
#     test_loader = DataLoader(test_data, num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True,
#                              pin_memory=True)
#     testing_batch_generator = get_training_batch(test_loader)
#
#     print("Initializing Networks")
#     model_xcp = xception(len(train_data.classes), load_pretrain=True)
#     optimizer_xcp = optim.Adam(model_xcp.parameters(), lr=opt.lr)
#     model_xcp.cuda()
#     cse_loss = nn.CrossEntropyLoss().cuda()
#     writer = SummaryWriter('%s/logs/%s' % (opt.save_dir, sig))
#
#     print("Start Training")
#     itr = opt.it_start
#     while itr != opt.it_end + 1:
#         batch_train, lb_train = next(training_batch_generator)
#         loss = train(model_xcp, optimizer_xcp, batch_train, lb_train, cse_loss)
#         write_tfboard(writer, [loss[0]], itr, name='TRAIN')
#
#         if itr % 100 == 0:
#             test_results = [0, 0]
#             for i in range(5):
#                 batch_test, lb_test = next(testing_batch_generator)
#                 a, b = test(model_xcp, batch_test, lb_test, cse_loss)
#                 test_results[0] += a
#                 test_results[1] += b
#             test_results[0] /= 5
#             test_results[1] /= 5
#             write_tfboard(writer, test_results, itr, name='TEST')
#             print("Eval: " + str(itr))
#         if itr % 1000 == 0:
#             torch.save({'module': model_xcp.state_dict()}, '%s/modules/%s/%d.pickle' % (opt.save_dir, sig, itr))
#             print("Save Model: {:d}".format(itr))
#
#         itr += 1
#
#
# if __name__ == '__main__':
#     main()
