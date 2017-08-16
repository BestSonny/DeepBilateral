import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import itertools
import shutil
import tqdm
import numpy as np
import time
from utils import AverageMeter, label_accuracy_score

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename+'.pth')
    if is_best:
        print 'save best model'
        shutil.copyfile(filename+'.pth', filename+'_best.pth')

class Trainer(object):
    def __init__(self, cuda, model, optimizer, scheduler, criterion,
                 train_dataset, val_dataset, max_iter, args):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.losses = AverageMeter()

        self.best_loss = float("inf")
        self.args = args
        self.learning_rate = args.learning_rate

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.model.eval()
        end = time.time()

        label_trues = []
        label_preds = []

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_dataset), total=len(self.val_dataset),
                desc='Testing on epoch %d' % self.epoch, ncols=80, leave=False):
            # target = target.long().squeeze()
            if self.cuda:
                data, target = data.cuda(), target.cuda(async=True)

            data_var = Variable(data, volatile=True)
            target_var = Variable(target, volatile=True)
            # compute output
            output = self.model(data_var)

            loss = self.criterion(output, target_var)
             # measure accuracy and record loss
            losses.update(loss.data[0], data.size(0))
            # dice_loss_.update(dice_loss.data[0], 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # imgs = data.cpu()
            # pred = output.data.max(1)[1].cpu().numpy()
            # true = target.cpu().numpy()
            # for img, lt, lp in zip(imgs, true, pred):
            #     label_trues.append(lt)
            #     label_preds.append(lp)

            if batch_idx % self.args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Softmax Loss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'.format(
                    batch_idx, len(self.val_dataset), batch_time=batch_time, softmax_loss=losses))
        # metrics = label_accuracy_score(
        # label_trues, label_preds, n_class=2)
        # metrics = np.array(metrics)
        # metrics *= 100

        return losses.avg

    def train(self):
        batch_time = AverageMeter()

        self.model.train()

        end = time.time()
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_dataset), total=len(self.train_dataset),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            # target = target.long().squeeze()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data_var, target_var = Variable(data), Variable(target)

            output = self.model(data_var)

            loss = self.criterion(output, target_var)
            # measure accuracy and record loss
            self.losses.update(loss.data[0], data.size(0))


            # compute gradient and do SGD step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.iteration = self.iteration + 1
            if self.iteration >= self.max_iter:
                break

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Softmax Loss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'.format(
                      self.epoch, batch_idx, len(self.train_dataset), batch_time=batch_time,
                      softmax_loss=self.losses))

        self.losses.reset()

    def run(self):
        for epoch in itertools.count(self.epoch):
            self.epoch = epoch

            if self.val_dataset:
                val_loss = self.validate()
                self.scheduler.step(val_loss)
            if self.iteration >= self.max_iter:
                break

            # acc, acc_cls, mean_iu, fwavacc = metrics
            is_best = val_loss < self.best_loss
            self.best_loss = min(val_loss, self.best_loss)
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
            }, is_best, self.args.checkpoint_dir + '/' + self.args.model_name)

            self.train()


def train_net(model, train_dataset, val_dataset, args):

    cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    optim = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optim, 'min', verbose=True)
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        scheduler = scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_iter=args.max_iter,
        args= args,
        criterion= nn.MSELoss()
    )

    if cuda:
        model = model.cuda()
        cudnn.benchmark = True

    trainer.run()
