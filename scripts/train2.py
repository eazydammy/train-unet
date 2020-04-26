import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import make_data_sampler, make_batch_data_sampler
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    
    # model and dataset
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')

    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        args.epochs = 160
    if args.lr is None or args.lr == 1e-4:
        args.lr = 0.01 / 8 * args.batch_size
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset("ade20k", split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset("ade20k", split='val', mode='val', **data_kwargs)
        self.args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        self.args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model(model="unet", in_channel=3, out_channel=13).to(self.device)

        # resume checkpoint if needed
        # if args.resume:
        #     if os.path.isfile(args.resume):
        #         _, ext = os.path.splitext(args.resume)
        #         assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
        #         print('Resuming training, loading {}...'.format(args.resume))
        #         self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(ignore_index=-1).to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = True
        epochs, max_iters = self.args.epochs, self.args.max_iters
        val_per_iters = self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        log_per_iters = self.args.log_iter * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()

        model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, _) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format("unet", "ade20k")
    filename = os.path.join(directory, filename)

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format("unet", "ade20k")
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    args.num_gpus = 1

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.device = "cpu"

    cudnn.benchmark = False
    args.lr = args.lr * args.num_gpus

    logger = setup_logger("semantic_segmentation", args.log_dir, filename='unet_ade20k_log.txt')
    logger.info("Using {} GPUs".format(1))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
