import math

class LearningRateScheduler(object):
    """ the reproduction of some tensorflow's learning rate decay strategies """

    def __init__(self, strategy, base_lr, nbr_epochs, nbr_iters_per_epoch, lr_decay_every=10, decay_rate = 0.1):
        self.base_lr = base_lr
        self.nbr_epochs = nbr_epochs
        self.lr_decay_every = lr_decay_every
        self.nbr_iters_per_epoch = nbr_iters_per_epoch
        self.total_iters = nbr_epochs * nbr_iters_per_epoch
        self.decay_rate = decay_rate
        self.strategy = strategy

        self.strategies = {'cosine': self.__cosine_decay,
                           'polynomial': self.__polynomial_decay,
                           'exponential': self.__exponential_decay,
                           'natural_exp': self.__natural_exp_decay,
                           'inverse_time': self.__inverse_time_decay}

    def __cosine_decay(self, epoch, i):
        step = epoch * self.nbr_iters_per_epoch + i
        cosine_decay = 0.5 * (1 + math.cos(1.0 * math.pi * step / self.total_iters))
        return cosine_decay * self.base_lr

    def __polynomial_decay(self, epoch, i):
        step = epoch * self.nbr_iters_per_epoch + i
        polynomial_decay = pow((1 - 1.0 * step / self.total_iters), 0.9)
        return polynomial_decay * self.base_lr

    def __exponential_decay(self, epoch, i):
        epoch_decay = pow(self.decay_rate, (epoch / self.lr_decay_every))
        return epoch_decay * self.base_lr

    def __natural_exp_decay(self, epoch, i):
        exp_decay = math.exp(-self.decay_rate * epoch / self.lr_decay_every)
        return exp_decay * self.base_lr

    def __inverse_time_decay(self, epoch, i):
        inverse_time_decay = (1 + self.decay_rate * epoch / self.lr_decay_every)
        return self.base_lr / inverse_time_decay

    def adjust_learning_rate(self, optimizer, i, epoch):
        lr = self.strategies[self.strategy](epoch, i)
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
        return lr

# ref https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/segmentation/option.py
def lr_parse(args):
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 180,
            'citys': 240,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.learning_rate is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.004,
            'citys': 0.004,
        }
        args.learning_rate = lrs[args.dataset.lower()] / 16 * args.batch_size
    print(args)
    return args