import torch
import torch.utils.data as data

from torch.utils.data.sampler import BatchSampler

__all__ = ['synchronize', 'is_main_process',
           'all_gather', 'make_data_sampler', 'make_batch_data_sampler',
           'reduce_dict', 'reduce_loss_dict']

def is_main_process():
    return True


def synchronize():
    return


def all_gather(data):
    return [data]


def reduce_dict(input_dict, average=True):
    return input_dict



def reduce_loss_dict(loss_dict):
    return loss_dict


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:

            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


if __name__ == '__main__':
    pass
