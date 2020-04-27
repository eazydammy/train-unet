import torch
import os
def save_checkpoint(weights, epoch, best_performance, is_best, model, backbone,
                    checkpoint_prefix_name, checkpoint_root_path, **kwargs):
    filename = '{}_{}_{}_{}.pt'.format(checkpoint_prefix_name,
                                        model,
                                        backbone,
                                        'best'if is_best else '')
    torch.save({
        'state_dict': weights,
        'best_performance': best_performance,
        'start_epoch': epoch
    }, os.path.join(checkpoint_root_path, filename))