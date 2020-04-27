import torch.utils.data as data
from .ade20k import ADE20K

datasets = {
        'ade20k': ADE20K
    }

def get_train_val_loader(name, image_size=384, data_path='./data/', batch_size=8, shuffle=True, dataloader_workers=4,
                         pin_memory=False, use_background=True, **kwargs):
    global datasets
    train_loader = data.DataLoader(datasets[name](mode='train', image_size=image_size, data_path=data_path, use_background=use_background),
                                   batch_size=batch_size, shuffle=shuffle, num_workers=dataloader_workers,
                                   pin_memory=pin_memory, drop_last=True)
    val_loader = data.DataLoader(datasets[name](mode='val', image_size=image_size, data_path=data_path, use_background=use_background),
                                 batch_size=batch_size, shuffle=False, num_workers=dataloader_workers,
                                 pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader

def get_dataset_tools(name, image_size=384, use_background=True, **kwargs):
    global datasets
    return datasets[name](mode='pred', image_size=image_size, use_background=use_background)