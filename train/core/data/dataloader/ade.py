"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class ADE20KSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    dataset_root : string
        Path to ADE20K folder.
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'ADEChallengeData2016'
    NUM_CLASS = 13

    def __init__(self, dataset_root='../datasets', split='test', mode=None, transform=None, crop_size=416, encode=False, **kwargs):
        super(ADE20KSegmentation, self).__init__(dataset_root, split, mode, transform, **kwargs)
        self.encode = encode
        self.crop_size = crop_size
        dataset_root = os.path.join(dataset_root, self.BASE_DIR)
        assert os.path.exists(dataset_root), "Please setup the dataset using ../datasets/ade20k.py"
        self.images, self.masks = _get_ade20k_pairs(dataset_root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + dataset_root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), dataset_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, encode_flag=self.encode, crop_size=self.crop_size)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask, encode_flag=self.encode, crop_size=self.crop_size)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img, encode_flag=self.encode), self._mask_transform(mask, encode_flage=self.encode)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask, encode_flage=False):
        if(encode_flage):
            width, height = mask.size
            mask_ = Image.new('L', (2 * width, height), (0))
            mask_.paste(mask, (0, 0, width, height))
            mask_.paste(mask, (width, 0, 2 * width, height))
            mask = mask_
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("floor, flooring", "road, route", "grass", "sidewalk, pavement",
                "earth, ground", "rug, carpet, carpeting", "field", "sand", "path",
                "stairs, steps", "stairway, staircase", "land, ground, soil", "step, stair")


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            # maskname = basename + '_seg' + '.png'
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths


if __name__ == '__main__':
    train_dataset = ADE20KSegmentation()
