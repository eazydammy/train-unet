import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

class ADE20K(data.Dataset):

    NBR_CLASSES = 13
    def __init__(self, mode='train', image_size=384, data_path='./data/', use_background=True):
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.data_path = data_path
        self.image_size = image_size * 1.083
        self.crop_size = image_size
        if not self.mode == 'pred':
            self.init_data()
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean, std=self.std
            )
        ])
        if use_background:
            ADE20K.NBR_CLASSES = 14
        self.use_background = use_background


    def __get_pairs(self, img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            base_filename, _ = os.path.splitext(filename)
            mask_filename = '{}.png'.format(base_filename)
            if os.path.isfile(os.path.join(mask_folder, mask_filename)):
                img_paths.append(os.path.join(img_folder, filename))
                mask_paths.append(os.path.join(mask_folder, mask_filename))
            else:
                raise RuntimeError('cannot find the mask image for {}'.format(mask_filename))
        return img_paths, mask_paths

    def init_data(self):
        if self.mode == 'train':
            img_folder = os.path.join(self.data_path, 'images/training')
            mask_folder = os.path.join(self.data_path, 'annotations/training')
            self.images, self.masks = self.__get_pairs(img_folder, mask_folder)
        elif self.mode == 'val':
            img_folder = os.path.join(self.data_path, 'images/validation')
            mask_folder = os.path.join(self.data_path, 'annotations/validation')
            self.images, self.masks = self.__get_pairs(img_folder, mask_folder)
        else:
            train_img_folder = os.path.join(self.data_path, 'images/training')
            train_mask_folder = os.path.join(self.data_path, 'annotations/training')
            train_img_paths, train_mask_paths = self.__get_pairs(train_img_folder, train_mask_folder)
            val_img_folder = os.path.join(self.data_path, 'images/validation')
            val_mask_folder = os.path.join(self.data_path, 'annotations/validation')
            val_img_paths, val_mask_paths = self.__get_pairs(val_img_folder, val_mask_folder)
            self.images = train_img_paths + val_img_paths
            self.masks = train_mask_paths + val_mask_paths
        if len(self.images) != len(self.masks):
            raise RuntimeError("the number of images and masks does not matching")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = Image.open(self.images[index])
        mask = Image.open(self.masks[index])

        if self.mode == 'train':
            img, mask = self.__preprocessing_for_train(img, mask)
        elif self.mode == 'val':
            img, mask = self.__preprocessing_for_validation(img, mask)

        return self.im_transform(img), self.__mask_transform(mask)

    def preprocessing_for_predict(self, img):
        ori = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean, std=self.std
            )
        ])
        prep = transform(ori)
        prep.unsqueeze_(0)
        return ori, prep

    def __preprocessing_for_validation(self, img, mask):
        w, h = img.size
        if h > w:
            rate = h / w
            img = img.resize((self.crop_size, int(self.crop_size * rate)), Image.BILINEAR)
            mask = mask.resize((self.crop_size, int(self.crop_size * rate)), Image.NEAREST)
        else:
            rate = w / h
            img = img.resize((int(self.crop_size * rate), self.crop_size), Image.BILINEAR)
            mask = mask.resize((int(self.crop_size * rate), self.crop_size), Image.NEAREST)

        w, h = img.size
        x = int((w - self.crop_size) // 2)
        y = int((h - self.crop_size) // 2)
        img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        mask = mask.crop((x, y, x + self.crop_size, y + self.crop_size))

        return img, mask

    def __preprocessing_for_train(self, img, mask):
        # random left-right flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random up-down flip
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random gaussian blur
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=2.0))

        # random resize
        long_size = int((random.random() * (2 - 0.5) + 0.5) * self.image_size)
        w, h = img.size
        if h > w:
            rate = h / w
            img = img.resize((int(long_size / rate), long_size), Image.BILINEAR)
            mask = mask.resize((int(long_size / rate), long_size), Image.NEAREST)
        else:
            rate = w / h
            img = img.resize((long_size, int(long_size / rate)), Image.BILINEAR)
            mask = mask.resize((long_size, int(long_size / rate)), Image.NEAREST)

        # padding
        w, h = img.size
        if min(h, w) < self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        # crop
        w, h = img.size
        crop_w = random.randint(0, w - self.crop_size)
        crop_h = random.randint(0, h - self.crop_size)
        img = img.crop((crop_w, crop_h, crop_w + self.crop_size, crop_h + self.crop_size))
        mask = mask.crop((crop_w, crop_h, crop_w + self.crop_size, crop_h + self.crop_size))

        # random rotation
        # if random.random() < 0.5:
        #     rotation_degree = random.randint(-10, 10)
        #     img = img.rotate(angle=rotation_degree, resample=Image.BILINEAR)
        #     mask = mask.rotate(angle=rotation_degree, resample=Image.NEAREST)

        return img, mask

    def __mask_transform(self, mask):
        target = np.array(mask).astype('int64') # 0~150, 151 classes
        if not self.use_background:
            target = target - 1 # -1~149, 150 classes
        return torch.from_numpy(target)