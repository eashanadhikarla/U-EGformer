import os
import random
import glob

import torchvision
from torch.utils import data
from PIL import Image, ImageOps

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np
import torch


class RandomErasing(object):
    '''
    Source: https://arxiv.org/abs/1708.04896

    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.11301495,0.11392263, 0.10859329]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, target, mask):
        if random.uniform(0, 1) > self.probability:
            return img, target, mask

        for attempt in range(100):
            area = img.width * img.height

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.width and h < img.height:
                x1 = random.randint(0, img.height - h)
                y1 = random.randint(0, img.width - w)
                erase_region = Image.new('RGB', (w, h), tuple(int(mean_i * 255) for mean_i in self.mean))
                img.paste(erase_region, (y1, x1))
                target.paste(erase_region, (y1, x1))
                mask.paste(erase_region, (y1, x1))
                return img, target, mask

        return img, target, mask


class LOLv1_DataGenerator(torch.utils.data.Dataset):
    def __init__(self, images_path, mode='train', image_size=None, task=None, num_samples=None, padding_type="constant", fill=0):
        self.images_path = images_path
        self.image_size = image_size
        self.task = task
        self.num_samples = num_samples
        self.mode = mode
        self.padding_type = padding_type
        self.fill = fill
        self.pad_if_needed = True  # Assuming padding is needed, adjust as necessary

        if self.mode == 'train' or self.mode == 'val':
            self.image_list_lowlight = self.images_path
        elif self.mode == 'test':
            self.image_list_lowlight = glob.glob(self.images_path + '*')
            if num_samples:
                self.image_list_lowlight = self.image_list_lowlight[:self.num_samples]

        if self.mode == 'train':
            random.shuffle(self.image_list_lowlight)

    def __flip_aug__(self, low, high, mask):
        if random.random() > 0.5:
            low  = ImageOps.flip(low)
            high = ImageOps.flip(high)
            mask = ImageOps.flip(mask)
        if random.random() > 0.5:
            low  = ImageOps.mirror(low)
            high = ImageOps.mirror(high)
            mask = ImageOps.mirror(mask)
        return low, high, mask

    def __get_params__(self, low, output_size):
        """Get parameters for a random crop."""
        w, h = low.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(w, h)}")

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __pad_if_needed__(self, img, output_size, padding_type="constant"):
        """Pad the image if its size is smaller than the output size."""
        width, height = img.size
        target_w, target_h = output_size

        if self.pad_if_needed and (width < target_w or height < target_h):
            padding_h = max(target_h - height, 0)
            padding_w = max(target_w - width, 0)
            padding = (padding_w // 2, padding_h // 2, padding_w - padding_w // 2, padding_h - padding_h // 2)

            if padding_type == "constant":
                img = ImageOps.expand(img, border=padding, fill=self.fill)
            elif padding_type == "reflect":
                img = ImageOps.expand(img, border=padding, fill=None)
                img = ImageOps.mirror(img)
        return img

    def __random_crop__(self, low, high, mask, output_size):
        """Crop the given images at a random location with optional padding."""
        low_padded = self.__pad_if_needed__(low, output_size, self.padding_type)
        high_padded = self.__pad_if_needed__(high, output_size, self.padding_type)
        mask_padded = self.__pad_if_needed__(mask, output_size, self.padding_type)

        i, j, h, w = self.__get_params__(low_padded, output_size)

        low_cropped = low_padded.crop((j, i, j + w, i + h))
        high_cropped = high_padded.crop((j, i, j + w, i + h))
        mask_cropped = mask_padded.crop((j, i, j + w, i + h))
        return low_cropped, high_cropped, mask_cropped

    def __augment__(self, img1, img2, mask, aug):
        if aug == 0:
            return img1, img2, mask
        elif aug < 3:
            # Flip based on the augmentation index
            if aug == 1:  # Flip Left/Right
                img1 = ImageOps.mirror(img1)
                img2 = ImageOps.mirror(img2)
                mask = ImageOps.mirror(mask)
            elif aug == 2:  # Flip Top/Bottom
                img1 = ImageOps.flip(img1)
                img2 = ImageOps.flip(img2)
                mask = ImageOps.flip(mask)
        elif aug == 3:
            # 90-degree rotation
            img1 = img1.rotate(-90)
            img2 = img2.rotate(-90)
            mask = mask.rotate(-90)
        elif aug > 3 and aug < 6:
            # Multiple 90-degree rotations based on the augmentation index
            k = aug - 2
            img1 = img1.rotate(-90 * k)
            img2 = img2.rotate(-90 * k)
            mask = mask.rotate(-90 * k)
        elif aug == 6:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.mirror(img1).rotate(-90)
            img2 = ImageOps.mirror(img2).rotate(-90)
            mask = ImageOps.mirror(mask).rotate(-90)
        elif aug == 7:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.flip(img1).rotate(-90)
            img2 = ImageOps.flip(img2).rotate(-90)
            mask = ImageOps.flip(mask).rotate(-90)
        return img1, img2, mask

    def __getitem__(self, index):
        data_lowlight_path = self.image_list_lowlight[index]
        patch_size = 256 # Training Patch Size

        # Open the image
        input_img = Image.open(data_lowlight_path).convert('RGB')

        if self.task == 'low':
            target_img = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            attention_mask = Image.open(data_lowlight_path.replace('low', 'low_masks_otsu')).convert('RGB')

        # Processing depending on the mode
        if self.mode == 'train':
            width, height = target_img.size
            row = random.randint(0, height - patch_size)
            column = random.randint(0, width - patch_size)
            aug = random.randint(0, 8)

            # Crop patch using PIL's cropping method
            input_img = input_img.crop((column, row, column + patch_size, row + patch_size))
            target_img = target_img.crop((column, row, column + patch_size, row + patch_size))
            attention_mask = attention_mask.crop((column, row, column + patch_size, row + patch_size))

            # Data Augmentations (Patch training)
            input_img, target_img, attention_mask = self.__augment__(
                input_img, target_img, attention_mask, aug
            )

            # Data Augmentations (Random crop over patches)
            input_img, target_img, attention_mask = self.__random_crop__(
                input_img, target_img, attention_mask, (patch_size,patch_size)
            )

        elif self.mode == 'val':
            # Validate on center crop
            if patch_size is not None:
                input_img = torchvision.transforms.functional.center_crop(input_img, (patch_size, patch_size))
                target_img = torchvision.transforms.functional.center_crop(target_img, (patch_size, patch_size))
                attention_mask = torchvision.transforms.functional.center_crop(attention_mask, (patch_size, patch_size))

        # Convert the images to tensor
        '''Note: The TF.to_tensor automatically scales the images’ intensity from a range of [0, 255] to a range of [0, 1],
        so we don't need to divide the tensors by 255 again for data normalization.'''
        input_img  = torchvision.transforms.functional.to_tensor(input_img)
        target_img = torchvision.transforms.functional.to_tensor(target_img)
        attention_mask = torchvision.transforms.functional.to_tensor(attention_mask)

        filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
        return input_img, target_img, attention_mask, filename

    def __len__(self):
        return len(self.image_list_lowlight)


class LOLv2_DataGenerator(data.Dataset):
    def __init__(self, images_path, mode='train', image_size=None, task=None, num_samples=None):
        self.images_path = images_path
        self.image_size = image_size
        # self.task = task
        self.num_samples = num_samples
        self.mode = mode
        if self.mode=='train' or self.mode=='val':
            self.image_list_lowlight = self.images_path
        if self.mode=='test':
            self.image_list_lowlight = glob.glob(self.images_path + '*')
            if num_samples:
                self.image_list_lowlight = self.image_list_lowlight[:self.num_samples]
        if self.mode == 'train':
            random.shuffle(self.image_list_lowlight)

    def __flip_aug__(self, low, high, mask):
        if random.random() > 0.5:
            low  = ImageOps.flip(low)
            high = ImageOps.flip(high)
            mask = ImageOps.flip(mask)
        if random.random() > 0.5:
            low  = ImageOps.mirror(low)
            high = ImageOps.mirror(high)
            mask = ImageOps.mirror(mask)
        return low, high, mask

    def __get_params__(self, low, output_size):
        """Get parameters for a random crop."""
        w, h = low.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(w, h)}")

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __random_crop__(self, low, high, mask, output_size):
        """
        Inspired from: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomCrop
        Crop the given image at a random location.
        """
        i, j, h, w = self.__get_params__(low, output_size)

        low = low.crop((j, i, j + w, i + h))
        high = high.crop((j, i, j + w, i + h))
        mask = mask.crop((j, i, j + w, i + h))
        return low, high, mask

    def __augment__(self, img1, img2, mask, aug):
        if aug == 0:
            return img1, img2, mask
        elif aug < 3:
            # Flip based on the augmentation index
            if aug == 1:  # Flip Left/Right
                img1 = ImageOps.mirror(img1)
                img2 = ImageOps.mirror(img2)
                mask = ImageOps.mirror(mask)
            elif aug == 2:  # Flip Top/Bottom
                img1 = ImageOps.flip(img1)
                img2 = ImageOps.flip(img2)
                mask = ImageOps.flip(mask)
        elif aug == 3:
            # 90-degree rotation
            img1 = img1.rotate(-90)
            img2 = img2.rotate(-90)
            mask = mask.rotate(-90)
        elif aug > 3 and aug < 6:
            # Multiple 90-degree rotations based on the augmentation index
            k = aug - 2
            img1 = img1.rotate(-90 * k)
            img2 = img2.rotate(-90 * k)
            mask = mask.rotate(-90 * k)
        elif aug == 6:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.mirror(img1).rotate(-90)
            img2 = ImageOps.mirror(img2).rotate(-90)
            mask = ImageOps.mirror(mask).rotate(-90)
        elif aug == 7:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.flip(img1).rotate(-90)
            img2 = ImageOps.flip(img2).rotate(-90)
            mask = ImageOps.flip(mask).rotate(-90)
        return img1, img2, mask

    def __getitem__(self, index):
        data_lowlight_path = self.image_list_lowlight[index]
        patch_size = 256 # Training Patch Size

        # Open the image
        input_img = Image.open(data_lowlight_path).convert('RGB')

        # if self.task == 'low':
        target_img = Image.open(data_lowlight_path.replace('Low', 'Normal').replace('low', 'normal')).convert('RGB')
        attention_mask = Image.open(data_lowlight_path.replace('Low', 'Mask')).convert('RGB')

        # Processing depending on the mode
        if self.mode == 'train':
            width, height = target_img.size
            row = random.randint(0, height - patch_size)
            column = random.randint(0, width - patch_size)
            aug = random.randint(0, 8)

            # Crop patch using PIL's cropping method
            input_img = input_img.crop((column, row, column + patch_size, row + patch_size))
            target_img = target_img.crop((column, row, column + patch_size, row + patch_size))
            attention_mask = attention_mask.crop((column, row, column + patch_size, row + patch_size))

            # Data Augmentations (Patch training)
            input_img, target_img, attention_mask = self.__augment__(
                input_img, target_img, attention_mask, aug
            )

            # Data Augmentations (Random crop over patches)
            input_img, target_img, attention_mask = self.__random_crop__(
                input_img, target_img, attention_mask, (patch_size,patch_size)
            )

        elif self.mode == 'val':
            # Validate on center crop
            if patch_size is not None:
                input_img = torchvision.transforms.functional.center_crop(input_img, (patch_size, patch_size))
                target_img = torchvision.transforms.functional.center_crop(target_img, (patch_size, patch_size))
                attention_mask = torchvision.transforms.functional.center_crop(attention_mask, (patch_size, patch_size))

        # Convert the images to tensor
        '''Note: The TF.to_tensor automatically scales the images’ intensity from a range of [0, 255] to a range of [0, 1],
        so we don't need to divide the tensors by 255 again for data normalization.'''
        input_img  = torchvision.transforms.functional.to_tensor(input_img)
        target_img = torchvision.transforms.functional.to_tensor(target_img)
        attention_mask = torchvision.transforms.functional.to_tensor(attention_mask)

        filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
        return input_img, target_img, attention_mask, filename

    def __len__(self):
        return len(self.image_list_lowlight)


class SICE_DataGenerator(data.Dataset):
    def __init__(self, images_path, mode='train', image_size=None, task=None, num_samples=None):
        self.images_path = images_path
        self.image_size = image_size
        self.task = task
        self.num_samples = num_samples
        self.mode = mode
        if self.mode=='train' or self.mode=='val':
            self.image_list_lowlight = self.images_path
        if self.mode=='test':
            self.image_list_lowlight = glob.glob(self.images_path + '*')
            if num_samples:
                self.image_list_lowlight = self.image_list_lowlight[:self.num_samples]
        if self.mode == 'train':
            random.shuffle(self.image_list_lowlight)
        # Initialize the RandomErasing augmentation
        self.__random_erasing__ = RandomErasing()

    def __flip_aug__(self, low, high, mask):
        if random.random() > 0.5:
            low  = ImageOps.flip(low)
            high = ImageOps.flip(high)
            mask = ImageOps.flip(mask)
        if random.random() > 0.5:
            low  = ImageOps.mirror(low)
            high = ImageOps.mirror(high)
            mask = ImageOps.mirror(mask)
        return low, high, mask

    def __get_params__(self, low, output_size):
        """Get parameters for a random crop."""
        w, h = low.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(w, h)}")

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __random_crop__(self, low, high, mask, output_size):
        """
        Inspired from: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomCrop
        Crop the given image at a random location.
        """
        i, j, h, w = self.__get_params__(low, output_size)

        low = low.crop((j, i, j + w, i + h))
        high = high.crop((j, i, j + w, i + h))
        mask = mask.crop((j, i, j + w, i + h))
        return low, high, mask

    def __augment__(self, img1, img2, mask, aug):
        if aug == 0:
            return img1, img2, mask
        elif aug < 3:
            # Flip based on the augmentation index
            if aug == 1:  # Flip Left/Right
                img1 = ImageOps.mirror(img1)
                img2 = ImageOps.mirror(img2)
                mask = ImageOps.mirror(mask)
            elif aug == 2:  # Flip Top/Bottom
                img1 = ImageOps.flip(img1)
                img2 = ImageOps.flip(img2)
                mask = ImageOps.flip(mask)
        elif aug == 3:
            # 90-degree rotation
            img1 = img1.rotate(-90)
            img2 = img2.rotate(-90)
            mask = mask.rotate(-90)
        elif aug > 3 and aug < 6:
            # Multiple 90-degree rotations based on the augmentation index
            k = aug - 2
            img1 = img1.rotate(-90 * k)
            img2 = img2.rotate(-90 * k)
            mask = mask.rotate(-90 * k)
        elif aug == 6:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.mirror(img1).rotate(-90)
            img2 = ImageOps.mirror(img2).rotate(-90)
            mask = ImageOps.mirror(mask).rotate(-90)
        elif aug == 7:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.flip(img1).rotate(-90)
            img2 = ImageOps.flip(img2).rotate(-90)
            mask = ImageOps.flip(mask).rotate(-90)
        return img1, img2, mask

    def __resize_image__(self, img):
        width, height = img.size
        if self.image_size is None:
            return img

        target_w, target_h = self.image_size
        aspect_ratio = width / height
        if width < height:
            width = target_w
            height = int(width / aspect_ratio)
        else:
            height = target_h
            width = int(height * aspect_ratio)
        return img.resize((width, height), Image.LANCZOS)

    def __getitem__(self, index):
        data_lowlight_path = self.image_list_lowlight[index]
        patch_size = 256 # 384 # Training Patch Size

        # Open the image
        input_img = Image.open(data_lowlight_path).convert('RGB')
        if self.mode == 'test':
            input_img = self.__resize_image__(input_img)

        if self.task == 'low':
            target_img = Image.open(data_lowlight_path.replace('low', 'gt')).convert('RGB')
            if self.mode == 'test':
                target_img = self.__resize_image__(target_img)
            # attention_mask = Image.open(data_lowlight_path.replace('low', 'low_masks')).convert('RGB')
            attention_mask = Image.open(data_lowlight_path.replace('low', 'low_masks_otsu')).convert('RGB')
            if self.mode == 'test':
                attention_mask = self.__resize_image__(attention_mask)

        elif self.task == 'over':
            target_img = Image.open(data_lowlight_path.replace('over', 'gt')).convert('RGB')
            if self.mode == 'test':
                target_img = self.__resize_image__(target_img)
            # attention_mask = Image.open(data_lowlight_path.replace('over', 'over_masks')).convert('RGB')
            attention_mask = Image.open(data_lowlight_path.replace('over', 'over_masks_otsu')).convert('RGB')
            if self.mode == 'test':
                attention_mask = self.__resize_image__(attention_mask)

        # Processing depending on the mode
        if self.mode == 'train':
            # width, height = target_img.size
            # if height < patch_size or width < patch_size:
            #     raise ValueError(f'{data_lowlight_path} Image dimensions ({width}, {height}) are smaller than patch_size ({patch_size})')
            # row = random.randint(0, height - patch_size)
            # column = random.randint(0, width - patch_size)
            aug = random.randint(0, 8)

            # # Crop patch using PIL's cropping method
            # input_img = input_img.crop((column, row, column + patch_size, row + patch_size))
            # target_img = target_img.crop((column, row, column + patch_size, row + patch_size))
            # attention_mask = attention_mask.crop((column, row, column + patch_size, row + patch_size))

            # Data Augmentations (Patch training)
            input_img, target_img, attention_mask = self.__augment__(
                input_img, target_img, attention_mask, aug
            )

            # Data Augmentations (Random crop over patches), reduces the img size by half
            input_img, target_img, attention_mask = self.__random_crop__(
                input_img, target_img, attention_mask, (patch_size,patch_size)
            )

            # # Apply RandomErasing to all three images
            # input_img, target_img, attention_mask = self.__random_erasing__(
            #     input_img, target_img, attention_mask
            # )

        elif self.mode == 'val':
            # Validate on center crop
            if patch_size is not None:
                input_img = torchvision.transforms.functional.center_crop(input_img, (patch_size, patch_size))
                target_img = torchvision.transforms.functional.center_crop(target_img, (patch_size, patch_size))
                attention_mask = torchvision.transforms.functional.center_crop(attention_mask, (patch_size, patch_size))

        # Convert the images to tensor
        '''Note: The TF.to_tensor automatically scales the images’ intensity from a range of [0, 255] to a range of [0, 1], so we don't need to divide the tensors by 255 again for data normalization.'''
        input_img  = torchvision.transforms.functional.to_tensor(input_img)
        target_img = torchvision.transforms.functional.to_tensor(target_img)
        attention_mask = torchvision.transforms.functional.to_tensor(attention_mask)

        filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
        # print(input_img.shape, target_img.shape, attention_mask.shape)
        return input_img, target_img, attention_mask, filename

    def __len__(self):
        return len(self.image_list_lowlight)


class ME_DataGenerator(torch.utils.data.Dataset):
    def __init__(self, images_path, mode='train', image_size=None, task=None, num_samples=None, padding_type="constant", fill=0):
        self.images_path = images_path
        self.image_size = image_size
        self.task = task
        self.num_samples = num_samples
        self.mode = mode
        self.padding_type = padding_type
        self.fill = fill
        self.pad_if_needed = True  # Adjust as necessary

        # Image list initialization
        if self.mode == 'train' or self.mode == 'val':
            self.image_list_lowlight = self.images_path
        elif self.mode == 'test':
            self.image_list_lowlight = glob.glob(self.images_path + '*')
            if num_samples:
                self.image_list_lowlight = self.image_list_lowlight[:self.num_samples]

        if self.mode == 'train':
            random.shuffle(self.image_list_lowlight)

        # Initialize the RandomErasing augmentation
        self.__random_erasing__ = RandomErasing()

    def __flip_aug__(self, low, high, mask):
        if random.random() > 0.5:
            low  = ImageOps.flip(low)
            high = ImageOps.flip(high)
            mask = ImageOps.flip(mask)
        if random.random() > 0.5:
            low  = ImageOps.mirror(low)
            high = ImageOps.mirror(high)
            mask = ImageOps.mirror(mask)
        return low, high, mask

    # ################
    # ## Type 2
    # ################
    def __get_params__(self, low, output_size):
        """Get parameters for a random crop."""
        w, h = low.size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(w, h)}")

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # def __random_crop__(self, low, high, mask, output_size):
    #     """
    #     Inspired from: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomCrop
    #     Crop the given image at a random location.
    #     """
    #     i, j, h, w = self.__get_params__(low, output_size)

    #     low = low.crop((j, i, j + w, i + h))
    #     high = high.crop((j, i, j + w, i + h))
    #     mask = mask.crop((j, i, j + w, i + h))
    #     return low, high, mask

    # ################
    # ## Type 3
    # ################
    def __pad_if_needed__(self, img, output_size, padding_type="constant"):
        """Pad the image if its size is smaller than the output size."""
        width, height = img.size
        target_w, target_h = output_size

        if self.pad_if_needed and (width < target_w or height < target_h):
            padding_h = max(target_h - height, 0)
            padding_w = max(target_w - width, 0)
            padding = (padding_w // 2, padding_h // 2, padding_w - padding_w // 2, padding_h - padding_h // 2)

            if padding_type == "constant":
                img = ImageOps.expand(img, border=padding, fill=self.fill)
            elif padding_type == "reflect":
                img = ImageOps.expand(img, border=padding, fill=None)
                img = ImageOps.mirror(img)

        return img

    def __random_crop__(self, low, high, mask, output_size):
        """Crop the given images at a random location with optional padding."""
        low_padded = self.__pad_if_needed__(low, output_size, self.padding_type)
        high_padded = self.__pad_if_needed__(high, output_size, self.padding_type)
        mask_padded = self.__pad_if_needed__(mask, output_size, self.padding_type)

        i, j, h, w = self.__get_params__(low_padded, output_size)

        low_cropped = low_padded.crop((j, i, j + w, i + h))
        high_cropped = high_padded.crop((j, i, j + w, i + h))
        mask_cropped = mask_padded.crop((j, i, j + w, i + h))
        return low_cropped, high_cropped, mask_cropped

    def __augment__(self, img1, img2, mask, aug):
        if aug == 0:
            return img1, img2, mask
        elif aug < 3:
            # Flip based on the augmentation index
            if aug == 1:  # Flip Left/Right
                img1 = ImageOps.mirror(img1)
                img2 = ImageOps.mirror(img2)
                mask = ImageOps.mirror(mask)
            elif aug == 2:  # Flip Top/Bottom
                img1 = ImageOps.flip(img1)
                img2 = ImageOps.flip(img2)
                mask = ImageOps.flip(mask)
        elif aug == 3:
            # 90-degree rotation
            img1 = img1.rotate(-90)
            img2 = img2.rotate(-90)
            mask = mask.rotate(-90)
        elif aug > 3 and aug < 6:
            # Multiple 90-degree rotations based on the augmentation index
            k = aug - 2
            img1 = img1.rotate(-90 * k)
            img2 = img2.rotate(-90 * k)
            mask = mask.rotate(-90 * k)
        elif aug == 6:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.mirror(img1).rotate(-90)
            img2 = ImageOps.mirror(img2).rotate(-90)
            mask = ImageOps.mirror(mask).rotate(-90)
        elif aug == 7:
            # Flip and then rotate 90 degrees
            img1 = ImageOps.flip(img1).rotate(-90)
            img2 = ImageOps.flip(img2).rotate(-90)
            mask = ImageOps.flip(mask).rotate(-90)
        return img1, img2, mask

    def __resize_image__(self, img):
        width, height = img.size
        if self.image_size is None:
            return img

        target_w, target_h = self.image_size
        aspect_ratio = width / height
        if width < height:
            width = target_w
            height = int(width / aspect_ratio)
        else:
            height = target_h
            width = int(height * aspect_ratio)
        return img.resize((width, height), Image.LANCZOS)

    def __getitem__(self, index):
        data_lowlight_path = self.image_list_lowlight[index]
        patch_size = 256 # 384 # Training Patch Size

        # Open the image
        input_img = Image.open(data_lowlight_path).convert('RGB')
        if self.mode == 'test':
            input_img = self.__resize_image__(input_img)

        if self.task == 'low' and self.mode == 'test':
            target_img = Image.open(data_lowlight_path.replace('low', 'expert_c_testing_set').replace('_N1.5.JPG', '.jpg')).convert('RGB')
            target_img = self.__resize_image__(target_img)
            attention_mask = Image.open(data_lowlight_path.replace('low', 'low_masks_otsu')).convert('RGB')
            attention_mask = self.__resize_image__(attention_mask)
        elif self.task == 'low' and self.mode == 'train' or self.task == 'low' and self.mode == 'val':
            target_img = Image.open(data_lowlight_path.replace('low', 'GT_IMAGES').replace('_N1.5.JPG', '.jpg')).convert('RGB')
            attention_mask = Image.open(data_lowlight_path.replace('low', 'low_masks_otsu')).convert('RGB')

        elif self.task == 'over' and self.mode == 'test':
            target_img = Image.open(data_lowlight_path.replace('over', 'expert_c_testing_set').replace('_P1.5.JPG', '.jpg')).convert('RGB')
            target_img = self.__resize_image__(target_img)
            attention_mask = Image.open(data_lowlight_path.replace('over', 'over_masks_otsu')).convert('RGB')
            attention_mask = self.__resize_image__(attention_mask)
        elif self.task == 'over' and self.mode == 'train' or self.task == 'over' and self.mode == 'val':
            target_img = Image.open(data_lowlight_path.replace('over', 'GT_IMAGES').replace('_P1.5.JPG', '.jpg')).convert('RGB')
            attention_mask = Image.open(data_lowlight_path.replace('over', 'over_masks_otsu')).convert('RGB')

        # Processing depending on the mode
        if self.mode == 'train':
            width, height = target_img.size
            if height < patch_size or width < patch_size:
                raise ValueError(f'{data_lowlight_path} Image dimensions ({width}, {height}) are smaller than patch_size ({patch_size})')
            row = random.randint(0, height - patch_size)
            column = random.randint(0, width - patch_size)

            # Crop patch using PIL's cropping method
            input_img = input_img.crop((column, row, column + patch_size, row + patch_size))
            target_img = target_img.crop((column, row, column + patch_size, row + patch_size))
            attention_mask = attention_mask.crop((column, row, column + patch_size, row + patch_size))

            aug = random.randint(0, 8)
            # Data Augmentations (Patch training)
            input_img, target_img, attention_mask = self.__augment__(
                input_img, target_img, attention_mask, aug
            )

            # Data Augmentations (Random crop over patches), reduces the img size by half
            input_img, target_img, attention_mask = self.__random_crop__(
                input_img, target_img, attention_mask, (patch_size, patch_size)
            )

            # # Apply RandomErasing to all three images
            # input_img, target_img, attention_mask = self.__random_erasing__(
            #     input_img, target_img, attention_mask
            # )

        elif self.mode == 'val':
            # Validate on center crop
            if patch_size is not None:
                input_img = torchvision.transforms.functional.center_crop(input_img, (patch_size, patch_size))
                target_img = torchvision.transforms.functional.center_crop(target_img, (patch_size, patch_size))
                attention_mask = torchvision.transforms.functional.center_crop(attention_mask, (patch_size, patch_size))

        # Convert the images to tensor
        '''Note: The TF.to_tensor automatically scales the images’ intensity from a range of [0, 255] to a range of [0, 1], so we don't need to divide the tensors by 255 again for data normalization.'''
        input_img  = torchvision.transforms.functional.to_tensor(input_img)
        target_img = torchvision.transforms.functional.to_tensor(target_img)
        attention_mask = torchvision.transforms.functional.to_tensor(attention_mask)

        filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
        # print(input_img.shape, target_img.shape, attention_mask.shape)
        return input_img, target_img, attention_mask, filename

    def __len__(self):
        return len(self.image_list_lowlight)

# ############
# ## Demo Test
# ############
# from utils import split_data
# traindir = '../../datasets/SICEV2/test/low/'
# train_paths, val_paths = split_data(traindir, split_percentage=1.0)
# dataset = SICE_DataGenerator(
#     train_paths, 
#     mode='train', 
#     image_size=(900, 600), 
#     task='low', 
#     num_samples=2
# )
# test_loader  = torch.utils.data.DataLoader(dataset, batch_size=4)

# for i, data in enumerate(test_loader):
#     img, tar, mask, _ = data
#     print(img.shape)
#     exit()