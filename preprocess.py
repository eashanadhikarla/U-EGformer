import os
import csv
import random
import argparse
import torch
import torchvision.transforms.functional as TF

from glob import glob
from tqdm import tqdm
from PIL import Image

def augment(img1, img2, mask, aug):
    if aug == 0:
        return img1, img2, mask
    elif aug < 3:
        img1 = img1.flip(aug)
        img2 = img2.flip(aug)
        mask = mask.flip(aug)
    elif aug == 3:
        img1 = torch.rot90(img1, dims=(1, 2))
        img2 = torch.rot90(img2, dims=(1, 2))
        mask = torch.rot90(mask, dims=(1, 2))
    elif aug > 3 and aug < 6:
        img1 = torch.rot90(img1, dims=(1, 2), k=(aug-2))
        img2 = torch.rot90(img2, dims=(1, 2), k=(aug-2))
        mask = torch.rot90(mask, dims=(1, 2), k=(aug-2))
    elif aug == 6:
        img1 = torch.rot90(img1.flip(1), dims=(1, 2))
        img2 = torch.rot90(img2.flip(1), dims=(1, 2))
        mask = torch.rot90(mask.flip(1), dims=(1, 2))
    elif aug == 7:
        img1 = torch.rot90(img1.flip(2), dims=(1, 2))
        img2 = torch.rot90(img2.flip(2), dims=(1, 2))
        mask = torch.rot90(mask.flip(2), dims=(1, 2))
    return img1, img2, mask

def random_crop(img_low, img_gt, img_mask, size):
    width, height = img_low.size
    max_attempts = 10  # Set a limit on the number of attempts to find unique coordinates
    attempts = 0

    while True:
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
        if attempts >= max_attempts:
            break
        attempts += 1

    right = left + size
    bottom = top + size
    
    return (img_low.crop((left, top, right, bottom)),
            img_gt.crop((left, top, right, bottom)),
            img_mask.crop((left, top, right, bottom)))

def process_images(train_path, dest_dir, num_crops, crop_size):
    lowlight_paths = glob(os.path.join(train_path, 'low', '*'))
    gt_paths = [p.replace('low', 'gt') for p in lowlight_paths]
    mask_paths = [p.replace('low', 'low_masks') for p in lowlight_paths]

    os.makedirs(dest_dir, exist_ok=True)

    for low_path, gt_path, mask_path in tqdm(zip(lowlight_paths, gt_paths, mask_paths), total=len(lowlight_paths)):
        for i in range(num_crops):
            img_low, img_gt, img_mask = random_crop(
                Image.open(low_path).convert('RGB'),
                Image.open(gt_path).convert('RGB'),
                Image.open(mask_path).convert('RGB'),
                crop_size)

            aug = random.randint(0, 8)
            img_low, img_gt, img_mask = augment(TF.to_tensor(img_low), TF.to_tensor(img_gt), TF.to_tensor(img_mask), aug)

            img_low = TF.to_pil_image(img_low)
            img_gt = TF.to_pil_image(img_gt)
            img_mask = TF.to_pil_image(img_mask)

            base_name = os.path.basename(low_path).replace('.JPG', f'_crop_{i}.JPG')
            low_dest = os.path.join(dest_dir, 'low', base_name)
            gt_dest = os.path.join(dest_dir, 'gt', base_name)
            mask_dest = os.path.join(dest_dir, 'low_masks', base_name)

            os.makedirs(os.path.dirname(low_dest), exist_ok=True)
            os.makedirs(os.path.dirname(gt_dest), exist_ok=True)
            os.makedirs(os.path.dirname(mask_dest), exist_ok=True)

            img_low.save(low_dest)
            img_gt.save(gt_dest)
            img_mask.save(mask_dest)

def create_csv(dest_dir):
    lowlight_paths = sorted(glob(os.path.join(dest_dir, 'low', '*')))
    gt_paths = sorted([p.replace('low', 'gt') for p in lowlight_paths])
    mask_paths = sorted([p.replace('low', 'low_masks') for p in lowlight_paths])

    with open(os.path.join(dest_dir, 'train.csv'), 'w', newline='') as csvfile:
        fieldnames = ['Input', 'Target', 'Mask', 'Dimension']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for low_path, gt_path, mask_path in zip(lowlight_paths, gt_paths, mask_paths):
            img_low = Image.open(low_path)
            img_gt = Image.open(gt_path)
            img_mask = Image.open(mask_path)

            assert img_low.size == img_gt.size == img_mask.size, f"Image dimensions do not match for {os.path.basename(low_path)}"

            writer.writerow({
                'Input': low_path,
                'Target': gt_path,
                'Mask': mask_path,
                'Dimension': f"{img_low.size[0]}x{img_low.size[1]}"
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and augment images.')
    parser.add_argument('--traindir', type=str, required=True, help='Directory containing the training images.')
    parser.add_argument('--destdir', type=str, required=True, help='Directory to save the processed images.')
    parser.add_argument('--num_crops', type=int, default=3, help='Number of random crops to be taken from one image.')
    parser.add_argument('--crop_size', type=int, default=256, help='Size of the random crop.')

    args = parser.parse_args()
    process_images(args.traindir, args.destdir, args.num_crops, args.crop_size)
    create_csv(args.destdir)





# def process_images(train_path, dest_dir):
#     lowlight_paths = glob(os.path.join(train_path, 'low', '*'))
#     gt_paths = [p.replace('low', 'gt') for p in lowlight_paths]
#     mask_paths = [p.replace('low', 'low_masks') for p in lowlight_paths]

#     os.makedirs(dest_dir, exist_ok=True)

#     for low_path, gt_path, mask_path in tqdm(zip(lowlight_paths, gt_paths, mask_paths), total=len(lowlight_paths), unit="image"):
#         img_low = Image.open(low_path).convert('RGB')
#         img_gt = Image.open(gt_path).convert('RGB')
#         img_mask = Image.open(mask_path).convert('RGB')

#         aug = random.randint(0, 8)
#         img_low, img_gt, img_mask = augment(TF.to_tensor(img_low), TF.to_tensor(img_gt), TF.to_tensor(img_mask), aug)

#         img_low = TF.to_pil_image(img_low)
#         img_gt = TF.to_pil_image(img_gt)
#         img_mask = TF.to_pil_image(img_mask)

#         low_dest = os.path.join(dest_dir, 'low', os.path.basename(low_path))
#         gt_dest = os.path.join(dest_dir, 'gt', os.path.basename(gt_path))
#         mask_dest = os.path.join(dest_dir, 'low_masks', os.path.basename(mask_path))

#         os.makedirs(os.path.dirname(low_dest), exist_ok=True)
#         os.makedirs(os.path.dirname(gt_dest), exist_ok=True)
#         os.makedirs(os.path.dirname(mask_dest), exist_ok=True)

#         img_low.save(low_dest)
#         img_gt.save(gt_dest)
#         img_mask.save(mask_dest)