import os
import time
import glob
import random
import numpy as np
import torch

from PIL import Image
from torch.utils import data
from model.UnifiedEGformer import GuidedIAT
from metrics import PSNR, SSIM
from collections import OrderedDict

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

## Load and Save Enhanced Images
def save_enhanced_images(dataset_path, modeldir, result_folder, start_idx, end_idx, model_quantize=False):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model = GuidedIAT(
        input_channels=3,
        transformer_blocks=5,
        embedding_dimension=48
    )

    device = 'mps' # 'cuda'
    model = torch.nn.DataParallel(model)
    model.to(device)

    # Load the checkpoint and adjust the state dict
    checkpoint = torch.load(modeldir, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = 'module.' + k  # add `module.`
        new_state_dict[name] = v

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict)
    model.eval()

    if model_quantize:
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        # model = quantize_dynamic(model, {torch.nn.LSTM}, dtype=torch.qint8)
        # model = quantize_dynamic(model, {torch.nn.GRU}, dtype=torch.qint8)

    print("\nModel loaded.")

    total_time = 0
    image_files = sorted(os.listdir(dataset_path))[start_idx:end_idx]
    num_images = len(image_files)

    for enum, image_name in enumerate(image_files):
        enhanced_image_path = os.path.join(result_folder, image_name)
        # Check if the enhanced image already exists
        if os.path.exists(enhanced_image_path):
            print(f"Skipping already processed image: {image_name}")
            continue

        print(f"{enum+1}. Processing {image_name} ...")
        image_path = os.path.join(dataset_path, image_name)

        with Image.open(image_path).convert('RGB') as img:
            # original_size = img.size
            # # Check if the image dimensions are larger than 1400x1000
            # if img.width > 512 or img.height > 512:
            #     img.thumbnail((512, 512))

            img_np = np.asarray(img).transpose(2, 0, 1) / 255.0
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)
            img_tensor = img_tensor.to(device)

            start_time = time.time()
            with torch.no_grad():
                _, _, enhanced_img, _, _, _ = model(img_tensor)
            end_time = time.time()
            inference_time = end_time-start_time
            total_time += inference_time

            enhanced_img_np = enhanced_img.squeeze().cpu().numpy().transpose(1, 2, 0)
            enhanced_img_pil = Image.fromarray((enhanced_img_np * 255).astype(np.uint8))
            enhanced_img_pil.save(os.path.join(result_folder, image_name))

            # Release memory
            del img, enhanced_img, enhanced_img_pil
            torch.cuda.empty_cache()

    print(f"Batch inference time: {total_time/num_images} seconds")
    # print("Enhanced images have been saved in the result directory.")
    # It might also be beneficial to clear the cache once more after the entire batch has been processed
    torch.cuda.empty_cache()
    return (total_time/num_images)


def save_all_enhanced_images(dataset_path, modeldir, result_folder, start_idx, end_idx, model_quantize=False):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model = GuidedIAT(
        input_channels=3,
        transformer_blocks=5,
        embedding_dimension=48
    )

    device = 'mps' # 'cuda'
    model = torch.nn.DataParallel(model)
    model.to(device)

    # Load the checkpoint and adjust the state dict
    checkpoint = torch.load(modeldir, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = 'module.' + k  # add `module.`
        new_state_dict[name] = v

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict)
    model.eval()

    if model_quantize:
        from torch.quantization import quantize_dynamic
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        # model = quantize_dynamic(model, {torch.nn.LSTM}, dtype=torch.qint8)
        # model = quantize_dynamic(model, {torch.nn.GRU}, dtype=torch.qint8)

    print("\nModel loaded.")

    total_time = 0
    image_files = sorted(os.listdir(dataset_path))[start_idx:end_idx]
    num_images = len(image_files)

    for enum, image_name in enumerate(image_files):
        enhanced_image_path = os.path.join(result_folder, image_name)
        # Check if the enhanced image already exists
        if os.path.exists(enhanced_image_path):
            print(f"Skipping already processed image: {image_name}")
            continue

        print(f"{enum+1}. Processing {image_name} ...")
        image_path = os.path.join(dataset_path, image_name)

        with Image.open(image_path).convert('RGB') as img:
            img_np = np.asarray(img).transpose(2, 0, 1) / 255.0
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)
            img_tensor = img_tensor.to(device)

            start_time = time.time()
            with torch.no_grad():
                local_mul, local_add, _, global_enhanced_output, attention_map, _ = model(img_tensor)
                # Calculate local enhanced output
                local_enhanced_output = img_tensor.mul(local_mul).add(local_add)
            end_time = time.time()
            inference_time = end_time-start_time
            total_time += inference_time

            # Save local enhanced output
            local_enhanced_output_np = local_enhanced_output.squeeze().cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            local_enhanced_output_pil = Image.fromarray((local_enhanced_output_np * 255).astype(np.uint8))
            local_enhanced_output_pil.save(os.path.join(result_folder, f"{os.path.splitext(image_name)[0]}_local.png"))

            # Save global enhanced output
            global_enhanced_output_np = global_enhanced_output.squeeze().cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            global_enhanced_output_pil = Image.fromarray((global_enhanced_output_np * 255).astype(np.uint8))
            global_enhanced_output_pil.save(os.path.join(result_folder, f"{os.path.splitext(image_name)[0]}.png"))

            # Save attention map
            attention_map_np = attention_map.squeeze().cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            attention_map_pil = Image.fromarray((attention_map_np * 255).astype(np.uint8))
            attention_map_pil.save(os.path.join(result_folder, f"{os.path.splitext(image_name)[0]}_attention_map.png"))

            # Release memory
            del img, local_enhanced_output, global_enhanced_output, attention_map_pil
            torch.cuda.empty_cache()

    print(f"Batch inference time: {total_time/num_images} seconds")
    # It might also be beneficial to clear the cache once more after the entire batch has been processed
    torch.cuda.empty_cache()
    return (total_time/num_images)


class Eval_DataGenerator(data.Dataset):
    def __init__(self, gt_path, result_path=None, matching_key=None):
        if matching_key.startswith(('mu')):
            self.gt_images = sorted(glob.glob(gt_path + '*.jpg'))
            self.result_images = {os.path.basename(path): path for path in glob.glob(result_path + '*_N1.5.JPG')}
        else:
            self.gt_images = sorted(glob.glob(gt_path + '*'))
            self.result_images = {os.path.basename(path): path for path in glob.glob(result_path + '*')}

    def __getitem__(self, index):
        gt_image_path = self.gt_images[index]
        gt_image_name = os.path.basename(gt_image_path)
        if matching_key.startswith(('mu')):
            # Change the extension of gt_image_name from .jpg to _N1.5.JPG to match the result images
            result_image_name = gt_image_name.replace('.jpg', '_N1.5.JPG')
            result_image_path = self.result_images.get(result_image_name)
        else:
            result_image_path = self.result_images.get(gt_image_name.replace("normal", "low"))

        if result_image_path:
            result_image = Image.open(result_image_path).convert('RGB')
            gt_image = Image.open(gt_image_path).convert('RGB')

            gt_image_np = np.asarray(gt_image, dtype=np.float32).transpose((2, 0, 1)) / 255
            result_image_np = np.asarray(result_image, dtype=np.float32).transpose((2, 0, 1)) / 255
            return result_image_np, gt_image_np, gt_image_name  # Ensure gt_image_name is returned as a string
        else:
            # Handle the case where the result image does not exist
            print(f"No matching result image found for {gt_image_path}")
            return None

    def __len__(self):
        return len(self.gt_images)


## Evaluation function
def evaluate(testdir, resultdir, matching_key):
    print("Running metrics for results ...")
    ssim_Fn = SSIM()
    psnr_Fn = PSNR()

    testdataset = Eval_DataGenerator(gt_path=testdir, result_path=resultdir, matching_key=matching_key)
    test_loader = torch.utils.data.DataLoader(testdataset)
    print(f"Total test images: {len(testdataset)}")

    total_ssim, total_psnr = 0.0, 0.0
    total_images = len(testdataset)

    for imgs in test_loader:
        enhanced_img_np, gt_image_np = imgs[0], imgs[1]
        ssim = ssim_Fn(gt_image_np, enhanced_img_np, as_loss=False)
        psnr = psnr_Fn(gt_image_np, enhanced_img_np)

        total_ssim += ssim.item()
        total_psnr += psnr.item()

    avg_ssim = total_ssim / total_images
    avg_psnr = total_psnr / total_images
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}\n")

if __name__ == "__main__":

    model_list = {
        # LOL-v2 Dataset (u - underexposure; o - overexposure)
        'lu-exp12'    : 'lolv2/lolv2_26_Jan_21h_28m',
        'lu-exp11'    : 'lolv2/lolv2_22_Jan_6h_29m',
        'lu-sota'     : 'lolv2/lolv2_15_Jan_19h_57m',
        'lu-exp10'    : 'lolv2/lolv2_16_Jan_21h_11m',
        'lu-exp9'     : 'lolv2/lolv2_1_Jan_23h_44m',
        'lu-exp8'     : 'lolv2/lolv2_11_Nov_0h_36m',
        'lu-exp7'     : 'lolv2/lolv2_29_Oct_15h_44m',
        'lu-exp6'     : 'lolv2/lolv2_29_Oct_5h_15m',
        'lu-exp3'     : 'lolv2/lolv2_28_Oct_20h_13m',

        # ME-v2 Dataset (u - underexposure; o - overexposure)
        'mu-exp2.1'   : 'mev2/mev2_13_Nov_22h_42m',
        'mu-exp2'     : 'mev2/mev2_13_Nov_21h_7m',
        'mu-exp1.3'   : 'mev2/mev2_13_Nov_15h_57m',
        'mu-exp1.2'   : 'mev2/mev2_13_Nov_11h_52m',
        'mu-exp1.1'   : 'mev2/mev2_13_Nov_0h_22m',
        'mu-exp1'     : 'mev2/mev2_11_Nov_21h_21m',

        # SICE-v2 Dataset (u - underexposure; o - overexposure)
        'su-exp24'    : 'sicev2/sicev2_2_Mar_16h_40m',
        'su-exp23'    : 'sicev2/sicev2_5_Nov_4h_44m',
        'su-exp22'    : 'sicev2/sicev2_3_Nov_3h_16m', 
        'su-exp21'    : 'sicev2/sicev2_2_Nov_19h_15m',
        'su-exp20'    : 'sicev2/sicev2_1_Nov_17h_10m',
        'su-exp19'    : 'sicev2/sicev2_1_Nov_1h_26m',
        'su-exp18'    : 'sicev2/sicev2_31_Oct_17h_0m',
        'su-exp17'    : 'sicev2/sicev2_26_Oct_13h_24m',
        'su-exp15'    : 'sicev2/sicev2_19_Oct_6h_53m',
        'su-exp14'    : 'sicev2/sicev2_16_Oct_19h_21m',
        'su-exp13'    : 'sicev2/sicev2_16_Oct_16h_47m',
        'su-exp12'    : 'sicev2/sicev2_5_Oct_13h_2m',
        'so-exp1.1'   : 'sicev2/sicev2_10_Nov_15h_43m',
        'so-exp1'     : 'sicev2/sicev2_8_Nov_19h_47m',

        # SICE-Grad
        # SICE-Mix
    }
    # ===================================
    model_in_eval = model_list['su-exp24']
    # ===================================
    print(f"Model in evaluation: {model_in_eval}")

    model_type     = ["psnr", "ssim"]
    mt             = model_type[1]
    task           = "low"
    model_quantize = False

    # rootdir = "../../datasets/" # llnet-server-1
    # rootdir = "../datasets/" # opod
    rootdir = "../../../research/llie/datasets/" # Local

    # Check for values in the dictionary and get the corresponding key
    matching_key = next((key for key, value in model_list.items() if value == model_in_eval), None)

    if matching_key:
        # Use the matching key to determine the dataset paths
        modeldir = f"./checkpoint/{model_in_eval}/egformer_sicev2_best_{mt}.pt"
        if matching_key.startswith('lu'):
            dataset_path = os.path.join(rootdir, "LOL-v2/Real_captured/Test/Low/")
            testdir = os.path.join(rootdir, "LOL-v2/Real_captured/Test/Normal/")
        elif matching_key.startswith('mu'):
            dataset_path = os.path.join(rootdir, "ME-v2/test/low_1/")
            testdir = os.path.join(rootdir, "ME-v2/test/expert_c_testing_set/")
        elif matching_key.startswith('mo'):
            dataset_path = os.path.join(rootdir, "ME-v2/test/over_1/")
            testdir = os.path.join(rootdir, "ME-v2/test/expert_c_testing_set/")
        elif matching_key.startswith('su'):
            dataset_path = os.path.join(rootdir, "SICEV2/test/low/")
            testdir = os.path.join(rootdir, "SICEV2/test/gt/")
        elif matching_key.startswith('so'):
            dataset_path = os.path.join(rootdir, "SICEV2/test/over/")
            testdir = os.path.join(rootdir, "SICEV2/test/gt/")

        result_folder = os.path.join(
            './results', os.path.basename(os.path.dirname(modeldir)), mt + '/')

    # ################# Step 1 #################
    # save_enhanced_images(dataset_path, modeldir, result_folder, model_quantize)
    '''
    I noticed that the max my GPU can hold is only 15 images at 
    a time, hence the following loop
    '''

    overall_inference_time = 0.0
    total_images = len(os.listdir(dataset_path))
    batch_size = 6
    num_batches = (total_images + batch_size - 1) // batch_size

    print(f"\nUsing model type: {mt}")

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        print(f"Processing batch {batch + 1}/{num_batches}: Images {start_idx} to {end_idx - 1}")

        oit = save_enhanced_images(dataset_path, modeldir, result_folder, start_idx, end_idx, model_quantize)
        # oit = save_all_enhanced_images(dataset_path, modeldir, result_folder, start_idx, end_idx, model_quantize)
        overall_inference_time += oit

    print(f"Average inference time: {overall_inference_time / num_batches} seconds")
    ##########################################

    ## Step 2
    # evaluate(testdir, result_folder, matching_key)
