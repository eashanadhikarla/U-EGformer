import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def calculate_ssim_map(img1, img2):
    # Convert images to grayscale if they are RGB
    if img1.shape[-1] == 3:
        img1 = Image.fromarray(img1, mode='RGB').convert('L')
        img1 = np.array(img1)
    if img2.shape[-1] == 3:
        img2 = Image.fromarray(img2, mode='RGB').convert('L')
        img2 = np.array(img2)

    # Calculate SSIM map
    _, ssim_map = ssim(img1, img2, full=True)

    return ssim_map


def save_ssim_map(output_path, gt_path, save_path):
    # Read images
    output_img = np.array(Image.open(output_path))
    gt_img = np.array(Image.open(gt_path))

    print("output_image, gt_image: ", output_img.shape, gt_img.shape, "\n")

    # Calculate SSIM map
    ssim_map_result = calculate_ssim_map(output_img, gt_img)

    # Save SSIM map
    # Replace <low...> with <normal...> in the output filename
    output_filename = os.path.basename(output_path)
    ssim_map_filename = f"{output_filename}_ssim_map.png"
    ssim_map_filepath = os.path.join(save_path, ssim_map_filename)
    print("ssim_map_filepath: ", ssim_map_filepath)

    # Convert the float SSIM map to uint8 before saving
    Image.fromarray((ssim_map_result * 255).astype(np.uint8)).save(ssim_map_filepath)


def process_images(output_folder, gt_folder):
    # List all files in the ground-truth folder
    gt_files = os.listdir(gt_folder)

    # Process each ground-truth file
    for gt_file in gt_files:
        # Construct paths for output and ground truth images
        gt_path = os.path.join(gt_folder, gt_file)
        output_path = os.path.join(output_folder, gt_file.replace("normal", "low"))

        print("gt_path: ", gt_path)
        print("output_path: ", output_path)

        # Calculate and save SSIM map
        save_ssim_map(output_path, gt_path, output_folder)


if __name__ == "__main__":
    # Define paths
    baseline_output_folder = "~/sota/iat/IAT_enhance/results/LOL-v2/"
    baseline_gt_folder     = "~/datasets/LOL-v2/Real_captured/Test/Normal"

    my_model_output_folder = "~/sota/unifiedEGformer (d2 branch)/results/lolv2_7_Feb_5h_27m_sota/ssim/"
    my_model_gt_folder     = "~/datasets/LOL-v2/Real_captured/Test/Normal"

    # Process my model images
    process_images(my_model_output_folder, my_model_gt_folder)
