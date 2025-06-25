import argparse
import os
import torch
import numpy as np
import time

from PIL import Image
from model.UnifiedEGformer import GuidedIAT

def main(args):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Load model
    model = GuidedIAT(
        input_channels=3,
        transformer_blocks=5,
        embedding_dimension=48
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Load model weights
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Load and preprocess input image
    img = Image.open(args.input).convert('RGB')
    
    # Resize image while keeping aspect ratio
    # width, height = img.size
    # new_width = 256
    # new_height = int(height * new_width / width)
    # img = img.resize((new_width, new_height))

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalize image if specified
    if args.normalize:
        img_tensor = img_tensor * 2.0 - 1.0

    # Perform inference
    start_time = time.time()
    with torch.no_grad():
        _, _, _, enhanced_img, _, _ = model(img_tensor)
    end_time = time.time()

    # enhanced_img = enhanced_img + 0.05*(img_tensor)

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # Post-process and save enhanced image
    enhanced_img_np = enhanced_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    enhanced_img_pil = Image.fromarray((enhanced_img_np * 255).astype(np.uint8))
    enhanced_output_name = os.path.basename(args.input)
    enhanced_img_pil.save(enhanced_output_name) # "enhanced_image.png")
    print("Enhanced image saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on input image")
    parser.add_argument("--weights", type=str, default="checkpoint/sice_grad/sice_grad_6_Mar_22h_37m/egformer_sicev2_best_ssim.pt", help="Path to model weights")
    parser.add_argument("--input", type=str, default="samples/demo-4.JPG", help="Path to input image")
    parser.add_argument("--normalize", action="store_true", help="Normalize input image")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device index")
    args = parser.parse_args()
    main(args)