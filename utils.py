import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from collections import OrderedDict

EPS = 1e-3
PI = 22.0 / 7.0

def split_data(data_dir, split_percentage=0.8):
    # List all image paths
    all_image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    random.shuffle(all_image_paths)

    # Handle the case where all data is for training
    if split_percentage == 1:
        return all_image_paths, []

    # Otherwise, split data into training and validation sets
    train_size = int(len(all_image_paths) * split_percentage)
    train_paths = all_image_paths[:train_size]
    val_paths = all_image_paths[train_size:]
    return train_paths, val_paths

def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params

def print_args(args):
    # Determine the length of the longest argument for alignment
    max_length = max(len(arg) for arg in vars(args))    
    for arg in vars(args):
        # Adjust the alignment using the format specifier
        print(f"{arg:<{max_length}} : {getattr(args, arg)}")

class clr:
    """
    Defining colors for the print syntax coloring
    """
    H   = '\033[35m' # Header
    B   = '\033[94m' # Blue
    G   = '\033[36m' # Green
    W   = '\033[93m' # Warning
    F   = '\033[91m' # Fail
    E   = '\033[0m'  # End 
    BD  = '\033[1m'  # Bold
    UL  = '\033[4m'  # Underline


def tupletype(s, n: int=None):
    """
    This will handle "arguments" in a tuple format

    Supported:
    --foo (1,2)
    --foo 1,2
    --foo "( 1, 2 )"

    Limited error checking, so this works too:
    --foo ((1,2))
    """
    s = s.replace("(", "").replace(")", "")
    mapped_int = map(int, s.split(","))
    return tuple(mapped_int)


def visualization(img, img_path, iteration):
    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img = img.cpu().numpy()

    for i in range(img.shape[0]):
        # save name
        name = str(iteration) + '_' + str(i) + '.jpg'
        img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0
        plt.imsave(os.path.join(img_path, name), img_single)


# Display the model summary
# The custom summary function handles both tuples and lists in output shapes
def custom_summary(model, input_tensor):
    def register_hook(module):
        def hook(module, input, output):
            nonlocal layer_count
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = input[0].shape
            summary[m_key]["input_shape_str"] = str(input[0].shape)
            if isinstance(output, tuple):
                summary[m_key]["output_shape"] = [out.shape for out in output]
                summary[m_key]["output_shape_str"] = [str(out.shape) for out in output]
            else:
                summary[m_key]["output_shape"] = output.shape
                summary[m_key]["output_shape_str"] = str(output.shape)

            params = sum(p.numel() for p in module.parameters())
            summary[m_key]["trainable"] = params
            summary[m_key]["non_trainable"] = 0

            layer_count += 1

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    layer_count = 0  # Initialize layer count
    model.apply(register_hook)
    model(input_tensor)
    for hook in hooks:
        hook.remove()

    total_params = 0
    total_output = 0
    trainable_params = 0
    total_output = 0
    summary_str = ""

    for layer in summary:
        total_params += summary[layer]["trainable"] + summary[layer]["non_trainable"]
        if isinstance(summary[layer]["output_shape"], list):
            total_output += sum([np.prod(shape) for shape in summary[layer]["output_shape"]])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        trainable_params += summary[layer]["trainable"]
        if isinstance(summary[layer]["output_shape_str"], list):
            output_shape_str = ", ".join(summary[layer]["output_shape_str"])
        else:
            output_shape_str = summary[layer]["output_shape_str"]
        summary_str += (
            f"{layer}\t"
            f"{output_shape_str}\t"
            f"{summary[layer]['trainable']}\n"
        )

    summary_str += f"=================================================================\n"
    summary_str += f"Total layers: {layer_count}\n"  # Display the total number of layers
    summary_str += f"Total params: {total_params}\n"
    summary_str += f"Trainable params: {trainable_params}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params}\n"
    summary_str += f"Total output: {total_output}\n"
    summary_str += f"=================================================================\n"

    print(summary_str)