# Unified-EGformer: Exposure Guided Lightweight Transformer for Mixed-Exposure Image Enhancement
Adhikarla et al

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://link.springer.com/chapter/10.1007/978-3-031-78110-0_17)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://arxiv.org/pdf/2407.13170)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/eashanadhikarla/U-EGformer)

Despite recent strides made by AI in image processing, the issue of mixed exposure, pivotal in many real-world scenarios like surveillance and photography, remains a challenge. Traditional image enhancement techniques and current transformer models are limited with primary focus on either overexposure or underexposure. To bridge this gap, we introduce the Unified-Exposure Guided Transformer (Unified-EGformer). Our proposed solution is built upon advanced transformer architectures, equipped with local pixel-level refinement and global refinement blocks for color correction and image-wide adjustments. We employ a guided attention mechanism to precisely identify exposure-compromised regions, ensuring its adaptability across various real-world conditions. U-EGformer, with a lightweight design featuring a memory footprint (peak memory) of only 1134 MB (0.1 Million parameters) and an inference time of **95 ms** (over **9x** faster than typical existing implementations, is a viable choice for real-time applications such as surveillance and autonomous navigation. Additionally, our model is highly generalizable, requiring minimal fine-tuning to handle multiple tasks and datasets with a single architecture.

## News
- Sept, 2024: New related paper released - [ExpoMamba](https://github.com/eashanadhikarla/ExpoMamba)
- Aug 17, 2024: Our paper got accepted in ICPR 2024!
- July 18, 2024: ArXiv released!

## Getting Started
Conda Environment Installation
```python
  # Create a new conda environment
  conda create --name uegformer python=3.8
  
  # Activate the environment  
  conda activate uegformer
  
  # Install PyTorch with CUDA support
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  
  # Install additional dependencies
  pip install timm einops pyyaml wandb opencv-python pillow matplotlib tqdm
  pip install pytorch-msssim torchinfo

  # OR simply use our environment
  conda env create -f env.yml
```

## Dataset Structure
```python
dataset/
├── low/               # Low-exposure images
├── over/              # Over-exposure images  
├── gt/                # Ground truth images
├── low_masks_otsu/    # Attention masks for low-exposure
└── over_masks_otsu/   # Attention masks for over-exposure
```

## Training
The training pipeline supports training on multiple datasets with flexible configurations, make sure to edit the config file accordingly.
```python
python3 fsdp_train.py
```
## Eval
```python
python3 evalnp_batch.py
```

# BibTeX Reference
If you found our paper and code useful, please consider citing our paper:
```
@inproceedings{adhikarla2024unified,
  title={Unified-EGformer: Exposure Guided Lightweight Transformer for Mixed-Exposure Image Enhancement},
  author={Adhikarla, Eashan and Zhang, Kai and VidalMata, Rosaura G and Aithal, Manjushree and Madhusudhana, Nikhil Ambha and Nicholson, John and Sun, Lichao and Davison, Brian D},
  booktitle={International Conference on Pattern Recognition},
  pages={260--275},
  year={2024},
  organization={Springer}
}
```

## Other related works
```
@inproceedings{
  adhikarla2024expomamba,
  title={ExpoMamba: Exploiting Frequency {SSM} Blocks for Efficient and Effective Image Enhancement},
  author={Eashan Adhikarla and Kai Zhang and John Nicholson and Brian D. Davison},
  booktitle={Workshop on Efficient Systems for Foundation Models II @ ICML2024},
  year={2024},
  url={https://openreview.net/forum?id=X9L6PatYhH}
}
```
