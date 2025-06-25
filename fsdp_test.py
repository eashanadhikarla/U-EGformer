import os
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dataloader import DataGenerator
from model.UnifiedEGformer import GuidedIAT
from utils import print_args
from trainer import test, dsttest

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '52355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_main(rank, world_size, args):
    setup(rank, world_size)

    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)
    else:
        rank = 0  # For non-distributed setup, rank will be 0

    if is_distributed and dist.get_rank() == 0:
        print("\n===================\nTest Configurations\n===================")
        print_args(args)
        print("\n")
    elif not is_distributed:
        print("\n===================\nTest Configurations\n===================")
        print_args(args)
        print("\n")

    testdataset  = DataGenerator(images_path=args.testdir, mode='test', image_size=args.image_size, task=args.task, num_samples=args.num_test_samples)
    sampler_ts = DistributedSampler(testdataset, rank=rank, num_replicas=world_size)

    cuda_kwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': False}
    test_kwargs  = {'batch_size': 1, 'sampler': sampler_ts}
    test_kwargs.update(cuda_kwargs)
    test_loader  = torch.utils.data.DataLoader(testdataset, **test_kwargs)

    torch.cuda.set_device(rank)

    model = GuidedIAT(in_dim=3, type=args.task).to(rank)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    epoch = 0
    best_psnr, best_ssim = 0.0, 0.0
    test(model, epoch, test_loader, best_psnr, best_ssim)

    cleanup()

if __name__ == '__main__':
    # Load testing configuration from YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Training configuration
    parser = argparse.ArgumentParser(
        description='Unified-Exposure Guided Transformer for Low-light Image Enhancement'
    )

    # Testing arguments
    training_args = config.get('testing', {})
    parser.set_defaults(**training_args)

    args = parser.parse_args()

    # As the rank variable was missing, I'm assuming it's initialized to 0 and world_size is 1 for single node testing.
    test_main(rank=0, world_size=1, args=args)