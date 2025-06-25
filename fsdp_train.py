import os
import random
import yaml
import argparse
import functools
import numpy as np
# Weights & Biases
import wandb
from datetime import datetime
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from warmup_scheduler import GradualWarmupScheduler
from dataloader import LOLv1_DataGenerator, LOLv2_DataGenerator, SICE_DataGenerator, ME_DataGenerator
from model.UnifiedEGformer import GuidedIAT
from trainer import train, validate, test, dsttrain, dstvalidate, dsttest
from loss import LossFunctions
from utils import tupletype, network_parameters, print_args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '52355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def save_model(model, args, rank, name, is_distributed): #, is_best_ssim, is_best_psnr):
    """
    Saves the model based on provided criteria.
    """
    # states = model.state_dict()
    # print(f"Saving the best {name} model")
    # torch.save(states, os.path.join(args.ckpt_dir, f"egformer_sicev2_best_{name}.pt"))

    if args.save_model and is_distributed:
        if rank == 0:
            states = model.state_dict()
            print(f"Saving the best {name} model")
            torch.save(states, os.path.join(args.ckpt_dir, f"egformer_sicev2_best_{name}.pt"))
    elif args.save_model and not is_distributed:
        states = model.state_dict()
        print(f"Saving the best {name} model")
        torch.save(states, os.path.join(args.ckpt_dir, f"egformer_sicev2_best_{name}.pt"))

def fsdp_main(rank, world_size, args):

    if args.wandb:
        # Wandb Project & Experiment Naming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dataset_name = "LOL-v2"
        task_name = "underexp"
        misc_description = "blocks5_indiv_noKL_hparamloss_warmup0.3k-1.2k"

        experiment_name = f"exp-{current_time}-{misc_description}"
        project_name = f"{dataset_name}-{task_name}"
        # Initialize Weights & Biases
        wandb.init(project=project_name, name=experiment_name)

    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)
    else:
        rank = 0  # For non-distributed setup, rank will be 0

    #######################
    ## Train Configurations
    #######################
    # Get the current date and time
    now = datetime.now()
    args.ckpt_dir = f"{args.ckpt_dir}_{now.day}_{now.strftime('%b')}_{now.hour}h_{now.minute}m"
    if is_distributed and dist.get_rank() == 0:
        if not os.path.exists(args.ckpt_dir) and args.save_model:
            os.makedirs(args.ckpt_dir)
        print("\n====================\nTrain Configurations\n====================")
        print_args(args)
        print("\n")
    elif not is_distributed:
        if not os.path.exists(args.ckpt_dir) and args.save_model:
            os.makedirs(args.ckpt_dir)
        print("\n====================\nTrain Configurations\n====================")
        print_args(args)
        print("\n")
    if args.wandb:
        # Log all hyperparameters
        wandb.config.update(vars(args))

    ##########################
    ## Data loading (Strategy)
    ##########################
    if "LOL-v1" in args.traindir:
        train_paths, val_paths = split_data(args.traindir, args.split_percentage)
        traindataset = LOLv1_DataGenerator(images_path=train_paths, mode='train', image_size=args.image_size, task=args.task)
        if is_distributed and dist.get_rank() == 0:
            print(f"Total training images: {len(traindataset)}")
        if val_paths:
            valdataset = LOLv1_DataGenerator(images_path=val_paths, mode='val', image_size=args.image_size, task=args.task)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total validation images: {len(valdataset)}")
        else:
            valdataset = None
        if args.testdir:
            testdataset = LOLv1_DataGenerator(images_path=args.testdir, mode='test', image_size=args.image_size, task=args.task, num_samples=args.num_test_samples)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total test images: {len(testdataset)}")

    elif "LOL-v2" in args.traindir:
        train_paths, val_paths = split_data(args.traindir, args.split_percentage)
        traindataset = LOLv2_DataGenerator(images_path=train_paths, mode='train', image_size=args.image_size) #, task=args.task)
        if is_distributed and dist.get_rank() == 0:
            print(f"Total training images: {len(traindataset)}")
        if val_paths:
            valdataset = LOLv2_DataGenerator(images_path=val_paths, mode='val', image_size=args.image_size) #, task=args.task)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total validation images: {len(valdataset)}")
        else:
            valdataset = None
        if args.testdir:
            testdataset = LOLv2_DataGenerator(images_path=args.testdir, mode='test', image_size=args.image_size, num_samples=args.num_test_samples) # task=args.task)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total test images: {len(testdataset)}")

    elif "SICE" in args.traindir:
        train_paths, val_paths = split_data(args.traindir, args.split_percentage)
        traindataset = SICE_DataGenerator(images_path=train_paths, mode='train', image_size=args.image_size, task=args.task)
        if is_distributed and dist.get_rank() == 0:
            print(f"Total training images: {len(traindataset)}")
        if val_paths:
            valdataset = SICE_DataGenerator(images_path=val_paths, mode='val', image_size=args.image_size, task=args.task)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total validation images: {len(valdataset)}")
        else:
            valdataset = None
        if args.testdir:
            testdataset = SICE_DataGenerator(images_path=args.testdir, mode='test', image_size=args.image_size, task=args.task, num_samples=args.num_test_samples)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total test images: {len(testdataset)}")

    elif "ME" in args.traindir:
        train_paths, val_paths = split_data(args.traindir, args.split_percentage)
        traindataset = ME_DataGenerator(images_path=train_paths, mode='train', image_size=args.image_size, task=args.task)
        if is_distributed and dist.get_rank() == 0:
            print(f"Total training images: {len(traindataset)}")
        if val_paths:
            valdataset = ME_DataGenerator(images_path=val_paths, mode='val', image_size=args.image_size, task=args.task)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total validation images: {len(valdataset)}")
        else:
            valdataset = None
        if args.testdir:
            testdataset = ME_DataGenerator(images_path=args.testdir, mode='test', image_size=args.image_size, task=args.task, num_samples=args.num_test_samples)
            if is_distributed and dist.get_rank() == 0:
                print(f"Total test images: {len(testdataset)}")

    if is_distributed:
        sampler_tr = DistributedSampler(traindataset, rank=rank, num_replicas=world_size, shuffle=True)
        sampler_vl = DistributedSampler(valdataset, rank=rank, num_replicas=world_size) if valdataset else None
        if args.testdir:
            sampler_ts = DistributedSampler(testdataset, rank=rank, num_replicas=world_size)
    else:
        # For non-distributed setup, use the default sampler
        sampler_tr = None
        sampler_vl = None
        if args.testdir:
            sampler_ts = None

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler_tr}
    val_kwargs   = {'batch_size': 1, 'sampler': sampler_vl} if valdataset else None
    if args.testdir:
        test_kwargs = {'batch_size': 1, 'sampler': sampler_ts}
    cuda_kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs) if valdataset else None
    if args.testdir:
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(traindataset, **train_kwargs)
    val_loader   = torch.utils.data.DataLoader(valdataset, **val_kwargs) if valdataset else None
    if args.testdir:
        test_loader  = torch.utils.data.DataLoader(testdataset, **test_kwargs)

    my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    #################
    ## Model Instance
    #################
    model = GuidedIAT(input_channels=3, transformer_blocks=args.MapGenerator_transformer_blocks, embedding_dimension=args.pos_emb_dim).to(rank)

    # modeldir = "./checkpoint/lolv2/lolv2_26_Jan_21h_28m/egformer_sicev2_best_ssim.pt"
    # checkpoint = torch.load(modeldir)
    # model.load_state_dict(checkpoint)

    model = FSDP(model) if is_distributed else model.to(rank)

    p_number = network_parameters(model)
    if is_distributed and dist.get_rank() == 0:
        print("Total params: ", p_number * world_size)
    elif not is_distributed:
        print("Total params: ", p_number)

    ############
    ## Optimizer
    ############
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    #######################
    ## Scheduler (Strategy)
    #######################
    if args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'WarmupCosine':
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup_epochs, eta_min=float(1e-6))
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    loss_functions = LossFunctions()
    best_psnr = 0.0  # initialize with a low value
    best_ssim = 0.0  # initialize with a low value

    ###########
    ## Training
    ###########
    init_start_event.record()
    if is_distributed and args.split_percentage == 1:
        for epoch in range(1, args.epochs + 1):
            if epoch in (args.warmuprestartepochforcosine):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs-epoch)

            dsttrain(args, model, rank, optimizer, epoch, loss_functions, train_loader, scheduler, wandb)
            prev_best_psnr, prev_best_ssim = best_psnr, best_ssim
            if args.testdir and epoch%args.testevery==0:
                best_psnr, best_ssim = dsttest(args, model, rank, world_size, epoch, test_loader, best_psnr, best_ssim, wandb)

            # scheduler.step()
            if best_psnr > prev_best_psnr:
                save_model(model, args, rank, "psnr", is_distributed)
            if best_ssim > prev_best_ssim:
                save_model(model, args, rank, "ssim", is_distributed)

    elif is_distributed:
        for epoch in range(1, args.epochs + 1):
            if epoch in (args.warmuprestartepochforcosine):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs-epoch)

            dsttrain(args, model, rank, optimizer, epoch, loss_functions, train_loader, scheduler, wandb)
            prev_best_psnr, prev_best_ssim = best_psnr, best_ssim
            best_psnr, best_ssim = dstvalidate(args, model, rank, world_size, epoch, val_loader, best_psnr, best_ssim, wandb)
            if args.testdir and epoch%args.testevery==0:
                best_psnr, best_ssim = dsttest(args, model, rank, world_size, epoch, test_loader, best_psnr, best_ssim, wandb)

            # scheduler.step()
            if best_psnr > prev_best_psnr:
                save_model(model, args, rank, "psnr", is_distributed)
            if best_ssim > prev_best_ssim:
                save_model(model, args, rank, "ssim", is_distributed)

    else:
        # single GPU training
        if args.split_percentage == 1:
            for epoch in range(1, args.epochs + 1):
                if epoch in (args.warmuprestartepochforcosine):
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs-epoch)

                train(args, model, optimizer, epoch, loss_functions, train_loader, scheduler, wandb)
                prev_best_psnr, prev_best_ssim = best_psnr, best_ssim
                if args.testdir and epoch%args.testevery==0:
                    best_psnr, best_ssim = test(args, model, epoch, test_loader, best_psnr, best_ssim, wandb)

                # scheduler.step()
                if best_psnr > prev_best_psnr:
                    save_model(model, args, rank, "psnr", is_distributed)
                if best_ssim > prev_best_ssim:
                    save_model(model, args, rank, "ssim", is_distributed)

        else:
            for epoch in range(1, args.epochs + 1):
                if epoch in (args.warmuprestartepochforcosine):
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs-epoch)
                train(args, model, optimizer, epoch, loss_functions, train_loader, scheduler, wandb)
                prev_best_psnr, prev_best_ssim = best_psnr, best_ssim
                best_psnr, best_ssim = validate(args, model, epoch, val_loader, best_psnr, best_ssim, wandb)
                if args.testdir and epoch%args.testevery==0:
                    best_psnr, best_ssim = test(args, model, epoch, test_loader, best_psnr, best_ssim, wandb)

                # scheduler.step()
                if best_psnr > prev_best_psnr:
                    save_model(model, args, rank, "psnr", is_distributed)
                if best_ssim > prev_best_ssim:
                    save_model(model, args, rank, "ssim", is_distributed)

    init_end_event.record()
    # # use a barrier to make sure training is done on all ranks
    # dist.barrier()

    if args.wandb:
        # At the end of training
        wandb.finish()

    if is_distributed and rank == 0:
        # Print elapsed time only for rank 0
        elapsed_time_seconds = (init_start_event.elapsed_time(init_end_event) / 1000)
        elapsed_time_hours = elapsed_time_seconds / 3600
        print(f"CUDA event elapsed time: {elapsed_time_hours:.4f} hours ({init_start_event.elapsed_time(init_end_event) / 1000} sec)")
    elif not is_distributed:
        # Print elapsed time only for rank 0
        elapsed_time_seconds = (init_start_event.elapsed_time(init_end_event) / 1000)
        elapsed_time_hours = elapsed_time_seconds / 3600
        print(f"CUDA event elapsed time: {elapsed_time_hours:.4f} hours ({init_start_event.elapsed_time(init_end_event) / 1000} sec)")

    if is_distributed:
        cleanup()  # Cleanup only for distributed training

def tuple_loader(loader, node):
    value = loader.construct_scalar(node)
    return tuple(map(int, value.strip('()').split(',')))

if __name__ == '__main__':
    yaml.add_constructor('!tuple', tuple_loader, Loader=yaml.SafeLoader)

    ## Load configuration from YAML file

    # with open('config_lolv1.yml', 'r') as file:
    #     config = yaml.safe_load(file)

    with open('./config/config_lolv2.yml', 'r') as file:
        config = yaml.safe_load(file)

    # with open('config_sicev2.yml', 'r') as file:
    #     config = yaml.safe_load(file)

    # with open('config_mev2.yml', 'r') as file:
    #     config = yaml.safe_load(file)

    # Training configuration
    parser = argparse.ArgumentParser(
        description='Unified-Exposure Guided Transformer for Low-light Image Enhancement'
    )

    # Data arguments
    data_args = config.get('data', {})
    parser.set_defaults(**data_args)

    # Training arguments
    training_args = config.get('training', {})
    parser.set_defaults(**training_args)

    # Model arguments
    model_args = config.get('model', {})
    parser.set_defaults(**model_args)

    # Debugging arguments
    debugging_args = config.get('debugging', {})
    parser.set_defaults(**debugging_args)

    args = parser.parse_args()

    torch.manual_seed(args.seed) # https://arxiv.org/abs/2109.08203
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    WORLD_SIZE = torch.cuda.device_count()

    if WORLD_SIZE > 1:
        # Multiple GPUs available
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)
    else:
        # Single GPU or CPU
        # rank and world_size are set to 0 and 1, respectively
        fsdp_main(0, 1, args)