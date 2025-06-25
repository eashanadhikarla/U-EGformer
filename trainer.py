import os
import json
import torch
import torch.distributed as dist
from metrics import PSNR, SSIM

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_images(imgs, titles, save_path):
    num_images = len(imgs)
    fig = plt.figure(figsize=(15, 5))
    
    for i in range(num_images):
        fig.add_subplot(1, num_images, i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def train(args, model, optimizer, epoch, loss_functions, train_loader, scheduler, wandb):
    model.train()
    total_loss = 0
    individual_losses = {}

    for iteration, imgs in enumerate(train_loader):
        low_img, high_img, target_attn_map = imgs[0].to(device), imgs[1].to(device), imgs[2].to(device)
        optimizer.zero_grad()
        local_mul, local_add, enhance_img, attn_map = model(low_img)

        # Loss calculation
        loss = 0
        if 'L1' in args.loss_funcs:
            l1_loss = 1 * loss_functions.l1_loss(enhance_img, high_img)
            loss += l1_loss
            individual_losses['L1'] = f'{l1_loss.item():.5f}'

        if 'SL1' in args.loss_funcs:
            sL1_loss = 1 * loss_functions.smooth_l1_loss(enhance_img, high_img)
            loss += sL1_loss
            individual_losses['SL1'] = f'{sL1_loss.item():.5f}'

        if 'Char' in args.loss_funcs:
            char_loss = 1 * loss_functions.charbonnier_loss(enhance_img, high_img)
            loss += char_loss
            individual_losses['Char'] = f'{char_loss.item():.5f}'

        if 'MSE' in args.loss_funcs:
            mse_loss = 0.5 * loss_functions.mse_loss(enhance_img, high_img)
            loss += mse_loss
            individual_losses['MSE'] = f'{mse_loss.item():.5f}'

        if 'Attention' in args.loss_funcs:
            # attention_loss = 0.8 * loss_functions.attention_loss(attn_map, target_attn_map)
            attention_loss = ( 1*loss_functions.charbonnier_loss(attn_map, target_attn_map) + 1*loss_functions.ssim_loss(attn_map, target_attn_map) )
            loss += attention_loss
            individual_losses['CHAR_SSIM'] = f'{attention_loss.item():.5f}'

        if 'SSIM' in args.loss_funcs:
            ssim_loss = 1 * loss_functions.ssim_loss(enhance_img, high_img)
            loss += ssim_loss
            individual_losses['SSIM'] = f'{ssim_loss.item():.5f}'

        if 'MSSSIM' in args.loss_funcs:
            ms_ssim_loss = 1 - loss_functions.ms_ssim_loss(enhance_img, high_img)
            loss += ms_ssim_loss
            individual_losses['MSSSIM'] = f'{ms_ssim_loss.item():.5f}'

        if 'VGG' in args.loss_funcs:
            vgg_loss = 0.5 * loss_functions.vgg_loss(enhance_img, high_img)
            loss += vgg_loss
            individual_losses['VGG'] = f'{vgg_loss.item():.5f}'

        if 'MAL1' in args.loss_funcs:
            mal1_loss = 1e-5 * loss_functions.mul_add_loss(mul, add, low_img, high_img)
            loss += mal1_loss
            individual_losses['MAL1'] = f'{mal1_loss.item():.5f}'

        if 'GHist' in args.loss_funcs:
            ghist_loss = 1 * loss_functions.gradient_hist_loss(enhance_img, high_img)
            loss += ghist_loss
            individual_losses['GH'] = f'{ghist_loss.item():.5f}'

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if args.wandb:
            wandb.log({
                'Loss/train': loss.item(),
                **individual_losses
            }, step=epoch)

    # average loss for non-distributed setup
    total_loss /= len(train_loader)
    print(f'Train Epoch: {epoch} \tLoss: {total_loss:.5f}')
    formatted_losses = ', '.join([f'{key}: {value}' for key, value in individual_losses.items()])
    print(f'({formatted_losses})')

    # if args.debug:
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        os.makedirs(args.ckpt_dir+'/progress')
    # Plot images after each epoch
    with torch.no_grad():
        model.eval()
        local_mul, local_add, enhance_img, attention_map = model(low_img)
        model.train()

        low_img = low_img.squeeze()
        local_mul = local_mul.squeeze()
        local_add = local_add.squeeze()

        # Create PIL image for the local enhanced output
        local_enhanced_output_np = to_pil_image((low_img.mul(local_mul).add(local_add)).cpu())
        attn_map_np = to_pil_image(attention_map.squeeze().cpu())
        enhance_img_np = to_pil_image(enhance_img.squeeze().cpu())
        low_img_np = to_pil_image(low_img.squeeze().cpu())
        high_img_np = to_pil_image(high_img.squeeze().cpu())

    # Save images to the checkpoint folder
    save_path = os.path.join(args.ckpt_dir+'/progress', f'epoch_{epoch}_images.png')
    save_images([low_img_np, local_enhanced_output_np, attn_map_np, enhance_img_np, high_img_np],
                ['Low Image', 'Local Enhanced Output', 'Attention Map', 'Enhanced Image', 'High Image'],
                save_path)

    # Wait for user input before moving to the next epoch
    input("Press Enter to continue to the next epoch...")

def validate(args, model, epoch, val_loader, best_psnr, best_ssim, wandb):
    model.eval()
    ssim, psnr = SSIM(), PSNR()
    ssim_sum, psnr_sum = 0, 0  # Initialize sums to accumulate values

    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].to(device), imgs[1].to(device)
            _, _, enhanced_img, _ = model(low_img)

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        ssim_sum += ssim_value
        psnr_sum += psnr_value

    SSIM_mean = ssim_sum / len(val_loader)
    PSNR_mean = psnr_sum / len(val_loader)

    if PSNR_mean > best_psnr:
        best_psnr = PSNR_mean
    if SSIM_mean > best_ssim:
        best_ssim = SSIM_mean

    print(f"Validation Epoch {epoch}")
    print("========================================================")
    print(f"Validation PSNR: {float(PSNR_mean):>8.4f}    |    Highest PSNR: {float(best_psnr):>8.4f}")
    print(f"Validation SSIM: {float(SSIM_mean):>8.4f}    |    Highest SSIM: {float(best_ssim):>8.4f}")
    print("========================================================")

    if args.wandb:
        wandb.log({
            'Metrics/SSIM': SSIM_mean,
            'Metrics/PSNR': PSNR_mean
        }, step=epoch)
    return best_psnr, best_ssim

def test(args, model, epoch, test_loader, best_psnr, best_ssim, wandb):
    model.eval()
    ssim, psnr = SSIM(), PSNR()
    ssim_sum, psnr_sum = 0.0, 0.0  # Initialize sums to accumulate values

    for i, imgs in enumerate(test_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].to(device), imgs[1].to(device)
            _, _, enhanced_img, _ = model(low_img)
        
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        ssim_sum += ssim_value
        psnr_sum += psnr_value

    SSIM_mean = ssim_sum / len(test_loader)
    PSNR_mean = psnr_sum / len(test_loader)

    if PSNR_mean > best_psnr:
        best_psnr = PSNR_mean
    if SSIM_mean > best_ssim:
        best_ssim = SSIM_mean

    print(f"Test Epoch {epoch}")
    print("========================================================")
    print(f"PSNR: {float(PSNR_mean):>8.4f}    |    Highest PSNR: {float(best_psnr):>8.4f}")
    print(f"SSIM: {float(SSIM_mean):>8.4f}    |    Highest SSIM: {float(best_ssim):>8.4f}")
    print("========================================================\n")
    
    if args.wandb:
        wandb.log({
            'Metrics/SSIM': SSIM_mean,
            'Metrics/PSNR': PSNR_mean
        }, step=epoch)
    return best_psnr, best_ssim


def dsttrain(args, model, rank, optimizer, epoch, loss_functions, train_loader, scheduler, wandb):
    model.train()
    ddp_loss = torch.zeros(1).to(rank)
    individual_losses = {}

    for iteration, imgs in enumerate(train_loader):
        low_img, high_img, target_attn_map = imgs[0].to(rank), imgs[1].to(rank), imgs[2].to(rank)
        optimizer.zero_grad()
        local_mul, local_add, enhance_img, attn_map = model(low_img)

        # Loss calculation
        loss = 0
        if 'L1' in args.loss_funcs:
            l1_loss = 1 * loss_functions.l1_loss(enhance_img, high_img)
            loss += l1_loss
            individual_losses['L1'] = f'{l1_loss.item():.5f}'

        if 'SL1' in args.loss_funcs:
            sL1_loss = 1 * loss_functions.smooth_l1_loss(enhance_img, high_img)
            loss += sL1_loss
            individual_losses['SL1'] = f'{sL1_loss.item():.5f}'

        if 'Char' in args.loss_funcs:
            char_loss = 1 * loss_functions.charbonnier_loss(enhance_img, high_img)
            loss += char_loss
            individual_losses['Char'] = f'{char_loss.item():.5f}'

        if 'MSE' in args.loss_funcs:
            mse_loss = 0.5 * loss_functions.mse_loss(enhance_img, high_img)
            loss += mse_loss
            individual_losses['MSE'] = f'{mse_loss.item():.5f}'

        if 'Attention' in args.loss_funcs:
            # attention_loss = 0.8 * loss_functions.attention_loss(attn_map, target_attn_map)
            attention_loss = ( 1*loss_functions.charbonnier_loss(attn_map, target_attn_map) + 1*loss_functions.ssim_loss(attn_map, target_attn_map) )
            loss += attention_loss
            individual_losses['CHAR_SSIM'] = f'{attention_loss.item():.5f}'

        if 'SSIM' in args.loss_funcs:
            ssim_loss = 1 * loss_functions.ssim_loss(enhance_img, high_img)
            loss += ssim_loss
            individual_losses['SSIM'] = f'{ssim_loss.item():.5f}'

        if 'MSSSIM' in args.loss_funcs:
            ms_ssim_loss = 1 - loss_functions.ms_ssim_loss(enhance_img, high_img)
            loss += ms_ssim_loss
            individual_losses['MSSSIM'] = f'{ms_ssim_loss.item():.5f}'

        if 'VGG' in args.loss_funcs:
            vgg_loss = 0.5 * loss_functions.vgg_loss(enhance_img, high_img)
            loss += vgg_loss
            individual_losses['VGG'] = f'{vgg_loss.item():.5f}'

        if 'MAL1' in args.loss_funcs:
            mal1_loss = 1e-5 * loss_functions.mul_add_loss(mul, add, low_img, high_img)
            loss += mal1_loss
            individual_losses['MAL1'] = f'{mal1_loss.item():.5f}'

        if 'GHist' in args.loss_funcs:
            ghist_loss = 1 * loss_functions.gradient_hist_loss(enhance_img, high_img)
            loss += ghist_loss
            individual_losses['GH'] = f'{ghist_loss.item():.5f}'

        loss.backward()
        optimizer.step()
        scheduler.step()
        ddp_loss[0] += loss.item()

        if args.wandb:
            wandb.log({
                'Total Training Loss': loss.item(),
                **{key: float(value) for key, value in individual_losses.items()}
            }, step=epoch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    ddp_loss[0] /= len(train_loader)

    if rank == 0:
        # if args.debug:
        print("Debugging mode")
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        # Plot images after each epoch (only rank 0)
        with torch.no_grad():
            model.eval()
            local_mul, local_add, enhance_img, attention_map = model(low_img)
            model.train()

        low_img = low_img.squeeze()
        local_mul = local_mul.squeeze()
        local_add = local_add.squeeze()

        # Create PIL image for the local enhanced output
        local_enhanced_output_np = to_pil_image((low_img.mul(local_mul).add(local_add)).cpu())
        attn_map_np = to_pil_image(attention_map.squeeze().cpu())
        enhance_img_np = to_pil_image(enhance_img.squeeze().cpu())
        low_img_np = to_pil_image(low_img.squeeze().cpu())
        high_img_np = to_pil_image(high_img.squeeze().cpu())

        # Save images to the checkpoint folder
        save_path = os.path.join(args.ckpt_dir+'/progress', f'epoch_{epoch}_images.png')
        save_images([low_img_np, local_enhanced_output_np, attn_map_np, enhance_img_np, high_img_np],
                    ['Low Image', 'Local Enhanced Output', 'Attention Map', 'Enhanced Image', 'High Image'],
                    save_path)

        # # Synchronize processes before continuing
        # dist.barrier()


def dstvalidate(args, model, rank, world_size, epoch, val_loader, best_psnr, best_ssim, wandb):
    model.eval()
    ssim, psnr = SSIM(), PSNR()
    ssim_list, psnr_list = [], []
    ddp_values = torch.zeros(2).to(rank)

    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].to(rank), imgs[1].to(rank)
            _, _, enhanced_img, attn_map = model(low_img)

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        ddp_values[0] += ssim_value
        ddp_values[1] += psnr_value

    dist.all_reduce(ddp_values, op=dist.ReduceOp.SUM)
    ddp_values /= world_size  # Average the values
    SSIM_mean = ddp_values[0] / len(val_loader)
    PSNR_mean = ddp_values[1] / len(val_loader)

    if PSNR_mean > best_psnr:
        best_psnr = PSNR_mean
    if SSIM_mean > best_ssim:
        best_ssim = SSIM_mean

    if rank == 0:
        print(f"Validation Epoch {epoch}")
        print("========================================================")
        print(f"Validation PSNR: {float(PSNR_mean):>8.4f}    |    Highest PSNR: {float(best_psnr):>8.4f}")
        print(f"Validation SSIM: {float(SSIM_mean):>8.4f}    |    Highest SSIM: {float(best_ssim):>8.4f}")
        print("========================================================")
    
    if args.wandb:
        wandb.log({
            'Metrics/SSIM': SSIM_mean,
            'Metrics/PSNR': PSNR_mean
        }, step=epoch)
    return best_psnr, best_ssim

def dsttest(args, model, rank, world_size, epoch, test_loader, best_psnr, best_ssim, wandb):
    model.eval()
    ssim, psnr = SSIM(), PSNR()
    ssim_list, psnr_list = [], []
    ddp_values = torch.zeros(2).to(rank)

    for i, imgs in enumerate(test_loader):
        with torch.no_grad():
            low_img, high_img = imgs[0].to(rank), imgs[1].to(rank)
            _, _, enhanced_img, attn_map = model(low_img)

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()
        ddp_values[0] += ssim_value
        ddp_values[1] += psnr_value

    dist.all_reduce(ddp_values, op=dist.ReduceOp.SUM)
    ddp_values /= world_size  # Average the values
    SSIM_mean = ddp_values[0] / len(test_loader)
    PSNR_mean = ddp_values[1] / len(test_loader)

    if PSNR_mean > best_psnr:
        best_psnr = PSNR_mean
    if SSIM_mean > best_ssim:
        best_ssim = SSIM_mean

    if rank == 0:
        print(f"Test Epoch {epoch}")
        print("========================================================")
        print(f"PSNR: {float(PSNR_mean):>8.4f}    |    Highest PSNR: {float(best_psnr):>8.4f}")
        print(f"SSIM: {float(SSIM_mean):>8.4f}    |    Highest SSIM: {float(best_ssim):>8.4f}")
        print("========================================================\n")
    
    if args.wandb:
        wandb.log({
            'Metrics/SSIM': SSIM_mean,
            'Metrics/PSNR': PSNR_mean
        }, step=epoch)
    return best_psnr, best_ssim