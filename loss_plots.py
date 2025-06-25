import os
import re
import matplotlib.pyplot as plt

# Path to your log file
rootdir = "/home/ubuntu/research/egformer/egformer/"
# log_filename = "lolv2_blocks5_indiv_newhparamloss_warmup0.3k-1.2k.out"
log_filename = "lolv2_blocks5_indiv_noKL_hparamloss_warmup0.3k-1.2k-2.out"
file_path = os.path.join(rootdir, log_filename)

# Regular expression patterns
epoch_pattern = r'Train Epoch: (\d+)'
loss_pattern = r'\(([^)]+)\)'

# Lists to store data for plotting
epochs = []
l1_losses = []
char_losses = []
mse_losses = []
char_ssim_losses = []
# kl_losses = []
ssim_losses = []
vgg_losses = []
mal1_losses = []

# Flags to control parsing
parse_losses = False
epoch = 0

# Read and parse the log file
with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    # Extract epoch information
    epoch_match = re.search(epoch_pattern, line)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        epochs.append(epoch)

    # Parse the line for individual losses
    if parse_losses:
        # Extract individual losses using regex
        match = re.search(loss_pattern, line)
        if match:
            losses_str = match.group(1)
            loss_pairs = re.findall(r'(\w+): (\d+\.\d+)', losses_str)

            # Print the extracted loss pairs for debugging
            # print(f'Epoch {epoch} Loss Pairs:', loss_pairs)

            # Convert the loss pairs to a dictionary
            losses = {k: float(v) for k, v in loss_pairs}

            l1_losses.append(losses.get('L1', 0))
            # char_losses.append(losses.get('Char', 0))
            mse_losses.append(losses.get('MSE', 0))
            char_ssim_losses.append(losses.get('CHAR_SSIM', 0))
            # kl_losses.append(losses.get('KL', 0))
            ssim_losses.append(losses.get('SSIM', 0))
            vgg_losses.append(losses.get('VGG', 0))
            # mal1_losses.append(losses.get('MAL1', 0))
            parse_losses = False  # Stop parsing the current line for individual losses

    # Check if the current line indicates the start of individual losses
    if 'Loss: ' in line:
        parse_losses = True

# Plot overall losses
plt.figure(figsize=(10, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(epochs, l1_losses, label='L1', color=colors[0])
# plt.plot(epochs, char_losses, label='Char', color=colors[0])
plt.plot(epochs, mse_losses, label='MSE', color=colors[1])
plt.plot(epochs, char_ssim_losses, label='CHAR_SSIM', color=colors[2])
# plt.plot(epochs, kl_losses, label='CHAR+SSIM', color=colors[2])
plt.plot(epochs, ssim_losses, label='SSIM', color=colors[3])
plt.plot(epochs, vgg_losses, label='VGG', color=colors[4])
# plt.plot(epochs, mal1_losses, label='MAL1', color=colors[5])
plt.title('Overall Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss Values')
plt.legend()
plt.grid(True)

# Save the overall loss plot to a file
output_overall_file_path = "./loss_plot_overall.png"
plt.savefig(output_overall_file_path)
print(f'Overall Loss Plot saved to: {output_overall_file_path}')

# Plot individual losses in a subplot grid
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# L1 Loss Plot
axs[0, 0].plot(epochs, l1_losses, label='L1', color=colors[0])
axs[0, 0].set_title('L1 Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# # Char Loss Plot
# axs[0, 0].plot(epochs, char_losses, label='Char', color=colors[0])
# axs[0, 0].set_title('Char Loss')
# axs[0, 0].legend()
# axs[0, 0].grid(True)

# MSE Loss Plot
axs[0, 1].plot(epochs, mse_losses, label='MSE', color=colors[1])
axs[0, 1].set_title('MSE Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# KL Loss Plot
axs[1, 0].plot(epochs, char_ssim_losses, label='CHAR_SSIM', color=colors[2])
# axs[1, 0].plot(epochs, kl_losses, label='CHAR_SSIM', color=colors[2])
axs[1, 0].set_title('CHAR_SSIM Loss')
axs[1, 0].legend()
axs[1, 0].grid(True)

# SSIM Loss Plot
axs[1, 1].plot(epochs, ssim_losses, label='SSIM', color=colors[3])
axs[1, 1].set_title('SSIM Loss')
axs[1, 1].legend()
axs[1, 1].grid(True)

# VGG Loss Plot
axs[2, 0].plot(epochs, vgg_losses, label='VGG', color=colors[4])
axs[2, 0].set_title('VGG Loss')
axs[2, 0].legend()
axs[2, 0].grid(True)

# # MAL1 Loss Plot
# axs[2, 1].plot(epochs, mal1_losses, label='MAL1', color=colors[5])
# axs[2, 1].set_title('MAL1 Loss')
# axs[2, 1].legend()
# axs[2, 1].grid(True)

# Adjust layout
plt.tight_layout()

# Save the individual losses subplot grid to a file
output_subplot_file_path = "./loss_plot_individual.png"
plt.savefig(output_subplot_file_path)
print(f'Individual Losses Subplot Plot saved to: {output_subplot_file_path}')
