import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from metrics import SSIM #, MS_SSIM
from pytorch_msssim import MS_SSIM

#################
# Perceptual Loss
#################
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.mse_loss = nn.MSELoss()
        self.layer_name_mapping = {
            '3' : "relu1_2",
            '8' : "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(self.mse_loss(pred_im_feature, gt_feature))
        return sum(loss)/len(loss)

class VGG_Loss(nn.Module):
    def __init__(self, vgg_model, _lambda_=0.2):
        super(VGG_Loss, self).__init__()
        self.loss_network = LossNetwork(vgg_model)
        self._lambda_ = _lambda_
        
    def forward(self, output, target):
        Lvgg = self.loss_network(output, target)
        minimizedLvgg = self._lambda_ * Lvgg
        return minimizedLvgg

# Color Loss
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k

#########################
# Gradient Histogram Loss
#########################
class GradientLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2

        assert(self.clip_radius>0 and self.clip_radius<=1)
        self.bin_width = 2*self.clip_radius/self.bin_num

        if self.bin_width*255<1:
            raise RuntimeError("bin width is too small")
        
        self.bin_mean = np.arange(-self.clip_radius+self.bin_width*0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L1' # 'L2'

        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]     # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]      # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]    # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src
        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None
        
        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])
            fx = torch.cuda.FloatTensor([fx])
            fy = torch.cuda.FloatTensor([fy])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)
        loss = self.criterion(output_gradient_x,target_gradient_x)+self.criterion(output_gradient_y,target_gradient_y)
        return loss

######################
# KL Divergence Loss 2
######################
class AttentionMapLoss(nn.Module):
    def __init__(self):
        super(AttentionMapLoss, self).__init__()

    def forward(self, A_pred, A_gt):
        epsilon = 1e-8  # Small constant to avoid division by zero

        # Clipping to ensure values are between epsilon and 1
        A_gt = torch.clamp(A_gt, epsilon, 1.0 - epsilon)
        A_pred = torch.clamp(A_pred, epsilon, 1.0 - epsilon)

        # Computing KL divergence
        kl_loss = torch.mean(A_gt * torch.log(A_gt / A_pred))
        return kl_loss

##################
# 'MUL' 'ADD' Loss
##################
class MUL_ADD(nn.Module):
    def __init__(self, alpha=0.001, beta=0.001):
        super(MUL_ADD, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, mul, add, low_img, high_img):
        # Compute ideal_add
        ideal_add = high_img - (high_img * 1e-8) / (low_img + 1e-8)
        # Compute ideal_mul using ideal_add
        ideal_mul = (high_img - ideal_add) / (low_img + 1e-8)

        # Compute losses
        loss_mul = F.smooth_l1_loss(mul, ideal_mul, reduction='mean')
        loss_add = F.smooth_l1_loss(add, ideal_add, reduction='mean')

        # Weighted sum of losses
        # total_muladd_loss = self.alpha * loss_mul + self.beta * loss_add
        total_muladd_loss = loss_mul + loss_add
        return total_muladd_loss

##################
# Charbonnier Loss
##################
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


class LossFunctions:
    def __init__(self):
        # L1 Loss
        self.l1_loss = nn.L1Loss()

        # Smooth-L1 Loss
        self.smooth_l1_loss = F.smooth_l1_loss

        # MSE Loss
        self.mse_loss = nn.MSELoss()

        # VGG Loss
        vgg_model = vgg16(weights="IMAGENET1K_V1").features[:16]
        vgg_model = vgg_model.cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_loss = VGG_Loss(vgg_model)
        self.vgg_loss.eval()

        # SSIM Loss
        self.ssim_loss = SSIM(channels=3)

        self.mul_add_loss = MUL_ADD()

        # MS-SSIM Loss
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True)

        # Gradient Histogram Loss
        self.gradient_hist_loss = GradientLoss()

        # Attention Map Loss (KL Divergence)
        self.attention_loss = AttentionMapLoss()

        # Charbonnier Loss
        self.charbonnier_loss = CharbonnierLoss()

        # Color Loss
        self.color = L_color()