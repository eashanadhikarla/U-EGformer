import math
import numbers
import einops
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import custom_summary


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
        # return x / (torch.sqrt(sigma+1e-5)).unsqueeze(-1) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class NextAttentionImplZ(nn.Module):
    '''
    Axis-based Multi-head Self-Attention (A-MSA)
    '''
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)
        return res


class NextAttentionZ(nn.Module):
    '''
    Axis-based Multi-head Self-Attention (row and col attention)
    '''
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)
        return x


class FeedForward(nn.Module):
    '''
    Dual Gated Feed-Forward Network (DGFN)
    '''
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    '''
    Axis-based Transformer Block
    '''
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self,
                in_features,
                hidden_features = None,
                out_features = None,
                act_layer = nn.GELU,
                drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Color Normalization
class Aff_channel(nn.Module):
    def __init__(self,
                dim,
                channel_first = True):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        if self.channel_first:
            x1 = torch.tensordot(x, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = x * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        return x2


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
            self,
            in_features,
            hidden_features = None,
            out_features = None,
            act_layer = nn.GELU,
            drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock_ln(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio = 4.,
            qkv_bias = False,
            qk_scale = None,
            drop = 0.,
            attn_drop = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = Aff_channel,
            init_values = 1e-4):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape

        norm_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = x + self.drop_path(self.gamma_1*self.conv2(self.attn(self.conv1(norm_x))))
        norm_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm2(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.gamma_2*self.mlp(norm_x))
        return x


# Short Cut Connection on Final Layer
class LocalEnhancement(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(LocalEnhancement, self).__init__()

        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # main blocks
        blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]

        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add


class query_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.ones((1, 10, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 10, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class EAF_featurefusion(nn.Module):
    def __init__(self, in_channels):
        super(EAF_featurefusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_features, global_features):
        # Apply convolutions to local features
        local_features_1 = F.relu(self.conv1(local_features))
        local_features_2 = F.relu(self.conv2(local_features))

        # Global pooling and fully connected layer for global features
        global_pool = self.global_pool(global_features)
        global_pool = torch.flatten(global_pool, 1)
        global_pool = self.fc(global_pool)
        global_pool = self.sigmoid(global_pool)
        global_pool = global_pool.unsqueeze(-1).unsqueeze(-1)

        # Multiply convolved local features with pooled global features
        fused_features = local_features_1 * global_pool + local_features_2
        return fused_features


def illuminance_map(rgb_image):
    # Weights for converting RGB to grayscale based on luminosity
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(rgb_image.device)
    # Ensure the input image is a torch tensor and has dimensions [Batch, Channels, Height, Width]
    if len(rgb_image.shape) == 3:
        rgb_image = rgb_image.unsqueeze(0)
    # Convert the RGB image to grayscale
    illuminance_map = F.conv2d(rgb_image, weights)
    return illuminance_map


class GlobalEnhancement(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=4):
        super(GlobalEnhancement, self).__init__()
        # Basic color matrix
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)

        # Sub-network for adaptive gamma base calculation
        self.adaptive_gamma_net = nn.Sequential(
            nn.Conv2d(10, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()  # To ensure gamma is within a reasonable range
        )

        # main blocks
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.gamma_linear = nn.Linear(out_channels, 1)
        self.color_linear = nn.Linear(out_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # # Illuminance map
        ilmap = illuminance_map(x)
        inv_ilmap = 1 - ilmap
        x = inv_ilmap.mul(x)

        x = self.conv_large(x)
        x = self.generator(x)

        # Reshape the output of generator to 4D tensor for adaptive_gamma_net
        batch_size, features, length = x.shape
        # Assuming 'length' can be factorized into two equal factors
        side = int(math.sqrt(length))
        reshaped_x = x.view(batch_size, features, side, side)

        # Calculate adaptive gamma base
        adaptive_gamma_base = self.adaptive_gamma_net(reshaped_x)

        # Proceed with the rest of the operations
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + adaptive_gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base

        return gamma, color


class GuidedAttentionMapGenerator(nn.Module):
    '''
    Generates Attention Maps of the exposure affected areas that is guiding IAT transformer for correcting exposures.

    The LLFormer's Axis-MSA is re-designed as a SwinTransformer workflow for mixed-exposure correction, it doesn't 
    inherently "recognize" mixed-exposure affected areas in the attention map in the sense of understanding the semantics 
    of the image. Instead, it learns to generate attention maps that correspond to areas that are potentially affected 
    by mixed exposure.
    '''
    def __init__(self, in_channels, num_heads=1, num_transformer_blocks=1, emb_dim=48):
        super(GuidedAttentionMapGenerator, self).__init__()
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks

        # self.pos_embed = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)

        # Change pos_embed as specified
        self.pos_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # 1x1 convolution to map 48 channels back to in_channels
        self.channel_map = nn.Conv2d(emb_dim, in_channels, kernel_size=1, bias=True)

        self.norm1 = LayerNorm(in_channels, LayerNorm_type='BiasFree')

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=in_channels,
                num_heads=num_heads,
                ffn_expansion_factor=2.66,
                bias=True,
                LayerNorm_type='BiasFree'
            ) for _ in range(num_transformer_blocks)
        ])

        self.norm2 = LayerNorm(in_channels, LayerNorm_type='BiasFree')
        self.mlp = Mlp(in_channels, int(in_channels * 4), act_layer=nn.GELU, drop=0.0)

    def forward(self, x):
        # x = x + self.pos_embed(x)
        x = x + self.channel_map(self.pos_embed(x))
        B, C, H, W = x.shape

        shortcut = x
        x = self.norm1(x)

        x = x.reshape(B, H, W, C) # Added for inference
        x = x.view(B, H, W, C)

        # Apply multiple TransformerBlocks
        for transformer_block in self.transformer_blocks:
            x = x.permute(0, 3, 1, 2)
            x = transformer_block(x)
            x = x.permute(0, 2, 3, 1).contiguous() # .view(B, C, H, W)
        x = x.view(B, C, H, W)

        # Feed-forward network
        x = self.norm2(x)
        x = shortcut + self.mlp(x.view(B, -1, C)).view(B, C, H, W)
        attention_map = x.transpose(1, 2).reshape(B, C, H, W)
        return attention_map


class LearnableDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnableDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class LearnableUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnableUpsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_transpose(x)


class GuidedIAT(nn.Module):
    def __init__(self, input_channels=3, transformer_blocks=3, embedding_dimension=48):
        super(GuidedIAT, self).__init__()

        # Initialize the Attention Map Generator
        self.attention_map_generator = GuidedAttentionMapGenerator(
            in_channels=input_channels, 
            num_transformer_blocks=transformer_blocks, 
            emb_dim=embedding_dimension
        )

        # Initialize Local Enhancement Block
        self.local_enhancement = LocalEnhancement(in_dim=input_channels)
        # Initialize Global Enhancement Block
        self.global_enhancement = GlobalEnhancement(in_channels=input_channels) #, type=enhancement_type)
        # Initialize Exposure Aware Attention Fusion module with random weights
        self.eaf_featurefusion = EAF_featurefusion(input_channels)
        self._initialize_weights()

    def apply_color_transformation(self, image, color_correction_matrix):
        original_shape = image.shape
        flattened_image = image.reshape(-1,3)
        transformed_image = torch.tensordot(flattened_image, color_correction_matrix, dims=[[-1], [-1]])
        reshaped_image = transformed_image.view(original_shape)
        return torch.clamp(reshaped_image, 1e-8, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_image):
        # Generate Attention Map
        attention_map = self.attention_map_generator(input_image)
        attended_image = input_image.mul(attention_map)

        # Process through Local Enhancement Block
        local_mul, local_add = self.local_enhancement(attended_image)
        local_enhanced_output = input_image.mul(local_mul).add(local_add)

        # Process through Global Enhancement Block
        gamma, color_matrix = self.global_enhancement(local_enhanced_output)
        b = local_enhanced_output.shape[0]
        local_enhanced_output = local_enhanced_output.permute(0,2,3,1)

        global_enhanced_output = torch.stack(
            [self.apply_color_transformation(local_enhanced_output[i,:,:,:], color_matrix[i,:,:])**gamma[i,:] for i in range(b)], 
            dim=0)

        local_enhanced_output = local_enhanced_output.permute(0,3,1,2)
        global_enhanced_output = global_enhanced_output.permute(0,3,1,2)

        ## Apply Exposure Aware Attention Fusion
        global_enhanced_output = self.eaf_featurefusion(global_enhanced_output)
        global_enhanced_output = self.eaf_featurefusion(local_enhanced_output, global_enhanced_output) + input_image

        return local_mul, local_add, local_enhanced_output, global_enhanced_output, attention_map, attended_image

########################
# Check model summary
########################
if __name__ == "__main__":
    total_params = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = GuidedIAT(input_channels=3, transformer_blocks=3, embedding_dimension=48).to(device)

    from torchinfo import summary
    input_size = (4, 3, 400, 600)
    summary(model1, input_size)