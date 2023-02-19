## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat
import random


##########################################################################
## Layer Norm

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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, v = qkv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = q

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class AttentionCross(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionCross, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.to_q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.to_q_dwconv = nn.Conv2d(dim, dim, 3, 1, padding=1, groups=dim)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, context):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q = self.to_q_dwconv(self.to_q(context))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super(LeFF, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = round(math.sqrt(hw * 2))
        ww = round(math.sqrt(hw // 2))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear2(x)

        return x


class StoTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., proj_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=1, token_mlp='leff'):
        super(StoTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.stride = stride
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size)  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size)  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size - 1
        relative_coords[:, :, 0] *= 2 * self.win_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

        self.norm1 = norm_layer(dim)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = nn.Identity()

        self.proj = nn.Linear(dim, dim)
        self.se_layer = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def attention(self, q, k, v, attn_mask=None):
        B_, h, N_, C_ = q.shape

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]  # [nW, N_, N_]
            mask = repeat(attn_mask, 'nW m n -> nW m (n d)', d=1)  # [nW, N_, N_]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_, N_ * 1) + mask.unsqueeze(1).unsqueeze(
                0)  # [1, nW, 1, N_, N_]
            # [B, nW, nh, N_, N_]
            attn = attn.view(-1, self.num_heads, N_, N_ * 1)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        y = (attn @ v).transpose(1, 2).reshape(B_, N_, h * C_)
        y = self.proj(y)
        return y

    def forward(self, x, mask=None):
        B, L, C = x.shape

        H = round(math.sqrt(L * 2))
        W = round(math.sqrt(L // 2))

        shortcut = x
        x = self.norm1(x)
        q = self.to_q(x)  # [B, L, C]
        v = self.to_v(x)

        q = rearrange(q, 'b (h w) c -> b h w c', h=H)
        v = rearrange(v, 'b (h w) c -> b h w c', h=H)
        k = q
        kv = torch.cat((k, v), dim=-1)

        x = x.view(B, H, W, C)

        if self.training:
            if mask != None:
                input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
                input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
                attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
                attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            ## Stochastic shift window
            H_offset = random.randint(0, self.win_size - 1)
            W_offset = random.randint(0, self.win_size - 1)

            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)

            if H_offset > 0:
                h_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -H_offset),
                            slice(-H_offset, None))
            else:
                h_slices = (slice(0, None),)
            if W_offset > 0:
                w_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -W_offset),
                            slice(-W_offset, None))
            else:
                w_slices = (slice(0, None),)

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask  # [nW, N_,N_]

            # cyclic shift
            shifted_q = torch.roll(q, shifts=(-H_offset, -W_offset), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-H_offset, -W_offset), dims=(1, 2))

            # partition windows
            q_windows = window_partition(shifted_q, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
            q_windows = q_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
            B_, N_, C_ = q_windows.shape
            q_windows = q_windows.reshape(B_, N_, self.num_heads, C_ // self.num_heads).permute(0, 2, 1, 3)

            kv_windows = window_partition(shifted_kv, self.win_size)  # nW*B, win_size, win_size, 2C
            kv_windows = kv_windows.view(-1, self.win_size * self.win_size, 2 * C)
            kv_windows = kv_windows.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
            k_windows, v_windows = kv_windows[0], kv_windows[1]

            attn_windows = self.attention(q_windows, k_windows, v_windows, attn_mask)

            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

            x = torch.roll(x, shifts=(H_offset, W_offset), dims=(1, 2))

            x = x.view(B, H * W, C)
            del attn_mask

        else:
            avg = torch.zeros((B, H * W, C)).cuda()
            NUM = 0
            for H_offset in range(self.win_size):
                for W_offset in range(self.win_size):
                    if mask != None:
                        input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
                        input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
                        attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
                        attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(
                            1)  # nW, win_size*win_size, win_size*win_size
                        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                                     float(0.0))
                    else:
                        attn_mask = None
                    shift_mask = torch.zeros((1, H, W, 1)).type_as(x)

                    if H_offset > 0:
                        h_slices = (slice(0, -self.win_size),
                                    slice(-self.win_size, -H_offset),
                                    slice(-H_offset, None))
                    else:
                        h_slices = (slice(0, None),)

                    if W_offset > 0:
                        w_slices = (slice(0, -self.win_size),
                                    slice(-self.win_size, -W_offset),
                                    slice(-W_offset, None))
                    else:
                        w_slices = (slice(0, None),)

                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            shift_mask[:, h, w, :] = cnt
                            cnt += 1

                    shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
                    shift_mask_windows = shift_mask_windows.view(-1,
                                                                 self.win_size * self.win_size)  # nW, win_size*win_size
                    shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                        2)  # nW, win_size*win_size, win_size*win_size
                    shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                        shift_attn_mask == 0, float(0.0))
                    attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask  # [nW, N_,N_]

                    shifted_q = torch.roll(q, shifts=(-H_offset, -W_offset), dims=(1, 2))
                    shifted_kv = torch.roll(kv, shifts=(-H_offset, -W_offset), dims=(1, 2))

                    # partition windows
                    q_windows = window_partition(shifted_q, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
                    q_windows = q_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
                    B_, N_, C_ = q_windows.shape
                    q_windows = q_windows.reshape(B_, N_, self.num_heads, C_ // self.num_heads).permute(0, 2, 1, 3)

                    kv_windows = window_partition(shifted_kv, self.win_size)  # nW*B, win_size, win_size, 2C
                    kv_windows = kv_windows.view(-1, self.win_size * self.win_size, 2 * C)
                    kv_windows = kv_windows.reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1,
                                                                                                             4)
                    k_windows, v_windows = kv_windows[0], kv_windows[1]

                    attn_windows = self.attention(q_windows, k_windows, v_windows, attn_mask)

                    attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
                    shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
                    # reverse cyclic shift
                    y = torch.roll(shifted_x, shifts=(H_offset, W_offset), dims=(1, 2))

                    y = y.view(B, H * W, C)
                    avg = NUM / (NUM + 1) * avg + y / (NUM + 1)
                    NUM += 1
                    del attn_mask
            x = avg
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicStoformerLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_mlp='leff'):
        super(BasicStoformerLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            StoTransformerBlock(dim=dim,
                                num_heads=num_heads, win_size=win_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                norm_layer=norm_layer, token_mlp=token_mlp)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x


# class EfficientAttention(nn.Module):
#
#     def __init__(self, dim, num_heads, bias):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.qkv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, input_):
#         n, _, h, w = input_.size()
#
#         qkv = self.qkv_dwconv(self.qkv(input_))
#         queries, values = qkv.chunk(2, dim=1)
#
#         queries = rearrange(queries, 'b c h w -> b c (h w)')
#         values = rearrange(values, 'b c h w -> b c (h w)')
#         keys = queries
#         # keys = self.keys(input_).reshape((n, self.key_channels, h * w))
#         # queries = self.queries(input_).reshape(n, self.key_channels, h * w)
#         # values = self.values(input_).reshape((n, self.value_channels, h * w))
#         head_key_channels = self.dim // self.num_heads
#         head_value_channels = self.dim // self.num_heads
#
#         attended_values = []
#         for i in range(self.num_heads):
#             key = F.softmax(keys[
#                             :,
#                             i * head_key_channels: (i + 1) * head_key_channels,
#                             :
#                             ], dim=2)
#             query = F.softmax(queries[
#                               :,
#                               i * head_key_channels: (i + 1) * head_key_channels,
#                               :
#                               ], dim=1)
#             value = values[
#                     :,
#                     i * head_value_channels: (i + 1) * head_value_channels,
#                     :
#                     ]
#             context = key @ value.transpose(1, 2)
#             attended_value = (
#                     context.transpose(1, 2) @ query
#             ).reshape(n, head_value_channels, h, w)
#             attended_values.append(attended_value)
#
#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.project_out(aggregated_values)
#         return reprojected_value
#
#
# class EfficientAttention2(nn.Module):
#
#     def __init__(self, dim, num_heads, bias):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#
#         self.to_q = nn.Conv2d(dim, dim, 1, bias=bias)
#         self.to_q_dwconv = nn.Conv2d(dim, dim, 3, 1, padding=1, groups=dim)
#
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, context):
#         b, c, h, w = x.shape
#
#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#
#         q = self.to_q_dwconv(self.to_q(context))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = q.softmax(dim=2)
#         k = k.softmax(dim=3)
#
#         attn = (k @ v.transpose(2, 3))
#
#         out = (attn.transpose(2, 3) @ q)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
#
#         out = self.project_out(out)
#         return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, depth, ffn_expansion_factor, bias, LayerNorm_type, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)
            for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="reflect")

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class NewRestormer(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],  # 2 3 3 4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=True,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 use_checkpoint=False,

                 ):
        super(NewRestormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.encoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor / 2., bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1 = TransformerLayer(dim=dim, num_heads=heads[0], depth=num_blocks[0],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerLayer(dim=dim * 2, num_heads=heads[1], depth=num_blocks[1],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerLayer(dim=dim * 4, num_heads=heads[2], depth=num_blocks[2],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.down3_4 = Downsample(int(dim * 4))  ## From Level 3 to Level 4
        self.latent = BasicStoformerLayer(dim=dim * 8, num_heads=heads[3], depth=num_blocks[3], qkv_bias=bias,
                                          use_checkpoint=use_checkpoint)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level3 = AttentionCross(dim * 2 ** 2, heads[2], bias=bias)
        self.decoder_level3 = TransformerLayer(dim=dim * 4, num_heads=heads[2], depth=num_blocks[2],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        # self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2 = AttentionCross(dim * 2 ** 1, heads[1], bias=bias)
        self.decoder_level2 = TransformerLayer(dim=dim * 2, num_heads=heads[1], depth=num_blocks[1],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = TransformerLayer(dim=dim * 2, num_heads=heads[0], depth=num_blocks[0],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.refinement = TransformerLayer(dim=dim * 2, num_heads=heads[0], depth=num_refinement_blocks,
                                           ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                           LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        # #### For Dual-Pixel Defocus Deblurring Task ####
        # self.dual_pixel_task = dual_pixel_task
        # if self.dual_pixel_task:
        #     self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels,
                                kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="reflect")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        _, _, H, W = inp_enc_level4.shape
        latent = rearrange(inp_enc_level4, "b c h w -> b (h w) c")
        latent = self.latent(latent)
        latent = rearrange(latent, "b (h w) c -> b c h w", h=H, w=W)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3, out_enc_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1


