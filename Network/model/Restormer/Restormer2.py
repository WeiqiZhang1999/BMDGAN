#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


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
from .cust_checkpoint import cust_checkpoint


USE_MEMORY_EFFICIENT_SiLU = True

if USE_MEMORY_EFFICIENT_SiLU:
    @torch.jit.script
    def silu_fwd(x):
        return x.mul(torch.sigmoid(x))


    @torch.jit.script
    def silu_bwd(x, grad_output):
        x_sigmoid = torch.sigmoid(x)
        return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))


    class SiLUJitImplementation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return silu_fwd(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            return silu_bwd(x, grad_output)


    def silu(x, inplace=False):
        return SiLUJitImplementation.apply(x)

else:
    def silu(x, inplace=False):
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return silu(x, self.inplace)

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
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x

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


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
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


class TransformerBlockWithCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_checkpoint):
        super(TransformerBlockWithCrossAttn, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.norm0 = LayerNorm(dim, LayerNorm_type)
        self.norm0_context = LayerNorm(dim, LayerNorm_type)
        self.attn_cross = CrossAttention(dim, num_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, context):
        return cust_checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context):
        x = x + self.attn_cross(self.norm0(x), self.norm0_context(context))
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLayerWithCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, depth, ffn_expansion_factor, bias, LayerNorm_type, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlockWithCrossAttn(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)
            for _ in range(depth)])

    def forward(self, x, context_list):
        for blk in self.blocks:
            x = blk(x, context_list.pop())
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


class TransformerLayer_CollectOut(nn.Module):
    def __init__(self, dim, num_heads, depth, ffn_expansion_factor, bias, LayerNorm_type, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)
            for _ in range(depth)])

    def forward(self, x):
        outs = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            outs.append(x)
        return outs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, bias=False, pre_norm=False):
        super(OverlapPatchEmbed, self).__init__()

        self.norm_in = nn.Identity()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1,
                              bias=False if pre_norm else bias, padding_mode="reflect")

        if pre_norm:
            self.norm_in = LayerNorm(in_c, "WithBias")

    def forward(self, x):
        _, _, h, w = x.shape
        x = to_3d(x)
        x = self.norm_in(x)
        x = to_4d(x, h, w)
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, n_feat_out=None, bias=False):
        super(Downsample, self).__init__()
        if n_feat_out is None:
            n_feat_out = n_feat * 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out // 4, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, n_feat_out=None, bias=False):
        super(Upsample, self).__init__()
        if n_feat_out is None:
            n_feat_out = n_feat // 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out * 4, kernel_size=3, stride=1, padding=1, bias=bias),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer2(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],  # 2 3 3 4
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=True,
                 use_checkpoint=False,
                 pre_norm_patch=False
                 ):
        super().__init__()
        LayerNorm_type = 'WithBias'

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, pre_norm=pre_norm_patch)

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
        self.latent = TransformerLayer(dim=dim * 8, num_heads=heads[3], depth=num_blocks[3], bias=bias,
                                       use_checkpoint=use_checkpoint, ffn_expansion_factor=ffn_expansion_factor,
                                       LayerNorm_type=LayerNorm_type)

        self.up4_3 = Upsample(int(dim * 8))  ## From Level 4 to Level 3
        self.decoder_level3 = TransformerLayer(dim=dim * 8, num_heads=heads[2], depth=num_blocks[2],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type,
                                               use_checkpoint=use_checkpoint)

        self.up3_2 = Upsample(dim * 8, dim * 2)  ## From Level 3 to Level 2
        self.decoder_level2 = TransformerLayer(dim=dim * 4, num_heads=heads[1], depth=num_blocks[1],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type,
                                               use_checkpoint=use_checkpoint)

        self.up2_1 = Upsample(dim * 4, dim)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = TransformerLayer(dim=dim * 2, num_heads=heads[0], depth=num_blocks[0],
                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                               LayerNorm_type=LayerNorm_type,
                                               use_checkpoint=use_checkpoint)

        self.refinement = TransformerLayer(dim=dim * 2, num_heads=heads[0], depth=num_refinement_blocks,
                                           ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                           LayerNorm_type=LayerNorm_type, use_checkpoint=use_checkpoint)

        self.output = nn.Conv2d(dim * 2, out_channels,
                                kernel_size=3, stride=1, padding=1, bias=bias, padding_mode="reflect")

        self.skip1_2 = Downsample(dim, bias=bias)
        self.skip1_3 = nn.Sequential(Downsample(dim, bias=bias), Downsample(dim * 2, bias=bias))

        self.skip2_1 = Upsample(dim * 2, bias=bias)
        self.skip2_3 = Downsample(dim * 2, bias=bias)

        self.skip3_1 = nn.Sequential(Upsample(dim * 4, bias=bias), Upsample(dim * 2, bias=bias))
        self.skip3_2 = Upsample(dim * 4, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = inp_enc_level4
        latent = self.latent(latent)

        level3_token = self.skip1_3(out_enc_level1) + self.skip2_3(out_enc_level2) + out_enc_level3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat((inp_dec_level3, level3_token), dim=1)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        level2_token = self.skip1_2(out_enc_level1) + out_enc_level2 + self.skip3_2(out_enc_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat((inp_dec_level2, level2_token), dim=1)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        level1_token = out_enc_level1 + self.skip2_1(out_enc_level2) + self.skip3_1(out_enc_level3)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat((inp_dec_level1, level1_token), dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1
