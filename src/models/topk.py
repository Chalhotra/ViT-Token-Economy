import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, Mlp
from timm.models.layers import DropPath

class Attention_TopK(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        self.init_n = 14 * 14 

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.keep_rate < 1:
            left_tokens = int(self.keep_rate * self.init_n)
            if left_tokens >= N - 1:
                return x, None, None

            cls_attn = attn[:, :, 0, 1:].mean(dim=1) 
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
            index = idx.unsqueeze(-1).expand(-1, -1, C)
            return x, index, idx

        return x, None, None

class Block_TopK(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_TopK(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, 
                                   proj_drop=drop, keep_rate=keep_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        tmp, index, idx = self.attn(self.norm1(x))
        x = x + self.drop_path(tmp)

        if index is not None:
            x = torch.cat([x[:, 0:1], torch.gather(x[:, 1:], dim=1, index=index)], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, idx

class TopKVisionTransformer(VisionTransformer):
    def __init__(self, pruning_locs, keep_rates, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_locs = pruning_locs
        self.keep_rates = keep_rates
        
        block_keep_rates = [1.0] * len(self.blocks)
        for i, loc in enumerate(pruning_locs):
            if loc < len(block_keep_rates):
                block_keep_rates[loc] = keep_rates[i]

        old_blocks = self.blocks
        self.blocks = nn.ModuleList()
        for i, block in enumerate(old_blocks):
            dim = block.norm1.normalized_shape[0]
            new_block = Block_TopK(
                dim=dim, num_heads=block.attn.num_heads, mlp_ratio=block.mlp.fc1.out_features / dim,
                qkv_bias=block.attn.qkv.bias is not None, keep_rate=block_keep_rates[i],
                norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
            )
            self.blocks.append(new_block)
        del old_blocks

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x