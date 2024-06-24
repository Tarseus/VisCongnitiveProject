import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from itertools import repeat
import collections.abc
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))

class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size = 224,
                 patch_size = 16,
                 channels = 3,
                 embedding_dim = 768):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(image_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.patch_count = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(channels, embedding_dim, kernel_size=patch_size, stride=patch_size, bias=True)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features):  
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(True)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 heads):
        super().__init__()
        mlp_ratio = 4.
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim,num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CustomVisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 channels=3,
                 class_count=100,
                 embedding_size=768,
                 transformer_depth=12,
                 attention_heads=12):
        super(CustomVisionTransformer, self).__init__()
        self.class_count = class_count
        self.embedding_size = embedding_size
        self.prefix_token_count = 1
        self.patch_embedding = PatchEmbedding(image_size=image_size, 
                                              patch_size=patch_size, 
                                              channels=channels,
                                              embedding_dim=embedding_size)
        patch_count = self.patch_embedding.patch_count
        self.classification_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        embedding_length = patch_count + self.prefix_token_count
        self.position_embedding = nn.Parameter(torch.randn(1, embedding_length, embedding_size) * 0.02)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(dim=embedding_size, heads=attention_heads) for _ in range(transformer_depth)])
        self.normalization = nn.LayerNorm(embedding_size)
        self.output_head = nn.Linear(embedding_size, class_count)
        self.initialize_weights()

    def initialize_weights(self):
        def custom_trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
            with torch.no_grad():
                tensor = tensor.normal_().fmod_(2).mul_(std).add_(mean)
                tensor = tensor.clamp_(min=a, max=b)
            return tensor
        custom_trunc_normal(self.position_embedding, std=0.02)
        nn.init.normal_(self.classification_token, std=1e-6)

    def apply_position_embedding(self, x):
        position_embedding = self.position_embedding
        to_concatenate = [self.classification_token.expand(x.size(0), -1, -1), x]
        x = torch.cat(to_concatenate, dim=1)
        x += position_embedding
        return x

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.apply_position_embedding(x)
        x = self.transformer_blocks(x)
        x = self.normalization(x)
        x = x[:, 0]
        x = self.output_head(x)
        return x
    
def vit_small_patch16_224():
    model = CustomVisionTransformer(patch_size=16,
                              embedding_size=384,
                              transformer_depth=12,
                              attention_heads=6)
    return model

def vit_base_patch16_224():
    model = CustomVisionTransformer(patch_size=16,
                              embedding_size=768,
                              transformer_depth=12,
                              attention_heads=12)
    return model