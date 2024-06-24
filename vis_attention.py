import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = torch.load('vit_small_patch16_224_cifar100_pretrain.pth', map_location='cpu')
    attention_weights = model.attention.qkv.weight.data