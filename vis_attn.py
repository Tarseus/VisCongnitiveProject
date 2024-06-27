import torch 
from torchvision import datasets
from torchvision.transforms import transforms
import csv
import torch.nn as nn
import vision_transformer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_attention(image, attn_weights, num_heads=6, img_size=224, patch_size=16):
    attn_weights = attn_weights.mean(axis=0)
    attn_weights = attn_weights[1:, 1:]
    num_patches_side = img_size // patch_size
    attn_map = np.zeros((img_size, img_size))
    for i in range(num_patches_side):
        for j in range(num_patches_side):
            patch_index = i * num_patches_side + j
            patch_attn = attn_weights[patch_index].reshape(num_patches_side, num_patches_side)
            patch_attn_avg = np.mean(patch_attn)
            attn_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] += patch_attn_avg
    image = image.permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    alpha_map = 1 - attn_map
    alpha_map = alpha_map * 0.5
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.detach().cpu().numpy())
    ax[0].set_title("Original Image")
    ax[1].imshow(image.detach().cpu().numpy())
    x, y = np.meshgrid(np.arange(attn_map.shape[1]), np.arange(attn_map.shape[0]))
    x = x.flatten()
    y = y.flatten()
    alpha_map_flattened = alpha_map.flatten()
    ax[1].scatter(x, y, c='black', alpha=alpha_map_flattened)
    ax[1].set_title("Attention Map")
    plt.show()
    
if __name__ == '__main__':
    model = vision_transformer.vit_small_patch16_224()
    model.load_state_dict(torch.load('vit_small_patch16_224_cifar100_pretrain.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs, attn_weight = model(images)
        print(attn_weight.shape)
        visualize_attention(images[0],
                            attn_weight[0].detach().cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        break