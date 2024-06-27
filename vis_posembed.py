import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch
from tqdm import tqdm
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def visualize_cosine_similarity(pos_embed):
    cos_sim_matrix = np.zeros((14,14,14,14))
    for i in tqdm(range(14), desc='Calculating cosine similarities'):
        for j in range(14):
            for k in range(14):
                for l in range(14):
                    cos_sim_matrix[i, j, k, l] = cosine_sim(pos_embed[i, j], pos_embed[k, l])
    fig, axes = plt.subplots(14, 14, figsize=(10, 9))
    fig.suptitle('Position embedding similarity', fontsize=20)
    for i in tqdm(range(14), desc='Visualizing cosine similarities'):
        for j in range(14):
            sns.heatmap(cos_sim_matrix[i, j], ax=axes[i, j], cmap='viridis', cbar=False)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_aspect('equal')
    cbar_ax = fig.add_axes([0.88, -0.5, 0.05, 2])
    sns.heatmap(np.array([[-1,1]]), cmap='viridis', cbar=False, ax=cbar_ax, cbar_kws={'label': 'Cosine similarity'}, vmin=-1, vmax=1)
    # cbar_ax.set_visible(False)
    fig.text(0.5, 0.04, 'Input patch column', ha='center', va='center', fontsize=16)
    fig.text(0.04, 0.5, 'Input patch row', ha='center', va='center', rotation='vertical', fontsize=16)
    for ax, col in zip(axes[-1], range(1, 15)):
        ax.set_xlabel(col)
    for ax, row in zip(axes[:,0], range(1, 15)):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=10)
    plt.subplots_adjust(right=0.9, wspace=0.05, hspace=0.05)
    plt.show()

if __name__ == '__main__':
    model = torch.load('vit_small_patch16_224_cifar100_pretrain.pth', map_location=torch.device('cpu'))
    pos_embed = model['position_embedding'].squeeze()[1:].cpu().numpy().reshape(14, 14, 384)
    visualize_cosine_similarity(pos_embed)