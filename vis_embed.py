import matplotlib.pyplot as plt
import numpy as np
import torch
import vision_transformer
from sklearn.decomposition import PCA

def transform_dict(pre_train_dict):
    new_dict = {}
    for key in pre_train_dict.keys():
        # print(key)
        if key.startswith('cls_token'):
            new_key = key.replace('cls_token', 'classification_token')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('pos_embed'):
            new_key = key.replace('pos_embed', 'position_embedding')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('patch_embed'):
            new_key = key.replace('patch_embed', 'patch_embedding')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('blocks'):
            new_key = key.replace('blocks', 'transformer_blocks')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('norm'):
            new_key = key.replace('norm', 'normalization')
            new_dict[new_key] = pre_train_dict[key]
        elif key.startswith('head'):
            new_key = key.replace('head', 'output_head')
            new_dict[new_key] = pre_train_dict[key]
    return new_dict

def adjust_contrast(image, factor=1.5):
    mean = np.mean(image)
    return np.clip((1 + factor) * (image - mean) + mean, 0, 1)

def visualize_filters(model):
    conv_weights = model.patch_embedding.proj.weight.data.cpu().numpy()
    conv_weights = (conv_weights - conv_weights.min()) / (conv_weights.max() - conv_weights.min())
    conv_weights_flat = conv_weights.reshape(conv_weights.shape[0], -1)
    pca = PCA(n_components=28)
    pca_result = pca.fit_transform(conv_weights_flat.T)
    pca_result_reshaped = pca_result.T.reshape((28, 3, 16, 16))
    filter_count_per_row = 7
    rows = 4
    fig, axes = plt.subplots(rows, filter_count_per_row, figsize=(filter_count_per_row * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < 28:
            adjusted_image = adjust_contrast(pca_result_reshaped[i].transpose((1, 2, 0)))
            ax.imshow(adjusted_image, cmap='gray')
        ax.axis('off')
    print('Visualizing filters...')
    plt.show()

if __name__ == '__main__':
    model = vision_transformer.vit_small_patch16_224()
    pre_train_dict = torch.load('vit_small_patch16_224_cifar100_pretrain.pth')
    model.load_state_dict(pre_train_dict)
    visualize_filters(model)