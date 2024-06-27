import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('training_results.csv')
df1['Epoch'] = df1['Epoch'] + 90
df2 = pd.read_csv('training_results_fromscratch.csv')
fig, axs = plt.subplots(3, 1, figsize=(8, 15))

axs[0].plot(df1['Epoch'], df1['Training Loss'], label='Training Loss - Pretrain+Finetune')
axs[0].plot(df2['Epoch'], df2['Training Loss'], label='Training Loss - From scratch', linestyle='--')
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(df1['Epoch'], df1['Train Accuracy'], label='Train Accuracy - Pretrain+Finetune')
axs[1].plot(df2['Epoch'], df2['Train Accuracy'], label='Train Accuracy - From scratch', linestyle='--')
axs[1].set_title('Train Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].legend()

axs[2].plot(df1['Epoch'], df1['Validation Accuracy'], label='Validation Accuracy - Pretrain+Finetune')
axs[2].plot(df2['Epoch'], df2['Validation Accuracy'], label='Validation Accuracy - From scratch', linestyle='--')
axs[2].set_title('Validation Accuracy')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy (%)')
axs[2].legend()
plt.subplots_adjust(hspace=0.8)
plt.tight_layout()
plt.savefig('train_log.png')