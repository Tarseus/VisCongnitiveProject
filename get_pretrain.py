import timm
import torch
import torch.nn as nn

model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=True)
model.head = nn.Linear(model.head.in_features, 100)
torch.save(model.state_dict(), 'vit_small_patch16_224_in1k.pth')