# import torch
# import torch.nn as nn

# encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256)
# net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

# inp = torch.rand(2, 75, 256)
# out = net(inp)
# print(out.shape)

from torchvision.models import resnet18

net = resnet18()
print(net)