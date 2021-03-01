from typing import MutableSequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward=None, \
                            affine=False, affine_dim=None, embd_method='maxpool'):
        super().__init__()
        self.affine = affine
        assert embd_method in ['maxpool', 'meanpool', 'last']
        self.embd_method = embd_method
        if self.affine:
            _inp = affine_dim
            self.affine = nn.Linear(input_dim, affine_dim)
        else:
            _inp = input_dim
        if dim_feedforward is None:
            dim_feedforward = _inp
        encoder_layer = nn.TransformerEncoderLayer(d_model=_inp, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def post_process(self, x):
        if self.embd_method == 'maxpool':
            x = x.transpose(1, 2)                               # out.shape => [batch_size, ft_dim, seq_len]      
            embd = F.max_pool1d(x, x.size(2), x.size(2))        # out.shape => [batch_size, ft_dim, 1]   
            embd = embd.squeeze()                               # out.shape => [batch_size, ft_dim]
        elif self.embd_method == 'meanpool':
            embd = torch.mean(x, dim=1)
        elif self.embd_method == 'last':
            embd = x[:, -1, :]
        return embd

    def forward(self, x, mask=None, src_key_padding_mask=None):
        # switch batch to dim-1, inp.shape => [seq_len, batch_size, ft_dim]
        x = x.transpose(0, 1)
        if self.affine:
            x = self.affine(x)
        out = self.encoder(x, mask, src_key_padding_mask)
        # switch back to batch first, inp.shape => [batch_size, seq_len, ft_dim]
        out = out.transpose(0, 1)
        out = self.post_process(out)
        return out
      

if __name__ == '__main__':
    net = TransformerEncoder(256, 2, nhead=4, dim_feedforward=256)
    inp = torch.rand(2, 75, 256)
    out = net(inp)
    print(out.shape)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))

# 1. 确认是不是batch first
# 2. 确认一下初始化规则