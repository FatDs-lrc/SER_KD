
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).float()


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class _TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers, 
        migrate from nn.TransformerEncoder, add output of each layer

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        hidden_states = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            hidden_states.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output, hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, nhead, dim_feedforward=None, \
                            affine=False, affine_dim=None, embd_method='maxpool'):
        super().__init__()
        self.affine = affine
        self.max_position_embeddings = 512
        assert embd_method in ['maxpool', 'meanpool', 'last']
        self.embd_method = embd_method
        if self.affine:
            _inp = affine_dim
            self.affine = nn.Linear(input_dim, affine_dim)
        else:
            _inp = input_dim
        if dim_feedforward is None:
            dim_feedforward = _inp
        
        # self.position_embeddings = nn.Embedding(self.max_position_embeddings, input_dim)
        self.position_embeddings = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(self.max_position_embeddings, input_dim, padding_idx=0),
                freeze=True
            )
        encoder_layer = nn.TransformerEncoderLayer(d_model=_inp, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = _TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers) # => 和 nn.TransformerEncoder没区别，多返回一个hidden states
        # self.linear = nn.Linear(_inp, _inp)
        # self.tanh = nn.Tanh()

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
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1)
        if self.affine:
            x = self.affine(x)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand([seq_len, batch_size])
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings * 0.01
        out, hidden_states = self.encoder(x, mask, src_key_padding_mask)
        # switch back to batch first, inp.shape => [batch_size, seq_len, ft_dim]
        out = out.transpose(0, 1)
        out = self.post_process(out)
        # out = self.tanh(self.linear(out))
        return out, hidden_states


class AlignNet(nn.Module):
    def __init__(self, template_dim, align_ft_dim, num_heads=4):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(align_ft_dim, template_dim),
            nn.Tanh()
        )
        self.query_affine = nn.Linear(template_dim, template_dim)
        self.template_dim = template_dim
        self.num_heads = num_heads
        self.dp = nn.Dropout(0.1)

        # self.multihead_attn = \
        #     nn.modules.activation.MultiheadAttention(template_dim, num_heads=num_heads, bias=False)
    
    def forward(self, template_ft, align_ft):
        template_ft = template_ft.transpose(0, 1)
        # print(template_ft.shape)
        # print(align_ft.shape)
        # input()
        to_align = self.affine(align_ft)
        scaling = float(self.template_dim) ** -0.5
        q = self.query_affine(template_ft)
        k = to_align
        tgt_len, bsz, embed_dim = q.size()
        assert embed_dim == self.template_dim
        head_dim = embed_dim // self.num_heads
        q = q * scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dp(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, k)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # out, attn_weight = self.multihead_attn(template_ft, to_align, to_align)
        return attn_output.transpose(0, 1)

'''
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
'''
if __name__ == '__main__':
    # net = TransformerEncoder(256, 2, nhead=4, dim_feedforward=256)
    # inp = torch.rand(2, 75, 256)
    # out, hidden_states = net(inp)
    # print(out.shape)
    # print(len(hidden_states))
    # for h in hidden_states:
    #     print(h.size())
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Total number of parameters : %.3f M' % (num_params / 1e6))

    # 1. 确认是不是batch first
    # 2. 确认一下初始化规则

    L = torch.rand(20, 2, 768)
    A = torch.rand(15, 2, 512)
    attn = AlignNet(template_dim=768, align_ft_dim=512, num_heads=4)
    out = attn(L, A)
    print(out.shape)
    # attn = nn.modules.activation.MultiheadAttention(768, num_heads=4)
    num_params = 0
    for param in attn.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))