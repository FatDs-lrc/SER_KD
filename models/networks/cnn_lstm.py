from torch import nn
import torch
import torch.nn.functional as F

class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf
    '''
    def __init__(self, input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.init_dim, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(self.init_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2) # Half-time dimension
        )

    def check_dim(self, input_dim):
        # Check input dimension, delta feature should be stack over channel.
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim/13), 13, (13//4)*self.hide_dim
        elif input_dim % 40 == 0:
            # Fbank feature
            return int(input_dim/40), 40, (40//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got :', input_dim)

    def view_input(self, feature, feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)

        return feature, feat_len

    def forward(self, feature, feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature, feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1], self.out_dim)
        return feature, feat_len

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling
    
    '''

    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.out_dim = sample_rate * \
            rnn_out_dim if sample_rate > 1 and sample_style == 'concat' else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj

        if self.sample_style not in ['drop', 'concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size, timestep, feature_dim = output.shape
            x_len = x_len//self.sample_rate

            if self.sample_style == 'drop':
                # Drop the unselected timesteps
                output = output[:, ::self.sample_rate, :].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep % self.sample_rate != 0:
                    output = output[:, :-(timestep % self.sample_rate), :]
                output = output.contiguous().view(batch_size, int(
                    timestep/self.sample_rate), feature_dim*self.sample_rate)
        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len

class Encoder(nn.Module):
    ''' Encoder (a.k.a. Listener in LAS)
        Encodes acoustic feature to latent representation, see config file for more details.'''

    def __init__(self, input_size, prenet, module, bidirection, dim, dropout, layer_norm, proj, sample_rate, sample_style):
        super(Encoder, self).__init__()

        # Hyper-parameters checking
        self.vgg = prenet == 'vgg'
        self.cnn = prenet == 'cnn'
        self.sample_rate = 1
        assert len(sample_rate) == len(dropout), 'Number of layer mismatch'
        assert len(dropout) == len(dim), 'Number of layer mismatch'
        num_layers = len(dim)
        assert num_layers >= 1, 'Encoder should have at least 1 layer'

        # Construct model
        module_list = []
        input_dim = input_size

        # Prenet on audio feature
        if self.vgg:
            vgg_extractor = VGGExtractor(input_size)
            module_list.append(vgg_extractor)
            input_dim = vgg_extractor.out_dim
            self.sample_rate = self.sample_rate*4
        if self.cnn:
            cnn_extractor = CNNExtractor(input_size, out_dim=dim[0])
            module_list.append(cnn_extractor)
            input_dim = cnn_extractor.out_dim
            self.sample_rate = self.sample_rate*4

        # Recurrent encoder
        if module in ['LSTM', 'GRU']:
            for l in range(num_layers):
                module_list.append(RNNLayer(input_dim, module, dim[l], bidirection, dropout[l], layer_norm[l],
                                            sample_rate[l], sample_style, proj[l]))
                input_dim = module_list[-1].out_dim
                self.sample_rate = self.sample_rate*sample_rate[l]
        else:
            raise NotImplementedError

        # Build model
        self.in_dim = input_size
        self.out_dim = input_dim
        self.layers = nn.ModuleList(module_list)

    def forward(self, input_x, enc_len=600):
        for _, layer in enumerate(self.layers):
            input_x, enc_len = layer(input_x, enc_len)
        input_x = input_x.transpose(1,2)
        input_x = F.max_pool1d(input_x, input_x.size(2), input_x.size(2)).squeeze()
        return input_x

if __name__ == '__main__':
    input_feat = torch.ones(3, 600, 120)
    # vgg_model = VGGExtractor(120)
    # out, _len = vgg_model(input_feat, 600)
    # print(out.shape, _len)
    encoder = Encoder(120, 'vgg', 'LSTM', \
        bidirection=True, 
        dim=[512,512,512,512,512], 
        dropout=[0,0,0,0,0], 
        layer_norm=[False,False,False,False,False],
        proj=[True,True,True,True,True],
        sample_rate=[1,1,1,1,1],
        sample_style="drop")
    out = encoder(input_feat, 600)
    print(out.shape)
    