import torch
import torch.nn as nn 
import torch.nn.functional as F

class EncCNN1d(nn.Module):
    def __init__(self, input_dim=130, channel=128, dropout=0.3):
        super(EncCNN1d, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(input_dim, channel, 10, stride=4, padding=4),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel, channel*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(channel*2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel*2, channel*4, 5, stride=2, padding=2),
            nn.BatchNorm1d(channel*4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1),
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, wav_data):
        # wav_data of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(wav_data.transpose(1, 2))
        out = out.transpose(1, 2)       # to (batch x seq x dim)
        out = self.dp(out)
        return out  

class EncCNN1dThin(nn.Module):
    def __init__(self, input_dim=130, channel=128, dropout=0.3):
        super(EncCNN1dThin, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(input_dim, channel, 10, stride=4, padding=4),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel, channel*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(channel*2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel*2, channel*2, 3, stride=2, padding=1),
            # nn.BatchNorm1d(channel*4),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1),
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, wav_data):
        # wav_data of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(wav_data.transpose(1, 2))
        out = out.transpose(1, 2)       # to (batch x seq x dim)
        out = self.dp(out)
        return out  

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=True):
        super().__init__()
        # self.conv_down = nn.Conv1d(in_channels, out_channels, kernel_size, \
        #     stride=2 if downsample else 1, padding=(kernel_size-1)//2)#, bias=False)
        self.conv_down = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.block1 = ResBlock(out_channels, out_channels, kernel_size)
        self.block2 = ResBlock(out_channels, out_channels, kernel_size)
    
    def forward(self, x):
        x = self.conv_down(x)
        x = self.bn(x)
        out = self.block1(x)
        out = self.block2(out)
        return out
        
class ResNetEncoder(nn.Module):
    def __init__(self, input_dim=130, channels=128):
        super().__init__()
        self.conv0 = nn.Conv1d(input_dim, channels, kernel_size=10, stride=2, bias=False)
        self.bn0 = nn.BatchNorm1d(channels)             # v2
        self.cnn_block1 = CNNBlock(channels, 2*channels, kernel_size=5, downsample=True)
        self.cnn_block2 = CNNBlock(2*channels, 4*channels, kernel_size=5, downsample=True)
        self.cnn_block3 = CNNBlock(4*channels, 4*channels, kernel_size=3, downsample=False)
        self.cnn_block4 = CNNBlock(4*channels, 2*channels, kernel_size=3, downsample=False)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = self.bn0(x)         # v2
        out = self.cnn_block1(x)
        out = self.cnn_block2(out)
        out = self.cnn_block3(out)
        out = self.cnn_block4(out)
        return out.transpose(1, 2)

if __name__ == '__main__':
    # x = torch.rand(2, 600, 130)
    # net = EncCNN1d()
    # out = net(x)
    # print(net)
    # print(out.size())
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Total number of parameters : %.3f M' % (num_params / 1e6))

    # x = torch.rand(2, 130, 600)
    # input_dim = 130
    # channel = 128
    # net1 = nn.Conv1d(input_dim, channel, 10, stride=2, padding=4)
    # net2 = nn.Conv1d(channel, channel*2, 5, stride=2, padding=2)
    # net3 = nn.Conv1d(channel*2, channel*4, 5, stride=2, padding=2)
    # net4 = nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1)
    # out1 = net1(x)
    # out2 = net2(out1)
    # out3 = net3(out2)
    # out4 = net4(out3)
    # print(x.shape)
    # print(out1.shape)
    # print(out2.shape)
    # print(out3.shape)
    # print(out4.shape)
    
    x = torch.rand(2, 600, 130)
    net = EncCNN1d(130, 256)
    out = net(x)
    print(net)
    print(out.size())
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters : %.3f M' % (num_params / 1e6))

    # from torchsummary import summary
    # net = ResNetEncoder(channels=64)
    # summary(net, (600, 130), device='cpu')
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print('Total number of parameters : %.3f M' % (num_params / 1e6))
