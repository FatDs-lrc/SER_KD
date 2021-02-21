import torch
from torch import nn 

class WavEncoder(nn.Module):
    def __init__(self, hidden_size=16):
        self.hidden_size = hidden_size
        super(WavEncoder, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, self.hidden_size, 15, stride=5, padding=1600),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(self.hidden_size, self.hidden_size*2, 15, stride=6),
            nn.BatchNorm1d(self.hidden_size*2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(self.hidden_size*2, self.hidden_size*4, 15, stride=6),
            nn.BatchNorm1d(self.hidden_size*4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(self.hidden_size*4, self.hidden_size*2, 15, stride=6),
        )

    def forward(self, wav_data):
        # wav_data: one channle and 16k sample rate 
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)
