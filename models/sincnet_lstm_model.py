
import torch
import os
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.rnn import LSTMEncoder
from models.networks.SincNet.sincnet import SincNetEncoder
from models.networks.fc import FcEncoder


class SincNetLSTMmodel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('--enc_hidden_size', type=int, default=16)
        parser.add_argument('--output_dim', type=int, default=4)
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--embd_method', type=str, default='maxpool')
        parser.add_argument('--bidirection', action='store_true')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.model_names = ['enc', 'rnn', 'C']
        self.netenc = SincNetEncoder()
        self.netrnn = LSTMEncoder(512, opt.hidden_size, embd_method='maxpool', bidirection=opt.bidirection)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        expand = 2 if opt.bidirection else 1
        self.netC = FcEncoder(opt.hidden_size * expand, cls_layers, dropout=0.3)
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.998)) # 0.999
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.signal = input['segments'].to(self.device)
        self.label = input['label'].to(self.device)
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        bs, seq_len, ndim = self.signal.size()
        self.signal = self.signal.view(-1, ndim)
        self.segments = self.netenc(self.signal)
        self.segments = self.segments.view(bs, seq_len, -1)
        self.feat = self.netrnn(self.segments)
        self.logits = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        loss = self.loss_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 

if __name__ == '__main__':
    model = SincNetEncoder()