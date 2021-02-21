
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.cnn_lstm import Encoder
from models.networks.fc import FcEncoder


class VggLSTMAudioModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim', type=int, default=120)
        parser.add_argument('--cls_layers', type=str, default='512,256')
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--rnn_layers', type=int, default=1)
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.model_names = ['A', 'C']
        self.netA = Encoder(opt.input_dim, 'vgg', 'LSTM', \
            bidirection=True, 
            dim=[opt.hidden_size,] * opt.rnn_layers, 
            dropout=[0,] * opt.rnn_layers, 
            layer_norm=[False,] * opt.rnn_layers,
            proj=[True,] * opt.rnn_layers,
            sample_rate=[1,] * opt.rnn_layers,
            sample_style="drop")
        self.netA = self.netA.float()
        
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        self.netC = FcEncoder(opt.hidden_size * 2, cls_layers, dropout=0.3)
            
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
        self.A_feat = input['A_feat'].float().to(self.device)
        self.label = input['label'].long().to(self.device)
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.feat = self.netA(self.A_feat)
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
