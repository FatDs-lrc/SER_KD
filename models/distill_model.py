
import torch
import os
import json
import torch.nn.functional as F

from models.base_model import BaseModel
from models.networks.rcn import EncCNN1d
from models.networks.fc import FcEncoder
from models.networks.classifier import BertClassifier
from models.networks.transformer import TransformerEncoder
from models.networks import tools
from models.utils.config import OptConfig


class DistillModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # all
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        # A
        parser.add_argument('--input_dim', type=int, default=130)
        parser.add_argument('--enc_channel', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=4)
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--nhead', type=int, default=4)
        parser.add_argument('--dim_feedforward', type=int, default=256)
        # V
        parser.add_argument('--bert_type', type=str, help='how many label types in this dataset')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--pretrained_path', type=str, help='where to load finetuned bert model')
        # optimizer
        parser.add_argument('--temperature', type=float, default=2.0, help='Teacher softmax temperature')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of CE loss')
        parser.add_argument('--kd_weight', type=float, default=1.0, help='weight of KD loss')
        parser.add_argument('--mse_weight', type=float, default=0.5, help='weight of KD loss')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'KD', 'mse']
        self.model_names = ['enc', 'rnn', 'L', 'C']
        self.pretrained_model = ['L']
        # net A
        self.netenc = EncCNN1d(opt.input_dim, opt.enc_channel)
        self.netrnn = TransformerEncoder(opt.enc_channel*2, opt.num_layers, opt.nhead, opt.dim_feedforward)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        # net L
        self.netL = BertClassifier.from_pretrained(opt.pretrained_path, \
                    num_classes=opt.output_dim, embd_method=opt.embd_method)
        # net C
        self.netC = FcEncoder(opt.enc_channel*2, cls_layers, dropout=0.3)
        
        self.cvNo = opt.cvNo
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_kd = torch.nn.KLDivLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # record lamda
            self.ce_weight = opt.ce_weight
            self.kd_weight = opt.kd_weight
            self.mse_weight = opt.mse_weight
            self.temperature = opt.temperature
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.signal = input['A_feat'].to(self.device)
        self.token_ids, self.attention_mask, self.label = input.values()
        self.token_ids = self.token_ids.to(self.device)
        self.attention_mask = self.attention_mask.to(self.device)
        self.label = self.label.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Teacher
        self.L_logits, self.hidden_states = self.netBC(self.token_ids, self.attention_mask)
        self.L_feat = torch.max(self.hidden_states, dim=1)[0]
        self.L_pred = F.softmax(self.L_logits, dim=-1)
        
        # Student
        self.segments = self.netenc(self.signal)
        self.feat = self.netrnn(self.segments)
        self.A_logits = self.netC(self.feat)
        self.A_pred = F.softmax(self.A_logits, dim=-1)
        self.A_log_pred = F.log_softmax(self.logits, dim=-1)
        self.pred = self.A_pred
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_KD = self.criterion_kd(self.A_log_pred, self.L_pred) * self.kd_weight
        self.loss_mse = self.criterion_mse(self.A_feat, self.L_feat) * self.mse_weight
        loss = self.loss_KD + self.loss_mse
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
