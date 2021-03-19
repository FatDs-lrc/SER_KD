
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.transformer import TransformerEncoder
from models.networks.rcn import EncCNN1d
from models.networks.classifier import FcClassifier

class MovieRawModel(BaseModel):
    '''
    A: DNN
    V: denseface + LSTM + maxpool
    L: bert + textcnn
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # to be compitate with train_no_test
        parser.add_argument('--cvNo', type=int, default=0)
        parser.add_argument('--no_test', action='store_true')
        parser.add_argument('--no_val', action='store_true')
        # parameters
        parser.add_argument('--input_dim', type=int, default=130)
        parser.add_argument('--enc_channel', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=6)
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--nhead', type=int, default=4)
        parser.add_argument('--dim_feedforward', type=int, default=256)
        # optimizer
        parser.add_argument('--temperature', type=float, default=2.0, help='Teacher softmax temperature')
        parser.add_argument('--kd_weight', type=float, default=1.0, help='weight of KD loss')
        parser.add_argument('--mse_weight', type=float, default=0.5, help='weight of KD loss')
        # resume
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--resume_dir', type=str, default="", help='resume epoch')
        parser.add_argument('--resume_epoch', type=int, default=-1, help='resume epoch')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['KD'] # KD
        self.model_names = ['enc', 'rnn', 'C']
        self.netenc = EncCNN1d(opt.input_dim, opt.enc_channel)
        self.netrnn = TransformerEncoder(opt.enc_channel*2, opt.num_layers, opt.nhead, opt.dim_feedforward)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')]
        # self.netC = FcEncoder(opt.enc_channel*2, cls_layers, dropout=0.3)
        self.netC = FcClassifier(opt.enc_channel*2, cls_layers, opt.output_dim, dropout=0.3)
        self.nhead = opt.nhead
        if self.isTrain:
            self.kd_weight = opt.kd_weight
            self.temperature = opt.temperature
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_kd = torch.nn.KLDivLoss(reduction='batchmean')
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
        self.L_logits = input['logits'].to(self.device)
        self.comparE = input['comparE'].to(self.device)
        self.len_comparE = input['len_comparE'].to(self.device)
        self.pesudo_label = input['label'].to(self.device)
        # self.hidden_states = input['hidden_states'].to(self.device)
        # self.len_hidden_states = input['len_hidden_states'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # batch_size, seq_len, _ = self.comparE.size()
        # mask_seq_len = seq_len // 8 + int(seq_len % 8 > 0)
        # self.attn_mask = torch.zeros([batch_size*self.opt.nhead, mask_seq_len, mask_seq_len]).long().to(self.segments)
        # self.key_mask = torch.zeros([batch_size, mask_seq_len]).long().to(self.segments)
        # for sample_num in range(batch_size):
        #     length = self.len_comparE[sample_num] // 8 + int(self.len_comparE[sample_num] % 8 > 0)
        #     self.attn_mask[sample_num*self.nhead:(sample_num+1)*self.nhead, 0:length, 0:length] = 1
        #     self.key_mask[sample_num, :length] = 1
        # self.attn_mask = self.attn_mask.bool()
        # self.key_mask = self.key_mask.bool()
        
        self.segments = self.netenc(self.comparE)
        self.feat, self.A_hidden_states = self.netrnn(self.segments) # mask=self.attn_mask, src_key_padding_mask=self.key_mask)
        self.logits, _ = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.L_pred = F.softmax(self.L_logits / self.temperature, dim=-1)
        self.A_log_pred = F.log_softmax(self.logits, dim=-1)
        self.loss_CE = self.criterion_ce(self.logits, self.pesudo_label)
        self.loss_KD = self.criterion_kd(self.A_log_pred, self.L_pred) * self.kd_weight
        self.total_loss = self.loss_KD # loss_CE
        self.total_loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
