
import torch
import os
import json
import torch.nn.functional as F
from collections import OrderedDict

from models.base_model import BaseModel
from models.networks.rnn import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.networks import tools
from models.early_fusion_model import EarlyFusionModel
from models.utils.config import OptConfig


class FeatureDistillModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--T_modality', type=str, help='which modality to use for Teacher model')
        parser.add_argument('--T_checkpoint', type=str, help='Teacher model checkpoints')
        parser.add_argument('--S_modality', type=str, help='which modality to use for Student model')
        parser.add_argument('--S_checkpoint', type=str, default=None, help='Student model checkpoints')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of CE loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of KD loss')
        
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'MSE']
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = getattr(opt, "embd_size_" + opt.S_modality.lower())
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # Student model
        self.S_modality = opt.S_modality
        self.S_checkpoint = opt.S_checkpoint
        assert len(self.S_modality) == 1, 'Only support one modality now'
        # acoustic model
        if 'A' in self.S_modality:
            self.model_names.append('SA')
            self.netSA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            
        # lexical model
        if 'L' in self.S_modality:
            self.model_names.append('SL')
            self.netSL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            
        # visual model
        if 'V' in self.S_modality:
            self.model_names.append('SV')
            self.netSV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
    
        self.cvNo = opt.cvNo
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # record lamda
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            # Init Teacher model
            self.T_modality = opt.T_modality
            self.T_checkpoint = opt.T_checkpoint
        
        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def setup(self, opt):
        """Load and print networks; create schedulers
           Load Teacher model if isTrain
           Load init Student model with pretrained network if giving S_checkpoint info 

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [tools.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            # load network T
            self.netT = self.load_encoder_module(self.T_checkpoint, self.T_modality)
            self.netTC = self.load_encoder_module(self.T_checkpoint, 'C')
            # load student network S
            if self.S_checkpoint:
                pretrained_netS = self.load_encoder_module(self.S_checkpoint, self.S_modality)
                netS = getattr(self, "netS" + self.S_modality)
                netS.load_state_dict(pretrained_netS.state_dict())
                print('[ Net Student ] Load parameters from ' + self.S_checkpoint)
            
            # init parameters
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                if name != self.S_modality or self.S_checkpoint is None:
                    net = tools.init_weights(net, opt.init_type, opt.init_gain)
                net = tools.to_gpu(net, opt.gpu_ids)
                setattr(self, 'net' + name, net)
        else:
            self.eval()
        self.print_networks(opt.verbose)
        self.post_process()
    
    def load_encoder_module(self, checkpoint_path, modality):
        config_path = os.path.join(checkpoint_path, 'train_opt.conf')
        config = self.load_from_opt_record(config_path)
        config.isTrain = False                             # teacher model should be in test mode
        config.gpu_ids = self.gpu_ids                      # set gpu to the same
        pretrained_encoder = EarlyFusionModel(config)
        pretrained_encoder.load_networks_cv(os.path.join(checkpoint_path, str(self.cvNo)))
        pretrained_encoder.cuda()
        pretrained_encoder.eval()
        return getattr(pretrained_encoder, 'net' + modality)
    
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in [self.S_modality, self.T_modality]:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L'in [self.S_modality, self.T_modality]:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in [self.S_modality, self.T_modality]:
            self.visual = input['V_feat'].float().to(self.device)
        
        self.label = input['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Teacher
        if self.isTrain:
            if 'A' == self.T_modality:
                self.T_feat = self.netT(self.acoustic)
            if 'L' == self.T_modality:
                self.T_feat = self.netT(self.lexical)
            if 'V' == self.T_modality:
                self.T_feat = self.netT(self.visual)
            self.T_logits, _ = self.netTC(self.T_feat) 
            self.T_pred = F.softmax(self.T_logits, dim=-1) 
        
        # Student
        if 'A' == self.S_modality:
            self.feat = self.netSA(self.acoustic)
        if 'L' == self.S_modality:
            self.feat = self.netSL(self.lexical)
        if 'V' == self.S_modality:
            self.feat = self.netSV(self.visual)
    
        self.logits, _ = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label) * self.ce_weight
        self.loss_MSE = self.criterion_mse(self.feat, self.T_feat) * self.mse_weight
        loss = self.loss_CE + self.loss_MSE
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
