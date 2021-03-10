
import torch
import os
import json
import torch.nn.functional as F

from models.base_model import BaseModel
from models.networks.rcn import EncCNN1d
from models.networks.fc import FcEncoder
from models.networks.classifier import bert_classifier, roberta_classifier
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

        self.netenc = EncCNN1d(opt.input_dim, opt.enc_channel)
        self.netrnn = TransformerEncoder(opt.enc_channel*2, opt.num_layers, opt.nhead, opt.dim_feedforward)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        self.netC = FcEncoder(opt.enc_channel*2, cls_layers, dropout=0.3)

        if 'roberta' in opt.bert_type:
            self.netBC = roberta_classifier(opt.output_dim, opt.bert_type)
        else:
            self.netBC = bert_classifier(opt.output_dim, opt.bert_type)
        
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
            self.T_logits /= self.temperature
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
        self.log_pred = F.log_softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label) * self.ce_weight
        self.loss_KD = self.criterion_kd(self.log_pred, self.T_pred) * self.kd_weight
        self.loss_mse = self.criterion_mse(self.feat, self.T_feat) * self.mse_weight
        loss = self.loss_CE + self.loss_KD + self.loss_mse
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
