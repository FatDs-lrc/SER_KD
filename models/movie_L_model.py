
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.transformer import TransformerEncoder, AlignNet
from models.networks.rcn import EncCNN1d
from models.networks.classifier import BertClassifier, FcClassifier

class MovieLModel(BaseModel):
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
        parser.add_argument('--mse_weight1', type=float, default=0.3, help='layer -1')
        parser.add_argument('--mse_weight2', type=float, default=0.2, help='layer -2')
        parser.add_argument('--mse_weight3', type=float, default=0.1, help='layer -3')
        parser.add_argument('--mse_weight4', type=float, default=0.1, help='layer -4')
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
        self.loss_names = ['KD', 'MSE1', 'MSE2', 'MSE3', 'MSE4'] # KD
        self.model_names = ['enc', 'rnn', 'C', '_teacher', 'align1', 'align2', 'align3', 'align4']
        self.pretrained_model = ['_teacher']
        self.netenc = EncCNN1d(opt.input_dim, opt.enc_channel)
        self.netrnn = TransformerEncoder(opt.enc_channel*2, opt.num_layers, opt.nhead, opt.dim_feedforward)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')]
        # self.netC = FcEncoder(opt.enc_channel*2, cls_layers, dropout=0.3)
        self.netC = FcClassifier(opt.enc_channel*2, cls_layers, opt.output_dim, dropout=0.3)
        self.nhead = opt.nhead

        self.netalign1 = AlignNet(768, opt.enc_channel*2, num_heads=4)
        self.netalign2 = AlignNet(768, opt.enc_channel*2, num_heads=4)
        self.netalign3 = AlignNet(768, opt.enc_channel*2, num_heads=4)
        self.netalign4 = AlignNet(768, opt.enc_channel*2, num_heads=4)
        self.align_layers = [1, 3, 5, 7]

        teacher_path = '/data4/lrc/movie_dataset/pretrained/bert_movie_model'
        self.net_teacher = BertClassifier.from_pretrained(
            teacher_path, num_classes=5, embd_method='max')
        self.net_teacher = self.net_teacher.eval()

        if self.isTrain:
            self.kd_weight = opt.kd_weight
            self.mse_weight1 = opt.mse_weight1
            self.mse_weight2 = opt.mse_weight2
            self.mse_weight3 = opt.mse_weight3
            self.mse_weight4 = opt.mse_weight4

            self.temperature = opt.temperature
            self.criterion_mse = torch.nn.MSELoss()
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
        self.input_ids = input['input_ids']
        self.mask = input['mask']
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
        self.feat, self.A_hidden = self.netrnn(self.segments) # mask=self.attn_mask, src_key_padding_mask=self.key_mask)
        self.logits, _ = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        if self.isTrain:
            self.L_pred = F.softmax(self.L_logits / self.temperature, dim=-1)
            with torch.no_grad():
                _, self.L_hidden = self.net_teacher(self.input_ids, self.mask)
                self.L_hidden = [self.L_hidden[i] for i in self.align_layers]
            self.A_log_pred = F.log_softmax(self.logits, dim=-1)
            self.A_hidden = self.A_hidden[-4:]
            self.align_out1 = self.netalign1(self.L_hidden[0], self.A_hidden[0])
            self.align_out2 = self.netalign1(self.L_hidden[1], self.A_hidden[1])
            self.align_out3 = self.netalign1(self.L_hidden[2], self.A_hidden[2])
            self.align_out4 = self.netalign1(self.L_hidden[3], self.A_hidden[3])
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_KD = self.criterion_kd(self.A_log_pred, self.L_pred) * self.kd_weight
        self.loss_MSE1 = self.criterion_mse(self.align_out1, self.L_hidden[0]) * self.mse_weight1
        self.loss_MSE2 = self.criterion_mse(self.align_out2, self.L_hidden[1]) * self.mse_weight2
        self.loss_MSE3 = self.criterion_mse(self.align_out3, self.L_hidden[2]) * self.mse_weight3
        self.loss_MSE4 = self.criterion_mse(self.align_out4, self.L_hidden[3]) * self.mse_weight4
        self.total_loss = self.loss_KD + self.loss_MSE1 + self.loss_MSE2 + self.loss_MSE3 + self.loss_MSE4
        self.total_loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
