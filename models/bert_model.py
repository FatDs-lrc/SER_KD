
import torch
import os
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.classifier import BertClassifier
from models.networks.classifier import RobertaClassifier
from transformers import AutoConfig


class BertModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--bert_type', type=str, help='how many label types in this dataset')
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
        self.model_names = ['BC']
        self.pretrained_model = ['BC']
        # Bert Model
        config = AutoConfig.from_pretrained(opt.bert_type)
        self.netBC = BertClassifier.from_pretrained(opt.bert_type, config=config, output_dim=opt.output_dim)
        # self.netBC = RobertaClassifier(config=config)
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
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
        self.token_ids, self.attention_mask, self.label = input.values()
        self.token_ids = self.token_ids.to(self.device)
        self.attention_mask = self.attention_mask.to(self.device)
        self.label = self.label.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.logits = self.netBC(self.token_ids, self.attention_mask)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        loss = self.loss_CE
        loss.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward() 
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 
