import os
import utils.utility as utility
import torch
from loss import Regularization_loss
from model import EMA

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        
        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.regular = Regularization_loss.Regularization(self.model,self.args,p=2)
        self.ema = EMA.EMA(0.9999)

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()
            
    def train(self,intra_anchors,cross_anchors):
        
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema.register(name, param.data)
        for batch, (inputs, labels, cams) in enumerate(self.train_loader):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            cams = cams.to(self.device)
            self.optimizer.zero_grad()
            
            features = self.mymodel(inputs)
            loss, intra_anchors1, cross_anchors1 = self.loss(features, labels, cams, intra_anchors, cross_anchors, epoch, lr)
            reg_loss = self.regular(self.model).to(self.device)
            total_loss = loss + reg_loss
            total_loss.backward()
            self.optimizer.step()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema.update(name, param.data)
            
            for i in range(len(intra_anchors1)):
                intra_anchors[i] = intra_anchors1[i].detach().clone()
                
            for i in range(len(cross_anchors)):
                cross_anchors[i] = cross_anchors1[i].detach().clone()
            
            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch+1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')
            
        self.loss.end_log(len(self.train_loader))
            
        return intra_anchors,cross_anchors
            
    def mymodel(self,inputs):
        _,f = self.model(inputs)
        return f
            
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs
        
    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.save(self, epoch, is_best=False)
        print("------------------")

        
        
        
        
        
        
        
        
        
    
