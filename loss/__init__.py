import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from loss import Intra_camera
from loss import Cross_camera

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckpt):
        super(Loss, self).__init__()
        print('[INFO] Making loss...')
        
        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'Intra_camera':
                loss_function = Intra_camera.Intra_camera(self.args)
                
            elif loss_type == 'Cross_camera':
                loss_function = Cross_camera.Cross_camera(self.args)
                
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
            
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])
        
        self.log = torch.Tensor()
        
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(self.device)
        
        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )
        print(self.loss)
        
    def forward(self,features,labels,cams,intra_anchors,cross_anchors,epoch,lr):
        losses = []
        for i, l in enumerate(self.loss):
            if self.args.model == 'resnet' and l['type'] == 'Intra_camera':
                # if epoch <= self.args.warm_up_epochs:
                loss1 = []
                #print(intra_anchors[0].requires_grad)
                intra_loss, intra_anchors1, cross_anchors1 = l['function'](features,labels,cams,intra_anchors,cross_anchors,epoch)
                # print('intra_loss:',intra_loss)
                loss1.append(intra_loss)
                loss1 = sum(loss1) / len(loss1)
                effective_loss = l['weight'] * loss1
                losses.append(effective_loss)
                self.log[-1, i] = effective_loss.item()
                # else:
                #     pass
                
            elif self.args.model == 'resnet' and l['type'] == 'Cross_camera':
                # if epoch <= self.args.warm_up_epochs:
                # pass
                # else:
                loss2 = []
                for j in range(len(intra_anchors1)):
                    intra_anchors[j] = intra_anchors1[j].detach().clone()
                cross_loss, intra_anchors1, cross_anchors1 = l['function'](features,labels,cams,intra_anchors,cross_anchors,epoch,lr)
                # print('cross_loss:',cross_loss)
                loss2.append(cross_loss)
                loss2 = sum(loss2) / len(loss2)
                effective_loss = l['weight'] * loss2
                losses.append(effective_loss)
                self.log[-1, i] = effective_loss.item()
  
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] = loss_sum.item()
        # print('self.log is:',self.log)
        return loss_sum,intra_anchors1,cross_anchors1
        
    
    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()
                
    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module
        
    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))
        
    def end_log(self, batches):
        self.log[-1].div_(batches)
        
    def display_loss(self, batch):
        # n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c ))#/ n_samples))

        return ''.join(log)
    
    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)
            
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
        
    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()