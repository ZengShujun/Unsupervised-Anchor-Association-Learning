import torch
from torch import nn
import torch.nn.functional as F

class Intra_camera(nn.Module):
    def __init__(self, args):
        super(Intra_camera, self).__init__()
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        
    def forward(self,features,labels,cams,intra_anchors,cross_anchors,epoch):
        
        labels = labels-1
        cams = cams-1
        features_n = self.normalize(features)  
        dist_pos_intra = []
        dist_neg = []
        
        labels_cam = []
        features_cam = []
        intra_anchors_n = []
        cross_anchors_n = []
        
        for i in range(self.args.num_cams):
            condition_cam = torch.eq(cams, i)
            labels_cam.append(labels[condition_cam])
            features_cam.append(features_n[condition_cam])
            intra_anchors_n.append(self.normalize(intra_anchors[i]).to(self.device))
            cross_anchors_n.append(self.normalize(cross_anchors[i]).to(self.device))
        if epoch <= self.args.warm_up_epochs:
            for i in range(self.args.num_cams):
                same_anchors_n = intra_anchors_n[i]
                same_labels_cam = labels_cam[i]
                same_intra_anchors_n = same_anchors_n[same_labels_cam]
                same_features_cam = features_cam[i]
                for j in range(len(same_labels_cam)):
                    image_feature = same_features_cam[j].unsqueeze(0)
                    anchor_feature = same_intra_anchors_n[j].unsqueeze(0)
                    dist_pos1 = self.euclidean_dist(image_feature, anchor_feature)
                    dist_pos = dist_pos1.squeeze(0)
                    dist_pos_intra.append(dist_pos)
                    
                dist_neg.append(self.association_ranking(features_cam[i], labels_cam[i], intra_anchors_n[i]))
                
            dist_pos_intra = torch.cat(dist_pos_intra, 0)
            dist_neg = torch.cat(dist_neg, 0)
            intra_loss = self.hinge_loss(dist_pos_intra, dist_neg)
            intra_anchors1 = self.update_intra_anchor(intra_anchors_n, features_cam, labels_cam, epoch)
        else:
            intra_loss = torch.Tensor([0]).squeeze(0).to(self.device)
            intra_anchors1 = self.update_intra_anchor(intra_anchors_n, features_cam, labels_cam, epoch)
            
        return intra_loss, intra_anchors1, cross_anchors
            
            
            
    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    
    def euclidean_dist(self, x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12)
        return dist
    
    def association_ranking(self, features_cam, labels_cam, intra_anchors_n):
        dist_all = self.euclidean_dist(features_cam, intra_anchors_n)
        dist_min1, rank1 = torch.topk(-dist_all, k=2, largest=True, sorted=True)
        non_match = torch.ne(labels_cam, rank1[:,0])
        dist_neg = torch.where(non_match, -dist_min1[:, 0], -dist_min1[:,1])
        return dist_neg
    
    def hinge_loss(self, dist_pos, dist_neg):
        loss = F.relu(dist_pos - dist_neg + self.args.margin).mean()
        return loss
    
    def update_intra_anchor(self, intra_anchors_n, features_cam, labels_cam,epoch):
        
        x=[]
        for i in range(len(intra_anchors_n)):
            x.append(intra_anchors_n[i].clone())
        if epoch <= self.args.warm_up_epochs:    
            for i in range(self.args.num_cams):
                same_anchors_n = x[i]
                same_labels_cam = labels_cam[i]
                same_features_cam = features_cam[i]
                same_intra_anchors_n = same_anchors_n[same_labels_cam]
                for j in range(len(same_labels_cam)):
                    image_feature = same_features_cam[j]
                    anchor_feature = same_intra_anchors_n[j]
                    diff = (anchor_feature - image_feature) * self.args.eta
                    same_anchors_n[same_labels_cam[j]] = anchor_feature - diff
                x[i] = same_anchors_n
            
            return x
        else:
            return x
                
                
            
           

    
        