import torch
from torch import nn

class Cross_camera(nn.Module):
    def __init__(self, args):
        super(Cross_camera, self).__init__()
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        
    def forward(self,features,labels,cams,intra_anchors,cross_anchors,epoch,lr):
        labels = labels-1
        cams = cams-1
        features_n = self.normalize(features)
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
        cross_anchors_batch_n = []
        for i in range(self.args.num_cams):
            cross_anchors_batch_n.append(cross_anchors_n[i][labels_cam[i]])
        if epoch > self.args.warm_up_epochs:
            cross_loss = self.loss(cross_anchors_n, cross_anchors_batch_n, labels_cam, labels, cams, features_n, epoch)
            cross_anchors1 = self.update_cross_anchor(cross_anchors_n, intra_anchors_n, cross_anchors_batch_n, labels_cam, labels, cams, features_n, epoch, lr)
        else:
            cross_loss = torch.Tensor([0]).squeeze(0).to(self.device)
            cross_anchors1 = self.update_cross_anchor(cross_anchors_n, intra_anchors_n, cross_anchors_batch_n, labels_cam, labels, cams, features_n, epoch, lr)
        return cross_loss, intra_anchors, cross_anchors1
        
        
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
    
    def loss(self, cross_anchors_n, cross_anchors_batch_n, labels_cam, labels, cams, features_n, epoch):
        e = self.edge(cross_anchors_n, cross_anchors_batch_n, labels_cam, epoch)
        #print('e is:',e)
        kys = list(e.keys())
        loss_all = []
        for i in range(len(labels)):
            cam1 = cams[i].item()
            label1 = labels[i].item()
            feature_n = features_n[i]
            for j in range(len(kys)):
                num = kys[j]
                if cam1 == num[0] and label1 == num[1]:
                    cam2 = num[2]
                    label2 = num[3]
                    label2 = torch.tensor(label2).to(self.device)
                    criterion = nn.CrossEntropyLoss()
                    output = torch.mm(feature_n.unsqueeze(0), cross_anchors_n[cam2].t())
                    loss1 = criterion(output, label2.unsqueeze(0))
                    label2 = label2.item()
                    loss2 = e[(cam1,label1,cam2,label2)] * loss1
                    loss_all.append(loss2)
        mean_cross_loss = sum(loss_all) / len(labels)
        return mean_cross_loss

    def edge(self, cross_anchors_n, cross_anchors_batch_n, labels_cam, epoch):
        e = {}
        for i in range(self.args.num_cams):
            other_anchors_n = []
            [other_anchors_n.append(cross_anchors_n[x]) for x in range(self.args.num_cams) if x is not i]
            other_cams = []
            [other_cams.append(x) for x in range(self.args.num_cams) if x is not i]
            same_anchors_n = cross_anchors_n[i]
            same_anchors_batch_n = cross_anchors_batch_n[i]
            same_labels_batch = labels_cam[i]
            for j in range(len(same_anchors_batch_n)):
                same_anchor = same_anchors_batch_n[j]
                same_label = same_labels_batch[j]
                self_score = torch.cosine_similarity(same_anchor, same_anchor, dim=0)
                cam1 = i
                label1 = same_label.item()
                e[(cam1,label1,cam1,label1)] = self_score
                for l in range(len(other_anchors_n)):
                    other_anchors_cam = other_anchors_n[l]
                    other_cam = other_cams[l]
                    dist = self.euclidean_dist(same_anchor.unsqueeze(0), other_anchors_cam)
                    _, rank = torch.topk(-dist, k=1, largest=True, sorted=True)
                    rank1_anchor = other_anchors_cam[rank.squeeze(1)]
                    dist = self.euclidean_dist(rank1_anchor, same_anchors_n)
                    _, rank1 = torch.topk(-dist, k=1, largest=True, sorted=True)
                    if rank1.squeeze(1) == same_label:
                        score = torch.cosine_similarity(same_anchor, rank1_anchor.squeeze(0), dim=0)
                        if score > self.args.threshold: 
                            cam2 = other_cam
                            label2 = rank.squeeze(0).squeeze(0)
                            label2 = label2.item()
                            e[(cam1,label1,cam2,label2)] = score
        return e
    
    def update_cross_anchor(self, cross_anchors_n, intra_anchors_n, cross_anchors_batch_n, labels_cam, labels, cams, features_n, epoch, lr):
        
        x=[]
        if epoch <= self.args.warm_up_epochs:
            for i in range(len(cross_anchors_n)):
                x.append(intra_anchors_n[i].clone())
        else:
            for i in range(len(cross_anchors_n)):
                x.append(cross_anchors_n[i].clone())
            e = self.edge(cross_anchors_n, cross_anchors_batch_n, labels_cam, epoch)
            kys = list(e.keys())
            for i in range(len(labels)):
                cam1 = cams[i].item()
                label1 = labels[i].item()
                feature_n = features_n[i]
                for j in range(len(kys)):
                    num = kys[j]
                    if cam1 == num[0] and label1 == num[1]:
                        cam2 = num[2]
                        label2 = num[3]
                        label2 = torch.tensor(label2).to(self.device)
                        update_anchors = x[cam2]
                        update_anchor = update_anchors[label2]
                        criterion = nn.CrossEntropyLoss()
                        output = torch.mm(feature_n.unsqueeze(0), cross_anchors_n[cam2].t())
                        loss = criterion(output, label2.unsqueeze(0))
                        label2 = label2.item()
                        update = update_anchor - lr * (1-loss) * e[(cam1,label1,cam2,label2)] * feature_n
                        update_anchors[label2] = update
                        x[cam2] = update_anchors
        return x
