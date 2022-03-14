import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
#from utils.functions import cmc, mean_ap, evaluate
from utils.functions import evaluate
from tqdm import tqdm

import more_itertools as mit

class Trainer():
    def __init__(self, args, model, loader, ckpt):
        self.args = args
        self.queryloader = loader.queryloader
        self.galleryloader = loader.galleryloader
        
        self.ckpt = ckpt
        self.model = model
        self.device = torch.device('cpu' if args.cpu else 'cuda')
 
  
    def mymodel(self,inputs):
        outputs = []
        fs = []
        for i in range(len(inputs)):
            output,f = self.model(inputs[i])
            output = output.unsqueeze(0)
            f = f.unsqueeze(0)
            outputs.append(output)
            fs.append(f)
        outputs = torch.cat(outputs, dim=0)
        fs = torch.cat(fs, dim=0)
        return outputs,fs
    
    def test(self):
        self.ckpt.write_log('[INFO] Test:')
        self.model.eval()
        
        self.ckpt.add_log(torch.zeros(1, 5))
        
        print('extracting query feats')
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(tqdm(self.queryloader)):
# =============================================================================
#             if batch_idx == 10:
#                 break
# =============================================================================
            if not self.args.cpu:
                imgs = imgs.to(self.device)
                pids = pids.to(self.device)
                camids = camids.to(self.device)                
            with torch.no_grad():
                b, n, s, c, h, w = imgs.size()
                assert(b == 1)
                imgs = imgs.view(b*n, s, c, h, w)
                features = self.get_features(self.model, imgs, self.args.test_num_tracks)
                features = torch.mean(features, 0)
                features = features.data.cpu()
                qf.append(features)
                q_pids.extend(pids.cpu().numpy())
                q_camids.extend(camids.cpu().numpy())
                torch.cuda.empty_cache()
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        
        print('extracting gallery feats')        
        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(tqdm(self.galleryloader)):
# =============================================================================
#             if batch_idx == 5:
#                 break
# =============================================================================
            if not self.args.cpu:
                imgs = imgs.to(self.device)
                pids = pids.to(self.device)
                camids = camids.to(self.device)
            with torch.no_grad():
                b, n, s, c, h, w = imgs.size()
                imgs = imgs.view(b*n, s, c, h, w)
                assert(b == 1)
                # handle chunked data
                features = self.get_features(self.model, imgs, self.args.test_num_tracks)
                features = torch.mean(features, 0) 
                features = features.data.cpu()
                gf.append(features)
                pids1 = []
                for i in range(5):
                    pids1.extend(pids.cpu().numpy())
                pids1 = np.asarray(pids1)
                g_pids.append(pids1)
                g_camids.extend(camids.cpu().numpy())
                torch.cuda.empty_cache()
        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        
        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
    
        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
        print("Results ----------")
        self.ckpt.log[-1, 0] = mAP
        rank = []
        cmc = torch.from_numpy(cmc)
        for i,r in enumerate(self.args.ranks):
            rank.append(cmc[r-1])
            self.ckpt.log[-1, i+1] = cmc[r-1]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank5: {:.4f} rank10: {:.4f} rank20: {:.4f} '.format(
            mAP,
            rank[0], rank[1], rank[2], rank[3],
        ))
        print("------------------")
    
    def get_features(self, model, imgs, test_num_tracks):
        """to handle higher seq length videos due to OOM error
        specifically used during test
        
        Arguments:
            model -- model under test
            imgs -- imgs to get features for
        
        Returns:
            features 
        """
    
        # handle chunked data
        all_features = []
    
        for test_imgs in mit.chunked(imgs, test_num_tracks):
            current_test_imgs = torch.stack(test_imgs)
            num_current_test_imgs = current_test_imgs.shape[0]
            b = current_test_imgs.size(0)
            t = current_test_imgs.size(1)
            current_test_imgs = current_test_imgs.view(b * t, current_test_imgs.size(2), current_test_imgs.size(3), current_test_imgs.size(4))
            outputs,features = model(current_test_imgs)
            features = features.view(b * t, -1)
            all_features.append(features)
    
        return torch.cat(all_features)
