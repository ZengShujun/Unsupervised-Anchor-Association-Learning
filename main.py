import torch

import data
import model
import loss
import trainers

from utils import utility
from config import args
from loss import tracklet_num

args.save_models = True
ckpt = utility.checkpoint(args)
loader = data.Data(args)
model = model.Model(args, ckpt)
loss = loss.Loss(args, ckpt) if not args.test_only else None
trainer = trainers.Trainer(args, model, loss, loader, ckpt)

intra_anchors = []
cross_anchors = []

num_tracklets = tracklet_num.get_tracklet_num(args.dataset_name)
for i in range(len(num_tracklets)):
    intra_anchors.append(torch.zeros([num_tracklets[i], args.num_f],dtype=torch.float))
    cross_anchors.append(torch.zeros([num_tracklets[i], args.num_f],dtype=torch.float))

n = 0
while not trainer.terminate():
    n += 1
    intra_anchors,cross_anchors = trainer.train(intra_anchors,cross_anchors)
    if n in args.savepth:
        trainer.test()