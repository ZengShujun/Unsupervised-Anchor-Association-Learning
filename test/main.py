import data
import model
import trainers

from utils import utility
from option import args
import os

load = ['trainpath']
r = []
for name in load:
    args.load = name
    name1 = "test1"
    args.load = os.path.join(args.load,'experiment',name1)
    for i in r:
        args.resume = i
        print("load:",args.load)
        print("Test epoch:",args.resume)
        ckpt = utility.checkpoint(args)
        ckpt.write_log('\nTest model_{}'.format(i))
        loader = data.Data(args)
        models = model.Model(args, ckpt)
        trainer = trainers.Trainer(args, models, loader, ckpt)
        trainer.test()
