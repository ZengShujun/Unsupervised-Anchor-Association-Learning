import argparse

parser = argparse.ArgumentParser(description='myRe-ID')

#1
parser.add_argument('--nThread', type=int, default=2, help='number of threads for data loading') #2
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
#2
parser.add_argument("--datadir", type=str, default="/MARS", help='dataset directory')
parser.add_argument('--data_train', type=str, default='Mars', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Mars', help='test dataset name')
#3
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--seq_len', default=16, type=int, help="sample num of frame every tracklet") 
parser.add_argument('--test-num-tracks', type=int, default=16, help="number of tracklets to pass to GPU during test (to avoid OOM error),4/16")
parser.add_argument('--ranks', nargs='+', default=[1, 5, 10, 20], type=int)
#4
parser.add_argument('--model', default='resnet', help='model name')
#5
parser.add_argument('--height', type=int, default=256, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--num_classes', type=int, default=625, help='num of x')
parser.add_argument('--num_f', type=int, default=2048, help='num of f')
#8
parser.add_argument("--resume", type=int, default=0, help='resume from specific checkpoint') 
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--load', type=str, default='/data', help='file name to load')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
