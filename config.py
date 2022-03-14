import argparse

parser = argparse.ArgumentParser(description='myRe-ID')

parser.add_argument('--dataset_dir', type=str, default='/MARS', help='the path of dataset')
parser.add_argument('--load', type=str, default='', help='file name to load')
parser.add_argument('--save', type=str, default='test1', help='file name to save') 
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--model', default='resnet', help='model name')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--loss', type=str, default='1*Intra_camera+1*Cross_camera', help='loss function configuration')
parser.add_argument('--dataset_name', type=str, default='MARS', help='train dataset name')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--num_cams', type=int, default=6, help='num of cameras')
parser.add_argument('--margin', type=float, default=0.5, help='Margin of triplet loss')
parser.add_argument('--eta', type=float, default=0.5, help='Learning rate to update anchors')
parser.add_argument('--threshold', type=float, default=0.75, help='threshold of cosine similarity')
parser.add_argument('--warm_up_epochs', type=int, default=27, help='Number of epochs to start tracklet association')
parser.add_argument('--savepth', nargs='+', default=[40], type=int)

parser.add_argument('--lr', type=float, default=0.03, help='learning rate')#2e-4
parser.add_argument('--optimizer', default='SGD', choices=('SGD','ADAM','NADAM','RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)') #change:ADAM
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', default=True, help='SGD nesterov')#action='store_true'
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')#5e-4,4e-5,2e-4,2.2e-4
parser.add_argument('--decay_type', type=str, default='step_8_20_33', help='learning rate decay type')#default='step_40_60_80','step'
parser.add_argument('--lr_decay', type=int, default=10, help='learning rate decay per N epochs')#default=40

parser.add_argument('--height', type=int, default=256, help='height of the input image')#256,224
parser.add_argument('--width', type=int, default=128, help='width of the input image')#256,112
parser.add_argument('--num_classes', type=int, default=625, help='num of x')
parser.add_argument('--num_f', type=int, default=2048, help='num of f')#2048

parser.add_argument('--nThread', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
