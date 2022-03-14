import os
import os.path as osp
import matplotlib.image as mpimg
from PIL import Image

from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class MyDataset(Dataset):
    def __init__(self,dataset_dir,transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.image_label_cam = []
        with open(osp.join(self.dataset_dir, 'traindata.txt'), 'r') as fp:
            content = fp.readlines()
            str_list = [s.rstrip().split(' ') for s in content]
            self.image_label_cam = [(x[0], int(x[1]), int(x[2])) for x in str_list]
        
    def __len__(self):
        return len(self.image_label_cam)
    
    def __getitem__(self, index):
        image_label_cam_pair = self.image_label_cam[index]
        imgpth = image_label_cam_pair[0]
        img = read_image(imgpth)
        if self.transform is not None:
            img = self.transform(img)
        label = image_label_cam_pair[1]
        cam = image_label_cam_pair[2]
        return img, label, cam