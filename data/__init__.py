from torchvision import transforms
from torch.utils.data import dataloader
from data.mars import MyDataset

class Data:
    def __init__(self, args):

        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        # if args.random_erasing:
        #     train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        # test_transform = transforms.Compose([
        #     transforms.Resize((args.height, args.width), interpolation=3),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        
        if not args.test_only:
            dataset = MyDataset(args.dataset_dir,train_transform)
            self.train_loader = dataloader.DataLoader(dataset, 
                                                      batch_size = args.batch_size,
                                                      shuffle = True,
                                                      num_workers = args.nThread,
                                                      drop_last = False)
            