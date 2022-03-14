from torchvision import transforms
from data.common import init_dataset
from data.mars import VideoDataset
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):

        # train_list = [
        #     transforms.Resize((args.height, args.width), interpolation=3),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ]

        # train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = init_dataset(args.data_train, args.datadir)
        
        pin_memory = False if args.cpu else True
        if args.data_test in ['Mars']:
            self.queryloader = dataloader.DataLoader(
                VideoDataset(dataset.query, seq_len=args.seq_len,
                             sample='dense', transform=test_transform),
                batch_size=args.test_batch, shuffle=False, num_workers=args.nThread,
                pin_memory=pin_memory, drop_last=False,
            )
            
            self.galleryloader = dataloader.DataLoader(
                VideoDataset(dataset.gallery, seq_len=args.seq_len,
                             sample='dense', transform=test_transform),
                batch_size=args.test_batch, shuffle=False, num_workers=args.nThread,
                pin_memory=pin_memory, drop_last=False,
            )
