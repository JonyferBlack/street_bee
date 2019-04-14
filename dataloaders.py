from torchvision import datasets, models, transforms
import os
import torch
from skimage.io import imread
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize
from torch.utils.data import Dataset
from glob import glob
import numpy as np




class SimpleSemanticDataset(Dataset):
    def __init__(self, path, do_norm = False):
        self.transform = ToTensor()
        if do_norm:
            self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.norm = None
        self.imgs = [imread(x) for x in glob(os.path.join(path,'imgs/*.png'))]
        self.masks = [imread(x)[:, :, np.newaxis] for x in glob(os.path.join(path,'masks/*.png'))]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img_tr = self.transform(img)
        if not self.norm == None:
            img_tr = self.norm(img_tr)
        return img_tr, self.transform(mask)


def init_data_loaders(input_size, data_dir, batch_size, image_sets = ['train', 'val', 'test']):
    """
    """
    img_net100_norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in image_sets}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in image_sets}
    print('done')
    return image_datasets, dataloaders_dict