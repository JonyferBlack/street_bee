from torchvision import datasets, models, transforms
import os
import torch
from skimage.io import imread
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torchvision.transforms.functional as F
import h5py


class SemanticDatabaseDataset(Dataset):
    def __init__(self, path, do_norm = False, images_key = 'images', masks_key = 'depths', img_dir='imgs', masks_dir = 'masks'):
        self.transform = ToTensor()
        if do_norm:
            self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.norm = None
        self.toPIL = ToPILImage()
        self.imgs, self.masks = self._read_database(path, images_key, masks_key)
        
    def _read_database(self, path, images_key, masks_key):
        images = None
        masks = None
        with h5py.File(path,'r') as f:
            data = f.get(images_key) 
            images = np.array(data)
            data = f.get(masks_key)
            masks = np.array(data)
        return images, masks
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img_tr = self.transform(img)

        if not self.norm == None:
            img_tr = self.norm(img_tr)
        mask_tr = mask
        if len(mask.shape) > 2:
            mask_tr = mask[:, :, 0]
        mask_tr = self.transform(mask)
        return img_tr, mask_tr


class SimpleSemanticDataset(Dataset):
    def __init__(self, path, do_norm = False, img_dir='imgs', masks_dir = 'masks'):
        self.transform = ToTensor()
        if do_norm:
            self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.norm = None
        self.toPIL = ToPILImage()
        self.imgs = self._scan_directory(os.path.join(path, img_dir))
        self.masks = self._scan_directory(os.path.join(path, masks_dir))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = imread(self.imgs[idx])
        mask = imread(self.masks[idx])
        img_tr = self.transform(img)

        if not self.norm == None:
            img_tr = self.norm(img_tr)
        mask_tr = mask
        if len(mask.shape) > 2:
            mask_tr = mask[:, :, 0]
        mask_tr = self.transform(mask)
        return img_tr, mask_tr

    def _scan_directory(self, source_directory):
        """
        Scan a directory for images, returning any images found with the
        extensions ``.jpg``, ``.JPG``, ``.jpeg``, ``.JPEG``, ``.gif``, ``.GIF``,
        ``.img``, ``.IMG``, ``.png``, ``.PNG``, ``.tif``, ``.TIF``, ``.tiff``,
        or ``.TIFF``.

        :param source_directory: The directory to scan for images.
        :type source_directory: String
        :return: A list of images found in the :attr:`source_directory`
        """
        # TODO: GIFs are highly problematic. It may make sense to drop GIF support.
        file_types = ['*.jpg', '*.bmp', '*.jpeg', '*.gif', '*.img', '*.png', '*.tiff', '*.tif']

        list_of_files = []
        if os.name == "nt":
            for file_type in file_types:
                list_of_files.extend(glob(os.path.join(os.path.abspath(source_directory), file_type)))
        else:
            file_types.extend([str.upper(str(x)) for x in file_types])
            for file_type in file_types:
                list_of_files.extend(glob(os.path.join(os.path.abspath(source_directory), file_type)))

        return list_of_files



class TestDataset(Dataset):
    def __init__(self, path):
        self.transform = ToTensor()
        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.imgs = [imread(x) for x in glob(os.path.join(path,'*.png'))]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img_tr = self.transform(img)
        return self.norm(img_tr)


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


