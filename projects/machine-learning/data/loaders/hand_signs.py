from torch.utils.data import Dataset
import h5py
import torch
import torchvision
import numpy as np

class HandSigns(Dataset):

    def __init__(self, train: bool, transform=None, target_transform=None):
        '''
        Note: the parameter 'transform' overriden by Normalize transform
        under the train-mode (i.e. train=True), but not under the test-mode
        (i.e. train=False).
        Note: the parameter 'target_transform' is applied regardless of 
        the mode.
        '''
        super(HandSigns, self).__init__()

        self.train = train
        self.target_transform = target_transform
        self.transform = transform
        
        prefix = "train" if self.train else "test"

        file_name = f"../data/{prefix}-hand-signs.h5"
        # raises FileNotFoundError()
        with h5py.File(name=file_name, mode="r") as f:
            self.set_x = torch.tensor(np.array(f[prefix + "_set_x"])) \
                .to(dtype=torch.float32)
            self.set_y = torch.tensor(np.array(f[prefix + "_set_y"]))
            
            self.class_labels = torch.tensor(f["list_classes"])
        
        # shape -> (m, c, h, w)
        self.set_x = torch.permute(self.set_x, dims=(0,3,1,2))

        if train:
            set_mean = torch.mean(input=self.set_x, dim=0, keepdim=False)
            set_std = torch.std(input=self.set_x, dim=0, keepdim=False)

            self.transform = torchvision.transforms.Normalize(
                mean=set_mean, std=set_std, inplace=False)
    
    def __len__(self):
        return len(self.set_x)
    
    def __getitem__(self, idx):
        image = self.set_x[idx]
        label = self.set_y[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    def get_norm_transform(self):
        '''Returns the mean and variance normalization transform computed 
        while loading the train dataset.

        Note: calling this method on an instance of HandSigns that was 
        initialized under the test mode (i.e. by passing train=False) will
        raise a RuntimeError.
        '''
        if not self.train:
            raise RuntimeError("Cannot call get_transform() under train=False")
        
        return self.transform
    
    def get_class_labels(self):
        return self.class_labels
