import os
import albumentations as alb
# import telepot
import pickle
import urllib

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.transforms as transforms
# from box_convolution import BoxConv2d
import pretrainedmodels as pm

from kekas.modules import Flatten, AdaptiveConcatPool2d

from skimage.transform import resize  # , rescale

from tqdm import tqdm_notebook
# from tg_tqdm import tg_tqdm
# import cv2


class SunRegionDataset(data_utils.Dataset):
    def __init__(self, path_to_df_pkl, path_to_fits_folder, height, width,
                 only_first_class=False, transformations=None, logarithm=True, max=None):
        """
        Args:
            path_to_df_pkl (string): path or url to pkl file represents pandas dataframe with labels
            path_to_image_folder (string): path to folder with fits
            height (int): image height
            width (int): image width
            only_first_class (bool): create dataset with only one letter represents first layer of Mctosh classes
            transformation: pytorch transforms for transforms and tensor conversion
        """
        if path_to_df_pkl.startswith('http'):
            with urllib.request.urlopen(path_to_df_pkl) as pkl:
                self.sunspots = pickle.load(pkl)
        else:
            self.sunspots = pickle.load(path_to_df_pkl)
        self.classes = np.asarray(self.sunspots.iloc[:, 2].unique())
        self.height = height
        self.width = width
        self.folder_path, self.dirs, self.files = next(os.walk(path_to_fits_folder))
        self.len = len(self.files)
        self.ind = list(range(self.len))
        self.transformations = transformations
        self.alb_transorms = alb.Compose([
                                         alb.RandomRotate90(p=0.1),
                                         alb.Rotate(75, p=0.1),
                                         # alb.Resize(224, 224, p=1),  #default 0.1
                                         # alb.RandomCrop(200, 200, p=0.1),
                                         alb.HorizontalFlip(),
                                         # alb.Transpose(),
                                         alb.VerticalFlip(),
                                         # alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                                         ], p=0.7)  # default 0.7
        self.to_tensor = transforms.ToTensor()
        self.only_first_class = only_first_class
        self.height = height
        self.width = width
        self.logarithm = logarithm
        self.first_classes = set([class_[0] for class_ in self.sunspots['class'].unique()])
        self.second_classes = set([class_[1] for class_ in self.sunspots['class'].unique()])
        self.third_classes = set([class_[2] for class_ in self.sunspots['class'].unique()])
        if max is None:
            self.max = self.find_max_dataset()
        else:
            self.max = max

    def __getitem__(self, index):
        file_path = os.path.join(self.folder_path, self.files[index])
        with fits.open(file_path) as fits_file:
            data = fits_file[0].data

        if self.transformations is None:
            if self.logarithm:
                data = self.log_normalize(data)
            data = self.normalize_data(data)
#             data = data.reshape(1, data.shape[0],data.shape[1]).repeat(3, axis=0)
            data = resize(data, (self.height, self.width), anti_aliasing=True)
            data = self.aug()(image=data)['image']  # augumentation
            data = self.to_tensor(data).float()  # uncomment for float
            data = data.repeat(3, 1, 1)
        else:
            data = self.transformations(data)

        mc_class = self.get_attr_region(self.files[index], self.sunspots, self.only_first_class)

        for ind, letter in enumerate(sorted(self.first_classes)):
            if letter == mc_class:
                num_class = ind

#         return (data, num_class, mc_class)
        return {"image": data, "label": num_class, "letter_label": mc_class}

    def __len__(self):
        return self.len

    def show_region(self, index):
        '''Plot region by index from dataset
        index: int, index of sample from dataset
        '''
        date, region = self.files[index].split('.')[1:3]
        file_path = os.path.join(self.folder_path, self.files[index])
        with fits.open(file_path) as fits_file:
            data = fits_file[0].data
        class_, size, location, number_ss = self.get_attr_region(self.files[index],
                                                                 self.sunspots,
                                                                 only_first_class=False,
                                                                 only_class=False)
        ax = plt.axes()
        ax.set_title(
            'Region {} on date {} with class {} on location {} with size {} and number_of_ss {}'
            .format(region, date, class_, location, size, number_ss))
        ax.imshow(data)
        # ax.annotate((24,12))

    def get_attr_region(self, filename, df, only_first_class=False, only_class=True):
        '''Get labels for regions
        '''
        date, region = filename.split('.')[1:3]
        reg_attr = df.loc[date[:-7], int(region[2:])]
        if only_first_class:
            return reg_attr['class'][0]
        elif (not only_class) and (only_first_class):
            class_, \
                size, \
                location, \
                number_ss = reg_attr[['class', 'size', 'location', 'number_of_ss']]
            return class_[0], size, location, number_ss
        elif (not only_class) and (not only_first_class):
            return reg_attr[['class', 'size', 'location', 'number_of_ss']]
        else:
            return reg_attr['class']

    def log_normalize(self, data):
        return np.sign(data) * np.log1p(np.abs(data))

    def normalize_data(self, data):
        return data / self.max

    def find_max_dataset(self):
        '''Find max value of pixels over all dataset
        '''
        m = []
        print('find max all over dataset')
        for file in tqdm_notebook(self.files):
            with fits.open(self.folder_path + file) as ff:
                m.append(np.nanmax(np.abs(ff[0].data)))
        return np.max(m)

    def aug(self):
        return self.alb_transorms

    def split_dataset(self, val_size=None, test_size=None):
        '''Spliting dataset in optional test, train, val datasets
        test_size (optional): float from 0 to 1.
        val_size (optional): float from 0 to 1.

        Returns datasets in order (train, valid, test)

        '''
        len_all = self.len
        test_split_size = int(np.floor(test_size * len_all)) if test_size else 0
        val_split_size = int(np.floor(val_size * len_all)) if val_size else 0
        train_split_size = len_all - test_split_size - val_split_size

        return data_utils.random_split(self, [train_split_size, val_split_size, test_split_size])


class Net(nn.Module):
    def __init__(
            self,
            num_classes: int = 7,
            p: float = 0.2,
            pooling_size: int = 2,
            last_conv_size: int = 1664,
            arch: str = "densenet169",
            pretrained: str = "imagenet") -> None:
        """A model to finetune.

        Args:
            num_classes: the number of target classes, the size of the last layer's output
            p: dropout probability
            pooling_size: the size of the result feature map after adaptive pooling layer
            last_conv_size: size of the flatten last backbone conv layer
            arch: the name of the architecture form pretrainedmodels
            pretrained: the mode for pretrained model from pretrainedmodels
        """
        super().__init__()
        net = pm.__dict__[arch](pretrained=None)
        modules = list(net.children())[:-1]  # delete last layer
        # add custom head
        modules += [nn.Sequential(
            # AdaptiveConcatPool2d is a concat of AdaptiveMaxPooling and AdaptiveAveragePooling
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(13312),
            nn.Dropout(p),
            nn.Linear(13312, num_classes)
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    """Determine what your model will do with your data.

    Args:
        model: the pytorch module to pass input in
        batch: the batch of data from the DataLoader

    Returns:
        The models forward pass results
    """
    inp = batch["image"]  # here we get an "image" from our dataset
    return model(inp)

