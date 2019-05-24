# import telepot
import pickle
import urllib

# from astropy.io import fits
# from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data as data_utils
# import torchvision

# from torchvision import datasets, models, transforms
# from torch.optim import lr_scheduler
# from torchvision.models.resnet import BasicBlock
# from box_convolution import BoxConv2d
# import pretrainedmodels as pm

from kekas import Keker, DataOwner  # , DataKek
# from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy  # , accuracy_score
# from kekas.modules import Flatten, AdaptiveConcatPool2d
# from kekas.callbacks import Callback, Callbacks, DebuggerCallback

from adabound import AdaBound

from sklearn.utils import class_weight
# from tqdm import tqdm_notebook
# from tg_tqdm import tg_tqdm

import warnings
# import cv2

from utils import SunRegionDataset, Net, step_fn

plt.ion()
warnings.filterwarnings("ignore")

tg_token = 'TOKEN'
tg_chat_id = 1234
ik_chat_id = 1234
sun_group_id = -1234

# define some things
url_pkl = 'https://raw.githubusercontent.com/iknyazeva/FitsProcessing/master/sunspot_1996_2017.pkl'
dataset_folder = 'ALLrescaled/'
path_to_save = ''

logdir = "logs"
lrlogdir = "lrlogs"
checkdir = 'check'

with urllib.request.urlopen(url_pkl) as pkl:
    sunspots = pickle.load(pkl)

print(sunspots.tail(5))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('\ndevice:', device)


max_value_pixel = 3977.35
batch_size = 64
adam_lr = 0.003
sgd_lr = 0.01
sgd_wd = 0.000
adam_wd = 0.0000
step_size = 30
num_epochs = 100


regions_dataset = SunRegionDataset(path_to_df_pkl=url_pkl, path_to_fits_folder=dataset_folder, height=100, width=100,
                                   only_first_class=True, logarithm=False, max=max_value_pixel)

train_dataset, val_dataset, test_dataset = regions_dataset.split_dataset(0.1, 0.1)

# with open('train_dataset.pkl', 'wb') as train:
#     pickle.dump(train_dataset, train)
# with open('val_dataset.pkl', 'wb') as val:
#     pickle.dump(val_dataset, val)
# with open('test_dataset.pkl', 'wb') as test:
#     pickle.dump(test_dataset, test)

# with open('train_dataset.pkl', 'rb') as train:
#     train_dataset = pickle.load(train)
# with open('val_dataset.pkl', 'rb') as val:
#     val_dataset = pickle.load(val)
# with open('test_dataset.pkl', 'rb') as test:
#     test_dataset = pickle.load(test)


train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataowner = DataOwner(train_loader, val_loader, test_loader)

# get weights for classes
label_wts = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique([class_[0] for class_ in sunspots['class']]), y=[class_[0] for class_ in sunspots['class']])

label_wts = torch.Tensor(label_wts).to(device)

w_criterion = nn.CrossEntropyLoss(weight=label_wts)
criterion = nn.CrossEntropyLoss()
model = Net()

# we use kekas framework for learning (https://github.com/belskikh/kekas/)
keker = Keker(model=model,
              dataowner=dataowner,
              criterion=w_criterion,
              step_fn=step_fn,
              target_key="label",
              metrics={"acc": accuracy},
              # opt=torch.optim.Adam,
              # opt=torch.optim.SGD,
              # opt_params={"weight_decay": 1e-5}
              # opt_params={"momentum": 0.99}
              opt=AdaBound,
              opt_params={'final_lr': 0.01,
                          'weight_decay': 5e-4}
              )

keker.freeze(model_attr='net')

keker.kek_one_cycle(max_lr=1e-6,
                    cycle_len=90,
                    momentum_range=(0.95, 0.85),
                    div_factor=10,
                    increase_fraction=0.3,
                    logdir=logdir,
                    cp_saver_params={
                        "savedir": checkdir,
                        "metric": "acc",
                        "n_best": 3,
                        "prefix": "check",
                        "mode": "max"
                    }
                    )


keker.load(checkdir + '/' + 'check.best.h5')

# FOR FINE TUNE ALL PARAMETERS OF NET

# keker.unfreeze(model_attr='net')

# keker.kek_one_cycle(max_lr=1e-6,
#                     cycle_len=90,
#                     momentum_range=(0.95, 0.85),
#                     div_factor=10,
#                     increase_fraction=0.3,
#                     logdir=logdir,
#                     cp_saver_params={
#                         "savedir": checkdir,
#                         "metric": "acc",
#                         "n_best": 3,
#                         "prefix": "check",
#                         "mode": "max"
#                     }
#                     )

# keker.load(checkdir + '/' + 'check.best.h5')

keker.predict(savepath="predicts")
