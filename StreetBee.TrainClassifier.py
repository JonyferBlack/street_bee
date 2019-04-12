"""
#%%
from google.colab import drive
drive.mount('/content/drive/')
os.chdir('/content/drive/My Drive/Colab Notebooks')
get_ipython().system('ls')
os.chdir('StreetBee')
get_ipython().system('ls')
"""

#%% [markdown]
# # StreetBee. Training classification
# ### by AIvanov
# 
# 2) Training set https://gist.github.com/donbobka/9d368b5161351fe4643a3676f0034a0c


#%%
import os
from IPython.display import Image
Image(filename=os.path.join('.', 'logo.jpg'))


#%%
import numpy as np
import cv2
import uuid
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from shutil import copyfile
import tarfile
import random
import matplotlib.pyplot as plt
import torch.nn as nn


#%%
debug = False
torch.cuda.empty_cache()
torch.cuda.device_count()


#%%
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cv2.setRNGSeed(seed)

#%% [markdown]
# ## Define model frome zoo and training ops
#%% [markdown]
# ### Setup hyper-parameters

#%%
data_dir = os.path.join('.', 'data_set')
contents = os.listdir(os.path.join(data_dir, 'val'))
# Number of classes
num_classes = len(contents)
# Batch size for training 64 - based on Google Benchmarks for K80
batch_size = 10 # 8 is for my GPU with 2Gb RAM
# Number of epochs to train for
num_epochs = 8
# learning rate
learning_rate=0.001
#momentum
momentum=0.9


#%%
# Detect if we have a GPU available
gpu_available = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu_available else "cpu")
device

#%% [markdown]
# ### Training and evaluating
#%% [markdown]
# #### Model default location and filename

#%%
model_dir = 'model'
model_st_path = os.path.join('.', model_dir, 'state_resNet50.pth')
model_path = os.path.join('.', model_dir, 'resNet50.pth')
model_ft = None

#%% [markdown]
# ### Load pretrained model

#%%
model_ft = torch.load(model_path)


#%%
model_ft.load_state_dict(torch.load(model_st_path))

#%% [markdown]
# #### Initialize model with desired output configuration

#%%
from training import initialize_model
model_ft, input_size = initialize_model(num_classes, model_ft, use_pretrained=True)
model_ft.eval()

#%% [markdown]
# ### Init loaders

#%%
from dataloaders import init_data_loaders


#%%
image_datasets, dataloaders_dict = init_data_loaders(input_size, data_dir, batch_size)


#%%
dataloaders_dict

#%% [markdown]
# ## Visualize data set, train model and visualize training

#%%
plt.rcParams['figure.figsize'] = [4, 4]

class_names = image_datasets['train'].classes

from confmetrics import norm

for inputs, classes in dataloaders_dict['train']:
    print(inputs.shape)
    sample = norm(inputs[0]).numpy().transpose(1, 2, 0)
    plt.imshow(sample)
    break

#%% [markdown]
# ### Train model

#%%
from training import train_model


#%%
torch.cuda.empty_cache()

#%% [markdown]
# ## Train

#%%
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()

if debug:
    print("Params to learn:")
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs)

#%% [markdown]
# ### Save best checkpoint

#%%
torch.save(model_ft.state_dict(), model_st_path)
torch.save(model_ft, model_path)

#%% [markdown]
# ### Plot training

#%%
# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method
ohist = []
ohist = [h.cpu().numpy() for h in hist]
plt.rcParams['figure.figsize'] = [15, 7]
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

#%% [markdown]
# ## Visualize test inference


#%%
from utils import visualize_model
visualize_model(model_ft, dataloaders_dict, device)


