import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import natsort
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import pickle
from sklearn.model_selection import train_test_split
from wild_dataset import wildSet
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from attention_model import ResNet_AT
from attention_model import *
import matplotlib.pyplot as plt
from torchsummary import summary
from warnings import filterwarnings
from resnet_ft import *
from senet import *

filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.set_device(1)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def get_loaders_test(dataframe, transform):
    test_set = wildSet(dataframe.Files, dataframe.labels, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=30)
    dataloaders = {"test": test_loader}
    return dataloaders


def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    return _structure


def test(test_loader, model, folder):
    txt_file = '/var/storage/cube-data/others/preds/' + folder + '.txt'
    running_corrects = 0
    model.eval()
    preds_ = []
    model.to(device)
    confusion_matrix = torch.zeros(7, 7)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for iter_, data in enumerate(tqdm(test_loader['test'])):
            X_train = torch.as_tensor(data[0], dtype=torch.float, device=device)
            outputs = model(X_train)
            _, preds = torch.max(outputs, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
    lines = 'Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise'
    with open(txt_file, 'w') as op_file:
        op_file.write(lines + "\n")
        for item in predlist:
            op_file.write("%s\n" % item.item())

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    files_path = '/var/storage/cube-data/others/Tar/cropped_aligned/'
    testing_og_path = '/var/storage/cube-data/others/test_set_Expr_Challenge.txt'
    with open(testing_og_path, 'r', encoding='utf-8-sig') as tog:
        testing_og_folders = tog.read().split()

    device = torch.device("cuda:1")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=112),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)

    path = '/var/storage/cube-data/others/checkpoint_resnet_reduced_set_87_4.pt'
    model = model_parameters(model, path)

    for each_testing_folder in testing_og_folders:
        df_test = pd.DataFrame()
        folder_path = os.path.join(files_path, each_testing_folder)
        files_in_folder = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        files_in_folder = [os.path.join(folder_path, i) for i in files_in_folder]
        df_test['Files'] = files_in_folder
        df_test['labels'] = np.ones(len(files_in_folder))
        test_loader = get_loaders_test(df_test, test_transform)
        test(test_loader, model, each_testing_folder)
        # break
