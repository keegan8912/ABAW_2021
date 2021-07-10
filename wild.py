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

writer = SummaryWriter(log_dir='/var/storage/cube-data/others/runs/'+datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_loaders_test(dataframe, transform):
    test_set = wildSet(dataframe.Files, dataframe.labels, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=15)
    dataloaders = {"test": test_loader}
    return dataloaders

def get_loaders(dataframe, transform, val_transform):
    train_set, val_set = train_test_split(dataframe, test_size=0.2)
    training_data = wildSet(train_set.Files, train_set.labels, transform=transform)
    trainloader = DataLoader(training_data, batch_size=512, shuffle=True, num_workers=35)
    # class_sample_count = np.unique(training_data.labels, return_counts=True)[1]
    # weight = len(training_data.labels) / (class_sample_count)
    # weight = 1. / weight
    # weight = [1 - w for w  in weight]
    val_data = wildSet(val_set.Files, val_set.labels, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=35)
    # total_len = len(dataframe)
    # val_split = 0.2
    # indices = list(range(total_len))
    # val_len = int(np.floor(val_split * total_len))
    # val_idx = np.random.choice(indices, size=val_len, replace=False)
    # val_sampler = SubsetRandomSampler(val_idx)
    # train_idx = list(set(indices) - set(val_idx))
    # train_sampler = SubsetRandomSampler(train_idx)

    # weight = [1, 1.91, 3.5, 4.02, 1.15, 1.29, 1.08]
    weight = [1, 1.06, 1.75, 2, 1.15, 1.29, 1.08]
    dataloaders = {"train": trainloader, "val": val_loader}
    return dataloaders, np.array(weight)


def save_ckp(state, checkpoint_dir):
    # f_path = os.path.join(checkpoint_dir, 'checkpoint_' +
    #                       datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt')
    f_path = os.path.join(checkpoint_dir, 'checkpoint_resnet_reduced_set_87_4cont.pt')
    torch.save(state, f_path)


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
    # model = torch.nn.DataParallel(_structure).cuda()
    return _structure
    # return model

# def accuracy(outputs, labels):
#     classes = torch.argmax(outputs, dim=1)
#     return torch.mean((classes == labels).float())


def train(dataloader, params):
    torch.cuda.empty_cache()
    device = params['device']
    EPOCHS = params['epochs']
    learning_rate = params['learning_rate']
    weights = params['weights']
    weights = torch.from_numpy(weights)
    useFER = params['useFER']
    useFace = params['useFace']
    useSEN = params['useSEN']
    continue_ = params['continue']
    savedModel = params['savedModel']
    feature_extract = params['feature_extract']
    loss_function = nn.CrossEntropyLoss(weight=weights.to(device).float())
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = F.nll_loss(weight=weights.to(device).float())

    # model = models.resnext50_32x4d(pretrained=True)
    if not useFER:
        model = resnet18_at()
        # model = models.resnet18(pretrained=True) #works better
        # # model = models.resnet18(pretrained=False)
        # num_ftrs = model.fc.in_features
        # model.fc = torch.nn.Linear(num_ftrs, 7)
    if useFER:
        print('Using FER')
        model = models.resnet18(num_classes=7)
        # model = models.resnet18(pretrained=False)

        path = '/var/storage/cube-data/others/Code/Resnet18_FER+_pytorch.pth.tar'
        # model = load_model(model, path)
        model = model_parameters(model, path)
        print('Setting grads False')
        # set_parameter_requires_grad(model, True) #comment this for next run
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
    if useFace:
        print('Using Face')
        model = resnet50(num_classes=8631)
        path = '/var/storage/cube-data/others/Code/resnet50_ft_weight.pkl'
        with open(path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
        print(model)
    if useSEN:
        print('Using SEN')
        model = senet50(num_classes=8631)
        path = '/var/storage/cube-data/others/Code/senet50_ft_weight.pkl'
        with open(path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
        print(model)
#add continue option

    # model = Net()
    model.to(device)
    # print(summary(model, input_size=(3,112,112)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=4e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    phases = ['train', 'val']
    start_epoch = 0
    if continue_:
        checkpoint = torch.load(savedModel)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Continuing')
    for epoch in range(start_epoch, EPOCHS):
        lr_scheduler.step()
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        # model = model.to(device)
        if epoch == 5:
            print('OK, lets see')
        for phase in phases:
            running_loss = 0
            running_corrects = 0
            val_running_loss = 0
            for iter_, data in enumerate(tqdm(dataloader[phase])):
                if phase == 'train':
                    model.train()
                if phase == 'val':
                    model.eval()
                    #images for one batch write out with labels - saveimg of torch?
                X_train = torch.as_tensor(data[0], dtype=torch.float, device=device)
                # X_train = X_train.unsqueeze(-1)
                y_train = torch.as_tensor(data[1], dtype=torch.long, device=device)
                # if epoch == 0 or epoch == 10:
                #     plt.figure(figsize=(10,6))
                #     temp = X_train[-1,:,:,:].detach().cpu().numpy()
                #     plt.imshow(np.moveaxis(temp, 0,-1))
                #     plt.show()

                outputs = model(X_train)
                
                # loss = F.nll_loss(outputs, y_train, weight=weights.to(device).float())
                loss = loss_function(outputs, y_train)
                # for params in model.parameters():
                #     params.grad = None
                optimizer.zero_grad()
                if phase == 'train':
                    # loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                if phase == 'val':
                    with torch.no_grad():    
                        val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == y_train.data)

            if epoch == EPOCHS - 1:  # save last epoch details
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                            }
                save_ckp(checkpoint, '/var/storage/cube-data/others/')


                # running_corrects += torch.sum(preds == y_train.data)

            if phase == 'train':
                epoch_loss = running_loss / len(dataloaders['train'].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
                print('EP:{}/{} {} Loss: {:.6f} Acc: {:.6f}'.format(epoch, EPOCHS, phase,
                                                                    epoch_loss, epoch_acc))
            if phase == 'val':
                val_ep_loss = val_running_loss / len(dataloaders['val'].dataset)
                val_epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
                print('EP:{}/{} {} Loss: {:.6f} Acc: {:.6f}'.format(epoch, EPOCHS, phase,
                                                                    val_ep_loss, val_epoch_acc))


            if phase == 'train':
                writer.add_scalar('Train Run Loss', running_loss, epoch)
                writer.add_scalar('Ep Loss', epoch_loss, epoch)
                writer.add_scalar('Tr Acc', epoch_acc, epoch)
            if phase == 'val':
                writer.add_scalar('Val Run Loss', val_running_loss, epoch)
                writer.add_scalar('Ep Loss', val_ep_loss, epoch)
                writer.add_scalar('Val Acc', val_epoch_acc, epoch)

#check if test acc is being calculated properly ; run resnet50 also
#see if i can reduce some overfitting
def test(test_loader, model):
    running_corrects = 0
    model.eval()
    preds_ = []
    model.to(device)
    confusion_matrix = torch.zeros(7, 7)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for iter_, data in enumerate(tqdm(test_loader['test'])):
            X_train = torch.as_tensor(data[0], dtype=torch.float, device=device)
            # X_train = X_train.unsqueeze(-1)
            y_train = torch.as_tensor(data[1], dtype=torch.long, device=device)
            outputs = model(X_train)
            _, preds = torch.max(outputs, 1)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, y_train.view(-1).cpu()])

            running_corrects += torch.sum(preds == y_train.data)
        epoch_acc = running_corrects.double() / len(test_loader['test'].dataset)
        y_trains = test_loader['test'].dataset.labels
        print(f"Test ACC: {epoch_acc}")
        print(classification_report(lbllist.detach().cpu().numpy(), predlist.detach().cpu().numpy()))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    # with open("extnd_23.pickle", "rb") as input_file:
    with open("df_collection_clean3.pickle", "rb") as input_file:
        df2 = pickle.load(input_file)
        # df_reduced = pickle.load(input_file)
    cls_1_df = df2[df2.labels == 0].iloc[::5, :]
    cls_4_df = df2[df2.labels == 4].iloc[::4, :]
    cls_5_df = df2[df2.labels == 5].iloc[::3, :]

    df_reduced = df2[(df2.labels != 0) & (df2.labels != 4) & (df2.labels != 5)]
    df_reduced = pd.concat([df_reduced, cls_1_df, cls_4_df, cls_5_df])
    df_reduced = df_reduced.sample(frac=1).reset_index(drop=True)

    df2 = df_reduced

    df_3 = df2[df2.labels == 3]
    df2 = pd.concat([df2, df_3])  # taken twice

    df_2 = df2[df2.labels == 2]
    df2 = pd.concat([df2, df_2.sample(frac=1).reset_index(drop=True)])  #

    df_1 = df2[df2.labels == 1]
    df2 = pd.concat([df2, df_1.sample(frac=0.8).reset_index(drop=True)])
    df_reduced = df2
    #
    # with open("df2_test.pickle", "rb") as input_file:
    #     df_test = pickle.load(input_file)

    # with open("extnd.pickle", "rb") as input_file:
    #     df2 = pickle.load(input_file)
    device = torch.device("cuda:1")
    # 'weights': torch.tensor([
    #     1.0,
    #     9.43,
    #     17.41,
    #     20.29,
    #     1.44,
    #     2.16,
    #     5.40, ]).cuda()
    params = {
    'learning_rate': 2e-3,
    'epochs': 50,
    'device': device,
    'feature_extract': True,
    'useFER': True,
    'useFace': False,
    'useSEN': False,
    'continue': False,
    'savedModel': '/var/storage/cube-data/others/checkpoint_resnet_reduced_set_87_4.pt'
    }
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.Resize(size=112),
                            transforms.Grayscale(num_output_channels=3),
                            # transforms.RandomRotation(10),
                            # transforms.RandomResizedCrop(112),
                            # transforms.Normalize(
                            # mean=[0.485, 0.456, 0.406],
                            # std=[0.229, 0.224, 0.225]
                            #     ),
                            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                            #                        hue=0.1),
                            # transforms.RandomAffine(degrees=7, translate=(0.1,0.1)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=7, fill=(0,))
    ])
    val_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(size=112),
                            transforms.Grayscale(num_output_channels=3),
                            # transforms.Normalize(
                            # mean=[0.485, 0.456, 0.406],
                            # std=[0.229, 0.224, 0.225]
                            #     )
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=112),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataloaders, weight = get_loaders(df_reduced, transform, val_transform)
    params['weights'] = weight
    input_size = 112
    train(dataloaders, params)
    with open("df2_test.pickle", "rb") as input_file:
        df_test = pickle.load(input_file)
    test_loader = get_loaders_test(df_test, test_transform)


    if params['useFER']:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)

        path = '/var/storage/cube-data/others/checkpoint_resnet_reduced_set_87_4cont.pt'
        # path = '/var/storage/cube-data/others/Code/Resnet18_FER+_pytorch.pth.tar'
        # loaded_model = load_model(model, path)
        model = model_parameters(model, path)
        # set_parameter_requires_grad(model, True) #comment this for next run

    if params['useSEN']:
        path = '/var/storage/cube-data/others/senet_face_checkpoint_.pt'
        print('Using SEN')
        model = senet50(num_classes=8631)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)
        model = load_model(model, path)
        print(model)

    # else:
    #     model = resnet18_at()
    #     path = '/var/storage/cube-data/others/checkpoint_resnet_reduced_set_87_4.pt'
    #     # model = load_model(model, path)
    #     model = model_parameters(model, path)
    test(test_loader, model)
