
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients
torch.manual_seed(0)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

model_cnn_robust = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)
def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    yp=model(X + delta)
    loss = nn.CrossEntropyLoss()(yp, y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()
# Training
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
import numpy as np

def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    
def Regularizer_FGSM(net,inputs,targets,criterion,lambda_=4,epsilon=0.1):
    delta = torch.zeros_like(inputs, requires_grad=True)
    inputs.requires_grad_()
    yp=net(inputs + delta)
    loss = nn.CrossEntropyLoss()(yp, targets)
    loss.backward(retain_graph=True)
    delta_pert=epsilon * delta.grad.detach().sign()
    net.zero_grad()
    outputs_pos=net(inputs+delta_pert)
    loss_pos=criterion(outputs_pos,targets)
    inputs.grad.data.zero_()
    grad_diff = torch.autograd.grad((loss_pos-loss), inputs, grad_outputs=torch.ones(targets.size()).to(device),create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    net.zero_grad()
    return delta_pert,torch.sum(lambda_*reg)/float(inputs.size(0))

def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    n=0.0
    regularizer=torch.tensor(0).float().cuda()
    for i,(X,y) in enumerate(loader):
        X,y = X.to(device), y.to(device)

        if opt:
            delta,regularizer=Regularizer_FGSM(model,X,y,nn.CrossEntropyLoss(),lambda_=4.0,**kwargs)
            X_pre_new=torch.clamp(X+delta,0,1)
            yp = model(X_pre_new)
            loss = nn.CrossEntropyLoss()(yp,y)+regularizer*0.2
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            delta = attack(model, X, y, **kwargs)
            X_pre_new=torch.clamp(X+delta,0,1)
            yp = model(X_pre_new)
            loss = nn.CrossEntropyLoss()(yp,y)
            
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        n+=X.shape[0]
    return total_err / n, total_loss / n
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
opt = optim.SGD(model_cnn_robust.parameters(), lr=1e-1)
print("epsilon=0.2,lambda=0.2")

for t in range(20):
    train_err, train_loss = epoch_adversarial(train_loader, model_cnn_robust, pgd_linf, opt,epsilon=0.1)
    test_err, test_loss = epoch(test_loader, model_cnn_robust)
    adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn_robust, pgd_linf)
    if t == 10:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-2
    print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
torch.save(model_cnn_robust.state_dict(), "./model_cnn_robust_FGSMR.pt")