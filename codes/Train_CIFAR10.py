'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd.gradcheck import zero_gradients
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from wide_resnet import *
from preact_resnet import *
#from utils_c import progress_bar
from utils import *
from torch.utils.data import Dataset,DataLoader
from utils import IPGD,TrainClock,torch_accuracy,AvgMeter,TrainClock
#from preact_resnet import *


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy 
best_clean_acc=0.0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True,num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG('VGG19')
net = PreActResNet18().cuda()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_resnet18_robust_ad_FGSMR_step8.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = 0
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def Regularizer_FGSM(net,inputs,targets,criterion,lambda_=4,epsilon=8/255.0):
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

# Training
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
    
def train(epoch,lam=0.5,epsilon=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    entropy_loss_all = 0.0
    sharingnodes_loss_all=0.0
    correct_clean = 0.0
    n=0.0
    all_regularier=0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        delta=torch.zeros(inputs.size()).to(device)
        delta.requires_grad = True
        delta.data,regularizer=Regularizer_FGSM(net,inputs,targets,nn.CrossEntropyLoss(),lambda_=4,epsilon=epsilon)       
        delta=clamp(delta,-epsilon,epsilon)       
        delta=delta.detach()
        net.train()
        outputs =net(clamp(inputs+delta,lower_limit,upper_limit))        
        optimizer.zero_grad()        
        loss_entropy = criterion(outputs, targets)
        loss=loss_entropy+regularizer*lam  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_regularier+=regularizer.item()*inputs.size(0)
        entropy_loss_all += loss_entropy.item()*inputs.size(0)       
        _,predicted_clean=outputs.max(1)
        correct_clean+=predicted_clean.eq(targets).sum().item()
        n+=targets.size(0)
    print()
    entropy_loss_all=entropy_loss_all/n
    correct_clean=correct_clean/n
    all_regularier=all_regularier/n
    
    print("entropy loss:{:.4f}  Acc:{:.4f} regularizer:{:.4f}".format(entropy_loss_all,correct_clean,all_regularier))


def test(epoch):
    global best_acc
    global best_clean_acc
    net.eval()
    test_loss_entropy = 0
    sharingnodes_loss_all=0.0
    correct_clean = 0
    correct_ad=0
    total = 0
    PgdAttack = IPGD(eps = 8.0/255,sigma = 4.0 / 255,nb_iter = 20, norm = np.inf,DEVICE = device)
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs.requires_grad=True
        
        outputs = net(inputs)
        loss_entropy = criterion(outputs, targets)        
        test_loss_entropy += loss_entropy.item()*inputs.size(0)
        
        adv_inp=PgdAttack.attack(net,inputs,targets)
        outputs_ad=net(adv_inp)
        
        _, predicted_ad = outputs_ad.max(1)
        correct_ad += predicted_ad.eq(targets).sum().item()

        _,predicted_clean=outputs.max(1)
        correct_clean+=predicted_clean.eq(targets).sum().item()

    print()
    
    # Save checkpoint.
    acc_adv_all = correct_ad/len(testset)
    acc_clean_all=correct_clean/len(testset)
    test_loss_entropy=test_loss_entropy/len(testset)
    print("loss_entropy:{:.4f} acc_adv:{:.4f} acc_clean:{:.4f}".format(test_loss_entropy,acc_adv_all,acc_clean_all))
    
    
    if acc_adv_all > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc_adv_all,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_resnet18_FGSMR_train.pth')
        best_acc = acc_adv_all
    if acc_clean_all > best_clean_acc:
        print('Saving clean acc..')
        state = {
            'net': net.state_dict(),
            'acc': acc_clean_all,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_resnet18_FGSMR_train_clean.pth')
        best_clean_acc = acc_clean_all
print("resnet resnet18 FGSMR train ")
print("lambda: 0.5,0.4,0.4-50,60,70")
print("epsilon: 8/255")
lam=0.5

tt=8.0
epsilon = (tt/ 255.) / std
for epoch in range(start_epoch, start_epoch+70):
    if (epoch-start_epoch)==50:
        lam=0.4
        for param_group in optimizer.param_groups:
            param_group["lr"] = 5e-4
    if (epoch-start_epoch)==60:
        lam=0.4
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-4
    train(epoch,lam=lam,epsilon=epsilon)
    test(epoch)
    