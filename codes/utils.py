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
from torch.utils.data import Dataset,DataLoader
import math
plt.switch_backend('agg')

def clip_eta(eta, norm, eps, is_gpu = True, DEVICE = torch.device('cuda:0')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12)
        eps = torch.tensor(eps)
        one = torch.tensor(1.0)
        if is_gpu:
            avoid_zero_div = avoid_zero_div.to(DEVICE)
            eps = eps.to(DEVICE)
            one = one.to(DEVICE)
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p = norm, dim = -1, keepdim = False)
            #print(normalize.size())
            #print(eps.size())
            normalize = torch.max(normalize, avoid_zero_div)
            #normalize = torch.unsqueeze()
            normalize.unsqueeze_(dim = -1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)
            #print(normalize.size())
            factor = torch.min(one, eps / normalize)
            factor = eps / normalize
            #print('eta', eta.size(), factor.size())

            eta = eta * factor

    return eta

class IPGD(object):

    _mean = torch.tensor(np.array([0.4914, 0.4822, 0.4465]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    _var = torch.tensor(np.array([0.2023, 0.1994, 0.2010]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 / 255.0, sigma = 3 / 255.0, nb_iter = 20, norm = np.inf, DEVICE = torch.device('cuda:0')):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = self._mean.to(DEVICE)
        self._var = self._var.to(DEVICE)

    def single_attck(self, net, inp, label, eta, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        #inp.retain_grad()
        adv_inp = inp + eta
        #adv_inp.retain_grad()
        #print("  eq", adv_inp.requires_grad)
        #adv_inp.requires_grad = True
        net.zero_grad()

        pred = net(adv_inp)
        #loss.backward()
        if target is not None:
            targets = torch.sum(pred[:, target])
            #print(targets.size())
            grad_sign = torch.autograd.grad(targets, adv_inp, only_inputs=True, retain_graph = False)[0].sign()

            #print(grad_sign.size(), torch.sum(grad_sign))
            #grad_sign = grad_sign.sign()
        else:
            loss = self.criterion(pred, label)
            #grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs = False)[0].sign()
            grad_sign = torch.autograd.grad(loss, adv_inp,only_inputs=True, retain_graph = False)[0].sign()
            #grad_sign = torch.autograd.backward(adv_inp, loss)
            #loss.backward()
            ##grad_sign = adv_inp.grads.sign()
            #print(np.sum(grad_sign))
            #print(torch.sum(grad_sign))
        #print(loss.item())

        #print("dd", grad_sign)
        #grad_sign = adv_inp.grad.sign()

        adv_inp = adv_inp + grad_sign * (self.sigma / self._var)
        tmp_adv_inp = adv_inp * self._var +  self._mean

        tmp_inp = inp * self._var + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._var
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._var

        return eta

    def attack(self, net, inp, label, target = None):

        eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attck(net, inp, label, eta, target)
            #print(i)

        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._var +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._var

        return adv_inp

    def get_batch_accuracy(self, net, inp, label):

        adv_inp = self.attack(net, inp ,label)

        pred = net(adv_inp)

        accuracy = torch_accuracy(pred, label, (1, ))[0].item()


        return accuracy


class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0
def torch_accuracy(output, target, topk = (1, )):
    '''
    param output, target: should be torch Variable
    '''
    #assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    #assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    #print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim = True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans

class AvgMeter(object):
    name = 'No name'
    def __init__(self, name = 'No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0
    def update(self, mean_var, count = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num
        

class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0
if __name__=="__main__":
    PgdAttack = IPGD(eps = 8.0/255,sigma = 4.0 / 255,nb_iter = 20, norm = np.inf,DEVICE = device)
    adv_inp=PgdAttack.attack(net,inputs,targets)