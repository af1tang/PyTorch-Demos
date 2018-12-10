#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:48:39 2018

@author: af1tang
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision #toy datasets

import numpy as np
import matplotlib.pyplot as plt

from inhouse.pytorch.models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch, model, data_loader, optimizer, criterion, log_interval = 200, mode = "lstm"):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        
        #resizing
        if mode == 'lstm':
            data.resize_(data.size(0),data.size(1)*data.size(3), data.size(2))
            target.resize_(data.size(0))
        elif mode == 'mlp':
            dim = int(data.numel() / data.size(0))
            data.resize_(data.size(0), dim)
            target.resize_(data.size(0))
        
        #reset gradients to zero
        optimizer.zero_grad()
        
        #forward prop
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  #get index of max log-probability 
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        
        #backprop
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        #logging
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    
    print('\nTrain Epoch: {} \tAccuracy: {:.6f}'.format(
                epoch, 100. * correct / len(data_loader.dataset)))

def test(epoch, model, data_loader, optimizer, critierion, cuda=False, mode = 'lstm'):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
        
        #resizing
        if mode == 'lstm':
            data.resize_(data.size(0),data.size(1)*data.size(3), data.size(2))
            target.resize_(data.size(0))
        elif mode == 'mlp':
            dim = int(data.numel() / data.size(0))
            data.resize_(data.size(0), dim)
            target.resize_(data.size(0))
        
        #reset gradients to zero
        optimizer.zero_grad()
        
        #inference
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] #argmax => index of class
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    print('Test Epoch: {} \tAccuracy: {:.6f}\n'.format(
                epoch, 100. * correct / len(data_loader.dataset)))

def get_data(batch_size=128):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../', train=False, download=True,
                    transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def pipeline(nepochs = 1000, input_size = 784, output_size = 10, 
             weight_size = 32, dropout = .5, mode = "lstm", nlayers = 2,
             learning_rate = 1e-3, momentum = .9, batch_size=128,
             criterion = nn.CrossEntropyLoss()):
    
    #model handling
    if mode == "rnn":
        model = LSTM(input_size, weight_size, output_size, nlayers, dropout)
    elif mode == 'cnn':
        model = CNN()
    else:
        model = MLP(input_size, output_size, weight_size, dropout)
    #initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
    #get data generators
    train_loader, test_loader = get_data(batch_size)
    #training and testing
    for epoch in range(nepochs):
        train(epoch, model, train_loader, optimizer, criterion)
        test(epoch, model, test_loader, optimizer, criterion)
    return model