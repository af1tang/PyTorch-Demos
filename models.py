#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:28:19 2018

@author: af1tang
"""

import torch
import torch.nn as nn
from inhouse.pytorch.layers import *

class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_weights, dropout):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_input, n_weights)
        self.linear2 = nn.Linear(n_weights, n_weights)
        self.linear3 = nn.Linear(n_weights, n_output)
        self.dropout = nn.Dropout(dropout)        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        h = self.relu(self.linear1(x))
        h = self.relu(self.linear2(h))
        h = self.relu(self.linear2(h))
        h = self.relu(self.linear2(h))
        h = self.dropout(h)
        yhat = self.linear3(h)
        return yhat


class RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, layers, dropout, return_sequence = False):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(n_input, n_hidden, layers, nonlinearity = 'tanh',
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.linear = nn.Linear(n_hidden, n_output)
        self.return_sequence = return_sequence
    
    def forward(self, x): 
        #x here is a FULL sequence.
        outputs, hidden = self.rnn(x)
        if self.return_sequence:
            yhat = self.linear(outputs)
        else:
            yhat = self.linear(outputs[:, -1, :])
        return yhat

class LSTM(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, layers, dropout, return_sequence = False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, layers, 
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.linear = nn.Linear(n_hidden, n_output)
        self.return_sequence = return_sequence
    
    def forward(self, x):
        outputs, hidden = self.lstm(x)
        if self.return_sequence:
            yhat = self.linear(outputs)
        else:
            yhat = self.linear(outputs[:, -1, :])
        return yhat

class EncoderRNN(nn.Module):
    def __init__(self, n_input, n_hidden, layers, dropout, mode = 'gru'):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(n_input, n_hidden, layers, 
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.gru = nn.GRU(n_input, n_hidden, layers, 
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.rnn = nn.RNN(n_input, n_hidden, layers, nonlinearity = 'tanh',
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.mode = mode

    
    def forward(self, x):
        if self.mode == 'lstm':
            outputs, hidden = self.lstm(x)
        elif self.mode == 'gru':
            outputs, hidden = self.gru(x)
        else:
            outputs, hidden = self.rnn(x)
        return outputs, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, attn_model, n_hidden, n_output, layers=1, dropout=.1, mode = "gru", soft=True):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(n_hidden, n_hidden, layers, 
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.lstm = nn.LSTM(n_hidden, n_hidden, layers, 
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.rnn = nn.RNN(n_hidden, n_hidden, layers, nonlinearity = 'tanh',
                          batch_first = True, dropout = (0 if layers ==1 else dropout))
        self.decoder = nn.Linear(n_hidden, n_output)
        #self.attn = Attn(attn_model, n_hidden)
        #self.concat = nn.Linear(n_hidden*2, n_hidden)
        self.softmax = nn.LogSoftmax(dim=1)
        self.soft = soft
        self.mode = mode
    
    def forward(self, x0, ht):
        #x0 is the <EOS> token!
        #yt outputs the 
        if self.mode =='gru':
            outputs, hidden = self.gru(x0, ht)
        elif self.mode =='lstm':
            outputs, hidden = self.lstm(x0, ht)
        else:
            outputs, hidden = self.rnn(x0, ht)
        if self.soft:
            yt = self.softmax(self.decoder(outputs[:, -1, :]))
        else:
            yt = self.decoder(outputs[:, -1, :]) 
        return yt
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.maxpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        yhat = self.fc3(out)
        return yhat
