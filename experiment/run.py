import time
import numpy as np
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
import pandas as pd
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from tensorboardX import SummaryWriter
import random

def t(a):
    h = a//3600
    m = (a-h*3600)//60
    s = a-3600*h-60*m
    print(str(h)+'h'+str(m)+'m'+str(s)+'s')
    
def method_A(params, store=False, save_path=None):
    model = setup_architecture(params)
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    
    trainpath = "/tf/online-continual-learning/datasets/20220331/train"
    train_set = torchvision.datasets.ImageFolder(trainpath, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(train_set))
    print(train_set.class_to_idx)
    train_loader = data.DataLoader(train_set, batch_size=params.batch, shuffle=True, num_workers=2,
                                       drop_last=True)
    
    train_loader_for_test = data.DataLoader(train_set, batch_size=params.test_batch, shuffle=True, num_workers=2,
                                       drop_last=True)
    
    testpath = "/tf/online-continual-learning/datasets/20220331/test"
    test_set = torchvision.datasets.ImageFolder(testpath, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(test_set))
    print(test_set.class_to_idx)
    test_loader = data.DataLoader(test_set, batch_size=params.test_batch, shuffle=True, num_workers=2,
                                       drop_last=True)    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultA_ep5_linear')
    
    start = time.time()
    
    for ep in range(params.epoch):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        loss = agent.train_learner_A(train_loader)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        writer.add_scalar('Train Loss', loss, ep)
        
        train_accuracy,train_recall,train_precision = agent.evaluate(train_loader_for_test)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        test_accuracy,test_recall,test_precision = agent.evaluate(test_loader)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        tra,trr,trp,tea,ter,tep = agent.classifier(train_loader_for_test,test_loader)
        
        writer.add_scalars('accuracy', {'train_accuracy':train_accuracy,'test_accuracy':test_accuracy,'train_accuracy_classifier':tra,'test_accuracy_classifier':tea}, ep)
        writer.add_scalars('recall', {'train_recall':train_recall,'test_recall':test_recall,'train_recall_classifier':trr,'test_recall_classifier':ter}, ep)
        writer.add_scalars('precision', {'train_precision':train_precision,'test_precision':test_precision,'train_precision_classifier':trp,'test_precision_classifier':tep}, ep)
        print("train_accuracy {}----train_recall {}----train_precision {}".format(train_accuracy,train_recall,train_precision))
        print("test_accuracy {}----test_recall {}----test_precision {}".format(test_accuracy,test_recall,test_precision))
        print("train_accuracy_classifier {}----train_recall_classifier {}----train_precision_classifier {}".format(tra,trr,trp))
        print("test_accuracy_classifier {}----test_recall_classifier {}----test_precision_classifier {}".format(tea,ter,tep))
        
        writer.add_scalar('epoch', ep, ep)
        
    end = time.time()
    t(int(end-start))
    writer.add_scalar('time', end-start, 1)
    
def method_B(params, store=False, save_path=None):
    model = setup_architecture(params)
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    
    trainpath = "/tf/online-continual-learning/datasets/20220331/train"                     
    train_set = torchvision.datasets.ImageFolder(trainpath, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(train_set))
    print(train_set.class_to_idx)
    train_loader_for_test = data.DataLoader(train_set, batch_size=params.test_batch, shuffle=True, num_workers=2,
                                       drop_last=True)
    
    index = [i for i in range(len(train_set))] 
    random.shuffle(index)
    
    
    testpath = "/tf/online-continual-learning/datasets/20220331/test"
    test_set = torchvision.datasets.ImageFolder(testpath, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(test_set))
    print(test_set.class_to_idx)
    test_loader = data.DataLoader(test_set, batch_size=params.test_batch, shuffle=True, num_workers=2,
                                       drop_last=True)
    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultB_t')
    
    totaltime = 0
    for run in range(params.num_runs):
        print('run',run+1)
        train_set_spilit = data.Subset(train_set,index[(len(train_set)//10)*run:(len(train_set)//10)*(run+1)])
        print('train_set_spilit',len(train_set_spilit))
        train_loader = data.DataLoader(train_set_spilit, batch_size=params.batch, shuffle=True, num_workers=2,
                                           drop_last=True)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        start = time.time()
        agent.train_learner_B(train_loader,run,writer)
        end2 = time.time()
        print('traintime')
        t(int(end2-start))
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        train_accuracy,train_recall,train_precision = agent.evaluate(train_loader_for_test)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        test_accuracy,test_recall,test_precision = agent.evaluate(test_loader)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        tra,trr,trp,tea,ter,tep = agent.classifier(train_loader_for_test,test_loader)
        
        writer.add_scalars('accuracy', {'train_accuracy':train_accuracy,'test_accuracy':test_accuracy,'train_accuracy_classifier':tra,'test_accuracy_classifier':tea}, run)
        writer.add_scalars('recall', {'train_recall':train_recall,'test_recall':test_recall,'train_recall_classifier':trr,'test_recall_classifier':ter}, run)
        writer.add_scalars('precision', {'train_precision':train_precision,'test_precision':test_precision,'train_precision_classifier':trp,'test_precision_classifier':tep}, run)
        print("train_accuracy {}----train_recall {}----train_precision {}".format(train_accuracy,train_recall,train_precision))
        print("test_accuracy {}----test_recall {}----test_precision {}".format(test_accuracy,test_recall,test_precision))
        print("train_accuracy_classifier {}----train_recall_classifier {}----train_precision_classifier {}".format(tra,trr,trp))
        print("test_accuracy_classifier {}----test_recall_classifier {}----test_precision_classifier {}".format(tea,ter,tep))
     
        end = time.time()
        t(int(end-start))
        writer.add_scalar('time', end-start, run)
        totaltime += end-start
        
    print('totaltime')
    t(int(totaltime))
