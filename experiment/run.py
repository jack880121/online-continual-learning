import time
import numpy as np
#from continuum.data_utils import setup_test_loader
from utils.name_match import agents
from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from experiment.tune_hyperparam import tune_hyper
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import pandas as pd
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from tensorboardX import SummaryWriter
import random

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
    train_loader = data.DataLoader(train_set, batch_size=params.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
    
    testpath = "/tf/online-continual-learning/datasets/20220331/test"
    test_set = torchvision.datasets.ImageFolder(testpath, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(test_set))
    print(test_set.class_to_idx)
    test_loader = data.DataLoader(test_set, batch_size=params.test_batch, shuffle=True, num_workers=0)
    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultA_ep10')
    
    start = time.time()
    
    for ep in range(params.epoch):
        loss = agent.train_learner_A(train_loader)
        writer.add_scalar('Train Loss', loss, ep)
        
        train_accuracy,train_recall,train_precision = agent.evaluate(train_loader)
        test_accuracy,test_recall,test_precision = agent.evaluate(test_loader)
        writer.add_scalars('accuracy', {'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}, ep)
        writer.add_scalars('recall', {'train_recall':train_recall,'test_recall':test_recall}, ep)
        writer.add_scalars('precision', {'train_precision':train_precision,'test_precision':test_precision}, ep)
        print("train_accuracy {}----train_recall {}----train_precision {}".format(train_accuracy,train_recall,train_precision))
        print("test_accuracy {}----test_recall {}----test_precision {}".format(test_accuracy,test_recall,test_precision))
        
        writer.add_scalar('epoch', ep, ep)
        
    end = time.time()
    print(end-start)
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
    train_loader_for_test = data.DataLoader(train_set, batch_size=params.batch, shuffle=True, num_workers=0,
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
    test_loader = data.DataLoader(test_set, batch_size=params.test_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultB_t')
    
    totaltime = 0
    for run in range(params.num_runs):
        train_set_spilit = data.Subset(train_set,index[(len(train_set)//10)*run:(len(train_set)//10)*(run+1)])
        print('train_set_spilit',len(train_set_spilit))
        train_loader = data.DataLoader(train_set_spilit, batch_size=params.batch, shuffle=True, num_workers=0,
                                           drop_last=True)
        start = time.time()
        agent.train_learner_B(train_loader,run,writer)
        end2 = time.time()
        print('traintime',end2-start)
        
        train_accuracy,train_recall,train_precision = agent.evaluate(train_loader_for_test)
        test_accuracy,test_recall,test_precision = agent.evaluate(test_loader)
        writer.add_scalars('accuracy', {'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}, run)
        writer.add_scalars('recall', {'train_recall':train_recall,'test_recall':test_recall}, run)
        writer.add_scalars('precision', {'train_precision':train_precision,'test_precision':test_precision}, run)
        print("train_accuracy {}----train_recall {}----train_precision {}".format(train_accuracy,train_recall,train_precision))
        print("test_accuracy {}----test_recall {}----test_precision {}".format(test_accuracy,test_recall,test_precision))
     
        end = time.time()
        print(end-start)
        writer.add_scalar('time', end-start, run)
        totaltime += end-start
        
    print('totaltime',totaltime)
