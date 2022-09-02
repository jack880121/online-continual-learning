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
    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultA_ep1')
    
    start = time.time()
    
    for ep in range(params.epoch):
        loss = agent.train_learner_A(train_loader)
        writer.add_scalar('Training Loss', loss, ep)
        
        accuracy,recall,precision,testloss = agent.evaluate(test_loader)
        writer.add_scalar('accuracy', accuracy, ep)
        writer.add_scalar('recall', recall, ep)
        writer.add_scalar('precision', precision, ep)
        writer.add_scalar('Testing Loss', testloss, ep)
        print("accuracy {}----recall {}----precision {}".format(accuracy,recall,precision))
        
        writer.add_scalar('epoch', ep, ep)
        
    end = time.time()
    print(end-start)
    
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
    
    writer = SummaryWriter('/tf/online-continual-learning/result/resultB_ep1')
    
    start = time.time()
    
    for run in range(params.num_runs):
        agent.train_learner_B(train_loader,run)
        accuracy,recall,precision,testloss = agent.evaluate(test_loader)
        writer.add_scalar('accuracy', accuracy, run)
        writer.add_scalar('recall', recall, run)
        writer.add_scalar('precision', precision, run)
        writer.add_scalar('Testing Loss', testloss, run)
        print("accuracy {}----recall {}----precision {}".format(accuracy,recall,precision))
     
    end = time.time()
    print(end-start)
