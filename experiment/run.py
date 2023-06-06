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
    writer = SummaryWriter('/tf/online-continual-learning/result/resultA_t')
    
    start = time.time()
    
    for ep in range(params.epoch):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        loss = agent.train_learner_A(train_loader)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        writer.add_scalar('stage1_train_loss', loss, ep)
        
        train_accuracy,train_recall,train_precision,t1 = agent.evaluate(train_loader_for_test)
        t(t1)
        print(t1/93056*1000,'ms')
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
           
        test_accuracy,test_recall,test_precision,t2 = agent.evaluate(test_loader)
        t(t2)
        print(t2/22400*1000,'ms')
        
#         writer.add_scalar('testtime_ncm_proj', t2/22400*1000, ep)
        writer.add_scalar('testtime_ncm_original', t2/22400*1000, ep)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        print("train_accuracy_ncm {}----train_recall_ncm {}----train_precision_ncm {}".format(train_accuracy,train_recall,train_precision))
        print("test_accuracy_ncm {}----test_recall_ncm {}----test_precision_ncm {}".format(test_accuracy,test_recall,test_precision)) 
        
        #linear classifier
        tra,trr,trp,tea,ter,tep = agent.classifier(train_loader_for_test,test_loader,writer,ep+5)
        
        print("train_accuracy_linear {}----train_recall_linear {}----train_precision_linear {}".format(tra,trr,trp))
        print("test_accuracy_linear {}----test_recall_linear {}----test_precision_linear {}".format(tea,ter,tep))
        
        writer.add_scalars('accuracy', {'train_accuracy_ncm':train_accuracy,'test_accuracy_ncm':test_accuracy,'train_accuracy_linear':tra,'test_accuracy_linear':tea}, ep)
        writer.add_scalars('recall', {'train_recall_ncm':train_recall,'test_recall_ncm':test_recall,'train_recall_linear':trr,'test_recall_linear':ter}, ep)
        writer.add_scalars('precision', {'train_precision_ncm':train_precision,'test_precision_ncm':test_precision,'train_precision_linear':trp,'test_precision_linear':tep}, ep)
        

#         writer.add_scalars('accuracy', {'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}, ep)
#         writer.add_scalars('recall', {'train_recall':train_recall,'test_recall':test_recall}, ep)
#         writer.add_scalars('precision', {'train_precision':train_precision,'test_precision':test_precision}, ep)
#         print("train_accuracy {}----train_recall {}----train_precision {}".format(train_accuracy,train_recall,train_precision))
#         print("test_accuracy {}----test_recall {}----test_precision {}".format(test_accuracy,test_recall,test_precision))
   
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
        end = time.time()
        print('traintime')
        t(int(end-start))
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
#         train_accuracy,train_recall,train_precision,t1 = agent.evaluate(train_loader_for_test)
#         t(t1)
#         print(t1/93056*1000,'ms')
        
#         if torch.cuda.is_available():
#             torch.backends.cudnn.benchmark = False
           
#         test_accuracy,test_recall,test_precision,t2 = agent.evaluate(test_loader)
#         t(t2)
#         print(t2/22400*1000,'ms')
        
#         writer.add_scalar('testtime_ncm_proj', t2/22400*1000, run)
#         writer.add_scalar('testtime_ncm_original', t2/22400*1000, run)

        
#         if torch.cuda.is_available():
#             torch.backends.cudnn.benchmark = False
            
#         print("train_accuracy_ncm {}----train_recall_ncm {}----train_precision_ncm {}".format(train_accuracy,train_recall,train_precision))
#         print("test_accuracy_ncm {}----test_recall_ncm {}----test_precision_ncm {}".format(test_accuracy,test_recall,test_precision))   
              
#         #other classifier    
#         tra,trr,trp,tea,ter,tep = agent.classifier(train_loader_for_test,test_loader,writer,run)
        
#         print("train_accuracy_conv {}----train_recall_conv {}----train_precision_conv {}".format(tra,trr,trp))
#         print("test_accuracy_conv {}----test_recall_conv {}----test_precision_conv {}".format(tea,ter,tep))
              
#         writer.add_scalars('accuracy', {'train_accuracy_ncm':train_accuracy,'test_accuracy_ncm':test_accuracy,'train_accuracy_conv':tra,'test_accuracy_conv':tea}, run)
#         writer.add_scalars('recall', {'train_recall_ncm':train_recall,'test_recall_ncm':test_recall,'train_recall_conv':trr,'test_recall_conv':ter}, run)
#         writer.add_scalars('precision', {'train_precision_ncm':train_precision,'test_precision_ncm':test_precision,'train_precision_conv':trp,'test_precision_conv':tep}, run)
        
#         writer.add_scalars('accuracy', {'train_accuracy_ncm':train_accuracy,'test_accuracy_ncm':test_accuracy}, run)
#         writer.add_scalars('recall', {'train_recall_ncm':train_recall,'test_recall_ncm':test_recall}, run)
#         writer.add_scalars('precision', {'train_precision_ncm':train_precision,'test_precision_ncm':test_precision}, run)

        #new classifier    
        tra,trr,trp,tea,ter,tep = agent.classifier(train_loader_for_test,test_loader,writer)
        
        print("train_accuracy_conv {}----train_recall_conv {}----train_precision_conv {}".format(tra,trr,trp))
        print("test_accuracy_conv {}----test_recall_conv {}----test_precision_conv {}".format(tea,ter,tep))
              
        writer.add_scalars('accuracy', {'train_accuracy_conv':tra,'test_accuracy_conv':tea}, run)
        writer.add_scalars('recall', {'train_recall_conv':trr,'test_recall_conv':ter}, run)
        writer.add_scalars('precision', {'train_precision_conv':trp,'test_precision_conv':tep}, run)
        
        end2 = time.time()
        t(int(end2-start))
        writer.add_scalar('time', end2-start, run)
        totaltime += end2-start
        
    print('totaltime')
    t(int(totaltime))
