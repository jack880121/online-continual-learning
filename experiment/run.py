import time
import numpy as np
from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader
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
        
    agent.train_learner_A(train_loader)
    accuracy,recall,precision = agent.evaluate(test_loader)
    print("accuracy {}----recall {}----precision {}".format(accuracy,recall,precision))

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
    
    writer = SummaryWriter('/tf/online-continual-learning/resultB_ep1_noaug')
    
    for run in range(params.num_runs):
        agent.train_learner_B(train_loader,run)
        accuracy,recall,precision = agent.evaluate(test_loader)
        writer.add_scalar('accuracy', accuracy, run)
        writer.add_scalar('recall', recall, run)
        writer.add_scalar('precision', precision, run)
        print("accuracy {}----recall {}----precision {}".format(accuracy,recall,precision))
    
    
'''
def method_A_test(params, store=False, save_path=None):
    model = setup_architecture(params)
    checkpoint = torch.load('/tf/online-continual-learning/model_state_dict3.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    
    path = "/tf/online-continual-learning/datasets/20220331/test"
    test_set = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.3333, 0.2919, 0.2369),(0.2554, 0.2322, 0.1586)) 
    ]))
    print(len(test_set))
    print(test_set.class_to_idx)
    test_loader = data.DataLoader(test_set, batch_size=params.test_batch, shuffle=True, num_workers=0)
    accuracy,recall,precision = agent.evaluate(test_loader)
    print("accuracy {}----recall {}----precision {}".format(accuracy,recall,precision))
'''

'''    
def multiple_run(params, store=False, save_path=None):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))
    
    accuracy_list = []
    model = setup_architecture(params)
#     checkpoint = torch.load('/tf/online-continual-learning/model_state_dict3.pt')
#     model.load_state_dict(checkpoint['model_state_dict'])
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    for run in range(params.num_runs):
        #tmp_acc = []
        #run_start = time.time()
        data_continuum.new_run(run)
        

        # prepare val data loader
        #test_loaders = setup_test_loader(data_continuum.test_data(), params)
        if params.online:
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                print("-----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                
#                 acc_array = agent.evaluate(test_loaders)
#                 tmp_acc.append(acc_array)
#             run_end = time.time()
#             print(
#                 "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
#                                                                                run_end - run_start))
#             accuracy_list.append(np.array(tmp_acc))
#         else:
#             x_train_offline = []
#             y_train_offline = []
#             for i, (x_train, y_train, labels) in enumerate(data_continuum):
#                 x_train_offline.append(x_train)
#                 y_train_offline.append(y_train)
#             print('Training Start')
#             x_train_offline = np.concatenate(x_train_offline, axis=0)
#             y_train_offline = np.concatenate(y_train_offline, axis=0)
#             print("----------run {} training-------------".format(run))
#             print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
#             agent.train_learner(x_train_offline, y_train_offline)
#             acc_array = agent.evaluate(test_loaders)
#             accuracy_list.append(acc_array)

#     accuracy_array = np.array(accuracy_list)
#     end = time.time()
#     if store:
#         result = {'time': end - start}
#         result['acc_array'] = accuracy_array
#         save_file = open(table_path + '/' + save_path, "wb")
#         pickle.dump(result, save_file)
#         save_file.close()
#     if params.online:
#         avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
#         print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
#         print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
#               .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
#     else:
#         print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
#         print("avg_end_acc {}".format(np.mean(accuracy_list)))
'''                

'''
def multiple_run_test(params, store=False, save_path=None):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    accuracy_list = []
    model = setup_architecture(params)
    checkpoint = torch.load('/tf/online-continual-learning/model_state_dict3.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    for run in range(params.num_runs):
        run_start = time.time()
        data_continuum.new_run(run)

        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        if params.online:
            accuracy,recall,precision = agent.evaluate(test_loaders)
            run_end = time.time()
            print("----run {}----accuracy {}----recall {}----precision {}-----test time {}".format(run, accuracy,recall,precision,run_end - run_start))
#             print(
#                 "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, acc_array,
#                                                                                run_end - run_start))

#     accuracy_array = np.array(acc_array)
#     end = time.time()
    
#     if params.online:
#         avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
#         print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
#         print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
#               .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
'''    

'''
def multiple_run(params, store=False, save_path=None):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = params.model_name + '_' + params.data_name + '.pkl'

    accuracy_list = []
    model = setup_architecture(params)
    model = maybe_cuda(model, params.cuda)
    opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
    agent = agents[params.agent](model, opt, params)
    for run in range(params.num_runs):
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run(run)
        

        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        if params.online:
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                print("-----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
            run_end = time.time()
            print(
                "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                               run_end - run_start))
            accuracy_list.append(np.array(tmp_acc))
        else:
            x_train_offline = []
            y_train_offline = []
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
            print('Training Start')
            x_train_offline = np.concatenate(x_train_offline, axis=0)
            y_train_offline = np.concatenate(y_train_offline, axis=0)
            print("----------run {} training-------------".format(run))
            print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
            agent.train_learner(x_train_offline, y_train_offline)
            acc_array = agent.evaluate(test_loaders)
            accuracy_list.append(acc_array)

    accuracy_array = np.array(accuracy_list)
    end = time.time()
    if store:
        result = {'time': end - start}
        result['acc_array'] = accuracy_array
        save_file = open(table_path + '/' + save_path, "wb")
        pickle.dump(result, save_file)
        save_file.close()
    if params.online:
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    else:
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print("avg_end_acc {}".format(np.mean(accuracy_list)))
'''

'''
def multiple_run_tune(defaul_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(defaul_params.data, defaul_params.cl_type, defaul_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    #store table
    # set up storing table
    table_path = load_yaml('config/global.yml', key='path')['tables']
    metric_list = ['Avg_End_Acc'] + ['Avg_End_Fgt'] + ['Time'] + ["Batch" + str(i) for i in range(defaul_params.num_val, data_continuum.task_nums)]
    param_list = list(tune_params.keys()) + metric_list
    table_columns = ['Run'] + param_list
    table_path = table_path + defaul_params.data
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = defaul_params.model_name + '_' + defaul_params.data_name + '.csv'
    df = pd.DataFrame(columns=table_columns)
    # store list
    accuracy_list = []
    params_keep = []
    for run in range(defaul_params.num_runs):
        tmp_acc = []
        tune_data = []
        run_start = time.time()
        data_continuum.new_run()
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), defaul_params)
        tune_test_loaders = test_loaders[:defaul_params.num_val]
        test_loaders = test_loaders[defaul_params.num_val:]
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < defaul_params.num_val:
                #collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == defaul_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, defaul_params, tune_params)
                    params_keep.append(best_params)
                    final_params = vars(defaul_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)

        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

        #store result
        result_dict = {'Run': run}
        result_dict.update(best_params)
        end_task_acc = tmp_acc[-1]
        for i in range(data_continuum.task_nums - defaul_params.num_val):
            result_dict["Batch" + str(i + defaul_params.num_val)] = end_task_acc[i]
        result_dict['Avg_End_Acc'] = np.mean(tmp_acc[-1])
        result_dict['Avg_End_Fgt'] = single_run_avg_end_fgt(np.array(tmp_acc))
        result_dict['Time'] = run_end - run_start
        df = df.append(result_dict, ignore_index=True)
        save_dataframe_csv(df, table_path, save_path)
    accuracy_list = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)
    end = time.time()
    final_result = {'Run': 'Final Result'}
    final_result['Avg_End_Acc'] = avg_end_acc
    final_result['Avg_End_Fgt'] = avg_end_fgt
    final_result['Time'] = end - start
    df = df.append(final_result, ignore_index=True)
    save_dataframe_csv(df, table_path, save_path)
    print('----------- Total {} run: {}s -----------'.format(defaul_params.num_runs, end - start))
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))



def multiple_run_tune_separate(default_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(default_params.data, default_params.cl_type, default_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if default_params.num_val == -1:
        # offline tuning
        default_params.num_val = data_continuum.data_object.task_nums
    #store table
    # set up storing table
    result_path = load_yaml('config/global.yml', key='path')['result']
    table_path = result_path + default_params.data + '/' + default_params.cl_type
    for i in default_params.trick:
        if default_params.trick[i]:
            trick_name = i
            table_path = result_path + default_params.data + '/' + default_params.cl_type + '/' + trick_name
            break
    print(table_path)
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = default_params.model_name + '_' + default_params.data_name + '_' + str(default_params.seed) + '.pkl'
    # store list
    accuracy_list = []
    params_keep = []
    if isinstance(default_params.num_runs, int):
        run_list = range(default_params.num_runs)
    else:
        run_list = default_params.num_runs
    for run in run_list:
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        if default_params.train_val:
            single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        else:
            single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

    end = time.time()
    accuracy_array = np.array(accuracy_list)
    result = {'seed': default_params.seed}
    result['time'] = end - start
    result['acc_array'] = accuracy_array
    result['ram'] = check_ram_usage()
    result['best_params'] = params_keep
    save_file = open(table_path + '/' + save_path, "wb")
    pickle.dump(result, save_file)
    save_file.close()
    print('----------- Total {} run: {}s -----------'.format(default_params.num_runs, end - start))
    print('----------- Seed {} RAM: {}s -----------'.format(default_params.seed, result['ram']))

def single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    test_loaders = test_loaders_full[default_params.num_val:]

    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            else:
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0),
                      np.concatenate(labels_offline, axis=0))]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders)
        tmp_acc.append(acc_array)



def single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    break

        data_continuum.reset_run()
        # set up
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            print("----------run {} training batch {}-------------".format(run, i))
            print('size: {}, {}'.format(x_train.shape, y_train.shape))
            agent.train_learner(x_train, y_train)
            acc_array = agent.evaluate(test_loaders_full)
            tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            x_train_offline.append(x_train)
            y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0), labels_offline)]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders_full)
        tmp_acc.append(acc_array)
'''

