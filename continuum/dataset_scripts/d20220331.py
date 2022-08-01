import numpy as np
import cv2
import os
from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns


class d20220331(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'd20220331'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(d20220331, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

    def my_load(self,root,n1,n2,run):
        imgsize = 200
        d0 = os.listdir(root+"/D000")
        d1 = os.listdir(root+"/D001")

        data = np.zeros([n1+n2,imgsize,imgsize,3])
        c1 = 0
        for i in range (len(d0)):
            img = cv2.imread(root+"/D000/"+d0[i+run*n1])[np.newaxis, :] 
            if img.shape[1]==imgsize and img.shape[2]==imgsize:
                c1 = c1+1
                data[c1-1] = img
                if c1 == n1:
                    break
        c2 = 0
        for i in range (len(d1)):
            img = cv2.imread(root+"/D001/"+d1[i+run*n2])[np.newaxis, :] 
            if img.shape[1]==imgsize and img.shape[2]==imgsize:
                c2 = c2+1
                data[c1+c2-1] = img
                if c2 == n2:
                    break
                    
        data = data[:c1+c2]
        
        label = np.zeros((c1+c2,), dtype=int)
        label[c1:] = 1

        return data,label
        
    def download_load(self,run):
        root = "/tf/online-continual-learning/datasets/20220331"
        
        self.train_data,self.train_label = self.my_load(root+'/train',1500,1500,run)
        print('train ok')
        self.test_data,self.test_label = self.my_load(root+'/test',500,500,0)
        print('test ok')
        #self.test_data = self.train_data[:500]
        #self.test_label = self.train_label[:500]
'''    
    def download_load(self,run):
        root = "/tf/online-continual-learning/datasets/20220331"
        
        if params.mode == 'train':
            self.train_data,self.train_label = self.my_load(root+'/train',1500,1500,run)
            print('train ok')
        else:
            self.test_data,self.test_label = self.my_load(root+'/test',500,500,0)
            print('test ok')
'''

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data,                                                                                                   self.test_label, 
                                                                                        self.task_nums, 200,
                                                                                        self.params.ns_type,                                                                                               self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
            '''
        if self.scenario == 'ni':
            if params.mode == 'train':
                self.train_set= construct_ns_multiple_wrapper_train(self.train_data,
                                                                                        self.train_label,
                                                                                        self.task_nums, 200,
                                                                                        self.params.ns_type,                                                                                               self.params.ns_factor,
                                                                                    plot=self.params.plot_sample)
            else:
                self.test_set = construct_ns_multiple_wrapper_test(self.test_data,                                                                                                                             self.test_label, 
                                                                                        self.task_nums, 200,
                                                                                        self.params.ns_type,                                                                                               self.params.ns_factor,
                                                                                    plot=self.params.plot_sample)
            '''
            '''
        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')
            '''

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

#    def new_run(self, **kwargs):
#        self.setup()
#        return self.test_set
    def new_run(self,run):
        self.download_load(run)
        self.setup()
        #return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                                                         self.params.ns_factor)
