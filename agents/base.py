from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle
from sklearn.metrics import recall_score,accuracy_score,precision_score

class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

#     @abstractmethod
#     def train_learner(self, x_train, y_train):
#         pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)
    
    def evaluate(self, test_loader):
        checkpoint = torch.load('/tf/online-continual-learning/result/model_state_dict_A_ep50.pt')
        self.old_labels = [0,1]
        self.buffer.current_index = checkpoint['buffer.current_index']
        self.buffer.buffer_img = checkpoint['buffer.buffer_img']
        self.buffer.buffer_label = checkpoint['buffer.buffer_label']
        
        self.model.eval()
        acc_array = np.zeros(len(test_loader))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            acc = AverageMeter()
            sk_recall = AverageMeter()
            sk_accuracy = AverageMeter()
            sk_precision = AverageMeter()
            losses = AverageMeter()
            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                    feature = self.model.features(batch_x)  # (batch_size, feature_size)
                    for j in range(feature.size(0)):  # Normalize
                        feature.data[j] = feature.data[j] / feature.data[j].norm()
                    
                    loss = self.criterion(feature, batch_y)
                    
                    feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                    means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                    #old ncm
                    means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                    means = means.transpose(1, 2)
                    feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                    dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                    _, pred_label = dists.min(1)
                    # may be faster
                    # feature = feature.squeeze(2).T
                    # _, preds = torch.matmul(means, feature).max(0)
                    correct_cnt = (np.array(self.old_labels)[
                                       pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    recall = recall_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
                    accuracy = accuracy_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
                    precision = precision_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
                else:
                    logits = self.model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                #acc.update(correct_cnt, batch_y.size(0))
                sk_accuracy.update(accuracy, batch_y.size(0))
                sk_precision.update(precision, batch_y.size(0))
                sk_recall.update(recall, batch_y.size(0))
                losses.update(loss, batch_y.size(0))
            accuracy = sk_accuracy.avg()
            precision = sk_precision.avg()
            recall = sk_recall.avg()
            loss = losses.avg()
            #acc_array[task] = acc.avg()
        return accuracy,recall,precision,loss