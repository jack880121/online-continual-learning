from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.utils import maybe_cuda, AverageMeter
import copy
from utils.loss import SupConLoss
import pickle
from sklearn.metrics import recall_score,accuracy_score,precision_score
from models.resnet import LinearClassifier,ConvClassifier

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
        self.lbl_inv_map = {}

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
        self.model.eval()
        
        checkpoint = torch.load('/tf/online-continual-learning/result/model_state_dict_B_t.pt')
        self.old_labels = [0,1]
        self.buffer.current_index = checkpoint['buffer.current_index']
        self.buffer.buffer_img = checkpoint['buffer.buffer_img']
        self.buffer.buffer_label = checkpoint['buffer.buffer_label']
        
        
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}    #{0: [], 1: []}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)            #(1037,1963)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.forward(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.model.forward(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y

        with torch.no_grad():
#             acc = AverageMeter()
            sk_recall = AverageMeter()
            sk_accuracy = AverageMeter()
            sk_precision = AverageMeter()
#             losses = AverageMeter()

            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            
            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                    feature = self.model.forward(batch_x)  # (batch_size, feature_size)
                    for j in range(feature.size(0)):  # Normalize
                        feature.data[j] = feature.data[j] / feature.data[j].norm()
                    
                    feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                    means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                    #old ncm
                    means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)  
                    means = means.transpose(1, 2)               # means.shape torch.Size([128, 5760, 2])
                    feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)   feature.shape torch.Size([128, 5760, 2])
                    dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)   dists.shape torch.Size([128, 2])
            
                    _, pred_label = dists.min(1)      # pred_label.shape torch.Size([128])
 
                    # may be faster
#                     feature = feature.squeeze(2).T
#                     _, pred_label = torch.matmul(means, feature).max(0)
                    
#                     correct_cnt = (np.array(self.old_labels)[
#                                        pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    
                    recall = recall_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()]) #(128,) 的0與1
                    accuracy = accuracy_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
                    precision = precision_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
#                     cri = torch.nn.MSELoss()
#                     loss = torch.sqrt(cri(pred_label.type(torch.cuda.FloatTensor),batch_y.type(torch.cuda.FloatTensor)))
                else:
                    logits = self.model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    recall = recall_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()]) #(128,) 的0與1
                    accuracy = accuracy_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])
                    precision = precision_score(batch_y.cpu().numpy(),np.array(self.old_labels)[pred_label.tolist()])

                sk_accuracy.update(accuracy.item(), 1)  #batch_y.size(0)
                sk_precision.update(precision.item(), 1)
                sk_recall.update(recall.item(), 1)
#                 losses.update(loss.item(), batch_y.size(0))
            accuracy = sk_accuracy.avg()
            precision = sk_precision.avg()
            recall = sk_recall.avg()
#             loss = losses.avg()
        return accuracy,recall,precision #,loss

    def train(self, train_loader, classifier, criterion, optimizer):
        self.model.eval()
        classifier.train()
        
        sk_recall = AverageMeter()
        sk_accuracy = AverageMeter()
        sk_precision = AverageMeter()
        losses = AverageMeter()

        for i, batch_data in enumerate(train_loader):
            batch_x, batch_y = batch_data
            batch_x = maybe_cuda(batch_x, self.cuda)
            batch_y = maybe_cuda(batch_y, self.cuda)

            # compute loss
            with torch.no_grad():
                features = self.model.features(batch_x)
            output = classifier(features.detach())
            loss = criterion(output.cpu(), batch_y.cpu())
            
            _, pred_label = torch.max(output.cpu(), 1)
            recall = recall_score(batch_y.cpu().numpy(),pred_label) #(128,) 的0與1
            accuracy = accuracy_score(batch_y.cpu().numpy(),pred_label)
            precision = precision_score(batch_y.cpu().numpy(),pred_label)

            sk_accuracy.update(accuracy.item(), 1)  #batch_y.size(0)
            sk_precision.update(precision.item(), 1)
            sk_recall.update(recall.item(), 1)
            losses.update(loss.item(),1)
            
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        accuracy = sk_accuracy.avg()
        precision = sk_precision.avg()
        recall = sk_recall.avg()
        loss = losses.avg()
        print('train accuracy',accuracy)
#         print('train precision',precision)
#         print('train recall',recall)
#         print('train loss',loss)
        return loss

    def test(self, test_loader, classifier):
        self.model.eval()
        classifier.eval()
        
        with torch.no_grad():
            sk_recall = AverageMeter()
            sk_accuracy = AverageMeter()
            sk_precision = AverageMeter()
            
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

            for i, batch_data in enumerate(test_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                
                features = self.model.features(batch_x)
                output = classifier(features)

                _, pred_label = torch.max(output.cpu(), 1)
                recall = recall_score(batch_y.cpu().numpy(),pred_label) #(128,) 的0與1
                accuracy = accuracy_score(batch_y.cpu().numpy(),pred_label)
                precision = precision_score(batch_y.cpu().numpy(),pred_label)

                sk_accuracy.update(accuracy.item(), 1)  #batch_y.size(0)
                sk_precision.update(precision.item(), 1)
                sk_recall.update(recall.item(), 1)

            accuracy = sk_accuracy.avg()
            precision = sk_precision.avg()
            recall = sk_recall.avg()
            
        return accuracy,recall,precision
            
    def classifier(self, train_loader, test_loader, writer, run):      #linear classifier
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
#         classifier = LinearClassifier()
        classifier = ConvClassifier()
        optimizer = torch.optim.SGD(classifier.parameters(),lr=0.1,momentum=0.9)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
            ce = ce.cuda()
            torch.backends.cudnn.benchmark = True
        
        for epoch in range(80):
            loss = self.train(train_loader, classifier, ce, optimizer)
            if run==0:
                writer.add_scalar('stage2_train_loss', loss, epoch)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        tra,trr,trp = self.test(train_loader, classifier)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        tea,ter,tep = self.test(test_loader, classifier)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            
        return tra,trr,trp,tea,ter,tep