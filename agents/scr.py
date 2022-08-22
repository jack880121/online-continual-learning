import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from tensorboardX import SummaryWriter
import time

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
    def train_learner_A(self, train_loader):
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        
        for i, batch_data in enumerate(train_loader):
            batch_x, batch_y = batch_data
            batch_x = maybe_cuda(batch_x, self.cuda)
            batch_y = maybe_cuda(batch_y, self.cuda)

            for j in range(self.mem_iters):
                mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                if mem_x.size(0) > 0:
                    mem_x = maybe_cuda(mem_x, self.cuda)
                    mem_y = maybe_cuda(mem_y, self.cuda)
                    combined_batch = torch.cat((mem_x, batch_x))
                    combined_labels = torch.cat((mem_y, batch_y))
                    combined_batch_aug = self.transform(combined_batch)
                    features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                    loss = self.criterion(features, combined_labels)
                    losses.update(loss, batch_y.size(0))
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

            # update mem
            self.buffer.update(batch_x, batch_y)

            if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                            .format(i, losses.avg())
                    )

        torch.save({
                'buffer.current_index': self.buffer.current_index,
                'buffer.buffer_img': self.buffer.buffer_img, 
                'buffer.buffer_label': self.buffer.buffer_label, 
                'model_state_dict': self.model.state_dict(),
                }, '/tf/online-continual-learning/result/model_state_dict_A_ep50.pt')
        return losses.avg()
        
    def train_learner_B(self, train_loader,run):
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        
        writer = SummaryWriter('/tf/online-continual-learning/result/resultB_ep1_noaug')

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                if i==((len(train_loader)//10)*(run+1)):
                    break
             
                if i>=((len(train_loader)//10)*run)  and i<((len(train_loader)//10)*(run+1)):
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)

                    for j in range(self.mem_iters):
                        mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                        if mem_x.size(0) > 0:
                            mem_x = maybe_cuda(mem_x, self.cuda)
                            mem_y = maybe_cuda(mem_y, self.cuda)
                            combined_batch = torch.cat((mem_x, batch_x))
                            combined_labels = torch.cat((mem_y, batch_y))
                            combined_batch_aug = self.transform(combined_batch)
                            features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                            loss = self.criterion(features, combined_labels)
                            losses.update(loss, batch_y.size(0))
                            self.opt.zero_grad()
                            loss.backward()
                            self.opt.step()

                    # update mem
                    self.buffer.update(batch_x, batch_y)
                    if i % 100 == 1 and self.verbose:
                            print(
                                '==>>> it: {}, avg. loss: {:.6f}, '
                                    .format(i, losses.avg())
                            )
            writer.add_scalar('Training Loss', losses.avg(), ep)
            writer.close()
            torch.save({
                    'epoch': ep,
                    'buffer.current_index': self.buffer.current_index,
                    'buffer.buffer_img': self.buffer.buffer_img, 
                    'buffer.buffer_label': self.buffer.buffer_label, 
                    'model_state_dict': self.model.state_dict(),
                    }, 'model_state_dict_B_ep1_noaug.pt')
