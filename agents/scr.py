import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from utils.setup_elements import input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomVerticalFlip
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
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.7, 1.)),
            RandomVerticalFlip(),
            RandomHorizontalFlip()#,
            #ColorJitter(brightness=0.2)
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
                    losses.update(loss.item(), 1)
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
                }, '/tf/online-continual-learning/result/model_state_dict_A_lr0.05epo10.pt')
        return losses.avg()
        
    def train_learner_B(self, train_loader,run,writer):
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)   #batch_x.shape torch.Size([10, 3, 200, 200])
                batch_y = maybe_cuda(batch_y, self.cuda)   #batch_y.shape torch.Size([10])
  

                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))   #torch.Size([30, 3, 200, 200])
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_batch_aug = self.transform(combined_batch)   #torch.Size([30, 3, 200, 200])
                        features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)    #torch.Size([30, 2, 128])
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss.item(), 1) #batch_y.size(0)
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

#                     combined_batch_aug = self.transform(batch_x) 不包含memory資料
#                     features = torch.cat([self.model.forward(batch_x).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
#                     loss = self.criterion(features, batch_y)
#                     losses.update(loss.item(), batch_y.size(0))
#                     self.opt.zero_grad()
#                     loss.backward()
#                     self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    print('==>>> it: {}, avg. loss: {:.6f}, '.format(i, losses.avg()))
                    
            writer.add_scalar('stage1_train_loss', losses.avg(), ep+run*self.epoch)

        torch.save({
                #'run': run,
                #'epoch': ep,
                'buffer.current_index': self.buffer.current_index,
                'buffer.buffer_img': self.buffer.buffer_img, 
                'buffer.buffer_label': self.buffer.buffer_label, 
                'model_state_dict': self.model.state_dict(),
                }, '/tf/online-continual-learning/result/model_state_dict_B_t.pt')
