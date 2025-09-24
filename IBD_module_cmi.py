from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy.core.numeric import True_
#from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import math
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import pdb
import argparse
import numpy as np
from utils import *
from dataset import *
from models.IFP import InformativeFeaturePackage
from loader.loader import IFD_network_loader
from networks import resnet, gan, deepinversion
import losses as L
from query_sample import *

def unique_shape(s_shapes):
    n_s = []    # stores index for unique shapes
    unique_shapes = []  # stores unique shapes
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)   # add new shape
            n += 1
        n_s.append(n)
    return n_s, unique_shapes
    
def reset_model(model, reset=1):
    if reset == 0:  # no reset
        return
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str( self.transform )
    
class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)
    
class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(self.device)
        self._ptr = 0  # Points to the current position that can be updated
        self.n_updates = 0  # Record the total number of updates

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape
        assert self.dim_feat == c, "%d, %d" % (self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

class Train_Gen_Module(nn.Module):
    def __init__(self, args, teacher, student, generator):
        super(Train_Gen_Module, self).__init__()
        self.args = args
        self.teacher = teacher 
        self.student = student
        self.generator = generator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bank_size = 40960
        self.n_neg = 4096
        self.head_dim=128
        self.cr = 0.8  #0.8
        self.cr_T = 0.2  #0.2
        
        if args.data == "CIFAR10":
            self.num_classes = 10
            self.img_size = (3, 32, 32)
        elif args.data == "CIFAR100":
            self.num_classes = 100
            self.img_size = (3, 32, 32)
        elif args.data == "tiny":
            self.num_classes = 200
            self.img_size = (3, 64, 64)
            
        self.pop = self.args.int_pop
        
        self.IFM_t = InformativeFeaturePackage(self.teacher, eps=0.03, attack_iter=10)

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        
        # BNS
        self.t_hooks = []
        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.t_hooks.append(deepinversion.DeepInversionHook(module, 0))
        self.s_hooks = []
        for module in self.student.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.s_hooks.append(deepinversion.DeepInversionHook(module, 0))
                
        # CMI
        self.cmi_hooks = []
        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.cmi_hooks.append(deepinversion.InstanceMeanHook(module))
        with torch.no_grad():
            self.teacher.eval()
            if args.data == "CIFAR10" or args.data == "CIFAR100":
                syn_inputs = torch.randn(size=(1, 3, 32, 32), device=self.device)
            else:
                syn_inputs = torch.randn(size=(1, 3, 64, 64), device=self.device)
            _ = self.teacher(syn_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("CMI dims: %d"%(cmi_feature.shape[1]))
            del syn_inputs
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        self.head = MLPHead(cmi_feature.shape[1], self.head_dim).to(self.device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.args.lr_G)
        
    
    def train(self, labels_prob):
        self.teacher.eval()
        self.student.eval()
        self.generator.train()
            
        noises = torch.randn(self.args.batch_size, self.args.gen_dim_z).to(self.device).requires_grad_()
        #labels_prob = torch.tensor([0.05, 0.05, 0.175, 0.175, 0.175, 0.175, 0.05, 0.05, 0.05, 0.05])
        labels = torch.multinomial(labels_prob, self.args.batch_size, replacement=True).sort()[0].to(self.device)
        #labels = torch.randint(low=0, high=self.num_classes, size=(self.args.batch_size,)).sort()[0].to(self.device)
        
        reset_model(self.generator, 1)
        optimizer_G = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [noises], "lr": self.args.lr_z}], self.args.lr_G, betas=[0.5, 0.999])
        
        best_loss = 1e6
        best_mem = None
        best_features = None
        labels_loss = torch.zeros(self.num_classes, dtype=torch.float).to(self.device)
        labels_count = torch.zeros(self.num_classes).to(self.device)
        
        for i in range(self.args.N_G):
        
            x = self.generator(noises)
            x_syn, x_syn_local = self.aug(x)
            
            latent_r, robust_latent_z, non_robust_latent_z, t_logit_z, t_logit_r, t_logit_non_r = self.IFM_t.find_features(x_syn, labels, pop_number=self.pop, forward_version=False)
            s_logit = self.student(x_syn)
            
            # Inversion Loss
            loss_adv = -L.dkl_loss(s_logit, t_logit_z)
            loss_bn = sum([mod.r_feature for mod in self.t_hooks])
            loss_oh = F.cross_entropy(t_logit_z, labels)
            uniform_logits = torch.ones_like(t_logit_non_r).to(self.device) * 0.1
            loss_uni = L.KT_loss_student(t_logit_non_r, uniform_logits)
            
            total_loss = (
                loss_bn*self.args.bn +
                loss_adv*self.args.adv + 
                loss_oh*self.args.oh +
                loss_uni*self.args.uni 
            )
            
            optimizer_G.zero_grad()
            self.optimizer_head.zero_grad()
            total_loss.backward()
            optimizer_G.step()
            self.optimizer_head.step()
            
            with torch.no_grad():
                if best_loss > total_loss.item() or best_mem is None:
                    best_loss = total_loss.item()
                    best_mem = x.data
            """ 
            # t-SNE
            if i%10 == 0:
                plot_tsne_d(self.args, x_syn, self.teacher, i)
            """     
        # update risk loss
        
        x_adv = pgd_attack(self.teacher, x_syn, t_logit_z, labels)
        t_logit_adv = self.teacher(x_adv)
        correct_class_logits = t_logit_adv.gather(1, labels.unsqueeze(1)).squeeze()
        temp_logits = t_logit_adv.clone()
        temp_logits.scatter_(1, labels.unsqueeze(1), -float('inf'))
        max_other_logits = temp_logits.max(dim=1).values
        margin = correct_class_logits - max_other_logits
        is_fooled = (margin < 99999)
        neg_margin_fooled_only = -margin * is_fooled.float()
        class_neg_margin_sum = torch.zeros(self.num_classes, device=t_logit_adv.device)
        class_fooled_count = torch.zeros(self.num_classes, device=t_logit_adv.device)
        class_neg_margin_sum.scatter_add_(0, labels, neg_margin_fooled_only)
        class_fooled_count.scatter_add_(0, labels, is_fooled.float())
        average_neg_margin_per_class = torch.where(
            class_fooled_count > 0,
            class_neg_margin_sum / class_fooled_count,
            torch.tensor(0.0, device=t_logit_adv.device)
        )
        labels_loss = average_neg_margin_per_class
        """
        x_adv = pgd_attack(self.teacher, x_syn, t_logit_z, labels)
        t_logit_adv = self.teacher(x_adv)
        kl_Loss1 = F.cross_entropy(t_logit_adv.detach(), labels.detach(), reduction='none')
        for i in range(self.args.batch_size):
            label_idx = labels[i].item()
            labels_loss[label_idx] += kl_Loss1[i]
            labels_count[label_idx] += 1
        labels_loss = labels_loss / (labels_count + 1e-8)
        """
        
        
        return best_mem.detach(), labels.detach(), total_loss.item(), loss_bn.item(), loss_adv.item(), loss_oh.item(), loss_uni.item(), labels_loss
        