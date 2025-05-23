import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class InformativeFeaturePackage(nn.Module):
    def __init__(self, model, eps=0.03, attack_iter=10, IFD_iter=50, IFD_lr=0.1):
        super(InformativeFeaturePackage, self).__init__()
        self.model = model

        # PGD-based IFD attack hyper-parameter
        self.eps = eps
        self.attack_iter = attack_iter
        self.alpha = self.eps/attack_iter*2.3
        self.eta = 1e-2

        # IFD hyper-parameter
        self.IFD_iter = IFD_iter  # 100
        self.IFD_lr = IFD_lr
        self.cw_c = 0.1
        self.pgd_c = 10
        self.beta = 0.3
        self.grad = 1

        # define loss
        self.mse = nn.MSELoss(reduction='none')

        # softplus
        self.softplus = nn.Softplus()

    @staticmethod
    def grad_on_off(model, switch=False):
        for param in model.parameters():
            param.requires_grad=switch

    @staticmethod
    def kl_div(p, lambda_r):
        delta = 1e-10
        p_var = p.var(dim=[2, 3])
        q_var = (lambda_r.squeeze(-1).squeeze(-1)) ** 2

        eq1 = p_var / (q_var + delta)
        eq2 = torch.log((q_var + delta) / (p_var + delta))

        kld = 0.5 * (eq1 + eq2 - 1)

        return kld.mean()

    @staticmethod
    def sample_latent(latent_r, lambda_r):
        eps = torch.normal(0, 1, size=lambda_r.size()).cuda()   # Generate standard normal noise
        return latent_r + lambda_r.mul(eps)


    def sample_robust_and_non_robust_latent(self, latent_r, lambda_r):

        var = lambda_r.square() # variance of noise  [256, 640, 1, 1]
        r_var = latent_r.var(dim=(2,3)).view(-1) # expend to 1-dim  [163840]=256*640
        index = (var > r_var.max()).float()
        return index
        

    def find_features(self, input, labels, pop_number, forward_version=False):  # pop_number=3 forward_version=True

        latent_r = self.model(input, pop=pop_number)    # get a hidden layer output  [256, 640, 8, 8]
        lambda_r = torch.zeros([*latent_r.size()[:2],1,1]).cuda().requires_grad_()  # [256, 640, 1, 1]
        optimizer = torch.optim.Adam([lambda_r], lr=self.IFD_lr)

        for i in range(self.IFD_iter):

            lamb = self.softplus(lambda_r)  # like ReLU, but smooth
            latent_z = self.sample_latent(latent_r.detach(), lamb)  # add noise
            outputs = self.model(latent_z.clone(), intermediate_propagate=pop_number)
            kl_loss = self.kl_div(latent_r.detach(), lamb)  # training goal is to make lambda_r more robust without deviating from latent_r
            ce_loss = F.cross_entropy(outputs, labels)  
            loss = ce_loss + self.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        robust_lambda_r = lambda_r.clone().detach()  # [256, 640, 1, 1]

        # robust and non-robust index
        robust_index  = self.sample_robust_and_non_robust_latent(latent_r, self.softplus(robust_lambda_r))  # [256, 640, 1, 1]
        non_robust_index = 1-robust_index

        # robust and non-robust feature
        robust_latent_z     = latent_r * robust_index
        non_robust_latent_z = latent_r * non_robust_index

        robust_outputs = self.model(robust_latent_z, intermediate_propagate=pop_number)
        _, robust_predicted = robust_outputs.max(1)

        non_robust_outputs = self.model(non_robust_latent_z, intermediate_propagate=pop_number)
        _, non_robust_predicted = non_robust_outputs.max(1)
        
        t_logit_z = self.model(latent_r, intermediate_propagate=pop_number)

        if forward_version:
            return latent_r, robust_latent_z, non_robust_latent_z, t_logit_z, robust_predicted, non_robust_predicted
        return latent_r, robust_latent_z, non_robust_latent_z, t_logit_z, robust_outputs, non_robust_outputs

