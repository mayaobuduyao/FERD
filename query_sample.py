import torch
import torch.nn as nn
import torch.nn.functional as F

import losses as L

def generate_fair_adv(model, x, target):
    device = x.device
    model.eval()
    
    x_adv = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    for _ in range(10):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), target)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + 2.0/255 * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - 8.0/255), x + 8.0/255)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    return x_adv

def generate_hee(args, model, x):
    device = x.device
    model.eval()
    
    x_hee = x + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    for _ in range(10):
        x_hee.requires_grad_()
        with torch.enable_grad():
            loss = L.Entropy_Loss(reduction="mean")(model(x_hee))
        grad = torch.autograd.grad(loss, [x_hee])[0]
        x_hee = x_hee.detach() + 0.03 * torch.sign(grad.detach())
        x_hee = torch.clamp(x_hee, 0.0, 1.0)
    model.train()

    return x_hee


### paper ###
def generate_adv(model, x, logits, target):
    device = x.device
    model.eval()
    
    x_adv = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    criterion_kl = nn.KLDivLoss(reduction='none')
    uniform_logits = torch.ones_like(logits).to(device) * 0.1
    
    for _ in range(10):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_logits = model(x_adv)
            loss1 = torch.sum(criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(logits, dim=1)))
            #loss2 = -torch.sum(criterion_kl(F.log_softmax(adv_logits, dim=1), uniform_logits))
            loss = loss1 
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + 2.0/255 * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - 8.0/255), x + 8.0/255)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    return x_adv

"""
def generate_adv(model, x, logits, target):
    device = x.device
    model.eval()
    
    x_adv = x.detach() + 0.001 * torch.torch.randn(x.shape).to(device).detach()
    criterion_kl = nn.KLDivLoss(reduction='none')
    uniform_logits = torch.ones_like(logits).to(device) * 0.1
    
    for _ in range(1):
        x_adv.requires_grad_()
        with torch.enable_grad():
            adv_logits = model(x_adv)
            loss1 = torch.sum(criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(logits, dim=1)))
            loss2 = -torch.sum(criterion_kl(F.log_softmax(adv_logits, dim=1), uniform_logits))
            loss = loss1 + loss2*10
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + 8.0/255 * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - 8.0/255), x + 8.0/255)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    return x_adv
"""  

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=20, clamp=(0, 1)):

    images = images.clone().detach().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    labels = labels.clone().detach().to(images.device)
    ori_images = images.clone().detach()

    # Initialize with small random noise within epsilon-ball
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, clamp[0], clamp[1])
    
    model.eval()

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        # Gradient ascent step
        adv_images = adv_images + alpha * grad.sign()
        # Projection (clip to epsilon-ball)
        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + delta, clamp[0], clamp[1]).detach()  
        
    model.train()

    return adv_images

