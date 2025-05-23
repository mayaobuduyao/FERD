import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

T = 1

def custom_softmax(logits):
    B, C = logits.shape
    probs = F.softmax(logits, dim=1)
    max_probs, max_indices = probs.max(dim=1)
    s_n = torch.full_like(probs, fill_value=0.0)
    rest_mass = 1.0 - max_probs
    uniform_rest = rest_mass / (C - 1) 
    s_n += uniform_rest.unsqueeze(1) 
    s_n.scatter_(1, max_indices.unsqueeze(1), max_probs.unsqueeze(1))
    return s_n
"""
def dkl_loss(logits_student, logits_teacher, T2=None, type="gen"):
    _, NUM_CLASSES = logits_student.shape
    delta_n = (logits_teacher.view(-1, NUM_CLASSES, 1) - logits_teacher.view(-1, 1, NUM_CLASSES))
    delta_a = (logits_student.view(-1, NUM_CLASSES, 1) - logits_student.view(-1, 1, NUM_CLASSES))
    # [batch_size, NUM_CLASSES, NUM_CLASSES] = [batch_size, NUM_CLASSES, 1] - [batch_size, 1, NUM_CLASSES]

    if T2 == None and type == "gen":  
        T2 = 100.0
        s_n = F.softmax(logits_teacher/T2, dim=1)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)  # [batch_size, NUM_CLASSES, NUM_CLASSES]
    elif T2 == None and type == "stu":
        s_n = custom_softmax(logits_teacher)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)
    else:
        for i in range(logits_teacher.shape[0]):
            logits_t = logits_teacher.clone()
            labels = logits_t.data.max(1)[1]
            logits_t[i] = logits_t[i] * T2[labels[i]]
        s_n = F.softmax(logits_t, dim=1)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / logits_student.size(0)
    loss_sce = -(F.softmax(logits_teacher / T, dim=-1).detach() * F.log_softmax(logits_student / T, dim=-1)).sum(1).mean()
    return loss_mse + loss_sce
"""
def dkl_loss(logits_student, logits_teacher, gamma=1, CLASS_PRIOR=None, GI=False, T2=100.0):
    _, NUM_CLASSES = logits_student.shape
    delta_n = (logits_teacher.view(-1, NUM_CLASSES, 1) - logits_teacher.view(-1, 1, NUM_CLASSES))
    delta_a = (logits_student.view(-1, NUM_CLASSES, 1) - logits_student.view(-1, 1, NUM_CLASSES))
    # GI with class prior
    if GI:
        assert CLASS_PRIOR is not None, 'CLASS_PRIOR information should be collected'
        with torch.no_grad():
            CLASS_PRIOR = torch.pow(CLASS_PRIOR, gamma)
            p_n = CLASS_PRIOR.view(-1, NUM_CLASSES, 1) @ CLASS_PRIOR.view(-1, 1, NUM_CLASSES)
    else:
        s_n = F.softmax(logits_teacher / T2, dim=1)
        s_n = torch.pow(s_n, gamma)
        p_n = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / p_n.sum()
    loss_sce = -(F.softmax(logits_teacher, dim=-1).detach() * F.log_softmax(logits_student, dim=-1)).sum(1).mean()
    return loss_mse + loss_sce

"""
def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction="batchmean")
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)
"""
def cross_entropy(outputs, labels):
    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, labels) 

def smooth_one_hot(labels, classes, smoothing_dict):
    batch_size = labels.size(0)
    one_hot = torch.zeros((batch_size, classes), device=labels.device)
    
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    for i in range(batch_size):
        label = labels[i].item()
        smoothing = smoothing_dict.get(label, 0.1)
        one_hot[i] = (1 - smoothing) * one_hot[i] + smoothing / classes

    return one_hot

"""
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist.to(device)
"""
"""
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    true_labels = torch.nn.functional.one_hot(true_labels, classes).float()
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (classes - 1)
    true_dist = true_labels * confidence + smoothing_value
    return true_dist
"""

def robust_fairness_loss(logits, labels, num_classes=10):
    probs = torch.nn.functional.softmax(logits, dim=1)
    classwise_acc = torch.zeros(num_classes).to(logits.device)
    class_counts = torch.zeros(num_classes).to(logits.device)

    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.sum() > 0:
            classwise_acc[cls] = (probs[mask].argmax(dim=1) == cls).float().mean()
            class_counts[cls] = mask.sum()

    avg_acc = classwise_acc.mean()
    fairness_loss = torch.std(classwise_acc) / avg_acc

    return fairness_loss

def div_loss(outpus):
    softmax_o_S = F.softmax(outpus, dim=1).mean(dim=0)
    loss_div = (softmax_o_S * torch.log10(softmax_o_S)).sum()
    return loss_div

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)
"""
def max_margin_loss(logits, labels, alpha=1.0):
    teacher_target_logits = logits[torch.arange(labels.size(0)), labels]  # Shape: (batch_size,)
    teacher_others_logits = logits.clone()
    teacher_others_logits[torch.arange(labels.size(0)), labels] = float('-inf') 
    max_teacher_others_logits = torch.max(teacher_others_logits, dim=1).values  # Shape: (batch_size,)
    loss_mm = -teacher_target_logits + max_teacher_others_logits*alpha    
    loss = loss_mm.mean()
    return loss
"""
def max_margin_loss(out, iden, threshold=-2):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)
    if threshold is None:
        return (-1 * real).mean() + margin.mean()
    else:
        return (-1 * real).mean() + torch.abs(margin-threshold).mean() 

def KT_loss_generator(student_logits, teacher_logits, reduction='batchmean'):
    divergence_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction=reduction)
    total_loss = - divergence_loss
    return total_loss
    
def KT_loss_student(student_logits, teacher_logits, reduction='batchmean'):
    divergence_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1), reduction=reduction)
    total_loss = divergence_loss   
    return total_loss
    
def nt_xent_loss(emb_cln, emb_adv, T=10):
    batch_size = emb_cln.shape[0]
    
    emb_all = torch.cat([emb_cln, emb_adv], dim=0)
    sim_matrix = F.cosine_similarity(emb_all.unsqueeze(1), emb_all.unsqueeze(0), dim=2)

    labels = torch.arange(batch_size, device=emb_cln.device)
    labels = torch.cat([labels, labels], dim=0)

    logits = sim_matrix / T
    loss = F.cross_entropy(logits, labels)
    return loss
    
def loss_ssim(x_adv, x_cln):
    ssim_values = []
    for channel in range(x_adv.shape[1]):
        adv_channel = x_adv[:, channel, :, :].unsqueeze(1)  # [128, 1, 32, 32]
        cln_channel = x_cln[:, channel, :, :].unsqueeze(1)  # [128, 1, 32, 32]
        ssim_channel = ssim(adv_channel, cln_channel, data_range=1.0) 
        ssim_values.append(ssim_channel)

    ssim_mean = torch.mean(torch.tensor(ssim_values))
    return ssim_mean.item()
    
    

class Entropy_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(Entropy_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        if self.reduction == "mean":
            return b.mean()
        elif self.reduction == "sum":
            return b.sum()
        elif self.reduction == "none":
            return b
