import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import TensorDataset , DataLoader
import torchattacks
import math
from PIL import Image
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def save_mem(images, labels, save_dir, epoch, type):
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    #save_path = os.path.join(save_dir, type)
    save_path = save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images_filename = os.path.join(save_path, "synthetic_images.npy")
    labels_filename = os.path.join(save_path, "synthetic_labels.npy")

    if epoch > 1:
        org_images = np.load(images_filename)
        org_labels = np.load(labels_filename)

        images = np.concatenate((org_images, images), 0)
        labels = np.concatenate((org_labels, labels), 0)

    np.save(images_filename, images)
    np.save(labels_filename, labels)


def save_batch_samples(imgs, save_dir, save_name, type, col=None, size=None, pack=True, individual=False):
    output = f"{save_dir}/{save_name}_{type}.png"
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        pack_imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        pack_imgs = Image.fromarray( pack_imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                pack_imgs = pack_imgs.resize(size)
            else:
                w, h = pack_imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                pack_imgs = pack_imgs.resize([_w, _h])
        pack_imgs.save(output)
    if individual:
        output_filename = output.replace('.png', '') 
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img.transpose(1, 2, 0)) 
            individual_output = f"{output_filename}_{idx}.png"
            img.save(individual_output)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack


@torch.no_grad()
def get_rank2_label(logit, y):
    batch_size = len(logit)
    tmp = logit.clone()
    tmp[torch.arange(batch_size), y] = -float("inf")
    return tmp.argmax(1)


def clean_test(model, test_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_test(model, test_loader):
    correct = 0
    model.eval()
    attack = torchattacks.PGD(
        model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True
    )
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            with torch.enable_grad():
                adv_data = attack(data, target)
            output = model(adv_data)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adv_test_l2(model, test_loader):
    correct = 0
    model.eval()
    # attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)

    attack = torchattacks.PGDL2(
        model, eps=128.0 / 255, alpha=15.0 / 255, steps=10, random_start=True
    )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            with torch.enable_grad():
                adv_data = attack(data, target)
            output = model(adv_data)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
            
            
def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, logfile="output.log"):
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="[%(asctime)s] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            filename=self.logfile,
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


def save_checkpoint(
        state,
        name,
        epoch,
        is_best,
        which_best,
        save_path,
        save_freq=10,
    ):
    filename = os.path.join(save_path, str(name) + "checkpoint_" + str(epoch) + ".tar")
    if epoch % save_freq == 0:
        if not os.path.exists(filename):
            torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            save_path, str(name) + "best_" + str(which_best) + "_checkpoint.tar"
        )
        torch.save(state, best_filename)


def select_gmm(args, teacher , mem_imgs , mem_lbls , target_imgs = None , selection_score = 0.5):
    ce = nn.CrossEntropyLoss(reduction='none')
    teacher.eval()
    datasets = TensorDataset(mem_imgs, mem_lbls)
    loader = DataLoader(datasets , batch_size=args.batch_size , shuffle=False)

    losses = torch.tensor([])
    with torch.no_grad():
        for step, batch in enumerate(loader):
            imgs, label = batch[0], batch[1]
            label = label.type(torch.LongTensor)
            imgs , label = imgs.cuda() , label.cuda()
            outputs = teacher(imgs)
            loss = ce(outputs , label)
            losses = torch.cat([losses , loss.detach().cpu()])

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    input_loss = losses.reshape(-1,1)  # reshape to [N, 1]

    # GMM
    gmm = GaussianMixture(n_components=2 , max_iter=10 , tol=1e-2 , reg_covar=5e-4)

    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob_corr = prob[: , gmm.means_.argmin()]

    pred_corr = prob_corr > selection_score

    if target_imgs == None:
        correct_imgs = mem_imgs[pred_corr, :]
        correct_lbls = mem_lbls[pred_corr]

    else:
        correct_imgs = target_imgs[pred_corr, :]
        correct_lbls = mem_lbls[pred_corr]

    return correct_imgs , correct_lbls, pred_corr
    
def aug(inputs_jit, lim=1. / 8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit
    
#####################
# t-SNE
#####################    
    
class FeatureExtractor:
    def __init__(self, model, layer_name="block3"):
        self.model = model
        self.features = None
        
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.hook_fn)
                break
    
    def hook_fn(self, module, input, output):
        self.features = output
    
    def get_features(self, x):
        self.model(x) 
        return self.features

### one generator ###
def extract_features(args, generator, teacher_model, num_samples=256):
    teacher_model.eval()
    features = []
    labels = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = FeatureExtractor(teacher_model, layer_name="block3")
    
    with torch.no_grad():
        for _ in range(num_samples // args.batch_size + 1):
            z = torch.randn(args.batch_size, args.gen_dim_z).to(device)
            gen_samples = generator(z)
            
            _ = feature_extractor.get_features(gen_samples)
            batch_features = feature_extractor.features.mean(dim=[2, 3]).cpu().numpy()  # [batch_size, channels]
            
            t_logits = teacher_model(gen_samples)
            batch_labels = t_logits.argmax(dim=1).cpu().numpy()
            
            features.append(batch_features)
            labels.append(batch_labels)
    
    return np.concatenate(features), np.concatenate(labels)
    
def plot_tsne(args, generator, teacher_model):
    features, labels = extract_features(args, generator, teacher_model, num_samples=args.batch_size)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    embeddings = tsne.fit_transform(features)  # [batch_size, 2]
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, (x, y) in enumerate(embeddings):
        marker = 'o'
        color = colors(np.where(unique_labels == labels[i])[0][0]/len(unique_labels))
        
        plt.scatter(x, y, c=[color], marker=marker, alpha=0.6, edgecolors='w', linewidths=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=8)
    
    plt.title('t-SNE Visualization of Generated Samples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    
    os.makedirs("./tsne", exist_ok=True)
    save_file = os.path.join("./tsne", args.experiment_name)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.close()
    
### two generator ###
def extract_features2(args, generator, teacher_model, num_samples=256, is_adv=False):
    teacher_model.eval()
    features = []
    labels = []
    generator_types = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = FeatureExtractor(teacher_model, layer_name="block3")
    
    with torch.no_grad():
        for _ in range(num_samples // args.batch_size + 1):
            z = torch.randn(args.batch_size, args.gen_dim_z).to(device)
            gen_samples = generator(z)
            
            _ = feature_extractor.get_features(gen_samples)
            batch_features = feature_extractor.features.mean(dim=[2, 3]).cpu().numpy()  # [batch_size, channels]
            
            t_logits = teacher_model(gen_samples)
            batch_labels = t_logits.argmax(dim=1).cpu().numpy()
            
            features.append(batch_features)
            labels.append(batch_labels)
            generator_types.append(['adv' if is_adv else 'cln']*len(batch_labels))
    
    return np.concatenate(features), np.concatenate(labels), np.concatenate(generator_types)
    
def plot_tsne2(args, cln_generator, adv_generator, teacher_model):
    cln_features, cln_labels, cln_types = extract_features2(args, cln_generator, teacher_model, num_samples=args.batch_size, is_adv=False)
    adv_features, adv_labels, adv_types = extract_features2(args, adv_generator, teacher_model, num_samples=args.batch_size, is_adv=True)
    
    all_features = np.concatenate([cln_features, adv_features])
    all_labels = np.concatenate([cln_labels, adv_labels])
    all_types = np.concatenate([cln_types, adv_types])
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    embeddings = tsne.fit_transform(all_features)  # [batch_size, 2]
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(all_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, (x, y) in enumerate(embeddings):
        marker = 'o' if all_types[i] == 'cln' else '^'
        color = colors(np.where(unique_labels == all_labels[i])[0][0]/len(unique_labels))
        
        plt.scatter(x, y, c=[color], marker=marker, alpha=0.6, edgecolors='w', linewidths=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=8)
    
    plt.title('t-SNE Visualization of Generated Samples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    
    os.makedirs("./tsne", exist_ok=True)
    save_file = os.path.join("./tsne", args.experiment_name)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.close()
    
### dataset ###
def extract_features_d(args, data, teacher_model, num_samples=256):
    teacher_model.eval()
    features = []
    labels = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = FeatureExtractor(teacher_model, layer_name="block3")
    
    with torch.no_grad():
        _ = feature_extractor.get_features(data)
        batch_features = feature_extractor.features.mean(dim=[2, 3]).cpu().numpy()  # [batch_size, channels]
            
        t_logits = teacher_model(data)
        batch_labels = t_logits.argmax(dim=1).cpu().numpy()
            
        features.append(batch_features)
        labels.append(batch_labels)
    
    return np.concatenate(features), np.concatenate(labels)
    
def plot_tsne_d(args, data, teacher_model):
    features, labels = extract_features_d(args, data, teacher_model, num_samples=args.batch_size)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    embeddings = tsne.fit_transform(features)  # [batch_size, 2]
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, (x, y) in enumerate(embeddings):
        marker = 'o'
        color = colors(np.where(unique_labels == labels[i])[0][0]/len(unique_labels))
        
        plt.scatter(x, y, c=[color], marker=marker, alpha=0.6, edgecolors='w', linewidths=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=8)
    
    plt.title('t-SNE Visualization of Generated Samples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(alpha=0.3)
    
    os.makedirs("./tsne", exist_ok=True)
    save_file = os.path.join("./tsne", args.experiment_name)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.close()
    
### two data ###
def extract_features_d2(args, data, teacher_model, num_samples=256, is_adv=False):
    teacher_model.eval()
    features = []
    labels = []
    generator_types = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = FeatureExtractor(teacher_model, layer_name="block3")  #layer4
    
    with torch.no_grad():
        for _ in range(num_samples // args.batch_size + 1):
            
            _ = feature_extractor.get_features(data)
            batch_features = feature_extractor.features.mean(dim=[2, 3]).cpu().numpy()  # [batch_size, channels]
            
            t_logits = teacher_model(data)
            batch_labels = t_logits.argmax(dim=1).cpu().numpy()
            
            features.append(batch_features)
            labels.append(batch_labels)
            generator_types.append(['adv' if is_adv else 'cln']*len(batch_labels))
    
    return np.concatenate(features), np.concatenate(labels), np.concatenate(generator_types)
    
def plot_tsne_d2(args, cln, adv, teacher_model):
    cln_features, cln_labels, cln_types = extract_features_d2(args, cln, teacher_model, num_samples=args.batch_size, is_adv=False)
    adv_features, adv_labels, adv_types = extract_features_d2(args, adv, teacher_model, num_samples=args.batch_size, is_adv=True)
    
    all_features = np.concatenate([cln_features, adv_features])
    all_labels = np.concatenate([cln_labels, adv_labels])
    all_types = np.concatenate([cln_types, adv_types])
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    embeddings = tsne.fit_transform(all_features)  # [batch_size, 2]
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(all_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, (x, y) in enumerate(embeddings):
        marker = 'o' if all_types[i] == 'cln' else 'o' #^
        color = colors(np.where(unique_labels == all_labels[i])[0][0]/len(unique_labels))
        
        plt.scatter(x, y, c=[color], marker=marker, alpha=0.6, edgecolors='w', linewidths=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=8)
    
    plt.title('t-SNE Visualization of Generated Samples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(alpha=0.3)
    
    os.makedirs("./tsne", exist_ok=True)
    save_file = os.path.join("./tsne", args.experiment_name)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    plt.close()
