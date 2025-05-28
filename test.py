import argparse
import datetime
import time
import warnings
import torch
import torch.nn.functional as F

import losses as L
import models
from models.wideresnet import WRN_34_20, WRN_34_10
import torch.nn.functional as F
from dataset import *
from kornia import augmentation
from torchvision import datasets, transforms
from utils import *
from hooks import DeepInversionHook
#from IBD_module_t import *
from IBD_module_cmi import *  #CMI
from query_sample import generate_adv
from models.IFP import InformativeFeaturePackage
from bat_loss.stop_to_lastclean import *
from bat_loss.stop_to_firstadv import *
from collections import OrderedDict

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Data-Free Adversarial Kownledge Distillation")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul

# model configuration
parser.add_argument("--teacher_model", type=str, choices=["WRN_34_20", "WRN_34_10", "PreActResNet34"], default="WRN_34_10")
parser.add_argument("--student_model", type=str, choices=["ResNet18", "MobileNet", "PreActResNet18"], default="ResNet18")
parser.add_argument("--target_dir", type=str, default="./checkpoints/",)

# generator configuration
parser.add_argument("--gen_dim_z", "-gdz", type=int, default=100, help="Dimension of generator input noise.",)
parser.add_argument("--gen_distribution", "-gd", type=str, default="normal", help="Input noise distribution: normal (default) or uniform.",)

# dataset configuration
parser.add_argument("--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "tiny"])
parser.add_argument("--data_path", type=str, default="~/datasets/", help="where is the dataset CIFAR-10")
parser.add_argument("--test_batch_size", type=int, default=256, metavar="N", help="input batch size for testing",)

# training configuration
parser.add_argument("--batch_size", type=int, default=256, metavar="N", help="input batch size for training",)
parser.add_argument("--epochs", type=int, default=220, metavar="N", help="number of epochs to train")
parser.add_argument("--warmup", type=int, default=20, metavar="N", help="start to train student")
parser.add_argument("--lr", type=float, default=0.01, metavar="N", help="learning rate of student model")
parser.add_argument("--lr_G", type=float, default=0.002, metavar="N", help="learning rate of generator")
parser.add_argument("--lr_z", type=float, default=0.01, help="learning rate of latent code")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
parser.add_argument("--weight_decay", default=5e-4, type=float,)
parser.add_argument("--N_S", type=int, default=400, metavar="N", help="iterations of student model")
parser.add_argument("--N_G", type=int, default=200, metavar="N", help="iterations of generator")
parser.add_argument("--adv", default=1, type=float)
parser.add_argument("--bn", default=5, type=float)
parser.add_argument("--oh", default=1, type=float)
parser.add_argument("--uni", default=5, type=float)

# other configuration
parser.add_argument("--result_dir", default="./checkpoints", help="directory of model for saving checkpoint")
parser.add_argument("--save_freq", "-s", default=50, type=int, metavar="N", help="save frequency")
parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
parser.add_argument("--save_images", default="./gen_images", help="directory of generative samples")
parser.add_argument("--print_freq", type=int, default=10, help="frequency of print information")
parser.add_argument("--experiment_name", type=str, help="the name of experiment")
parser.add_argument("--int_pop", type=int, default=3, help="get Intermediate feature")

args = parser.parse_args()
    
if args.data == "CIFAR10" or args.data == "CIFAR100":
    img_size = 32
    img_shape = (3, 32, 32)
    nc = 3
else:
    img_size = 64
    img_shape = (3, 64, 64)
    nc = 3
    
if args.data == "CIFAR100":
    NUM_CLASSES = 100
elif args.data == "CIFAR10":
    NUM_CLASSES = 10
elif args.data == "tiny":
    NUM_CLASSES = 200
    
if args.data == "tiny":
    target_path = "/mnt/beegfs/home/zhengxiao/RoBen/models/tiny/Linf/preactresnet34.tar"
else:
    target_path = os.path.join(args.target_dir, "pretrained", args.data, f"{args.teacher_model}.pt")

exp_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
checkpoint_path = os.path.join(args.result_dir, args.data, args.experiment_name, exp_time, "checkpoints")
save_mem_dir = os.path.join(args.result_dir, args.data, args.experiment_name, exp_time, "runs_imgs")
if not os.path.exists(save_mem_dir):
    os.makedirs(save_mem_dir)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
logger = Logger(os.path.join(args.result_dir, args.data, args.experiment_name, exp_time, "output.log"))

if args.seed is not None:
    random_seed(args.seed)
    
# Standard Augmentation
std_aug = augmentation.container.ImageSequential(
    augmentation.RandomCrop(size=[img_shape[-2], img_shape[-1]], padding=4),
    augmentation.RandomHorizontalFlip(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### DataSet AT ###
def train_model_ds_at(args, cln_generator, student_model, teacher_model, optimizer_s, epoch):

    cln_generator.eval()
    teacher_model.eval()
    student_model.train()
        
    dataset = SyntheticDataset(root=save_mem_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data_iter = DataIter(data_loader)

    # train the student model
    for step in range(args.N_S):
        x_cln, labels = data_iter.next()
        x_cln, labels = x_cln.to(device), labels.to(device)
        labels = F.one_hot(labels, NUM_CLASSES).float()
        x_cln = std_aug(x_cln)
        
        with torch.no_grad(): 
            logits_t_cln = teacher_model(x_cln).detach()
        logits_s_cln = student_model(x_cln.detach())
        
        x_adv = generate_adv(student_model, x_cln, logits_t_cln, labels)
        logits_s_adv = student_model(x_adv.detach())
        
        loss_dkl = L.KT_loss_generator(logits_s_cln, logits_t_cln)
        loss_rob = L.KT_loss_generator(logits_s_adv, logits_t_cln)
        loss_ce = 0
        
        loss = 5.0/6*loss_dkl + 1.0/6*loss_rob

        optimizer_s.zero_grad()
        loss.backward()
        optimizer_s.step()
        
    #plot_tsne_d2(args, x_cln, x_adv, teacher_model)
        
    return loss, loss_dkl, loss_rob, loss_ce

### DataSet ###
def train_model_ds(args, cln_generator, student_model, teacher_model, optimizer_s, epoch):

    cln_generator.eval()
    teacher_model.eval()
    student_model.train()
    
    dataset = SyntheticDataset(root=save_mem_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data_iter = DataIter(data_loader)

    # train the student model
    for step in range(args.N_S):
        x_cln, labels = data_iter.next()
        x_cln, labels = x_cln.to(device), labels.to(device)
        x_cln = std_aug(x_cln)
        
        with torch.no_grad(): 
            t_cln_logits = teacher_model(x_cln).detach()
        s_cln_logits = student_model(x_cln.detach())
        
        loss_dkl = L.KT_loss_generator(s_cln_logits, t_cln_logits)
        loss = loss_dkl

        optimizer_s.zero_grad()
        loss.backward()
        optimizer_s.step()
        
    plot_tsne_d(args, x_cln, teacher_model)

    return loss, loss_dkl
    
        
def main():
    logger.info(args)
    
    if args.data == "CIFAR10" or args.data == "CIFAR100":
        testset = getattr(datasets, args.data)(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    else:
        data_root = "/mnt/beegfs/home/zhengxiao/data/tiny-imagenet-200"
        testset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    # get teacher model
    if args.teacher_model != "PreActResNet34":
        teacher_model = IFD_network_loader(args.teacher_model, num_classes=NUM_CLASSES, mean=0.0, std=1.0).to(device)
        state_dict = torch.load(target_path, map_location=device)
        teacher_model.load_state_dict(state_dict)
    else:
        teacher_model = IFD_network_loader(args.teacher_model, num_classes=NUM_CLASSES, mean=0.0, std=1.0).to(device)
        checkpoint = torch.load(target_path, map_location=device) 
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        teacher_model.load_state_dict(new_state_dict)
    
    teacher_acc = clean_test(teacher_model, test_loader)
    logger.info("Teacher Acc: %.4f", teacher_acc)
    
    # get student model
    student_model = getattr(models, args.student_model)(num_classes=NUM_CLASSES).to(device)
    student_model = nn.DataParallel(student_model).to(device)
    
    student_acc = clean_test(student_model, test_loader)
    logger.info("Student Acc: %.4f\n", student_acc)

    # get generator model
    cln_generator = models.Generator(nz=args.gen_dim_z, ngf=64, img_size=img_size, nc=nc)
    cln_generator = nn.DataParallel(cln_generator).to(device)

    # set student optimizer
    optimizer_s = optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=args.epochs)
            
    cln_net_helper = Train_Gen_Module(args, teacher_model, student_model, cln_generator)
    
    # start train
    best_stu1_acc = 0
    best_stu2_acc = 0
    loss_s=0
    class_max = 0.2
    class_min = 0.05
    labels_prob = torch.ones(NUM_CLASSES, dtype=torch.float).to(device)*(1/NUM_CLASSES)
    
    for epoch in range(1, args.epochs+1):
        
        ### Train generator
        best_mem, labels, loss_cln_G, loss_bn, loss_adv, loss_oh, loss_uni, labels_loss = cln_net_helper.train(labels_prob)
        save_mem(best_mem.data, labels, save_mem_dir, epoch, "cln")
        if epoch % args.print_freq == 0:
            save_batch_samples(best_mem, args.save_images, args.experiment_name, "cln")
            logger.info("Cln generator loss: %.4f loss_bn: %.4f loss_adv: %.4f loss_oh: %.4f loss_uni: %.4f" \
                %(loss_cln_G, loss_bn, loss_adv, loss_oh, loss_uni))
            
        ### Train student
        if epoch > args.warmup:
            loss, loss_dkl, loss_rob, loss_ce = train_model_ds_at(args, cln_generator, student_model, teacher_model, optimizer_s, epoch)
            #loss_s, loss_dkl = train_model_mem(args, best_mem, labels, student_model, teacher_model, optimizer_s, epoch)
            if epoch % args.print_freq == 0:
                logger.info("Student loss: %.4f loss_dkl: %.4f loss_rob: %.4f loss_ce: %.4f", loss, loss_dkl, loss_rob, loss_ce)
        
        scheduler_s.step()
        
        # Update labels_prob
        min_val = labels_loss.min()
        max_val = labels_loss.max()
        labels_prob = (labels_loss - min_val) / (max_val - min_val + 1e-8)
        labels_prob = labels_prob * (class_max - class_min) + class_min
        labels_prob = labels_prob / labels_prob.sum()
        print("labels_prob: ", labels_prob)
        print("labels: ", labels)
        
        if epoch % args.print_freq == 0:
            # Student Acc
            stu1_acc = clean_test(student_model, test_loader)
            logger.info("Epoch %d Finish, Student Acc %.4f\n" %(epoch, stu1_acc))
            
            if epoch >= 0:
                # Save generator
                save_checkpoint(
                    {  
                        "epoch": epoch,
                        "model_state_dict": cln_generator.state_dict(),
                    },
                    "gen",
                    epoch,
                    False,
                    "cln",
                    save_path=checkpoint_path,
                    save_freq=args.save_freq,
                )
            
                # Save student checkpoint
                is_best = stu1_acc > best_stu1_acc
                best_stu1_acc = max(stu1_acc, best_stu1_acc)
                save_checkpoint(
                    {  
                        "epoch": epoch,
                        "model_state_dict": student_model.state_dict(),
                        "optimizer": optimizer_s.state_dict(),
                        "nature_acc": float(stu1_acc),
                    },
                    "stu1",
                    epoch,
                    is_best,
                    "cln",
                    save_path=checkpoint_path,
                    save_freq=args.save_freq,
                )
            
    logger.info("Best Student ACC %.4f", best_stu1_acc)


if __name__ == "__main__":
    main()
