from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math


from cifar import CIFAR100, MetaCIFAR100



import collections
import numpy as np
import scipy
from scipy.stats import t
import tqdm

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import statistics
import time

from resnet import resnet12
import torchvision.transforms as transforms
from PIL import Image


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss






def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []

    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(testloader))
        for idx, data in pbar:
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)


            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            pbar.set_description("Acg acc {:.4f}".format(statistics.mean(acc)))

    return mean_confidence_interval(acc)



def validate(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg





def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apply_2d_rotation(input_tensor1, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.

    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor1.dim() >= 2
    input_tensor = input_tensor1.clone()

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )




def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot





def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='45,60,75', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12')
    parser.add_argument('--dataset', type=str, default='FC100')


    parser.add_argument('--use_trainval', action='store_true', help='using trainval')

    # distillation
    parser.add_argument('--distill', type=str, default='kd')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-b', '--beta', type=float, default=0.5, help='weight balance for KD')
    parser.add_argument('-l', '--lam', type=float, default=1.0, help='weight balance for SS losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')



    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # setting for meta-learning
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--graftversion', type=int, default=0)

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    
    opt.trial = '_'
    if opt.use_trainval:
        opt.trial = '_trainval'
    

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_distilled'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    opt.model_name = 'S:{}_{}_{}_l:{}'.format(opt.model_s, opt.dataset, opt.distill, opt.lam)



    opt.model_name = '{}{}_graft_{}'.format(opt.model_name, opt.trial, opt.graftversion)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def entropy(x,n=10):
    x=x.reshape(-1)
    scale=(x.max()-x.min())/n
    entropy=0
    for i in range(n):
        p=torch.sum((x>=x.min()+i*scale)*(x<x.min()+(i+1)*scale),dtype=torch.float)/len(x)
        if p!=0:
            entropy-=p*torch.log(p)
    return entropy

def grafting(module_list,opt,epoch):
    other_folder = opt.save_folder[:len(opt.save_folder)-1] + str((opt.graftversion+1)%2)
    graft_path = '{}/ckpt_graft_{graft}_epoch_{epoch}.pth'.format(other_folder,graft=(opt.graftversion+1)%2,epoch=epoch)
    print('grafting {} epoch {} from {}',opt.graftversion,epoch,graft_path)
    while not os.access(path=graft_path, mode=os.R_OK):
        time.sleep(100)
    try:
        checkpoint = torch.load(graft_path)['model']
    except:
        time.sleep(100)
        checkpoint = torch.load(graft_path)['model']

    
    net = module_list[0]
    #model_t = module_list[-1]
    #checkpoint = model_t.state_dict()

    model=collections.OrderedDict()
    for i,(key,u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            w=round(0.4*(np.arctan(500*((float(entropy(u).cpu())-float(entropy(checkpoint[key]).cpu())))))/np.pi+1/2,2)
        model[key]=u*w+checkpoint[key]*(1-w)
    net.load_state_dict(model)



class Rot_Block(nn.Module):
    def __init__(self,dim=640):
        super(Rot_Block, self).__init__()
        self.conv_rot = nn.Conv2d(dim,dim,3,1,1)
        self.bn_rot = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.relu(self.bn_rot(self.conv_rot(x)))


def main():

    best_acc = 0

    opt = parse_option()


    train_partition = 'trainval' if opt.use_trainval else 'train'

    if opt.dataset == 'FC100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
        transform_fc100 = [
            transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize_cifar100
            ]),

            transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                normalize_cifar100
            ])
        ]
        train_trans, test_trans = transform_fc100

        train_set = CIFAR100(args=opt, partition=train_partition, transform=train_trans)
        n_data = len(train_set)
        train_loader = DataLoader(train_set,
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 60
    else:
        raise NotImplementedError(opt.dataset)

    # model

    model_s = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls)

    model_s.rotm = nn.Sequential(Rot_Block(640),Rot_Block(640),Rot_Block(640),Rot_Block(640))
    
    model_s.rotm_avgpool = nn.AdaptiveAvgPool2d(1)
    model_s.rotm_class = nn.Linear(640,4)

    #model_s = nn.DataParallel(model_s.cuda())

    data = torch.randn(2, 3, 32, 32)



    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss


    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)


    #module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True


    #teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    #print('teacher accuracy: ', teacher_acc)



    for epoch in tqdm.tqdm(range(1, opt.epochs + 1)):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, val_loader)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        print('epoch {} train loss {:.4f} train acc {:.4f}'.format(epoch,train_loss,train_acc))
        fp=open(opt.model_path+'log_'+str(opt.graftversion)+'.txt','a+')
        fp.write('epoch {} train loss {:.4f} train acc {:.4f}\n'.format(epoch,train_loss,train_acc))
        fp.close()
        test_acc, test_acc_top5, test_loss = validate(val_loader, module_list[0] , criterion_cls, opt) #model_s
        print('epoch {} test loss {:.4f} test acc {:.4f} test acc top 5 {:.4f}'.format(epoch,test_loss,test_acc,test_acc_top5))

        fp=open(opt.model_path+'log_'+str(opt.graftversion)+'.txt','a+')
        fp.write('epoch {} test loss {:.4f} test acc {:.4f} test acc top 5 {:.4f}\n'.format(epoch,test_loss,test_acc,test_acc_top5))
        fp.close()


        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_graft_{graft}_epoch_{epoch}.pth'.format(graft=opt.graftversion,epoch=epoch))
            torch.save(state, save_file)


        grafting(module_list,opt,epoch)

    # save the last model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_graft_{graft}_last.pth'.format(opt.model_s,graft=opt.graftversion))
    torch.save(state, save_file)

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))


def train(epoch, train_loader, module_list, criterion_list, optimizer, opt, val_loader):


    for module in module_list:
        module.train()

    #module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    #model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in tqdm.tqdm(enumerate(train_loader)):
        if opt.distill in ['contrast']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data

        rot_img = create_4rotations_images(input)
        labels_rotation = create_rotations_labels(len(input), input.device)
        rot_img_size =  rot_img.size(0)

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            #input = input.cuda()
            target = target.cuda()
            labels_rotation = labels_rotation.cuda()
            index = index.cuda()
            if opt.distill in ['contrast']:
                contrast_idx = contrast_idx.cuda()

        inps = rot_img[:input.size(0)]


        preact = False
        if opt.distill in ['abound', 'overhaul']:
            preact = True
        _, logit_s = model_s(inps.cuda(), is_feat=True)

        lt_s,_ = model_s(rot_img.cuda(),is_feat=True)
        logits_z_s=model_s.rotation(lt_s[3])

        #with torch.no_grad():
        #    feat_t, logit_t = model_t(input, is_feat=True)
        #    feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target) + opt.lam * criterion_cls(logits_z_s, labels_rotation)
        #loss_div = criterion_div(logit_s, logit_t)


        loss = loss_cls#opt.beta * loss_cls + opt.gamma * loss_div

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()


        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
