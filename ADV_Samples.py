"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import os
import lib.adversary as adversary

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
    if args.adv_type == 'FGSM':
        adv_noise = 0.05
    elif args.adv_type == 'BIM':
        adv_noise = 0.01
    elif args.adv_type == 'DeepFool':
        if args.net_type == 'resnet':
            if args.dataset == 'cifar10':
                adv_noise = 0.18
            elif args.dataset == 'cifar100':
                adv_noise = 0.03
            else:
                adv_noise = 0.1
        else:
            if args.dataset == 'cifar10':
                adv_noise = 0.6
            elif args.dataset == 'cifar100':
                adv_noise = 0.1
            else:
                adv_noise = 0.5

    # load networks
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((125.3/255, 123.0/255, 113.9/255), \
                                                                (63.0/255, 62.1/255.0, 66.7/255.0)),])
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584
        if args.dataset == 'cifar10':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 10
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.03 / 2
        elif args.dataset == 'cifar100':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 8
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 8
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 * 2 / 8
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.06 / 5
        else:
            if args.adv_type == 'FGSM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.21 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.16 * 2 / 5
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.07 / 2
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), \
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        min_pixel = -2.42906570435
        max_pixel = 2.75373125076
        if args.dataset == 'cifar10':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        elif args.dataset == 'cifar100':
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 8
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 4
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.13 / 4
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 2
        else:
            if args.adv_type == 'FGSM':
                random_noise_size = 0.25 / 4
            elif args.adv_type == 'BIM':
                random_noise_size = 0.13 / 2
            elif args.adv_type == 'DeepFool':
                random_noise_size = 0.126
            elif args.adv_type == 'CWL2':
                random_noise_size = 0.05 / 1 
            
    model.cuda()
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    
    print('Attack: ' + args.adv_type  +  ', Dist: ' + args.dataset + '\n')
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0
    
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda()) 
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            
        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        if args.adv_type == 'FGSM': 
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            if args.net_type == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        elif args.adv_type == 'BIM': 
            gradient = torch.sign(inputs.grad.data)
            for k in range(5):
                inputs = torch.add(inputs.data, adv_noise, gradient)
                inputs = torch.clamp(inputs, min_pixel, max_pixel)
                inputs = Variable(inputs, requires_grad=True)
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                gradient = torch.sign(inputs.grad.data)
                if args.net_type == 'densenet':
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        if args.adv_type == 'DeepFool':
            _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                             args.num_classes, step_size=adv_noise, train_mode=False)
            adv_data = adv_data.cuda()
        elif args.adv_type == 'CWL2':
            _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, 'l2', crop_frac=1.0)
        else:
            adv_data = torch.add(inputs.data, adv_noise, gradient)
            
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
        
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)


        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        output = model(Variable(adv_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        
        output = model(Variable(noisy_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    
if __name__ == '__main__':
    main()
