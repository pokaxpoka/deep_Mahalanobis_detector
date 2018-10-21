"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
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
    if args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
    else:
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        
    # load networks
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    model.cuda()
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

    # measure the performance
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    base_line_list = []
    ODIN_best_tnr = [0, 0, 0]
    ODIN_best_results = [0 , 0, 0]
    ODIN_best_temperature = [-1, -1, -1]
    ODIN_best_magnitude = [-1, -1, -1]
    for T in T_list:
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, args.net_type, test_loader, magnitude, temperature, args.outf, True)
            out_count = 0
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude)) 
            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist) 
                lib_generation.get_posterior(model, args.net_type, out_test_loader, magnitude, temperature, args.outf, False)
                if temperature == 1 and magnitude == 0:
                    test_results = callog.metric(args.outf, ['PoT'])
                    base_line_list.append(test_results)
                else:
                    val_results = callog.metric(args.outf, ['PoV'])
                    if ODIN_best_tnr[out_count] < val_results['PoV']['TNR']:
                        ODIN_best_tnr[out_count] = val_results['PoV']['TNR']
                        ODIN_best_results[out_count] = callog.metric(args.outf, ['PoT'])
                        ODIN_best_temperature[out_count] = temperature
                        ODIN_best_magnitude[out_count] = magnitude
                out_count += 1
    
    # print the results
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('Baseline method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in base_line_list:
        print('out_distribution: '+ out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('')
        count_out += 1
        
    print('ODIN method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in ODIN_best_results:
        print('out_distribution: '+ out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('temperature: ' + str(ODIN_best_temperature[count_out]))
        print('magnitude: '+ str(ODIN_best_magnitude[count_out]))
        print('')
        count_out += 1

    
    
if __name__ == '__main__':
    main()