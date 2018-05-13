#!/usr/bin/python2
# coding=utf-8
from __future__ import print_function
import pandas as pd
import os
import sys
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import transforms as transforms
from dataloader import lunanod
from dataloader import CROPSIZE
from dataloader import simpleCrop
from models.capsnet import CapsNet3D, MarginLoss, config
from utils import progress_bar

blklst = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.111258527162678142285870245028',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.250397690690072950000431855143',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.295420274214095686326263147663',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.776800177074349870648765614630',
          '1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311']
fold = 0   # the subset for test
neptime = 2
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
logging.basicConfig(filename='log-'+str(fold), level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Nodule Classifier Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--crop', action='store_true', help='crop 3D bounding box from preprocess images')
    parser.add_argument('--save_dir', '-s', default='./checkpoint-{}/'.format(fold), type=str, metavar='PATH',
                        help='directory to save checkpoint (default: ./checkpoint-{fold}/)')
    parser.add_argument('--checkpoint', '--ckpt', default='', type=str, metavar='FILENAME',
                        help='filename of the previous checkpoint (default: none)')
    args = parser.parse_args()
    return args

def crop1(preprocesspath, croppath, anno_csv, cropsize):
    pdframe = pd.read_csv(anno_csv,
                          names=['seriesuid', 'coordZ', 'coordY', 'coordX', 'diameter_mm', 'malignant'])
    srslst = pdframe['seriesuid'].tolist()[1:]
    crdxlst = pdframe['coordX'].tolist()[1:]
    crdylst = pdframe['coordY'].tolist()[1:]
    crdzlst = pdframe['coordZ'].tolist()[1:]
    dimlst = pdframe['diameter_mm'].tolist()[1:]
    mlglst = pdframe['malignant'].tolist()[1:]

    annoList = []
    for srs, crdx, crdy, crdz, dim, mlg in zip(srslst, crdxlst, crdylst, crdzlst, dimlst, mlglst):
        annoList.append([srs, crdx, crdy, crdz, dim, mlg])

    if not os.path.exists(croppath):
        os.makedirs(croppath)

    for idx, anno in enumerate(annoList):
        fname = anno[0]
        pid = fname.split('-')[0]
        crdx = int(float(anno[1]))
        crdy = int(float(anno[2]))
        crdz = int(float(anno[3]))
        dim = int(float(anno[4]))
        data = np.load(os.path.join(preprocesspath, pid + '_clean.npy'))
        bgx = max(0, crdx - cropsize / 2)
        bgy = max(0, crdy - cropsize / 2)
        bgz = max(0, crdz - cropsize / 2)
        cropdata = np.ones((cropsize, cropsize, cropsize)) * 170
        cropdatatmp = np.array(data[0, bgz:min(data.shape[1], bgz+cropsize), bgy:min(data.shape[2], bgy+cropsize), bgx:min(data.shape[3], bgx+cropsize)])
        cropdata[:cropdatatmp.shape[0], :cropdatatmp.shape[1], :cropdatatmp.shape[2]] = np.array(cropdatatmp)
        assert cropdata.shape == (cropsize, cropsize, cropsize)
        np.save(os.path.join(croppath, fname + '.npy'), cropdata)

def crop2(preprocesspath, croppath, anno_csv, config):
    pdframe = pd.read_csv(anno_csv,
                          names=['seriesuid', 'coordZ', 'coordY', 'coordX', 'diameter_mm', 'malignant'],
                          header=0)
    srslst = pdframe['seriesuid'].tolist()
    crdxlst = pdframe['coordX'].tolist()
    crdylst = pdframe['coordY'].tolist()
    crdzlst = pdframe['coordZ'].tolist()
    dimlst = pdframe['diameter_mm'].tolist()
    mlglst = pdframe['malignant'].tolist()

    annoList = []
    for srs, crdx, crdy, crdz, dim, mlg in zip(srslst, crdxlst, crdylst, crdzlst, dimlst, mlglst):
        annoList.append([srs, crdx, crdy, crdz, dim, mlg])

    if not os.path.exists(croppath):
        os.makedirs(croppath)

    cropper = simpleCrop(config, 'test')

    for idx, anno in enumerate(annoList):
        fname = anno[0]
        pid = fname.split('-')[0]
        crdx = int(float(anno[1]))
        crdy = int(float(anno[2]))
        crdz = int(float(anno[3]))
        dim = int(float(anno[4]))
        target = np.array([crdz, crdy, crdx, dim])
        data = np.load(os.path.join(preprocesspath, pid + '_clean.npy'))
        crop_img, _ = cropper(data, target)
        crop_img = crop_img[0,...]
        crop_img = crop_img.astype(np.float32)
        np.save(os.path.join(croppath, fname + '.npy'), crop_img)
        progress_bar(idx, len(annoList), 'Crop nodule: {}/{}'.format(idx, len(annoList)))
    print('Crop done.')

def get_mean_and_std(croppath, blklst):
    # Calculate mean std
    pixvlu, npix = 0, 0
    for fname in os.listdir(croppath):
        if fname.endswith('.npy'):
            if fname[:-4] in blklst:
                continue
            data = np.load(os.path.join(croppath, fname))
            pixvlu += np.sum(data)
            npix += np.prod(data.shape)
    pixmean = pixvlu / float(npix)

    pixvlu = 0
    for fname in os.listdir(croppath):
        if fname.endswith('.npy'):
            if fname[:-4] in blklst:
                continue
            data = np.load(os.path.join(croppath, fname))-pixmean
            pixvlu += np.sum(data * data)
    pixstd = np.sqrt(pixvlu / float(npix-1))
    print('mean:{}'.format(pixmean))
    print('std:{}'.format(pixstd))
    logging.info('mean '+str(pixmean)+' std '+str(pixstd))
    return pixmean, pixstd

def get_lr(epoch):
    if epoch < 150*neptime:
        lr = 0.1 #args.lr
    elif epoch < 250*neptime:
        lr = 0.01
    else:
        lr = 0.001
    return lr

def get_train_test_loader(anno_csv, croppath, testfold, transform_train, transform_test):
    trfnamelst = []
    trlabellst = []
    trfeatlst = []
    tefnamelst = []
    telabellst = []
    tefeatlst = []

    dataframe = pd.read_csv(anno_csv,
                            names=['seriesuid', 'coordZ', 'coordY', 'coordX', 'diameter_mm', 'malignant'], header=0)
    alllst = dataframe['seriesuid'].tolist()
    labellst = dataframe['malignant'].tolist()
    crdxlst = dataframe['coordX'].tolist()
    crdylst = dataframe['coordY'].tolist()
    crdzlst = dataframe['coordZ'].tolist()
    dimlst = dataframe['diameter_mm'].tolist()

    # Make a test dataset
    print('Using subset{} as test split.'.format(testfold))
    teidlst = []
    for fname in os.listdir('/data/LUNA16/subset'+str(testfold)+'/'):
        if fname.endswith('.mhd'):
            teidlst.append(fname[:-4])

    for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
        if srsid.split('-')[0] in blklst:
            continue

        # crop raw pixel as feature
        data = np.load(os.path.join(croppath, srsid + '.npy'))
        bgx = data.shape[2]/2-CROPSIZE/2
        bgy = data.shape[1]/2-CROPSIZE/2
        bgz = data.shape[0]/2-CROPSIZE/2
        assert bgx==bgy==bgz==0
        data = np.array(data[bgz:bgz+CROPSIZE, bgy:bgy+CROPSIZE, bgx:bgx+CROPSIZE])
        feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))
        # print(feat.shape)
        if srsid.split('-')[0] in teidlst:
            tefnamelst.append(srsid + '.npy')
            telabellst.append(int(label))
            tefeatlst.append(feat)
        else:
            trfnamelst.append(srsid + '.npy')
            trlabellst.append(int(label))
            trfeatlst.append(feat)


    trainset = lunanod(croppath, trfnamelst, trlabellst, trfeatlst,
                       train=True, transform=transform_train, target_transform=None, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)

    testset = lunanod(croppath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test,
                      target_transform=None, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    return trainloader, testloader

def get_transform(pixmean, pixstd):
    # Data transforms
    logging.info('==> Preparing data..')  # Random Crop, Zero out, flip, scale,
    transform_train = transforms.Compose([
        transforms.RandomCrop(CROPSIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((pixmean), (pixstd)),
    ])
    return transform_train, transform_test

def train(net, criterion, optimizer, trainloader, epoch):
    logging.info('\nEpoch: '+str(epoch))
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_loss = 0
    correct = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, probs = net(inputs)
        idx += len(targets)
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(probs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('ep '+str(epoch)+' tracc '+str(correct/float(total)*100.)+' lr '+str(lr))
    logging.info('ep '+str(epoch)+' tracc '+str(correct/float(total)*100.)+' lr '+str(lr))

def test(net, criterion, testloader, epoch, args, savemodelpath):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0

    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs, probs = net(inputs)
        idx += len(targets)
        loss = criterion(probs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(probs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint of best_acc
    acc = 100.*correct/total
    if acc > best_acc:
        logging.info('Saving..')
        state_dict = net.module.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        state = {'epoch': epoch,
                 'save_dir': savemodelpath,
                 'state_dict': state_dict,
                 'args': args,
                 'lr': get_lr(epoch),
                 'best_acc': best_acc}
        torch.save(state, os.path.join(savemodelpath, 'ckpt.t7'))
        best_acc = acc


    # Save every 50 epochs
    state_dict = net.module.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    state = {'epoch': epoch,
             'save_dir': savemodelpath,
             'state_dict': state_dict,
             'args': args,
             'lr': get_lr(epoch),
             'best_acc': best_acc}
    if epoch % 50 == 0:
        logging.info('Saving..')
        torch.save(state, savemodelpath+'ckpt'+str(epoch)+'.t7')

    # Show and log metrics
    print('teacc '+str(acc)+' bestacc '+str(best_acc))
    print()
    logging.info('teacc '+str(acc)+' bestacc '+str(best_acc))

def main():
    args = parse_args()
    preprocesspath = '/data/preprocess/luna_preprocess/'
    croppath = './crop/'
    anno_csv = './data/annotationdetclsconvfnl_v3.csv'
    global best_acc, start_epoch, fold

    if args.crop:
        crop2(preprocesspath, croppath, anno_csv, config)
        return

    # prepare Model
    net = CapsNet3D(routing_iterations=3, n_classes=2)
    savemodelpath = args.save_dir
    if not os.path.exists(savemodelpath):
        os.makedirs(savemodelpath)

    if args.checkpoint:
        # Load checkpoint.
        logging.info('==> Resuming from {}'.format(args.checkpoint))
        checkpoint = torch.load(os.path.join(savemodelpath, args.checkpoint))
        net.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch'] + 1
    else:
        logging.info('==> Building model..')

    # Prepare train and test data splits
    pixmean, pixstd = get_mean_and_std(croppath, blklst)
    transform_train, transform_test = get_transform(pixmean, pixstd)
    trainloader, testloadter = get_train_test_loader(anno_csv, croppath, fold, transform_train, transform_test)

    # Define loss function (criterion) and optimizer
    criterion = MarginLoss(0.9, 0.1, 0.5)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if use_cuda:
        net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))  # use all GPU
        net = torch.nn.DataParallel(net, device_ids=[0])  # use all GPU
        criterion = criterion.cuda()
        cudnn.benchmark = False

    # Train and test(validate)
    for epoch in range(start_epoch, start_epoch + 350*neptime):
        train(net, criterion, optimizer, trainloader, epoch)
        test(net, criterion, testloadter, epoch, args, savemodelpath)

if __name__ == '__main__':
    status = main()
    sys.exit(status)
