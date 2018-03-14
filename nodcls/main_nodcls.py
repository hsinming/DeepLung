#!/usr/bin/python2
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import transforms as transforms
from dataloader import lunanod
from dataloader import CROPSIZE
import pandas as pd
import os
import argparse
from sklearn.ensemble import GradientBoostingClassifier as gbt
import pickle
from models.dpn3d import DPN92_3D
from utils import progress_bar

import logging
import numpy as np


gbtdepth = 1
fold = 9   # the subset for test
blklst = []
logging.basicConfig(filename='log-'+str(fold), level=logging.INFO)
parser = argparse.ArgumentParser(description='Nodule Classifier Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--crop', default=False, type=bool, help='crop 3D bounding box from preprocess images')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

preprocesspath = '/data/LUNA16/preprocess_all_subsets/'
croppath = '/data/LUNA16/crop/'
anno_csv = './data/annotationdetclsconvfnl_v3.csv'


def crop(preprocesspath, croppath, anno_csv, cropsize):
    pdframe = pd.read_csv(anno_csv,
                          names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
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
        cropdatatmp = np.array(data[0, bgx:bgx + cropsize, bgy:bgy + cropsize, bgz:bgz + cropsize])
        cropdata[cropsize / 2 - cropdatatmp.shape[0] / 2:cropsize / 2 - cropdatatmp.shape[0] / 2 + cropdatatmp.shape[0], \
        cropsize / 2 - cropdatatmp.shape[1] / 2:cropsize / 2 - cropdatatmp.shape[1] / 2 + cropdatatmp.shape[1], \
        cropsize / 2 - cropdatatmp.shape[2] / 2:cropsize / 2 - cropdatatmp.shape[2] / 2 + cropdatatmp.shape[2]] = np.array(2 - cropdatatmp)
        assert cropdata.shape[0] == cropsize and cropdata.shape[1] == cropsize and cropdata.shape[2] == cropsize
        np.save(os.path.join(croppath, fname + '.npy'), cropdata)

if args.crop:
    crop(preprocesspath, croppath, anno_csv, CROPSIZE)

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
pixstd = np.sqrt(pixvlu / float(npix))
print('mean:{}'.format(pixmean))
print('std:{}'.format(pixstd))
logging.info('mean '+str(pixmean)+' std '+str(pixstd))


# Data transforms
logging.info('==> Preparing data..') # Random Crop, Zero out, x z flip, scale, 
transform_train = transforms.Compose([
    transforms.RandomCrop(CROPSIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)), # need to cal mean and std, revise norm func
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
tefnamelst = []
telabellst = []
tefeatlst = []


dataframe = pd.read_csv(anno_csv,
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]


# test id
teidlst = []
for fname in os.listdir('/data/LUNA16/subset'+str(fold)+'/'):
    if fname.endswith('.mhd'):
        teidlst.append(fname[:-4])

mxx = mxy = mxz = mxd = 0
for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid.split('-')[0] in blklst:
        continue

    # crop raw pixel as feature
    data = np.load(os.path.join(croppath, srsid + '.npy'))
    bgx = data.shape[0]/2-CROPSIZE/2
    bgy = data.shape[1]/2-CROPSIZE/2
    bgz = data.shape[2]/2-CROPSIZE/2
    data = np.array(data[bgx:bgx+CROPSIZE, bgy:bgy+CROPSIZE, bgz:bgz+CROPSIZE])
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


for idx in xrange(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in xrange(len(tefeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    tefeatlst[idx][-1] /= mxd



trainset = lunanod(croppath, trfnamelst, trlabellst, trfeatlst,
                   train=True, transform=transform_train, target_transform=None, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)

testset = lunanod(croppath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test,
                  target_transform=None, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)


# Model
savemodelpath = './checkpoint-'+str(fold)+'/'
if args.resume:
    # Load checkpoint.
    logging.info('==> Resuming from checkpoint..')
    checkpoint = torch.load(savemodelpath+'ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    logging.info('==> Building model..')
    net = DPN92_3D()

neptime = 2
def get_lr(epoch):
    if epoch < 150*neptime:
        lr = 0.1 #args.lr
    elif epoch < 250*neptime:
        lr = 0.01
    else:
        lr = 0.001
    return lr


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = False #True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



# Training
def train(epoch):
    logging.info('\nEpoch: '+str(epoch))
    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_loss = 0
    correct = 0
    total = 0
    trainfeat = np.zeros((len(trfnamelst), 2560+CROPSIZE*CROPSIZE*CROPSIZE+1))
    trainlabel = np.zeros((len(trfnamelst),))
    idx = 0
    for batch_idx, (inputs, targets, feat) in enumerate(trainloader):
        if use_cuda:
            # print(len(inputs), len(targets), len(feat), type(inputs[0]), type(targets[0]), type(feat[0]))
            # print(type(targets), type(inputs), len(targets))
            # targetarr = np.zeros((len(targets),))
            # for idx in xrange(len(targets)):
                # targetarr[idx] = targets[idx]
            # print((Variable(torch.from_numpy(targetarr)).data).cpu().numpy().shape)
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, dfeat = net(inputs) 
        # add feature into the array
        # print(torch.stack(targets).data.numpy().shape, torch.stack(feat).data.numpy().shape)
        # print((dfeat.data).cpu().numpy().shape)
        trainfeat[idx:idx+len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
        for i in xrange(len(targets)):
            trainfeat[idx+i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
            trainlabel[idx+i] = np.array((targets[i].data).cpu().numpy())
        idx += len(targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    m = gbt(max_depth=gbtdepth, random_state=0)
    m.fit(trainfeat, trainlabel)
    gbttracc = np.mean(m.predict(trainfeat) == trainlabel)
    print('ep '+str(epoch)+' tracc '+str(correct/float(total))+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    logging.info('ep '+str(epoch)+' tracc '+str(correct/float(total))+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    return m

def test(epoch, m):
    global best_acc
    global best_acc_gbt
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testfeat = np.zeros((len(tefnamelst), 2560+CROPSIZE*CROPSIZE*CROPSIZE+1))
    testlabel = np.zeros((len(tefnamelst),))
    idx = 0
    for batch_idx, (inputs, targets, feat) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, dfeat = net(inputs)
        # add feature into the array
        testfeat[idx:idx+len(targets), :2560] = np.array((dfeat.data).cpu().numpy())
        for i in xrange(len(targets)):
            testfeat[idx+i, 2560:] = np.array((Variable(feat[i]).data).cpu().numpy())
            testlabel[idx+i] = np.array((targets[i].data).cpu().numpy())
        idx += len(targets)

        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print(testlabel.shape, testfeat.shape, testlabel)#, trainfeat[:, 3])
    gbtteacc = np.mean(m.predict(testfeat) == testlabel)
    if gbtteacc > best_acc_gbt:
        pickle.dump(m, open('gbtmodel-'+str(fold)+'.sav', 'wb'))
        logging.info('Saving gbt ..')
        state = {
            'net': net.module if use_cuda else net,
            'epoch': epoch,
        }
        if not os.path.exists(savemodelpath):
            os.makedirs(savemodelpath)
        torch.save(state, savemodelpath+'ckptgbt.t7')
        best_acc_gbt = gbtteacc
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logging.info('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.exists(savemodelpath):
            os.makedirs(savemodelpath)

        # Save checkpoint of best_acc
        torch.save(state, savemodelpath+'ckpt.t7')
        best_acc = acc
    logging.info('Saving..')
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.exists(savemodelpath):
        os.makedirs(savemodelpath)
    if epoch % 50 == 0:
        torch.save(state, savemodelpath+'ckpt'+str(epoch)+'.t7')
    # best_acc = acc
    print('teacc '+str(acc)+' bestacc '+str(best_acc)+' gbttestaccgbt '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))
    logging.info('teacc '+str(acc)+' bestacc '+str(best_acc)+' ccgbt '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))

for epoch in range(start_epoch, start_epoch + 350*neptime):
    m = train(epoch)
    test(epoch, m)