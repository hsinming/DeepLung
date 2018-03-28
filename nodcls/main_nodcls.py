#!/usr/bin/python2
from __future__ import print_function
import pandas as pd
import os
import argparse
from sklearn.ensemble import GradientBoostingClassifier as gbt
import cPickle as pickle
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
from models.dpn3d import DPN92_3D, config
from utils import progress_bar



gbtdepth = 2
fold = 0   # the subset for test
blklst = []
logging.basicConfig(filename='log-'+str(fold), level=logging.INFO)
parser = argparse.ArgumentParser(description='Nodule Classifier Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--crop', action='store_true', help='crop 3D bounding box from preprocess images')
parser.add_argument('--save_dir', '-s', default='./checkpoint-{}/'.format(fold), type=str, metavar='PATH',
                    help='directory to save checkpoint (default: ./checkpoint-{fold}/)')
parser.add_argument('--checkpoint', '--ckpt', default='ckpt.t7', type=str, metavar='FILENAME',
                    help='filename of the previous checkpoint (default: none)')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
best_acc_gbt = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

preprocesspath = '/data/preprocess/luna_preprocess/'
croppath = './crop/'
anno_csv = './data/annotationdetclsconvfnl_v3.csv'

# Crop the preprocess image to make datasets
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

def crop2(preprocesspath, croppath, anno_csv, cropsize):
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

if args.crop:
    try:
        crop2(preprocesspath, croppath, anno_csv, CROPSIZE)
        print('Crop is done.')
    except:
        pass


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
                        names=['seriesuid', 'coordZ', 'coordY', 'coordX', 'diameter_mm', 'malignant'])
alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]


# test id
print('Using subset{} as test split.'.format(fold))
teidlst = []
for fname in os.listdir('/data/LUNA16/subset'+str(fold)+'/'):
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)

testset = lunanod(croppath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test,
                  target_transform=None, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)


# Model
savemodelpath = args.save_dir
if not os.path.exists(savemodelpath):
    os.makedirs(savemodelpath)

net = DPN92_3D()

if args.checkpoint:
    # Load checkpoint.
    logging.info('==> Resuming from {}'.format(args.checkpoint))
    checkpoint = torch.load(os.path.join(savemodelpath, args.checkpoint))
    net.load_state_dict(checkpoint['state_dict'])
    m = pickle.load(open(os.path.join(savemodelpath,'gbtmodel.sav'), 'rb'))
    best_acc = checkpoint['best_acc']
    best_acc_gbt = checkpoint['best_acc_gbt']
    start_epoch = checkpoint['epoch'] + 1
else:
    logging.info('==> Building model..')
    m = gbt(max_depth=gbtdepth, random_state=0)


neptime = 2
def get_lr(epoch):
    if epoch < 150*neptime:
        lr = 0.1 #args.lr
    elif epoch < 250*neptime:
        lr = 0.01
    else:
        lr = 0.001
    return lr

# Define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    criterion = criterion.cuda()
    cudnn.benchmark = False #True

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
        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    m.fit(trainfeat, trainlabel)
    gbttracc = 100.*np.mean(m.predict(trainfeat) == trainlabel)
    print('ep '+str(epoch)+' tracc '+str(correct/float(total)*100.)+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    logging.info('ep '+str(epoch)+' tracc '+str(correct/float(total)*100.)+' lr '+str(lr)+' gbtacc '+str(gbttracc))
    return m

#Validation
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
        progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print(testlabel.shape, testfeat.shape, testlabel)#, trainfeat[:, 3])
    gbtteacc = 100.*np.mean(m.predict(testfeat) == testlabel)
    if gbtteacc > best_acc_gbt:
        pickle.dump(m, open(os.path.join(savemodelpath,'gbtmodel.sav'.format(fold)), 'wb'))
        logging.info('Saving gbt ..')
        state_dict = net.module.state_dict()
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        state = {'epoch': epoch,
                 'save_dir': savemodelpath,
                 'state_dict': state_dict,
                 'args': args,
                 'lr': get_lr(epoch),
                 'best_acc': best_acc,
                 'best_acc_gbt': best_acc_gbt}
        torch.save(state, os.path.join(savemodelpath, 'ckptgbt.t7'))
        best_acc_gbt = gbtteacc

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
                 'best_acc': best_acc,
                 'best_acc_gbt': best_acc_gbt}
        torch.save(state, os.path.join(savemodelpath, 'ckpt.t7'))
        best_acc = acc


    # Save every 50 epochs
    logging.info('Saving..')
    state_dict = net.module.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    state = {'epoch': epoch,
             'save_dir': savemodelpath,
             'state_dict': state_dict,
             'args': args,
             'lr': get_lr(epoch),
             'best_acc': best_acc,
             'best_acc_gbt': best_acc_gbt}
    if epoch % 50 == 0:
        torch.save(state, savemodelpath+'ckpt'+str(epoch)+'.t7')

    # Show and log metrics
    print('teacc '+str(acc)+' bestacc '+str(best_acc)+' gbtteacc '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))
    print()
    logging.info('teacc '+str(acc)+' bestacc '+str(best_acc)+' gbtteacc '+str(gbtteacc)+' bestgbt '+str(best_acc_gbt))


for epoch in range(start_epoch, start_epoch + 350*neptime):
    m = train(epoch)
    test(epoch, m)
