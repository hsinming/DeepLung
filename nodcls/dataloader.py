from __future__ import print_function
import os
import os.path
import numpy as np
from scipy.ndimage import zoom
import torch.utils.data as data
import warnings
CROPSIZE = 32

class lunanod(data.Dataset):
    def __init__(self, npypath, fnamelst, labellst, featlst, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                file = os.path.join(npypath, fentry)
                self.train_data.append(np.load(file))
                self.train_labels.append(label)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((len(fnamelst), CROPSIZE, CROPSIZE, CROPSIZE))
            self.train_len = len(fnamelst)
            print('train_data:', self.train_data.shape, '\ntrain_labels:', len(self.train_labels), '\ntrain_feat:',
                  len(self.train_feat))

        else:
            self.test_data = []
            self.test_labels = []
            self.test_feat = featlst
            for label, fentry in zip(labellst, fnamelst):
                if isinstance(fentry, np.ndarray):   # fentry is a numpy array
                    self.test_data.append(fentry)
                    self.test_labels.append(label)
                else:
                    file = os.path.join(npypath, fentry)
                    self.test_data.append(np.load(file))
                    self.test_labels.append(label)
            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape((len(fnamelst), CROPSIZE, CROPSIZE, CROPSIZE))
            self.test_len = len(fnamelst)
            print('test_data:', self.test_data.shape, '\ntest_labels:', len(self.test_labels), '\ntest_feat:', len(self.test_feat))

    def __getitem__(self, index):
        if self.train:
            img, target, feat = self.train_data[index], self.train_labels[index], self.train_feat[index]
        else:
            img, target, feat = self.test_data[index], self.test_labels[index], self.test_feat[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.shape, target.shape, feat.shape)
        # print(target)

        return img, target, feat

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len


class simpleCrop():
    def __init__(self, config, phase):
        self.crop_size = config['crop_size']
        self.scaleLim = config['scaleLim']
        self.radiusLim = config['radiusLim']
        self.jitter_range = config['jitter_range']
        self.isScale = config['augtype']['scale'] and phase == 'train'
        self.stride = config['stride']
        self.filling_value = config['filling_value']
        self.phase = phase

    def __call__(self, imgs, target):
        if self.isScale:
            radiusLim = self.radiusLim
            scaleLim = self.scaleLim
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = np.array(self.crop_size).astype('int')
        if self.phase == 'train':
            jitter_range = target[3] * self.jitter_range
            jitter = (np.random.rand(3) - 0.5) * jitter_range
        else:
            jitter = 0
        start = (target[:3] - crop_size / 2 + jitter).astype('int')
        pad = [[0, 0]]
        for i in range(3):
            if start[i] < 0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i] + crop_size[i] > imgs.shape[i + 1]:
                rightpad = start[i] + crop_size[i] - imgs.shape[i + 1]
            else:
                rightpad = 0
            pad.append([leftpad, rightpad])
        imgs = np.pad(imgs, pad, 'constant', constant_values=self.filling_value)
        crop = imgs[:, start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
               start[2]:start[2] + crop_size[2]]

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        if self.isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.filling_value)

        return crop, coord
