from __future__ import print_function
import os
import os.path
import numpy as np
import torch.utils.data as data
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
                if not isinstance(fentry, str):   # fentry is a numpy array
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