#!/usr/bin/python2
from __future__ import print_function, division
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec



cropfnlist = glob('./crop/*.npy')
cropfnlist = sorted(cropfnlist)
prep_root = '/data/preprocess/luna_preprocess/'
save_crop_root = './crop_pic/'

if not os.path.exists(save_crop_root):
    os.makedirs(save_crop_root)

for idx, fn in enumerate(cropfnlist):
    crop_img = np.load(fn)
    crop_size = crop_img.shape[0]
    nodule_id = os.path.basename(fn).split('.npy')[0]
    print('crop shape: ', crop_img.shape)

    plt.close('all')
    fig = plt.figure(figsize=(6, 6), dpi=96)
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.10, right=0.90, top=0.95, bottom=0.05)
    ax1 = plt.subplot(gs[0])
    plt.axis('off')
    ax1.set_title(nodule_id, fontsize=10)

    # draw in voxel coordinates
    ax1.imshow(np.mean(crop_img[int(crop_size*0.5)-1:int(crop_size*0.5)+1, :, :], axis=0), cmap='gray', interpolation='catrom')

    # Show or Save the picture
    plt.savefig(os.path.join(save_crop_root, '{}.jpg'.format(nodule_id)))
    #plt.show()