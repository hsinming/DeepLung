#!/usr/bin/python2
#coding=utf-8
import torch
from torch.autograd import Variable
from models.capsnet import CapsNet3D, CapsNetWithReconstruction3D, ReconstructionNet3D
from models.capsnet import CapsNetDeep
import sys

def test_capsnetdeep():
    net = CapsNetDeep(routing_iterations=3, n_classes=2)
    x = Variable(torch.randn(1, 1, 28, 28))
    out, prob = net(x)

    print('out', out)
    print('prob', prob)

def test_capsnet3d():
    net = CapsNet3D(routing_iterations=3, n_classes=10)
    x = Variable(torch.randn(1, 1, 28, 28, 28))
    out, prob = net(x)

    print('out', out)
    print('prob', prob)

def test_capsnet3drecon():
    caps3d = CapsNet3D(routing_iterations=3, n_classes=2)
    reconstruction_model = ReconstructionNet3D(n_dim=16, n_classes=2)
    net = CapsNetWithReconstruction3D(capsnet=caps3d, reconstruction_net=reconstruction_model)
    input = Variable(torch.randn(1, 1, 32, 32, 32))
    target = Variable(torch.LongTensor([1]))
    recon, prob = net(input, target)

    print('recon', recon)
    print('prob', prob)

if __name__ == '__main__':
    status = test_capsnet3d()
    sys.exit(status)

